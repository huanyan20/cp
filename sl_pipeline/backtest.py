"""Daily SL backtest with T+1 execution and trading_env-compatible costs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from metrics_utils import calculate_metrics
from research_pipeline import write_metrics_json
from settings import load_settings
from sl_pipeline.allocator import MarketContext, PortfolioAllocator, PortfolioState
from trading_env import COMMISSION_RATE, SLIPPAGE_RATE, TAX_RATE_SELL, SLIPPAGE_MULTIPLIER

if TYPE_CHECKING:
    from settings import AppSettings

TRADING_DAYS_PER_YEAR = 252


@dataclass
class BacktestConfig:
    vol_window: int = 20
    min_vol_obs: int = 5
    vol_floor: float = 0.05
    initial_value: float = 1.0


def rolling_annualized_vol(log_returns: pd.Series, window: int, min_obs: int, vol_floor: float) -> float:
    """Annualized volatility from trailing log returns."""
    recent = pd.to_numeric(log_returns, errors="coerce").dropna().tail(window)
    if len(recent) < min_obs:
        return vol_floor
    return max(float(recent.std() * np.sqrt(TRADING_DAYS_PER_YEAR)), vol_floor)


def build_trading_calendar(
    enriched: dict[str, pd.DataFrame],
    scores: dict[str, pd.Series],
    *,
    test_start: str,
    test_end: str,
) -> list[pd.Timestamp]:
    """Intersection of OOS score dates and price data in [test_start, test_end]."""
    start = pd.Timestamp(test_start)
    end = pd.Timestamp(test_end)
    date_sets: list[set[pd.Timestamp]] = []
    for ticker, series in scores.items():
        if ticker not in enriched:
            continue
        idx = pd.to_datetime(series.dropna().index)
        mask = (idx >= start) & (idx <= end)
        if mask.any():
            date_sets.append(set(idx[mask]))
    if not date_sets:
        return []
    common = set.intersection(*date_sets)
    return sorted(common)


def trade_cost_rate(prev_weight: float, target_weight: float) -> float:
    """One-way cost rate for a weight change (commission + slippage + sell tax)."""
    delta = abs(target_weight - prev_weight)
    if delta < 1e-8:
        return 0.0
    rate = COMMISSION_RATE + SLIPPAGE_RATE + SLIPPAGE_MULTIPLIER * (delta ** 2)
    if target_weight < prev_weight:
        rate += TAX_RATE_SELL
    return rate


def execute_rebalance(
    portfolio_value: float,
    prev_weights: dict[str, float],
    target_weights: dict[str, float],
    tickers: list[str],
    open_returns: dict[str, float],
) -> tuple[dict[str, float], float, float]:
    """Apply target weights, deduct friction, return (weights, turnover, new_value)."""
    turnover = 0.0
    total_cost = 0.0
    final_weights: dict[str, float] = {}

    for ticker in tickers:
        prev = float(prev_weights.get(ticker, 0.0))
        target = float(target_weights.get(ticker, 0.0))
        
        # Apply Limit Up/Down Blocking at Open
        open_r = open_returns.get(ticker, 0.0)
        if np.isfinite(open_r):
            if open_r >= 0.095 and target > prev:
                target = prev  # Block buying if limit up
            elif open_r <= -0.095 and target < prev:
                target = prev  # Block selling if limit down

        delta = abs(target - prev)
        turnover += delta
        if delta > 1e-8:
            trade_amount = delta * portfolio_value
            total_cost += trade_amount * trade_cost_rate(prev, target)
        final_weights[ticker] = target

    new_value = max(portfolio_value - total_cost, 1e-12)
    return final_weights, turnover, new_value


def build_vols_as_of(
    enriched: dict[str, pd.DataFrame],
    tickers: list[str],
    date: pd.Timestamp,
    *,
    vol_window: int,
    min_vol_obs: int,
    vol_floor: float,
) -> dict[str, float]:
    vols: dict[str, float] = {}
    for ticker in tickers:
        if ticker not in enriched:
            vols[ticker] = vol_floor
            continue
        df = enriched[ticker]
        hist = df.loc[df.index <= date, "log_return"]
        vols[ticker] = rolling_annualized_vol(hist, vol_window, min_vol_obs, vol_floor)
    return vols


def simulate_period(
    enriched: dict[str, pd.DataFrame],
    scores: dict[str, pd.Series],
    allocator: PortfolioAllocator,
    tickers: list[str],
    *,
    test_start: str,
    test_end: str,
    config: BacktestConfig | None = None,
    market_context: MarketContext | None = None,
    initial_state: PortfolioState | None = None,
) -> dict:
    """Run OOS backtest: signal at t, trade before earning return on t+1."""
    cfg = config or BacktestConfig()
    dates = build_trading_calendar(enriched, scores, test_start=test_start, test_end=test_end)
    if len(dates) < 2:
        raise ValueError("Backtest calendar has fewer than 2 OOS days.")

    if initial_state:
        portfolio_value = initial_state.portfolio_value
        peak_value = initial_state.peak_value
        positions = dict(initial_state.positions)
        cash_weight = initial_state.cash_weight
    else:
        portfolio_value = cfg.initial_value
        peak_value = portfolio_value
        positions: dict[str, float] = {}
        cash_weight = 1.0

    portfolio_hist = [portfolio_value]
    daily_returns: list[float] = []
    positions_hist: list[list[float]] = []
    cash_hist: list[float] = []
    turnover_hist: list[float] = []

    for i in range(len(dates) - 1):
        signal_date = dates[i]
        return_date = dates[i + 1]

        score_row = {
            ticker: float(scores[ticker].loc[signal_date])
            for ticker in tickers
            if ticker in scores and signal_date in scores[ticker].index and np.isfinite(scores[ticker].loc[signal_date])
        }
        if not score_row:
            continue

        vols = build_vols_as_of(
            enriched,
            tickers,
            signal_date,
            vol_window=cfg.vol_window,
            min_vol_obs=cfg.min_vol_obs,
            vol_floor=cfg.vol_floor,
        )
        rolling_mdd = (peak_value - portfolio_value) / max(peak_value, 1e-12)
        state = PortfolioState(
            positions=dict(positions),
            cash_weight=cash_weight,
            portfolio_value=portfolio_value,
            peak_value=peak_value,
            rolling_mdd=float(rolling_mdd),
        )

        daily_market_context = market_context
        if daily_market_context is None and len(enriched) > 0:
            first_ticker = next(iter(enriched.keys()))
            df = enriched[first_ticker]
            col = "macro_^TWII_log_return"
            if col in df.columns:
                try:
                    past = df.loc[:signal_date, col].dropna().tail(60)
                    if len(past) >= 20:
                        mom = float(np.sum(past))
                        print(f"[{signal_date.date()}] TWII 60d mom: {mom:.4f}")
                        if mom < -0.02:
                            daily_market_context = MarketContext(macro_guard_level="CRITICAL")
                        elif mom < 0.01:
                            daily_market_context = MarketContext(macro_guard_level="WARN")
                except KeyError:
                    pass

        target = allocator.allocate(score_row, vols, state, daily_market_context)

        open_returns = {}
        log_returns = {}
        for ticker in tickers:
            if ticker in enriched and return_date in enriched[ticker].index:
                # get() allows fallback if open_return is missing for older parquet caches
                open_returns[ticker] = float(enriched[ticker].loc[return_date].get("open_return", enriched[ticker].loc[return_date, "log_return"]))
                log_returns[ticker] = float(enriched[ticker].loc[return_date, "log_return"])
            else:
                open_returns[ticker] = 0.0
                log_returns[ticker] = 0.0

        # Apply overnight drift to PV and weights
        daily_open_rets = {t: np.exp(open_returns[t]) - 1.0 for t in tickers}
        overnight_pnl = sum(portfolio_value * positions.get(t, 0.0) * daily_open_rets[t] for t in tickers)
        
        prev_value = portfolio_value
        portfolio_value += overnight_pnl
        
        if portfolio_value > 1e-8:
            positions = {
                t: (positions.get(t, 0.0) * prev_value * (1.0 + daily_open_rets[t])) / portfolio_value
                for t in tickers if abs(positions.get(t, 0.0)) > 1e-8
            }
        else:
            positions = {}

        positions, turnover, portfolio_value = execute_rebalance(
            portfolio_value,
            positions,
            target.target_weights,
            tickers,
            open_returns,
        )
        cash_weight = max(0.0, 1.0 - sum(positions.values()))

        # Intraday return
        intraday_pnl = 0.0
        for ticker in tickers:
            weight = positions.get(ticker, 0.0)
            if weight <= 1e-8:
                continue
            intraday_ret = np.exp(log_returns[ticker] - open_returns[ticker]) - 1.0
            intraday_pnl += portfolio_value * weight * intraday_ret
            
        portfolio_value = max(portfolio_value + intraday_pnl, 1e-12)
        if portfolio_value > peak_value:
            peak_value = portfolio_value

        daily_returns.append((portfolio_value / max(prev_value, 1e-12)) - 1.0)
        positions_hist.append([float(positions.get(ticker, 0.0)) for ticker in tickers])
        cash_hist.append(cash_weight)
        turnover_hist.append(turnover)
        portfolio_hist.append(portfolio_value)

    return {
        "daily_returns": daily_returns,
        "positions": positions_hist,
        "cash_weights": cash_hist,
        "turnover": turnover_hist,
        "portfolio_hist": portfolio_hist,
        "n_days": len(daily_returns),
        "test_start": test_start,
        "test_end": test_end,
        "final_positions": positions,
        "final_cash_weight": cash_weight,
        "final_portfolio_value": portfolio_value,
        "final_peak_value": peak_value,
    }


def build_sl_seed_metrics(
    *,
    horizon: int,
    seed: int,
    allocator: str = "rule",
    settings: AppSettings | None = None,
) -> dict:
    """Metrics JSON template for SL walk-forward (Gate-compatible namespace)."""
    settings = settings or load_settings()
    return {
        "strategy": "sl_rule",
        "allocator": allocator,
        "algo": "sl_lightgbm",
        "horizon": horizon,
        "seed": seed,
        "cash_mode": "enabled",
        "enable_cash_action": True,
        "enable_margin_short": False,
        "train_test_period": "Walk-Forward-SL",
        "timesteps": 0,
        "env_config_version": None,
        "env_config_hash": None,
        "env_config": {
            "strategy": "sl_rule",
            "allocator": allocator,
            "horizon": horizon,
            "vol_target": 0.18,
            "top_k": settings.research.default_topk,
        },
        "overall": {},
        "periods": {},
        "skipped_periods": {},
    }


def metrics_from_backtest(
    backtest_result: dict,
    tickers: list[str],
    *,
    period_name: str,
    test_start: str,
    test_end: str,
) -> dict:
    """Compute Gate-ready metrics for one OOS period."""
    metrics = calculate_metrics(
        backtest_result["portfolio_hist"],
        backtest_result["positions"],
        backtest_result["cash_weights"],
        backtest_result["daily_returns"],
        backtest_result["turnover"],
        tickers,
    )
    metrics["test_start"] = test_start
    metrics["test_end"] = test_end
    metrics["period"] = period_name
    metrics["n_days"] = backtest_result.get("n_days", len(backtest_result["daily_returns"]))
    return metrics


def sl_metrics_path(
    results_dir: Path,
    *,
    horizon: int,
    seed: int,
    allocator: str = "rule",
) -> Path:
    return results_dir / f"metrics_sl_{allocator}_h{horizon}_seed{seed}.json"


def persist_sl_metrics(metrics: dict, path: Path) -> Path:
    write_metrics_json(metrics, str(path))
    return path
