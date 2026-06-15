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
from sl_pipeline.allocator import MarketContext, PortfolioAllocator, PortfolioState, TargetPortfolio
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
    liquidity_stress: bool = False
    cost_multiplier: float = 1.0


def rolling_annualized_vol(log_returns: pd.Series, window: int, min_obs: int, vol_floor: float) -> float:
    """Annualized volatility from trailing log returns."""
    recent = pd.to_numeric(log_returns, errors="coerce").dropna().tail(window)
    if len(recent) < min_obs:
        return vol_floor
    return max(float(recent.std() * np.sqrt(TRADING_DAYS_PER_YEAR)), vol_floor)


def build_trends_as_of(
    enriched: dict[str, pd.DataFrame],
    tickers: list[str],
    date: pd.Timestamp,
    *,
    trend_window: int = 60,
) -> dict[str, float]:
    """Calculate the absolute momentum (cumulative log return) over a window."""
    trends: dict[str, float] = {}
    for ticker in tickers:
        if ticker not in enriched:
            trends[ticker] = 1.0
            continue
        df = enriched[ticker]
        idx = df.index.searchsorted(date, side="right") - 1
        if idx < 0:
            trends[ticker] = 1.0
            continue
        start_idx = max(0, idx - trend_window + 1)
        hist_ret = pd.to_numeric(df["log_return"].iloc[start_idx : idx + 1], errors="coerce").dropna()
        if len(hist_ret) < 10:
            trends[ticker] = 1.0
            continue
        trends[ticker] = float(np.sum(hist_ret))
    return trends


def build_ma_distance_as_of(
    enriched: dict[str, pd.DataFrame],
    tickers: list[str],
    date: pd.Timestamp,
    *,
    window: int = 20,
) -> dict[str, float]:
    """Calculate the relative distance of the current price to the moving average."""
    distances: dict[str, float] = {}
    for ticker in tickers:
        if ticker not in enriched:
            distances[ticker] = 0.0
            continue
        df = enriched[ticker]
        idx = df.index.searchsorted(date, side="right") - 1
        if idx < 0:
            distances[ticker] = 0.0
            continue
        start_idx = max(0, idx - window + 1)
        hist_ret = pd.to_numeric(df["log_return"].iloc[start_idx : idx + 1], errors="coerce").dropna()
        if len(hist_ret) < window // 2:
            distances[ticker] = 0.0
            continue
        
        # Price path starting from 1.0
        price_path = np.exp(hist_ret.cumsum())
        current_p = float(price_path.iloc[-1])
        ma = float(price_path.mean())
        distances[ticker] = (current_p / ma) - 1.0
    return distances


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
    cost_multiplier: float = 1.0,
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
            total_cost += trade_amount * trade_cost_rate(prev, target) * cost_multiplier
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
        idx = df.index.searchsorted(date, side="right") - 1
        if idx < 0:
            vols[ticker] = vol_floor
            continue
        start_idx = max(0, idx - vol_window + 1)
        hist = df["log_return"].iloc[start_idx : idx + 1]
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
        positions = dict(initial_state.positions)
        cash_weight = initial_state.cash_weight
        position_cum_rets = dict(initial_state.position_cum_rets)
        position_peaks = dict(initial_state.position_peaks)
        cooldown_days = dict(initial_state.cooldown_days)
    else:
        portfolio_value = cfg.initial_value
        positions: dict[str, float] = {}
        cash_weight = 1.0
        position_cum_rets: dict[str, float] = {}
        position_peaks: dict[str, float] = {}
        cooldown_days: dict[str, int] = {}

    portfolio_hist = [portfolio_value]
    daily_returns: list[float] = []
    positions_hist: list[list[float]] = []
    cash_hist: list[float] = []
    turnover_hist: list[float] = []

    macro_eval = {}
    macro_path = Path("capital_flow_analysis/data/overnight_gap_features_1d.csv")
    if market_context is None and macro_path.exists():
        try:
            df_macro = pd.read_csv(macro_path)
            if "tw_trade_date" in df_macro.columns:
                df_macro["tw_trade_date"] = pd.to_datetime(df_macro["tw_trade_date"])
                df_macro.set_index("tw_trade_date", inplace=True)
                
                features = pd.DataFrame(index=df_macro.index)
                if "sox_ret" in df_macro.columns: features["sox_ret"] = df_macro["sox_ret"]
                if "vix_ret" in df_macro.columns: features["vix_ret"] = df_macro["vix_ret"]
                if "BTC_ret" in df_macro.columns: features["btc_ret"] = df_macro["BTC_ret"]
                if "USD_JPY_ret" in df_macro.columns: features["jpy_strength"] = -df_macro["USD_JPY_ret"]
                
                for col in features.columns:
                    roll = features[col].rolling(window=120, min_periods=20)
                    std = roll.std(ddof=1).replace(0.0, np.nan)
                    features[col + "_z"] = (features[col] - roll.mean()) / std
                
                for dt, row in features.iterrows():
                    level = "OK"
                    if pd.notna(row.get("sox_ret_z")) and row["sox_ret_z"] <= -3.0: level = "CRITICAL"
                    elif pd.notna(row.get("vix_ret_z")) and row["vix_ret_z"] >= 3.0: level = "CRITICAL"
                    elif pd.notna(row.get("btc_ret_z")) and row["btc_ret_z"] <= -3.0: level = "CRITICAL"
                    elif pd.notna(row.get("jpy_strength_z")) and row["jpy_strength_z"] >= 3.0: level = "CRITICAL"
                    elif pd.notna(row.get("sox_ret_z")) and row["sox_ret_z"] <= -2.0: level = "WARN"
                    elif pd.notna(row.get("vix_ret_z")) and row["vix_ret_z"] >= 2.0: level = "WARN"
                    elif pd.notna(row.get("btc_ret_z")) and row["btc_ret_z"] <= -2.0: level = "WARN"
                    elif pd.notna(row.get("jpy_strength_z")) and row["jpy_strength_z"] >= 2.0: level = "WARN"
                    macro_eval[pd.Timestamp(dt).date()] = level
        except Exception as e:
            print(f"Failed to load macro guard features: {e}")

    for i in range(len(dates) - 1):
        signal_date = dates[i]
        return_date = dates[i + 1]

        for t in list(cooldown_days.keys()):
            if cooldown_days[t] > 1:
                cooldown_days[t] -= 1
            else:
                del cooldown_days[t]

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
        trends = build_trends_as_of(
            enriched,
            tickers,
            signal_date,
            trend_window=60,
        )
        short_trends = build_trends_as_of(
            enriched,
            tickers,
            signal_date,
            trend_window=20,
        )
        lookback_window = 126
        recent_hist = portfolio_hist[-lookback_window:]
        rolling_peak = max(recent_hist)
        rolling_mdd = (rolling_peak - portfolio_value) / max(rolling_peak, 1e-12)

        position_mdds = {}
        for ticker in positions:
            if ticker in position_cum_rets and position_peaks.get(ticker, 1.0) > 0:
                peak = position_peaks[ticker]
                current = position_cum_rets[ticker]
                position_mdds[ticker] = (peak - current) / peak

        state = PortfolioState(
            positions=dict(positions),
            cash_weight=cash_weight,
            portfolio_value=portfolio_value,
            peak_value=rolling_peak,
            rolling_mdd=float(rolling_mdd),
            position_mdds=position_mdds,
            position_cum_rets=dict(position_cum_rets),
            position_peaks=dict(position_peaks),
            cooldown_days=dict(cooldown_days),
        )

        daily_market_context = market_context
        if daily_market_context is None:
            level = "OK"
            market_trends = build_trends_as_of(enriched, ["^TWII", "^IXIC"], signal_date, trend_window=60)
            market_trends_fast = build_trends_as_of(enriched, ["^TWII", "^IXIC"], signal_date, trend_window=20)
            twii_down = market_trends.get("^TWII", 1.0) < 0.0
            ixic_down = market_trends.get("^IXIC", 1.0) < 0.0
            twii_down_fast = market_trends_fast.get("^TWII", 1.0) < -0.05
            ixic_down_fast = market_trends_fast.get("^IXIC", 1.0) < -0.05
            
            if twii_down and ixic_down:
                level = "CRITICAL"
            elif (twii_down_fast and ixic_down_fast):
                # Faster trigger if market drops >5% in 20 days
                level = "WARN"
            elif twii_down or ixic_down:
                level = "WARN"

            sig_date_obj = signal_date.date()
            if sig_date_obj in macro_eval:
                gap_level = macro_eval[sig_date_obj]
                if gap_level == "CRITICAL" or (gap_level == "WARN" and level == "OK"):
                    level = gap_level
            
            daily_market_context = MarketContext(macro_guard_level=level)

        ma_20_distance = build_ma_distance_as_of(enriched, tickers, signal_date, window=20)
        ma_60_distance = build_ma_distance_as_of(enriched, tickers, signal_date, window=60)
        ma_distances = {20: ma_20_distance, 60: ma_60_distance}

        target = allocator.allocate(
            score_row, 
            vols, 
            state, 
            market_context=daily_market_context, 
            trends=trends,
            short_trends=short_trends,
            ma_distances=ma_distances,
        )

        ma_filter_enabled = getattr(allocator.config, "enable_ma_filter", False)
        ts_enabled = getattr(allocator.config, "enable_trailing_stop", False)
        cooldown_duration = getattr(allocator.config, "cooldown_duration", 10)
        ts_thresh = getattr(allocator.config, "trailing_stop_threshold", 0.15)
        ma_filter_windows = getattr(allocator.config, "ma_filter_windows", [20])

        for t in positions:
            if target.target_weights.get(t, 0.0) == 0.0:
                is_cooldown_triggered = False
                if ts_enabled and position_mdds.get(t, 0.0) >= ts_thresh:
                    is_cooldown_triggered = True
                if ma_filter_enabled:
                    for w in ma_filter_windows:
                        if ma_distances.get(w, {}).get(t, 1.0) < 0.0:
                            is_cooldown_triggered = True
                            break
                if is_cooldown_triggered:
                    cooldown_days[t] = cooldown_duration

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

        if cfg.liquidity_stress:
            prev_total = sum(positions.values())
            target_total = sum(target.target_weights.values())
            # If allocator is dumping > 30% of portfolio at once (panic sell)
            if prev_total - target_total > 0.30:
                for t in positions:
                    if target.target_weights.get(t, 0.0) < positions[t]:
                        open_returns[t] = -0.098  # Force limit down gap
                        log_returns[t] = -0.098

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
            cost_multiplier=cfg.cost_multiplier,
        )
        cash_weight = max(0.0, 1.0 - sum(positions.values()))

        # Intraday return
        intraday_pnl = 0.0
        for ticker in tickers:
            weight = positions.get(ticker, 0.0)
            if weight <= 1e-8:
                if ticker in position_cum_rets:
                    del position_cum_rets[ticker]
                if ticker in position_peaks:
                    del position_peaks[ticker]
                continue
            
            intraday_ret = np.exp(log_returns[ticker] - open_returns[ticker]) - 1.0
            intraday_pnl += portfolio_value * weight * intraday_ret
            
            if ticker not in position_cum_rets:
                position_cum_rets[ticker] = 1.0 + intraday_ret
                position_peaks[ticker] = max(1.0, position_cum_rets[ticker])
            else:
                daily_ret = np.exp(log_returns[ticker])
                position_cum_rets[ticker] *= daily_ret
                position_peaks[ticker] = max(position_peaks[ticker], position_cum_rets[ticker])
            
        portfolio_value = max(portfolio_value + intraday_pnl, 1e-12)

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
        "final_peak_value": portfolio_value, # Obsolete, just send PV
        "final_position_cum_rets": position_cum_rets,
        "final_position_peaks": position_peaks,
        "final_cooldown_days": cooldown_days,
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
            "vol_target": settings.research.sl_target_vol,
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
