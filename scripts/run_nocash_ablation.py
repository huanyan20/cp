"""No-cash ablation test: Milestone 3B diagnostic.

Purpose:
  Pure top-K equal-weight allocation with NO risk engine.
  No vol target, no MDD regime, no min_score filter, no trend filter.
  Answers: "If we FORCE 100% deployment, does the SL signal have positive absolute alpha?"

Usage:
  python scripts/run_nocash_ablation.py --horizon 20 --model lightgbm --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(".")
from data_loader import fetch_multi_asset_data
from data_pipeline.universe_builder import get_universe_builder
from metrics_utils import calculate_metrics
from research_pipeline import PERIODS, build_period_plan
from settings import load_settings
from sl_pipeline.allocator import PortfolioAllocator, PortfolioState, TargetPortfolio, MarketContext
from sl_pipeline.backtest import metrics_from_backtest, simulate_period
from sl_pipeline.signal_generator import (
    DEFAULT_LGBM_PARAMS,
    DEFAULT_XGB_PARAMS,
    SignalGenerator,
    SignalGeneratorConfig,
    normalize_model_backend,
)
from stock_universe import MACRO_TICKERS_RL

SETTINGS = load_settings()
REPORTS_DIR = Path("reports/milestone_3b_nocash")


class NoCashAllocator(PortfolioAllocator):
    """ABLATION: Pure Top-K equal-weight. ALWAYS 100% deployed. No risk engine.

    This bypasses ALL cash-forcing mechanisms:
    - No min_score filter
    - No trend/MA filter
    - No MDD regime cap
    - No vol target
    - No trailing stop
    - Equal weight on top-K at all times
    """

    class _Config:
        """Minimal stub so backtest.py getattr calls don't crash."""
        enable_ma_filter: bool = False
        enable_trend_filter: bool = False
        enable_trailing_stop: bool = False
        cooldown_duration: int = 0
        trailing_stop_threshold: float = 999.0
        ma_filter_windows: list = None

        def __init__(self):
            self.ma_filter_windows = []

    def __init__(self, top_k: int = 5) -> None:
        self.top_k = top_k
        self.config = NoCashAllocator._Config()


    def allocate(
        self,
        scores: dict[str, float],
        vols: dict[str, float],
        state: PortfolioState,
        market_context: MarketContext | None = None,
        trends: dict[str, float] | None = None,
        short_trends: dict[str, float] | None = None,
        ma_distances: dict | None = None,
    ) -> TargetPortfolio:
        if not scores:
            # Fallback: equal weight on all available
            return TargetPortfolio(target_weights={}, cash_weight=1.0)

        # Sort by score descending, pick top-k
        sorted_tickers = sorted(scores, key=lambda t: scores[t], reverse=True)
        top_k = sorted_tickers[: self.top_k]

        # Equal weight → 100% deployment always
        w = 1.0 / len(top_k)
        target_weights = {t: w for t in top_k}
        return TargetPortfolio(target_weights=target_weights, cash_weight=0.0)

    def reset_regime(self) -> None:
        pass


def test_fetch_start(test_start: str) -> str:
    return str(int(test_start[:4]) - 1) + test_start[4:]


def resolve_period(name: str) -> dict:
    period = next((p for p in PERIODS if p["name"] == name), None)
    if period is None:
        raise ValueError(f"Unknown period {name!r}")
    return period


def build_generator(horizon: int, model_backend: str, seed: int) -> SignalGenerator:
    backend = normalize_model_backend(model_backend)
    if backend == "lightgbm":
        params = dict(DEFAULT_LGBM_PARAMS)
        params["random_state"] = seed
    else:
        params = dict(DEFAULT_XGB_PARAMS)
        params["random_state"] = seed
    config = SignalGeneratorConfig(
        horizon=horizon,
        model_backend=backend,
        lgbm_params=params if backend == "lightgbm" else dict(DEFAULT_LGBM_PARAMS),
        xgb_params=params if backend == "xgboost" else dict(DEFAULT_XGB_PARAMS),
        use_ic_filtered_features=True,
    )
    return SignalGenerator(config)


def run_ablation(
    horizon: int,
    model_backend: str,
    seed: int,
    top_k: int,
    out_dir: Path,
) -> dict:
    print(f"\n{'='*60}")
    print(f"  NO-CASH ABLATION | {model_backend.upper()} | h={horizon}d | top_k={top_k} | seed={seed}")
    print(f"{'='*60}")

    plan = build_period_plan()
    builder = get_universe_builder("dynamic")
    allocator = NoCashAllocator(top_k=top_k)

    all_daily_returns: list[float] = []
    all_positions: list[list[float]] = []
    all_cash_weights: list[float] = []
    all_turnover: list[float] = []
    period_results: dict[str, dict] = {}
    current_state: PortfolioState | None = None

    for planned in plan:
        name = planned["name"]
        if planned.get("skip_reason"):
            continue

        period = resolve_period(name)
        train_end = planned.get("effective_train_end", period["train_end"])
        test_end = planned.get("effective_test_end", period["test_end"])

        print(f"\n  Period: {name} | {period['train_start']} ~ {test_end}")
        period_tickers = builder.build_universe(period["train_start"], top_n=45)

        train_data = fetch_multi_asset_data(
            tickers=period_tickers,
            start_date=period["train_start"],
            end_date=train_end,
            macro_tickers=[],
        )
        test_data = fetch_multi_asset_data(
            tickers=period_tickers,
            start_date=test_fetch_start(period["test_start"]),
            end_date=test_end,
            macro_tickers=[],
        )

        generator = build_generator(horizon, model_backend, seed)
        try:
            scores, summary = generator.fit_period(
                train_data, test_data, train_end=train_end, test_start=period["test_start"]
            )
        except Exception as e:
            print(f"  [!] Failed: {e}")
            continue

        combined_test = {**train_data, **test_data}
        backtest = simulate_period(
            combined_test,
            scores,
            allocator,
            period_tickers,
            test_start=period["test_start"],
            test_end=test_end,
            initial_state=current_state,
        )
        current_state = PortfolioState(
            positions=backtest["final_positions"],
            cash_weight=backtest["final_cash_weight"],
            portfolio_value=backtest["final_portfolio_value"],
            peak_value=backtest["final_peak_value"],
            position_cum_rets=backtest.get("final_position_cum_rets", {}),
            position_peaks=backtest.get("final_position_peaks", {}),
            cooldown_days=backtest.get("final_cooldown_days", {}),
        )

        period_metrics = metrics_from_backtest(
            backtest, period_tickers,
            period_name=name,
            test_start=period["test_start"],
            test_end=test_end,
        )
        period_results[name] = period_metrics

        all_daily_returns.extend(backtest["daily_returns"])
        all_positions.extend(backtest["positions"])
        all_cash_weights.extend(backtest["cash_weights"])
        all_turnover.extend(backtest["turnover"])

        m = period_metrics
        print(
            f"  Return={m['total_return']*100:+.1f}%  MDD={m['max_drawdown']*100:.1f}%  "
            f"Sortino={m.get('sortino',0):.2f}  AvgCash={m['avg_cash_weight']*100:.0f}%"
        )

    if not all_daily_returns:
        raise ValueError("No OOS data produced.")

    cum = np.cumprod(1.0 + np.array(all_daily_returns))
    hist = [1.0] + list(cum)
    last_tickers = list(period_results and list(period_results.keys()) and [])
    # get tickers from last period
    last_plan = [p for p in plan if not p.get("skip_reason")]
    if last_plan:
        last_tickers = builder.build_universe(resolve_period(last_plan[-1]["name"])["train_start"], top_n=45)

    overall = calculate_metrics(hist, all_positions, all_cash_weights, all_daily_returns, all_turnover, last_tickers)

    result = {
        "horizon": horizon,
        "model_backend": model_backend,
        "seed": seed,
        "top_k": top_k,
        "mode": "NO_CASH_ABLATION",
        "overall": overall,
        "periods": period_results,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ablation_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n  OVERALL: Return={overall['total_return']*100:+.1f}%  MDD={overall['max_drawdown']*100:.1f}%  "
          f"Sortino={overall.get('sortino',0):.2f}  AvgCash={overall['avg_cash_weight']*100:.0f}%")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="No-cash ablation: force 100% deployment to test raw signal alpha"
    )
    parser.add_argument("--horizon", type=int, default=20, choices=(5, 10, 20, 60))
    parser.add_argument("--model", default="lightgbm", choices=["lightgbm", "xgboost"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=5, help="Number of stocks to hold")
    args = parser.parse_args(argv)

    out_dir = REPORTS_DIR / f"{args.model}_{args.horizon}d_top{args.top_k}_seed{args.seed}"
    result = run_ablation(
        horizon=args.horizon,
        model_backend=args.model,
        seed=args.seed,
        top_k=args.top_k,
        out_dir=out_dir,
    )

    # Print comparison table
    print("\n" + "=" * 60)
    print("  ABLATION RESULT: No-Cash vs Risk-Engine")
    print("=" * 60)
    print("Period        | Return  | MDD    | Sortino | AvgCash")
    print("-" * 60)
    for name, m in result["periods"].items():
        print(
            f"{name:13s} | {m['total_return']*100:+6.1f}%  | {m['max_drawdown']*100:5.1f}%  | "
            f"{m.get('sortino',0):6.2f}  | {m['avg_cash_weight']*100:.0f}%"
        )
    o = result["overall"]
    print("-" * 60)
    print(
        f"{'OVERALL':13s} | {o['total_return']*100:+6.1f}%  | {o['max_drawdown']*100:5.1f}%  | "
        f"{o.get('sortino',0):6.2f}  | {o['avg_cash_weight']*100:.0f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
