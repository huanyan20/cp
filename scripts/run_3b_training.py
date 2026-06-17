"""Milestone 3B: LightGBM/XGBoost 三分類選股模型完整訓練腳本。

用法:
    python scripts/run_3b_training.py --horizon 20 --model lightgbm --seed 42
    python scripts/run_3b_training.py --horizon 20 --model xgboost --seed 42
    python scripts/run_3b_training.py --horizon 20 --model both --seeds 42,43,44

輸出 (reports/milestone_3b/):
  - ic_filtered_features.json       <- 通過 IC 篩選的特徵
  - {model}_{horizon}d_seed{seed}/  <- 每個組合的結果資料夾
    - scores_{period}.csv           <- OOS 分數
    - predictions_{period}.csv      <- 完整預測面板 (含 p_bot / p_mid / p_top)
    - summary_{period}.json         <- 訓練摘要 (class distribution, feature importance)
  - walk_forward_summary.md         <- 最終 Walk-Forward 績效摘要報告
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(".")
from data_loader import fetch_multi_asset_data
from data_pipeline.universe_builder import get_universe_builder
from metrics_utils import calculate_metrics
from research_pipeline import PERIODS, build_period_plan, write_metrics_json
from settings import load_settings
from sl_pipeline.backtest import (
    BacktestConfig,
    build_sl_seed_metrics,
    metrics_from_backtest,
    simulate_period,
    sl_metrics_path,
)
from sl_pipeline.allocator import PortfolioState
from sl_pipeline.rule_based_allocator import RISK_CONFIGS, RISK_V2, RuleBasedAllocator, RuleBasedAllocatorConfig
from sl_pipeline.signal_generator import (
    DEFAULT_LGBM_PARAMS,
    DEFAULT_XGB_PARAMS,
    IC_FILTERED_FEATURES,
    SignalGenerator,
    SignalGeneratorConfig,
    normalize_model_backend,
)
from stock_universe import MACRO_TICKERS_RL

SETTINGS = load_settings()
REPORTS_DIR = Path("reports/milestone_3b")

HORIZON_CHOICES = (5, 10, 20, 60)


def test_fetch_start(test_start: str) -> str:
    return str(int(test_start[:4]) - 1) + test_start[4:]


def resolve_period(name: str) -> dict:
    period = next((p for p in PERIODS if p["name"] == name), None)
    if period is None:
        raise ValueError(f"Unknown period {name!r}")
    return period


def build_allocator() -> RuleBasedAllocator:
    return RuleBasedAllocator(
        risk_config=RISK_V2,
        config=RuleBasedAllocatorConfig(
            top_k=SETTINGS.research.default_topk,
            max_single_weight=SETTINGS.risk_limits.max_single_weight,
            target_vol_annual=SETTINGS.research.sl_target_vol,
            trailing_stop_threshold=SETTINGS.research.sl_trailing_stop,
        ),
    )


def build_generator(horizon: int, model_backend: str, seed: int) -> SignalGenerator:
    backend = normalize_model_backend(model_backend)
    if backend == "lightgbm":
        params = dict(DEFAULT_LGBM_PARAMS)
        params["random_state"] = seed
        return SignalGenerator(
            SignalGeneratorConfig(
                horizon=horizon,
                model_backend=backend,
                lgbm_params=params,
                use_ic_filtered_features=True,
            )
        )
    else:
        params = dict(DEFAULT_XGB_PARAMS)
        params["random_state"] = seed
        return SignalGenerator(
            SignalGeneratorConfig(
                horizon=horizon,
                model_backend=backend,
                xgb_params=params,
                use_ic_filtered_features=True,
            )
        )


def run_walk_forward(
    *,
    horizon: int,
    model_backend: str,
    seed: int,
    out_dir: Path,
) -> dict:
    """Full walk-forward with 3-class classifier on IC-filtered features."""
    print(f"\n{'='*60}")
    print(f"  Model: {model_backend.upper()} | Horizon: {horizon}d | Seed: {seed}")
    print(f"{'='*60}")

    plan = build_period_plan()
    builder = get_universe_builder("dynamic")
    allocator = build_allocator()

    all_daily_returns: list[float] = []
    all_positions: list[list[float]] = []
    all_cash_weights: list[float] = []
    all_turnover: list[float] = []
    period_results: dict[str, dict] = {}
    current_state: PortfolioState | None = None
    period_cache = []

    for planned in plan:
        name = planned["name"]
        if planned.get("skip_reason"):
            print(f"  [!] Skipping {name}: {planned['skip_reason']}")
            continue

        period = resolve_period(name)
        train_end = planned.get("effective_train_end", period["train_end"])
        test_end = planned.get("effective_test_end", period["test_end"])

        print(f"\n  Period: {name} | Train: {period['train_start']} ~ {train_end} | Test: {period['test_start']} ~ {test_end}")

        # Dynamic universe per period
        period_tickers = builder.build_universe(period["train_start"], top_n=45)
        print(f"  Universe: {len(period_tickers)} tickers")

        train_data = fetch_multi_asset_data(
            tickers=period_tickers,
            start_date=period["train_start"],
            end_date=train_end,
            macro_tickers=["^TWII"],  # Option A: enable TWII market regime features
        )
        test_data = fetch_multi_asset_data(
            tickers=period_tickers,
            start_date=test_fetch_start(period["test_start"]),
            end_date=test_end,
            macro_tickers=["^TWII"],
        )

        generator = build_generator(horizon, model_backend, seed)

        try:
            scores, summary = generator.fit_period(
                train_data,
                test_data,
                train_end=train_end,
                test_start=period["test_start"],
            )
        except Exception as e:
            print(f"  [!] Period {name} failed: {e}")
            continue

        # Save period artifacts
        period_dir = out_dir / name
        period_dir.mkdir(parents=True, exist_ok=True)
        generator.save_scores(scores, period_dir / f"scores_{name}_h{horizon}.csv")
        generator.save_summary(summary, period_dir / f"summary_{name}_h{horizon}.json")
        generator.save_prediction_panel(period_dir / f"predictions_{name}_h{horizon}.csv")

        print(f"  Train rows: {summary.n_train_rows} | Test rows: {summary.n_test_rows}")
        print(f"  Class distribution (0=bot,1=mid,2=top): {summary.train_class_distribution}")
        print(f"  Top features: {list(summary.feature_importance_top10.keys())[:5]}")

        # Backtest
        combined_test = {**train_data, **test_data}
        allocator.reset_regime()
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
            backtest,
            period_tickers,
            period_name=name,
            test_start=period["test_start"],
            test_end=test_end,
        )
        period_results[name] = period_metrics
        period_cache.append({
            "combined_test": combined_test,
            "scores": scores,
            "period_tickers": period_tickers,
            "test_start": period["test_start"],
            "test_end": test_end,
        })

        all_daily_returns.extend(backtest["daily_returns"])
        all_positions.extend(backtest["positions"])
        all_cash_weights.extend(backtest["cash_weights"])
        all_turnover.extend(backtest["turnover"])

        m = period_metrics
        print(
            f"  Return: {m['total_return']*100:.1f}% | "
            f"MDD: {m['max_drawdown']*100:.1f}% | "
            f"Sortino: {m.get('sortino', 0):.2f} | "
            f"Sharpe: {m.get('sharpe', 0):.2f}"
        )

    if not all_daily_returns:
        raise ValueError("No OOS backtest data produced.")

    # Overall metrics
    cum_returns = np.cumprod(1.0 + np.array(all_daily_returns))
    portfolio_history = [1.0] + list(cum_returns)
    last_tickers = period_cache[-1]["period_tickers"] if period_cache else []
    overall_metrics = calculate_metrics(
        portfolio_history,
        all_positions,
        all_cash_weights,
        all_daily_returns,
        all_turnover,
        last_tickers,
    )

    result = {
        "horizon": horizon,
        "model_backend": model_backend,
        "seed": seed,
        "n_ic_features": len(IC_FILTERED_FEATURES),
        "overall": overall_metrics,
        "periods": period_results,
    }

    # Save run summary
    summary_path = out_dir / "run_summary.json"
    summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def format_results_table(all_results: list[dict]) -> str:
    """Format all walk-forward results as a markdown table."""
    rows = []
    for r in all_results:
        m = r["overall"]
        rows.append({
            "Model": r["model_backend"].upper(),
            "Horizon": f"{r['horizon']}d",
            "Seed": r["seed"],
            "Total Return": f"{m['total_return']*100:.1f}%",
            "MDD": f"{m['max_drawdown']*100:.1f}%",
            "Sortino": f"{m.get('sortino', 0):.2f}",
            "Sharpe": f"{m.get('sharpe', 0):.2f}",
            "Avg Turnover": f"{m.get('turnover', 0):.3f}",
        })
    df = pd.DataFrame(rows)
    return df.to_markdown(index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Milestone 3B: Train 3-class LightGBM/XGBoost on IC-filtered features"
    )
    parser.add_argument("--horizon", type=int, default=20, choices=HORIZON_CHOICES)
    parser.add_argument(
        "--model",
        default="lightgbm",
        choices=["lightgbm", "xgboost", "both"],
        help="Model backend(s) to train",
    )
    parser.add_argument(
        "--seeds",
        default="42",
        help="Comma-separated list of random seeds (e.g. '42,43,44')",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    models = ["lightgbm", "xgboost"] if args.model == "both" else [args.model]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save IC-filtered feature list
    ic_features_path = REPORTS_DIR / "ic_filtered_features.json"
    ic_features_path.write_text(
        json.dumps({"features": IC_FILTERED_FEATURES, "n": len(IC_FILTERED_FEATURES)}, indent=2),
        encoding="utf-8",
    )
    print(f"IC-filtered features ({len(IC_FILTERED_FEATURES)}): {IC_FILTERED_FEATURES}")

    all_results = []
    for model_backend in models:
        for seed in seeds:
            out_dir = args.output_dir or (
                REPORTS_DIR / f"{model_backend}_{args.horizon}d_seed{seed}"
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            result = run_walk_forward(
                horizon=args.horizon,
                model_backend=model_backend,
                seed=seed,
                out_dir=out_dir,
            )
            all_results.append(result)

    # Print final comparison table
    print("\n" + "=" * 70)
    print("  MILESTONE 3B: Walk-Forward Results Summary")
    print("=" * 70)
    table = format_results_table(all_results)
    print(table)

    # Save final summary report
    report_path = REPORTS_DIR / "walk_forward_summary.md"
    report_lines = [
        "# Milestone 3B: LightGBM/XGBoost 三分類選股模型結果\n",
        f"## IC-Filtered Features ({len(IC_FILTERED_FEATURES)} features)\n",
        "依照 Feature IC Dashboard (Milestone 3A) 篩選後，只保留 `|IC| > 0.02 @ 20d` 的黃金特徵進行訓練。\n",
        "\n## Walk-Forward 回測績效\n",
        table,
        "\n\n## 完整 Period 明細\n",
    ]
    for r in all_results:
        report_lines.append(
            f"\n### {r['model_backend'].upper()} | Horizon={r['horizon']}d | Seed={r['seed']}\n"
        )
        period_rows = []
        for period_name, m in r["periods"].items():
            period_rows.append({
                "Period": period_name,
                "Return": f"{m['total_return']*100:.1f}%",
                "MDD": f"{m['max_drawdown']*100:.1f}%",
                "Sortino": f"{m.get('sortino', 0):.2f}",
                "Sharpe": f"{m.get('sharpe', 0):.2f}",
            })
        if period_rows:
            report_lines.append(pd.DataFrame(period_rows).to_markdown(index=False))
            report_lines.append("\n")

    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nReport saved: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
