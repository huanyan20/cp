"""Walk-forward SL CLI: SignalGenerator (S1) + RuleBasedAllocator backtest (S2)."""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

from data_loader import fetch_multi_asset_data
from data_pipeline.universe_builder import get_universe_builder
from metrics_utils import calculate_metrics
from research_pipeline import PERIODS, build_period_plan, write_metrics_json
from settings import load_settings
from sl_pipeline.backtest import (
    build_sl_seed_metrics,
    metrics_from_backtest,
    simulate_period,
    sl_metrics_path,
    BacktestConfig,
)
from sl_pipeline.gate import run_sl_promotion_gate, save_sl_gate_result
from sl_pipeline.allocator import PortfolioState
from sl_pipeline.rule_based_allocator import (
    RuleBasedAllocator,
    RuleBasedAllocatorConfig,
)
from sl_pipeline.signal_generator import (
    DEFAULT_LGBM_PARAMS,
    SignalGenerator,
    SignalGeneratorConfig,
)
from stock_universe import MACRO_TICKERS_RL

SETTINGS = load_settings()


def test_fetch_start(test_start: str) -> str:
    """One-year lookback before OOS start (matches research_pipeline.build_eval_env)."""
    return str(int(test_start[:4]) - 1) + test_start[4:]


def resolve_period(name: str) -> dict:
    period = next((p for p in PERIODS if p["name"] == name), None)
    if period is None:
        choices = ", ".join(p["name"] for p in PERIODS)
        raise SystemExit(f"Unknown period {name!r}; choose one of: {choices}")
    return period


def build_allocator(name: str) -> RuleBasedAllocator:
    key = (name or "rule").strip().lower()
    if key != "rule":
        raise ValueError(
            f"Unsupported allocator {name!r}; only 'rule' is implemented in S2."
        )
    return RuleBasedAllocator(
        RuleBasedAllocatorConfig(
            top_k=SETTINGS.research.default_topk,
            max_single_weight=SETTINGS.risk_limits.max_single_weight,
            target_vol_annual=SETTINGS.research.sl_target_vol,
            trailing_stop_threshold=SETTINGS.research.sl_trailing_stop,
        )
    )


def run_single_period(
    period_name: str,
    *,
    horizon: int = 5,
    allocator_name: str | None = None,
    seed: int = 42,
    output_dir: Path | None = None,
    tickers: list[str] | None = None,
    write_metrics: bool = False,
    results_dir: Path | None = None,
) -> dict:
    """Train LightGBM on one walk-forward period; optionally run allocator backtest."""
    plan = build_period_plan()
    planned = next((p for p in plan if p["name"] == period_name), None)
    if planned is None:
        raise ValueError(f"Period {period_name} not in current plan.")
    if planned.get("skip_reason"):
        raise ValueError(planned["skip_reason"])

    period = resolve_period(period_name)
    train_end = planned.get("effective_train_end", period["train_end"])
    test_end = planned.get("effective_test_end", period["test_end"])
    if tickers is None:
        builder = get_universe_builder("dynamic")
        tickers = builder.build_universe(period["train_start"], top_n=45)

    train_data = fetch_multi_asset_data(
        tickers=tickers,
        start_date=period["train_start"],
        end_date=train_end,
        macro_tickers=MACRO_TICKERS_RL,
    )
    test_data = fetch_multi_asset_data(
        tickers=tickers,
        start_date=test_fetch_start(period["test_start"]),
        end_date=test_end,
        macro_tickers=MACRO_TICKERS_RL,
    )

    lgbm_params = dict(DEFAULT_LGBM_PARAMS)
    lgbm_params["random_state"] = seed
    generator = SignalGenerator(
        SignalGeneratorConfig(horizon=horizon, lgbm_params=lgbm_params)
    )
    scores, summary = generator.fit_period(
        train_data,
        test_data,
        train_end=train_end,
        test_start=period["test_start"],
    )

    out_dir = output_dir or Path(tempfile.mkdtemp(prefix=f"sl_{period_name}_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = out_dir / f"scores_{period_name}_h{horizon}.csv"
    summary_path = out_dir / f"summary_{period_name}_h{horizon}.json"
    generator.save_scores(scores, scores_path)
    generator.save_summary(summary, summary_path)

    wide = generator.scores_to_wide(scores)
    result: dict = {
        "period": period_name,
        "horizon": horizon,
        "allocator": allocator_name,
        "seed": seed,
        "output_dir": str(out_dir),
        "scores_path": str(scores_path),
        "summary_path": str(summary_path),
        "n_tickers": len(scores),
        "n_oos_days": wide.shape[0],
        "score_mean": wide.mean().mean(),
        "score_std": float(np.nanstd(np.asarray(wide, dtype=float))),
        "summary": json.loads(summary_path.read_text(encoding="utf-8")),
    }

    if allocator_name:
        allocator = build_allocator(allocator_name)
        combined_test = {**train_data, **test_data}
        backtest = simulate_period(
            combined_test,
            scores,
            allocator,
            tickers,
            test_start=period["test_start"],
            test_end=test_end,
        )
        period_metrics = metrics_from_backtest(
            backtest,
            tickers,
            period_name=period_name,
            test_start=period["test_start"],
            test_end=test_end,
        )
        result["backtest"] = {
            "n_days": backtest["n_days"],
            "metrics": period_metrics,
        }
        metrics_path = out_dir / f"metrics_{period_name}_h{horizon}.json"
        metrics_path.write_text(
            json.dumps(period_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        result["period_metrics_path"] = str(metrics_path)

        if write_metrics:
            seed_metrics = build_sl_seed_metrics(
                horizon=horizon,
                seed=seed,
                allocator=allocator_name,
                settings=SETTINGS,
            )
            seed_metrics["periods"][period_name] = period_metrics
            seed_metrics["overall"] = period_metrics
            out_results = results_dir or SETTINGS.paths.results_dir
            path = sl_metrics_path(
                out_results, horizon=horizon, seed=seed, allocator=allocator_name
            )
            write_metrics_json(seed_metrics, str(path))
            result["metrics_path"] = str(path)

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def run_walk_forward_sl(
    *,
    horizon: int = 5,
    allocator_name: str = "rule",
    seed: int = 42,
    output_dir: Path | None = None,
    tickers: list[str] | None = None,
    results_dir: Path | None = None,
    run_gate: bool = False,
) -> dict:
    plan = build_period_plan()
    allocator = build_allocator(allocator_name)
    if tickers is None:
        builder = get_universe_builder("dynamic")
        # Initialize default but dynamically update per period if requested
        tickers = builder.build_universe(
            plan[0]["name"][:4] + "-01-01", top_n=45
        )  # Fallback baseline

    seed_metrics = build_sl_seed_metrics(
        horizon=horizon,
        seed=seed,
        allocator=allocator_name,
        settings=SETTINGS,
    )

    all_daily_returns: list[float] = []
    all_positions: list[list[float]] = []
    all_cash_weights: list[float] = []
    all_turnover: list[float] = []
    current_state: PortfolioState | None = None
    period_cache = []

    for planned in plan:
        name = planned["name"]
        if planned.get("skip_reason"):
            seed_metrics["skipped_periods"][name] = planned["skip_reason"]
            continue

        period = resolve_period(name)
        train_end = planned.get("effective_train_end", period["train_end"])
        test_end = planned.get("effective_test_end", period["test_end"])

        # Dynamically fetch universe for this specific period
        builder = get_universe_builder("dynamic")
        period_tickers = builder.build_universe(period["train_start"], top_n=45)

        train_data = fetch_multi_asset_data(
            tickers=period_tickers,
            start_date=period["train_start"],
            end_date=train_end,
            macro_tickers=MACRO_TICKERS_RL,
        )
        test_data = fetch_multi_asset_data(
            tickers=period_tickers,
            start_date=test_fetch_start(period["test_start"]),
            end_date=test_end,
            macro_tickers=MACRO_TICKERS_RL,
        )

        lgbm_params = dict(DEFAULT_LGBM_PARAMS)
        lgbm_params["random_state"] = seed
        generator = SignalGenerator(
            SignalGeneratorConfig(horizon=horizon, lgbm_params=lgbm_params)
        )
        scores, _ = generator.fit_period(
            train_data,
            test_data,
            train_end=train_end,
            test_start=period["test_start"],
        )

        if output_dir:
            out_d = Path(output_dir)
            out_d.mkdir(parents=True, exist_ok=True)
            scores_path = out_d / f"scores_{name}_h{horizon}.csv"
            generator.save_scores(scores, scores_path)

        combined_test = {**train_data, **test_data}
        period_cache.append({
            "combined_test": combined_test,
            "scores": scores,
            "period_tickers": period_tickers,
            "test_start": period["test_start"],
            "test_end": test_end,
        })
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
        seed_metrics["periods"][name] = period_metrics

        all_daily_returns.extend(backtest["daily_returns"])
        all_positions.extend(backtest["positions"])
        all_cash_weights.extend(backtest["cash_weights"])
        all_turnover.extend(backtest["turnover"])

    if not all_daily_returns:
        raise ValueError("No OOS backtest data produced.")

    cum_returns = np.cumprod(1.0 + np.array(all_daily_returns))
    portfolio_history = [1.0] + list(cum_returns)
    overall_metrics = calculate_metrics(
        portfolio_history,
        all_positions,
        all_cash_weights,
        all_daily_returns,
        all_turnover,
        period_tickers,  # Note: strictly we should pass dynamic per-day tickers to overall metrics, but period_tickers from last fold is a fallback.
    )
    seed_metrics["overall"] = overall_metrics

    out_results = results_dir or SETTINGS.paths.results_dir
    path = sl_metrics_path(
        out_results, horizon=horizon, seed=seed, allocator=allocator_name
    )
    write_metrics_json(seed_metrics, str(path))

    # --- LIQUIDITY STRESS TEST ---
    stress_state: PortfolioState | None = None
    stress_returns = []
    promo_stress_state: PortfolioState | None = None
    promo_stress_returns = []
    for p_data in period_cache:
        stress_bt = simulate_period(
            p_data["combined_test"],
            p_data["scores"],
            allocator,
            p_data["period_tickers"],
            test_start=p_data["test_start"],
            test_end=p_data["test_end"],
            initial_state=stress_state,
            config=BacktestConfig(liquidity_stress=True)
        )
        stress_state = PortfolioState(
            positions=stress_bt["final_positions"],
            cash_weight=stress_bt["final_cash_weight"],
            portfolio_value=stress_bt["final_portfolio_value"],
            peak_value=stress_bt["final_peak_value"],
        )
        stress_returns.extend(stress_bt["daily_returns"])

        promo_bt = simulate_period(
            p_data["combined_test"],
            p_data["scores"],
            allocator,
            p_data["period_tickers"],
            test_start=p_data["test_start"],
            test_end=p_data["test_end"],
            initial_state=promo_stress_state,
            config=BacktestConfig(cost_multiplier=3.0)
        )
        promo_stress_state = PortfolioState(
            positions=promo_bt["final_positions"],
            cash_weight=promo_bt["final_cash_weight"],
            portfolio_value=promo_bt["final_portfolio_value"],
            peak_value=promo_bt["final_peak_value"],
        )
        promo_stress_returns.extend(promo_bt["daily_returns"])
        
    stress_cum = np.cumprod(1.0 + np.array(stress_returns))
    stress_hist = [1.0] + list(stress_cum)
    stress_mdd = 0.0
    if len(stress_hist) > 0:
        peak = stress_hist[0]
        for v in stress_hist:
            if v > peak:
                peak = v
            dd = (peak - v) / peak
            if dd > stress_mdd:
                stress_mdd = dd

    promo_cum = np.cumprod(1.0 + np.array(promo_stress_returns))
    promo_hist = [1.0] + list(promo_cum)
    promo_mdd = 0.0
    if len(promo_hist) > 0:
        peak = promo_hist[0]
        for v in promo_hist:
            if v > peak:
                peak = v
            dd = (peak - v) / peak
            if dd > promo_mdd:
                promo_mdd = dd
                
    stress_summary = {
        "tests": {
            "hard_disaster_stress": {
                "total_return": stress_hist[-1] - 1.0,
                "max_drawdown": stress_mdd
            },
            "promotion_stress": {
                "total_return": promo_hist[-1] - 1.0,
                "max_drawdown": promo_mdd
            }
        }
    }
    stress_path = out_results / "stress_summary.json"
    stress_path.parent.mkdir(parents=True, exist_ok=True)
    stress_path.write_text(json.dumps(stress_summary, indent=2), encoding="utf-8")
    # -----------------------------

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "walk_forward_manifest.json").write_text(
            json.dumps(
                {
                    "horizon": horizon,
                    "allocator": allocator_name,
                    "seed": seed,
                    "metrics_path": str(path),
                    "overall": overall_metrics,
                    "periods": list(seed_metrics["periods"].keys()),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    result = {
        "horizon": horizon,
        "allocator": allocator_name,
        "seed": seed,
        "metrics_path": str(path),
        "overall": overall_metrics,
        "periods": seed_metrics["periods"],
        "skipped_periods": seed_metrics["skipped_periods"],
    }

    if run_gate:
        gate_result, raw_summary, _ = run_sl_promotion_gate(
            path,
            min_seeds=SETTINGS.research.promotion_min_seeds,
        )
        gate_path = save_sl_gate_result(
            gate_result,
            raw_summary,
            results_dir=out_results,
            horizon=horizon,
            allocator=allocator_name,
            metrics_path=str(path),
            seed=seed,
        )
        result["promotion_gate"] = {
            "can_promote": gate_result.can_promote,
            "risk_level": gate_result.risk_level,
            "summary": gate_result.summary,
            "gates_passed": sum(1 for g in gate_result.gates if g.passed),
            "gates_total": len(gate_result.gates),
        }
        result["gate_result_path"] = str(gate_path)

    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SL walk-forward: LightGBM scores + rule allocator backtest"
    )
    parser.add_argument(
        "--period",
        default=None,
        help="Single period name (omit for all planned periods)",
    )
    parser.add_argument("--horizon", type=int, default=5, choices=(5, 10))
    parser.add_argument(
        "--allocator", default=None, help="S2: rule-based allocator (use 'rule')"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--write-metrics",
        action="store_true",
        help="Write Gate-compatible metrics JSON to results_dir",
    )
    parser.add_argument(
        "--gate",
        action="store_true",
        help="S3: run promotion_gate on SL metrics after walk-forward",
    )
    args = parser.parse_args(argv)

    out = args.output_dir
    if out is None and args.period:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path(tempfile.gettempdir()) / f"sl_{args.period or 'all'}_{stamp}"

    if args.period:
        result = run_single_period(
            args.period,
            horizon=args.horizon,
            allocator_name=args.allocator,
            seed=args.seed,
            output_dir=out,
            write_metrics=args.write_metrics or args.allocator is not None,
        )
        payload = {
            k: result[k]
            for k in (
                "period",
                "horizon",
                "allocator",
                "output_dir",
                "n_tickers",
                "n_oos_days",
                "score_mean",
                "score_std",
            )
            if k in result
        }
        if "backtest" in result:
            m = result["backtest"]["metrics"]
            payload["backtest"] = {
                "return_pct": round(m["total_return"] * 100, 2),
                "mdd_pct": round(m["max_drawdown"] * 100, 2),
                "sortino": round(m["sortino"], 2),
                "turnover": round(m["turnover"], 4),
                "avg_cash": round(m["avg_cash_weight"] * 100, 2),
            }
        if "metrics_path" in result:
            payload["metrics_path"] = result["metrics_path"]
        print(json.dumps(payload, indent=2))
        print(f"Scores: {result['scores_path']}")
        if "period_metrics_path" in result:
            print(f"Period metrics: {result['period_metrics_path']}")
        return 0

    if args.allocator is None:
        raise SystemExit("Full walk-forward requires --allocator rule (S2 backtest).")

    result = run_walk_forward_sl(
        horizon=args.horizon,
        allocator_name=args.allocator,
        seed=args.seed,
        output_dir=out,
        run_gate=args.gate,
    )
    overall = result["overall"]
    payload = {
        "horizon": result["horizon"],
        "allocator": result["allocator"],
        "metrics_path": result["metrics_path"],
        "overall_return_pct": round(overall["total_return"] * 100, 2),
        "overall_mdd_pct": round(overall["max_drawdown"] * 100, 2),
        "overall_sortino": round(overall["sortino"], 2),
        "periods": list(result["periods"].keys()),
        "skipped_periods": result.get("skipped_periods", {}),
    }
    if "promotion_gate" in result:
        payload["promotion_gate"] = result["promotion_gate"]
        payload["gate_result_path"] = result["gate_result_path"]
    print(json.dumps(payload, indent=2))
    if "promotion_gate" in result:
        status = "APPROVED" if result["promotion_gate"]["can_promote"] else "BLOCKED"
        safe_summary = (
            result["promotion_gate"]["summary"]
            .replace("\u2713", "[PASS]")
            .replace("\u2717", "[FAIL]")
        )
        print(f"\nSL Promotion Gate: {status} - {safe_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
