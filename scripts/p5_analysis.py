"""P5 Analysis: Generate baseline, ablation, and stress summary JSONs.

Usage
-----
    python p5_analysis.py baseline  [--dir results_dir] [--output path]
    python p5_analysis.py ablation  [--dir results_dir] [--output path]
    python p5_analysis.py stress    [--dir results_dir] [--output path]
    python p5_analysis.py all       [--dir results_dir]

Outputs
-------
    results_dir/baseline_summary.json  — market benchmark comparison
    results_dir/ablation_summary.json  — overnight-features on/off delta
    results_dir/stress_summary.json    — fee/slippage sensitivity
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np

from settings import AppSettings, load_settings

SETTINGS = load_settings()

def _build_stress_scenarios(settings: AppSettings) -> dict[str, dict[str, object]]:
    """Build the four stress-test scenarios from StressSettings (env-configurable)."""
    s = settings.stress
    return {
        "base": {
            "fee_rate": s.base_fee_rate,
            "tax_rate": s.base_tax_rate,
            "description": (
                f"standard broker ({s.base_fee_rate*100:.4f}% one-way "
                f"+ {s.base_tax_rate*100:.3f}% sell tax)"
            ),
        },
        "high_fee": {
            "fee_rate": s.high_fee_rate,
            "tax_rate": s.high_fee_tax_rate,
            "description": (
                f"high fee ({s.high_fee_rate*100:.4f}% one-way "
                f"+ {s.high_fee_tax_rate*100:.3f}% sell tax)"
            ),
        },
        "high_slippage": {
            "fee_rate": s.high_slippage_fee_rate,
            "tax_rate": s.high_slippage_tax_rate,
            "description": (
                f"high slippage ({s.high_slippage_fee_rate*100:.4f}% one-way "
                f"+ {s.high_slippage_tax_rate*100:.3f}% sell tax)"
            ),
        },
        "worst_case": {
            "fee_rate": s.worst_case_fee_rate,
            "tax_rate": s.worst_case_tax_rate,
            "description": (
                f"worst case ({s.worst_case_fee_rate*100:.4f}% one-way "
                f"+ {s.worst_case_tax_rate*100:.3f}% sell tax)"
            ),
        },
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_metrics_files(results_dir: str) -> list[dict]:
    """Read all valid metrics_*.json files that match the canonical naming pattern."""
    records = []
    for path in glob.glob(os.path.join(results_dir, "metrics_*.json")):
        filename = os.path.basename(path)
        if not re.fullmatch(
            r"metrics_(ppo|sac)_(enabled|disabled)(_with_features)?_wf_seed\d+\.json",
            filename,
        ):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            print(f"[WARN] Cannot read {path}: {exc}")
            continue
        if "overall" not in data:
            continue
        records.append({"path": path, "filename": filename, **data})
    return records


def _oos_date_range(records: list[dict]) -> tuple[str, str]:
    """Infer OOS date range from all period data across the given records."""
    starts: list[str] = []
    ends: list[str] = []
    for rec in records:
        for pm in rec.get("periods", {}).values():
            if pm.get("test_start"):
                starts.append(pm["test_start"])
            if pm.get("test_end"):
                ends.append(pm["test_end"])
    if not starts or not ends:
        return ("2024-07-01", str(date.today()))
    return (min(starts), max(ends))


def _count_oos_trading_days(records: list[dict]) -> int:
    """Estimate total OOS trading days (approx 126 per half-year period)."""
    periods_seen: set[str] = set()
    for rec in records:
        for pname in rec.get("periods", {}):
            periods_seen.add(pname)
    return len(periods_seen) * 126 if periods_seen else 504


def _price_series_metrics(prices: Any) -> dict[str, float]:
    """Compute return/risk metrics from a price series (requires pandas Series)."""
    prices = prices.dropna()
    if len(prices) < 2:
        return {"total_return": 0.0, "sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0}
    daily_rets = prices.pct_change().dropna().values.astype(float)
    total_return = float(prices.iloc[-1] / prices.iloc[0] - 1.0)
    mean_ret = float(np.mean(daily_rets))
    std_ret = float(np.std(daily_rets)) or 1e-8
    sharpe = float(mean_ret / std_ret * np.sqrt(252))
    neg = daily_rets[daily_rets < 0]
    downside = float(np.std(neg)) if len(neg) > 0 else 1e-8
    sortino = float(mean_ret / downside * np.sqrt(252))
    peak = float(prices.iloc[0])
    mdd = 0.0
    for v in prices.astype(float):
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        if dd > mdd:
            mdd = dd
    return {
        "total_return": round(total_return, 6),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "max_drawdown": round(mdd, 6),
    }


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------


def run_baseline(
    results_dir: str = "results_dir",
    output_path: str | None = None,
    cache_ttl_days: int = 7,
) -> Path:
    """Fetch market benchmarks over the OOS period and write baseline_summary.json.

    Requires network access (yfinance).  In test environments, patch
    ``p5_analysis.yf`` or pass ``output_path`` to a temp file.

    Parameters
    ----------
    cache_ttl_days:
        If the output file already exists and was written within this many days,
        skip the network fetch and return the cached file.  Set to 0 to disable.
    """
    out = Path(output_path or os.path.join(results_dir, "baseline_summary.json"))

    # N3: cache check — avoid repeated yfinance calls within TTL window
    if cache_ttl_days > 0 and out.exists():
        import time  # noqa: PLC0415

        age_days = (time.time() - out.stat().st_mtime) / 86400
        if age_days < cache_ttl_days:
            print(
                f"[baseline] cache hit ({age_days:.1f}d old, TTL={cache_ttl_days}d): {out}"
            )
            return out

    try:
        import yfinance as yf  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "yfinance is required for baseline generation.  "
            "Install with: pip install yfinance"
        ) from exc

    records = _read_metrics_files(results_dir)
    start_date, end_date = _oos_date_range(records)
    print(f"[baseline] OOS period: {start_date} → {end_date}")

    result: dict[str, Any] = {
        "period": {"start": start_date, "end": end_date},
        "generated_at": datetime.now().isoformat(),
    }

    benchmarks = {
        "Semi_2x": ("Semi_2x", "2x Leveraged CTBC Semiconductor ETF (00891.TW * 2)"),
        "0050": ("0050.TW", "Taiwan 50 ETF (元大台灣50)"),
        "buy_and_hold": ("0050.TW", "buy-and-hold proxy via 0050.TW (equal-weight tech30 approximation)"),
    }

    for key, (ticker, description) in benchmarks.items():
        try:
            if ticker == "Semi_2x":
                raw = yf.download("00891.TW", start=start_date, end=end_date, progress=False)
                close = raw["Close"].squeeze()
                daily_rets = close.pct_change().dropna()
                lev_rets = daily_rets * 2.0
                prices = (1 + lev_rets).cumprod()
                # Use close.index for prices but need to insert the initial value
                prices.loc[close.index[0]] = 1.0
                prices = prices.sort_index()
                metrics = _price_series_metrics(prices)
            else:
                raw = yf.download(ticker, start=start_date, end=end_date, progress=False)
                close = raw["Close"].squeeze()
                metrics = _price_series_metrics(close)

            metrics["description"] = description
            result[key] = metrics
            print(f"[baseline] {key}: total_return={metrics['total_return']:.2%}")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Cannot fetch {ticker}: {exc}")
            result[key] = {
                "total_return": 0.0,
                "sharpe": 0.0,
                "sortino": 0.0,
                "max_drawdown": 0.0,
                "description": description,
                "error": str(exc),
            }

    out.write_text(json.dumps(result, indent=4, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] wrote {out}")
    return out



# ---------------------------------------------------------------------------
# Ablation
# ---------------------------------------------------------------------------


def run_ablation(
    results_dir: str = "results_dir",
    output_path: str | None = None,
    feature_name: str = "overnight_features",
) -> Path:
    """Compare with-features vs without-features metrics and write ablation_summary.json."""
    records = _read_metrics_files(results_dir)

    # Partition records by feature flag
    with_feat: dict[tuple, dict] = {}
    without_feat: dict[tuple, dict] = {}
    for rec in records:
        algo = rec.get("algo", "")
        cash_mode = rec.get("cash_mode", "")
        seed = rec.get("seed", "")
        key = (algo, cash_mode, seed)
        if "_with_features_" in rec["filename"]:
            with_feat[key] = rec
        else:
            without_feat[key] = rec

    matched_keys = sorted(set(with_feat) & set(without_feat))

    metric_keys = ["sortino", "total_return", "max_drawdown", "sharpe", "turnover", "win_rate"]

    def _avg_overall(recs: list[dict]) -> dict[str, float]:
        agg: dict[str, list[float]] = {k: [] for k in metric_keys}
        for r in recs:
            overall = r.get("overall", {})
            for k in metric_keys:
                if k in overall:
                    agg[k].append(float(overall[k]))
        return {k: round(float(np.mean(v)), 6) if v else 0.0 for k, v in agg.items()}

    if not matched_keys:
        print("[WARN] No matching seed pairs found for ablation (need both with- and without-features files).")
        entry: dict[str, Any] = {
            "matched_seeds": [],
            "verdict": "no matched seeds — cannot compare",
            "with_feature": {},
            "without_feature": {},
            "delta": {},
        }
    else:
        with_avg = _avg_overall([with_feat[k] for k in matched_keys])
        without_avg = _avg_overall([without_feat[k] for k in matched_keys])
        delta = {
            k: round(with_avg.get(k, 0.0) - without_avg.get(k, 0.0), 6)
            for k in metric_keys
        }

        sortino_up = delta.get("sortino", 0.0) > 0
        mdd_down = delta.get("max_drawdown", 0.0) < 0  # lower MDD is better
        verdict = (
            "feature improves sortino and drawdown"
            if sortino_up and mdd_down
            else "feature improves sortino"
            if sortino_up
            else "feature improves drawdown only"
            if mdd_down
            else "feature does not improve core metrics"
        )

        matched_seeds = sorted({k[2] for k in matched_keys})
        print(f"[ablation] matched seeds: {matched_seeds}")
        print(f"[ablation] sortino delta: {delta.get('sortino', 0.0):+.4f}")
        print(f"[ablation] verdict: {verdict}")

        entry = {
            "matched_seeds": matched_seeds,
            "verdict": verdict,
            "with_feature": with_avg,
            "without_feature": without_avg,
            "delta": delta,
        }

    result = {feature_name: entry}
    out = Path(output_path or os.path.join(results_dir, "ablation_summary.json"))
    out.write_text(json.dumps(result, indent=4, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] wrote {out}")
    return out


# ---------------------------------------------------------------------------
# Stress
# ---------------------------------------------------------------------------


def apply_cost_drag(
    total_return: float,
    avg_daily_turnover: float,
    trading_days: int,
    fee_rate: float,
    tax_rate: float,
) -> dict[str, float]:
    """Estimate cost-adjusted return under a given fee/tax scenario.

    Cost model
    ----------
    Per unit of portfolio turnover, a round-trip costs:
        buy_fee + sell_fee + sell_tax = 2 × fee_rate + tax_rate

    Total drag over the OOS period:
        cost_drag = avg_daily_turnover × (2 × fee_rate + tax_rate) × trading_days

    Adjusted return:
        stressed = (1 + total_return) × max(1 − cost_drag, 0) − 1
    """
    round_trip_cost = 2.0 * fee_rate + tax_rate
    total_cost_drag = avg_daily_turnover * round_trip_cost * trading_days
    stressed = (1.0 + total_return) * max(1.0 - total_cost_drag, 0.0) - 1.0
    return {
        "total_return": round(stressed, 6),
        "cost_drag": round(total_cost_drag, 6),
        "daily_cost_bps": round(avg_daily_turnover * round_trip_cost * 10_000, 4),
    }


def run_stress(
    results_dir: str = "results_dir",
    output_path: str | None = None,
    settings: AppSettings | None = None,
) -> Path:
    """Apply fee/slippage scenarios to the best model and write stress_summary.json.

    Parameters
    ----------
    settings:
        Optional ``AppSettings`` instance.  When *None*, uses the module-level
        ``SETTINGS`` singleton (loaded from env vars / defaults).
    """
    records = _read_metrics_files(results_dir)
    if not records:
        raise RuntimeError(f"No valid metrics files found in {results_dir}")

    _settings = settings or SETTINGS
    stress_scenarios = _build_stress_scenarios(_settings)

    # Best model by overall Sortino
    best = max(records, key=lambda r: r.get("overall", {}).get("sortino", 0.0))
    overall = best["overall"]
    base_return = float(overall.get("total_return", 0.0))
    avg_turnover = float(overall.get("turnover", 0.0))
    trading_days = _count_oos_trading_days([best])

    source_label = (
        f"{best.get('algo', '?')} / {best.get('cash_mode', '?')} / seed{best.get('seed', '?')}"
    )
    print(f"[stress] Best model: {source_label}")
    print(
        f"[stress] base_return={base_return:.2%}  "
        f"avg_daily_turnover={avg_turnover:.4%}  "
        f"trading_days={trading_days}"
    )

    tests: dict[str, dict] = {}
    for name, params in stress_scenarios.items():
        scenario = apply_cost_drag(
            total_return=base_return,
            avg_daily_turnover=avg_turnover,
            trading_days=trading_days,
            fee_rate=params["fee_rate"],
            tax_rate=params["tax_rate"],
        )
        scenario["description"] = params["description"]
        scenario["fee_rate"] = params["fee_rate"]
        scenario["tax_rate"] = params["tax_rate"]
        tests[name] = scenario
        print(
            f"  {name:16s}: return={scenario['total_return']:+.2%}  "
            f"drag={scenario['cost_drag']:.2%}"
        )

    result: dict[str, Any] = {
        "source": source_label,
        "baseline_return": base_return,
        "avg_daily_turnover": avg_turnover,
        "trading_days": trading_days,
        "generated_at": datetime.now().isoformat(),
        "tests": tests,
    }

    out = Path(output_path or os.path.join(results_dir, "stress_summary.json"))
    out.write_text(json.dumps(result, indent=4, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] wrote {out}")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="P5 analysis: generate baseline / ablation / stress summary JSONs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python p5_analysis.py ablation\n"
            "  python p5_analysis.py stress --dir results_dir\n"
            "  python p5_analysis.py all\n"
            "  python p5_analysis.py baseline  # requires network\n"
        ),
    )
    parser.add_argument(
        "command",
        choices=["baseline", "ablation", "stress", "all"],
        help="Which analysis to run ('all' runs ablation + stress + baseline).",
    )
    parser.add_argument(
        "--dir",
        default=str(SETTINGS.paths.results_dir),
        metavar="RESULTS_DIR",
        help="Path to results directory (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help="Override output file path (single-command mode only).",
    )
    args = parser.parse_args()

    if args.command == "baseline":
        run_baseline(args.dir, args.output)
    elif args.command == "ablation":
        run_ablation(args.dir, args.output)
    elif args.command == "stress":
        run_stress(args.dir, args.output)
    else:  # all
        run_ablation(args.dir)
        run_stress(args.dir)
        try:
            run_baseline(args.dir)
        except ImportError as exc:
            print(f"[SKIP] baseline skipped — {exc}")


if __name__ == "__main__":
    main()
