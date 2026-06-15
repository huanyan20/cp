"""Read-only SL h10 status checker.

This script reports the active SL candidate state without generating new
metrics, orders, or gate artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from settings import load_settings
from sl_pipeline.candidate import CURRENT_CANDIDATE_ID, utc_now_iso
from sl_pipeline.gate import (
    promotion_result_to_dict,
    read_sl_metric_files,
    run_sl_promotion_gate,
    sl_gate_result_path,
)


WATCH_PERIODS = ("2022_BEAR", "2024H2", "2025H1")


def _round(value: Any, digits: int = 4) -> float | None:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _overall_rows(records: list[dict]) -> list[dict]:
    rows = []
    for record in sorted(records, key=lambda r: int(r.get("seed", -1))):
        overall = record.get("overall", {})
        rows.append(
            {
                "seed": record.get("seed"),
                "candidate_id": record.get("candidate_id"),
                "return": _round(overall.get("total_return")),
                "max_drawdown": _round(overall.get("max_drawdown")),
                "sortino": _round(overall.get("sortino")),
                "turnover": _round(overall.get("turnover")),
                "avg_cash_weight": _round(overall.get("avg_cash_weight")),
                "source": Path(record.get("path", "")).name,
            }
        )
    return rows


def _weak_period_rows(records: list[dict]) -> list[dict]:
    rows = []
    for record in sorted(records, key=lambda r: int(r.get("seed", -1))):
        for period in WATCH_PERIODS:
            metrics = record.get("periods", {}).get(period)
            if not metrics:
                continue
            rows.append(
                {
                    "seed": record.get("seed"),
                    "period": period,
                    "return": _round(metrics.get("total_return")),
                    "max_drawdown": _round(metrics.get("max_drawdown")),
                    "sortino": _round(metrics.get("sortino")),
                    "turnover": _round(metrics.get("turnover")),
                    "avg_cash_weight": _round(metrics.get("avg_cash_weight")),
                    "long_exposure": _round(metrics.get("long_exposure")),
                    "top_holdings": metrics.get("top_holdings", {}),
                }
            )
    return rows


def _stale_gate_warning(results_dir: Path, records: list[dict], horizon: int) -> str | None:
    gate_path = sl_gate_result_path(results_dir, horizon=horizon, allocator="rule", seed=None)
    if not gate_path.exists():
        return f"{gate_path.name} is missing."
    if not records:
        return "No metrics are available to compare against the multiseed gate."
    latest_metric_mtime = max(float(r.get("metrics_modified_at") or 0.0) for r in records)
    gate_mtime = gate_path.stat().st_mtime
    if gate_mtime < latest_metric_mtime:
        return (
            f"{gate_path.name} is older than the latest h{horizon} metrics; "
            "regenerate the multiseed gate before making promotion decisions."
        )
    try:
        gate_payload = json.loads(gate_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return f"{gate_path.name} is not valid JSON."
    gate_candidate = gate_payload.get("candidate_id")
    if gate_candidate != CURRENT_CANDIDATE_ID:
        return (
            f"{gate_path.name} candidate_id={gate_candidate!r} does not match "
            f"active candidate_id={CURRENT_CANDIDATE_ID!r}."
        )
    return None


def _dry_run_status(root: Path) -> dict[str, Any]:
    signal_path = root / "signal.json"
    diff_path = root / "trade_guard_diff.json"
    report_path = root / "results_dir" / "daily_dry_run_report.json"
    status: dict[str, Any] = {
        "signal_path": str(signal_path),
        "diff_path": str(diff_path),
        "report_path": str(report_path),
        "available": {
            "signal": signal_path.exists(),
            "diff": diff_path.exists(),
            "daily_report": report_path.exists(),
        },
        "warnings": [],
    }
    signal = {}
    diff = {}
    latest_report = {}
    try:
        if signal_path.exists():
            signal = json.loads(signal_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        status["warnings"].append("signal.json is not valid JSON.")
    try:
        if diff_path.exists():
            diff = json.loads(diff_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        status["warnings"].append("trade_guard_diff.json is not valid JSON.")
    try:
        if report_path.exists():
            history = json.loads(report_path.read_text(encoding="utf-8"))
            if isinstance(history, list) and history:
                latest_report = history[-1]
    except json.JSONDecodeError:
        status["warnings"].append("daily_dry_run_report.json is not valid JSON.")

    plan = diff.get("plan", {}) if diff else {}
    buys = plan.get("buys", {}) or {}
    sells = plan.get("sells", {}) or {}
    risk = diff.get("risk_checks", {}) if diff else {}
    signal_metadata = signal.get("metadata", {}) if signal else {}
    status.update(
        {
            "signal_id": signal.get("signal_id") if signal else None,
            "signal_candidate_id": signal_metadata.get("candidate_id"),
            "signal_gate_status": signal_metadata.get("gate_status"),
            "risk_check_passed": risk.get("passed"),
            "observed_total_exposure": _round(risk.get("observed_total_exposure")),
            "observed_max_single_weight": _round(risk.get("observed_max_single_weight")),
            "diff_order_counts": {"buys": len(buys), "sells": len(sells)},
            "latest_report_order_counts": {
                "buys": latest_report.get("generated_buys"),
                "sells": latest_report.get("generated_sells"),
            },
        }
    )
    if latest_report:
        if latest_report.get("generated_buys") != len(buys):
            status["warnings"].append("daily report buy count differs from trade_guard diff.")
        if latest_report.get("generated_sells") != len(sells):
            status["warnings"].append("daily report sell count differs from trade_guard diff.")
    return status


def build_status(results_dir: Path, horizon: int) -> dict[str, Any]:
    settings = load_settings()
    records = [
        record
        for record in read_sl_metric_files(results_dir)
        if int(record.get("horizon", 0)) == horizon
    ]
    active_records = [
        record for record in records if record.get("candidate_id") == CURRENT_CANDIDATE_ID
    ]
    records_for_gate = active_records or records
    result, raw_summary, _ = run_sl_promotion_gate(
        results_dir=results_dir,
        target_horizon=horizon,
        target_candidate_id=CURRENT_CANDIDATE_ID if active_records else None,
    )
    stale_warning = _stale_gate_warning(results_dir, records_for_gate, horizon)
    warnings = []
    if not active_records:
        warnings.append(
            f"No h{horizon} metrics are tagged with active candidate_id={CURRENT_CANDIDATE_ID}."
        )
    if stale_warning:
        warnings.append(stale_warning)
    return {
        "generated_at": utc_now_iso(),
        "candidate_id": CURRENT_CANDIDATE_ID,
        "horizon": horizon,
        "results_dir": str(results_dir),
        "promotion_state": "dry_run_only" if not result.core_gate_approved else "gate_approved",
        "gate": promotion_result_to_dict(result),
        "summary": raw_summary,
        "seed_metrics": _overall_rows(records_for_gate),
        "weak_periods": _weak_period_rows(records_for_gate),
        "dry_run": _dry_run_status(settings.paths.root_dir),
        "warnings": warnings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only SL h10 status check.")
    parser.add_argument("--dir", type=Path, default=Path("results_dir"))
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--fail-on-blocked", action="store_true")
    args = parser.parse_args(argv)

    status = build_status(args.dir, args.horizon)
    print(json.dumps(status, indent=2, ensure_ascii=False))
    if args.fail_on_blocked and not status["gate"]["core_gate_approved"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
