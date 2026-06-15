"""Promotion Gate helpers for SL walk-forward metrics (S3)."""

from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

from promotion_gate import PromotionResult, run_promotion_gate
from settings import load_settings
from sl_pipeline.candidate import (
    CURRENT_CANDIDATE_ID,
    current_candidate_metadata,
    utc_now_iso,
)

SETTINGS = load_settings()

SL_METRICS_PATTERN = re.compile(
    r"metrics_sl_(?P<allocator>\w+)_h(?P<horizon>\d+)_seed(?P<seed>\d+)\.json$"
)

METRIC_KEYS = (
    "sortino",
    "max_drawdown",
    "total_return",
    "avg_cash_weight",
    "cash_weight_std",
    "cash_corr_next_return",
    "turnover",
    "win_rate",
)


def read_sl_metric_files(results_dir: str | Path) -> list[dict]:
    """Load SL metrics JSON files (isolated from RL metrics_*.json namespace)."""
    results_dir = str(results_dir)
    records: list[dict] = []
    for path in glob.glob(os.path.join(results_dir, "metrics_sl_*.json")):
        filename = os.path.basename(path)
        match = SL_METRICS_PATTERN.fullmatch(filename)
        if not match:
            continue
        with open(path, encoding="utf-8") as handle:
            try:
                data = json.load(handle)
            except json.JSONDecodeError as exc:
                print(f"[WARN] Cannot read {path}: {exc}")
                continue
        if "overall" not in data or not data["overall"]:
            continue
        metadata = data.get("candidate_metadata", {})
        allocator = match.group("allocator")
        horizon = int(match.group("horizon"))
        candidate_id = (
            data.get("candidate_id")
            or metadata.get("candidate_id")
            or f"legacy_sl_{allocator}_h{horizon}"
        )
        label_mode = data.get("label_mode") or metadata.get("label_mode") or "legacy"
        records.append(
            {
                "path": path,
                "metrics_modified_at": (
                    Path(path).stat().st_mtime if Path(path).exists() else None
                ),
                "allocator": allocator,
                "horizon": horizon,
                "variant": f"sl_{allocator}_h{horizon}",
                "candidate_id": candidate_id,
                "label_mode": label_mode,
                **data,
            }
        )
    return records


def _classify_sl_cash_behavior(raw: dict) -> str:
    if raw.get("avg_cash_weight_mean", 0.0) < 0.01:
        return "weak cash usage"
    if raw.get("cash_weight_std_mean", 0.0) < 0.01:
        return "static cash"
    return "active cash"


def build_sl_raw_summary(records: list[dict]) -> list[dict]:
    """Convert SL metric files to promotion_gate raw_summary rows (sorted by Sortino)."""
    if not records:
        return []

    rows: list[dict] = []
    for record in records:
        overall = record.get("overall", {})
        rows.append(
            {
                "algo": record.get("algo", "sl_lightgbm"),
                "cash_mode": record.get("cash_mode", "enabled"),
                "variant": record.get("variant", "sl_rule"),
                "allocator": record.get("allocator", "rule"),
                "horizon": record.get("horizon", 5),
                "candidate_id": record.get("candidate_id", "legacy"),
                "label_mode": record.get("label_mode", "legacy"),
                "generated_at": record.get("generated_at"),
                "metrics_modified_at": record.get("metrics_modified_at"),
                "seed": record.get("seed"),
                "path": record.get("path", ""),
                **{key: float(overall.get(key, 0.0)) for key in METRIC_KEYS},
            }
        )

    frame = pd.DataFrame(rows)
    raw_summary: list[dict] = []
    group_cols = [
        "algo",
        "cash_mode",
        "variant",
        "allocator",
        "horizon",
        "candidate_id",
        "label_mode",
    ]
    for keys, group in frame.groupby(group_cols):
        algo, cash_mode, variant, allocator, horizon, candidate_id, label_mode = keys
        entry: dict[str, Any] = {
            "algo": algo,
            "cash_mode": cash_mode,
            "variant": variant,
            "allocator": allocator,
            "horizon": int(horizon),
            "candidate_id": candidate_id,
            "label_mode": label_mode,
            "seeds": sorted(int(s) for s in group["seed"].unique()),
            "source_files": sorted(str(p) for p in group["path"].dropna().unique()),
            "metric_generated_at": sorted(
                str(v) for v in group["generated_at"].dropna().unique()
            ),
        }
        modified = group["metrics_modified_at"].dropna()
        if not modified.empty:
            entry["latest_metrics_mtime"] = float(modified.max())
        for key in METRIC_KEYS:
            mean = float(group[key].mean())
            std = float(group[key].std(ddof=0)) if len(group) > 1 else 0.0
            entry[f"{key}_mean"] = mean
            entry[f"{key}_std"] = std
        entry["cash_behavior"] = _classify_sl_cash_behavior(entry)
        raw_summary.append(entry)

    raw_summary.sort(
        key=lambda item: (
            -item.get("sortino_mean", 0.0),
            item.get("max_drawdown_mean", 0.0),
            -item.get("total_return_mean", 0.0),
        )
    )
    return raw_summary


def build_sl_period_dataframe(records: list[dict]) -> pd.DataFrame:
    """Period-level DataFrame for SL promotion gate consistency checks."""
    rows: list[dict] = []
    for record in records:
        for period_name, metrics in record.get("periods", {}).items():
            rows.append(
                {
                    "algo": record.get("algo", "sl_lightgbm"),
                    "cash_mode": record.get("cash_mode", "enabled"),
                    "variant": record.get("variant", "sl_rule"),
                    "candidate_id": record.get("candidate_id", "legacy"),
                    "label_mode": record.get("label_mode", "legacy"),
                    "seed": record.get("seed"),
                    "period": period_name,
                    "test_start": metrics.get("test_start"),
                    "test_end": metrics.get("test_end"),
                    **{key: float(metrics.get(key, 0.0)) for key in METRIC_KEYS},
                }
            )
    return pd.DataFrame(rows)


def promotion_result_to_dict(result: PromotionResult) -> dict:
    return {
        "core_gate_approved": result.core_gate_approved,
        "full_gate_approved": result.full_gate_approved,
        "risk_level": result.risk_level,
        "summary": result.summary,
        "gates": [
            {
                "name": gate.name,
                "passed": gate.passed,
                "message": gate.message,
                "details": gate.details,
            }
            for gate in result.gates
        ],
    }


def run_sl_promotion_gate(
    metrics: dict | str | Path | None = None,
    *,
    results_dir: str | Path | None = None,
    min_seeds: int | None = None,
    sortino_threshold: float | None = None,
    max_drawdown_limit: float | None = None,
    turnover_limit: float | None = None,
    baseline_summary: dict[str, Any] | None = None,
    target_horizon: int | None = None,
    target_candidate_id: str | None = None,
) -> tuple[PromotionResult, list[dict], pd.DataFrame]:
    """Run promotion gate on SL metrics (no RL baseline/ablation/stress gates)."""
    sortino_threshold = (
        sortino_threshold
        if sortino_threshold is not None
        else SETTINGS.research.promotion_sortino_threshold
    )
    max_drawdown_limit = (
        max_drawdown_limit
        if max_drawdown_limit is not None
        else SETTINGS.research.promotion_max_drawdown
    )
    turnover_limit = (
        turnover_limit
        if turnover_limit is not None
        else SETTINGS.research.promotion_turnover_limit
    )

    records: list[dict]
    if metrics is not None:
        if isinstance(metrics, (str, Path)):
            path = Path(metrics)
            with path.open(encoding="utf-8") as handle:
                data = json.load(handle)
            records = [
                {
                    "path": str(path),
                    "allocator": data.get("allocator", "rule"),
                    "horizon": data.get("horizon", 5),
                    "variant": f"sl_{data.get('allocator', 'rule')}_h{data.get('horizon', 5)}",
                    **data,
                }
            ]
        else:
            records = [
                {
                    "allocator": metrics.get("allocator", "rule"),
                    "horizon": metrics.get("horizon", 5),
                    "variant": f"sl_{metrics.get('allocator', 'rule')}_h{metrics.get('horizon', 5)}",
                    **metrics,
                }
            ]
    else:
        records = read_sl_metric_files(results_dir or SETTINGS.paths.results_dir)

    if target_horizon is not None:
        records = [r for r in records if r.get("horizon") == target_horizon]
    if target_candidate_id is not None:
        records = [r for r in records if r.get("candidate_id") == target_candidate_id]

    if baseline_summary is None:
        try:
            baseline_path = Path(results_dir or SETTINGS.paths.results_dir) / "baseline_summary.json"
            if baseline_path.exists():
                with open(baseline_path, encoding="utf-8") as f:
                    baseline_summary = json.load(f)
        except Exception as e:
            print(f"[WARN] Could not load baseline_summary: {e}")

    try:
        stress_path = Path(results_dir or SETTINGS.paths.results_dir) / "stress_summary.json"
        if stress_path.exists():
            with open(stress_path, encoding="utf-8") as f:
                stress_summary = json.load(f)
        else:
            stress_summary = None
    except Exception as e:
        print(f"[WARN] Could not load stress_summary: {e}")
        stress_summary = None

    min_seeds = min_seeds if min_seeds is not None else SETTINGS.research.promotion_min_seeds
    raw_summary = build_sl_raw_summary(records)
    period_df = build_sl_period_dataframe(records)
    result = run_promotion_gate(
        raw_summary=raw_summary,
        period_df=period_df if not period_df.empty else None,
        baseline_summary=baseline_summary,
        ablation_summary=None,
        stress_summary=stress_summary,
        min_seeds=min_seeds,
        sortino_threshold=sortino_threshold,
        max_drawdown_limit=max_drawdown_limit,
        turnover_limit=turnover_limit,
        require_active_cash=False,
    )
    return result, raw_summary, period_df


def sl_gate_result_path(
    results_dir: Path,
    *,
    horizon: int,
    allocator: str = "rule",
    seed: int | None = None,
) -> Path:
    suffix = f"_seed{seed}" if seed is not None else "_multiseed"
    return results_dir / f"sl_gate_result_{allocator}_h{horizon}{suffix}.json"


def save_sl_gate_result(
    result: PromotionResult,
    raw_summary: list[dict],
    *,
    results_dir: Path,
    horizon: int,
    allocator: str = "rule",
    metrics_path: str | None = None,
    seed: int | None = None,
    candidate_metadata: dict[str, Any] | None = None,
) -> Path:
    """Persist SL gate outcome for experiment_report / audit."""
    candidate_id = (
        raw_summary[0].get("candidate_id")
        if raw_summary
        else CURRENT_CANDIDATE_ID
    )
    candidate_metadata = candidate_metadata or current_candidate_metadata(
        horizon=horizon,
        allocator=allocator,
        seed=seed,
    )
    source_metric_files = sorted(
        {
            str(path)
            for entry in raw_summary
            for path in entry.get("source_files", [])
        }
    )
    if metrics_path:
        source_metric_files.append(metrics_path)
        source_metric_files = sorted(set(source_metric_files))
    payload = {
        "strategy": "sl_rule",
        "allocator": allocator,
        "horizon": horizon,
        "candidate_id": candidate_id,
        "candidate_metadata": candidate_metadata,
        "generated_at": utc_now_iso(),
        "metrics_path": metrics_path,
        "source_metric_files": source_metric_files,
        "promotion_gate": promotion_result_to_dict(result),
        "summary": raw_summary,
    }
    path = sl_gate_result_path(results_dir, horizon=horizon, allocator=allocator, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
