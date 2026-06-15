import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env_config import (
    ENV_CONFIG_VERSION,
    build_env_config_snapshot,
    get_current_env_config_hash,
)
from promotion_gate import run_promotion_gate
from settings import load_settings
from sl_pipeline.comparison import (
    build_sl_vs_rl_comparison,
    gate_comparison_markdown,
    overall_comparison_markdown,
    period_comparison_markdown,
)
from sl_pipeline.gate import (
    promotion_result_to_dict,
    read_sl_metric_files,
    run_sl_promotion_gate,
)
from sl_pipeline.candidate import CURRENT_CANDIDATE_ID, utc_now_iso

SETTINGS = load_settings()

METRICS = [
    ("sortino", "OOS Sortino", "number"),
    ("max_drawdown", "Max Drawdown", "pct"),
    ("total_return", "Total Return", "pct"),
    ("avg_cash_weight", "Avg Cash", "pct"),
    ("cash_weight_std", "Cash Std", "pct"),
    ("cash_corr_next_return", "Cash/NextRet Corr", "number"),
    ("turnover", "Turnover", "pct"),
    ("win_rate", "Win Rate", "pct"),
]

BASELINE_METRICS = [
    "buy_and_hold",
    "Semi_2x",
    "0050",
]


def _read_metric_files(results_dir: str):
    records = []
    for path in glob.glob(os.path.join(results_dir, "metrics_*.json")):
        filename = os.path.basename(path)
        match = re.fullmatch(
            r"metrics_(ppo|sac)_(enabled|disabled)(_with_features)?_wf_seed\d+\.json",
            filename,
        )
        if not match:
            continue
        with open(path, encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception as exc:
                print(f"[WARN] Cannot read {path}: {exc}")
                continue
        if "overall" not in data:
            continue
        variant = "with_features" if match.group(3) else "base"
        records.append({"path": path, "variant": variant, **data})
    return records


def _filter_records_by_env_config(
    records: list[dict],
    env_config_hash: str | None = None,
    env_config_version: str | None = None,
    current_env_only: bool = True,
) -> tuple[list[dict], list[str]]:
    """Keep metrics comparable; exclude mixed reward/env generations by default."""
    notes: list[str] = []
    if not records:
        return records, notes

    if env_config_hash:
        matched = [r for r in records if r.get("env_config_hash") == env_config_hash]
        if not matched:
            notes.append(
                f"No metrics matched env_config_hash={env_config_hash}; "
                f"found hashes: {sorted({r.get('env_config_hash', 'legacy') for r in records})}."
            )
        else:
            notes.append(f"Filtered to env_config_hash={env_config_hash} ({len(matched)} file(s)).")
        return matched, notes

    if env_config_version:
        matched = [
            r for r in records
            if r.get("env_config_version") == env_config_version
        ]
        if not matched:
            notes.append(
                f"No metrics matched env_config_version={env_config_version}; "
                f"found versions: {sorted({r.get('env_config_version', 'legacy') for r in records})}."
            )
        else:
            notes.append(
                f"Filtered to env_config_version={env_config_version} ({len(matched)} file(s))."
            )
        return matched, notes

    if not current_env_only:
        hashes = {r.get("env_config_hash") for r in records if r.get("env_config_hash")}
        if len(hashes) > 1:
            notes.append(
                "Multiple env_config_hash values detected; including all (--include-all-env-configs)."
            )
        return records, notes

    current_hash = get_current_env_config_hash(SETTINGS)
    matched = [r for r in records if r.get("env_config_hash") == current_hash]
    legacy = [r for r in records if "env_config_hash" not in r]

    if matched:
        if legacy:
            notes.append(
                f"Excluded {len(legacy)} legacy metric file(s) without env_config_hash "
                f"(current={current_hash}, version={ENV_CONFIG_VERSION})."
            )
        if len({r.get("env_config_hash") for r in matched}) > 1:
            notes.append("Multiple env_config_hash values remain after filter.")
        else:
            notes.append(
                f"Using current env config only: version={ENV_CONFIG_VERSION}, hash={current_hash} "
                f"({len(matched)} file(s))."
            )
        return matched, notes

    if legacy:
        notes.append(
            f"No metrics tagged with current env_config_hash={current_hash}; "
            f"falling back to {len(legacy)} legacy file(s)."
        )
        return legacy, notes

    return records, notes


def _read_optional_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            return {}


def _fmt(value, kind):
    if pd.isna(value):
        value = 0.0
    if kind == "pct":
        return f"{value * 100:.2f}%"
    return f"{value:.2f}"


def _fmt_mean_std(mean, std, kind):
    if pd.isna(std):
        std = 0.0
    return f"{_fmt(mean, kind)} +/- {_fmt(std, kind)}"


def _sl_summary_markdown(raw_summary: list[dict]) -> str:
    if not raw_summary:
        return "No SL baseline metrics found (`metrics_sl_*.json`).\n"
    rows = []
    for entry in raw_summary:
        rows.append(
            {
                "Strategy": entry.get("variant", "sl_rule"),
                "Candidate": entry.get("candidate_id", "legacy"),
                "Label Mode": entry.get("label_mode", "legacy"),
                "Horizon": f"{entry.get('horizon', 5)}d",
                "Seeds": len(entry.get("seeds", [])),
                "OOS Sortino": _fmt_mean_std(
                    entry.get("sortino_mean", 0.0),
                    entry.get("sortino_std", 0.0),
                    "number",
                ),
                "Max Drawdown": _fmt_mean_std(
                    entry.get("max_drawdown_mean", 0.0),
                    entry.get("max_drawdown_std", 0.0),
                    "pct",
                ),
                "Total Return": _fmt_mean_std(
                    entry.get("total_return_mean", 0.0),
                    entry.get("total_return_std", 0.0),
                    "pct",
                ),
                "Turnover": _fmt_mean_std(
                    entry.get("turnover_mean", 0.0),
                    entry.get("turnover_std", 0.0),
                    "pct",
                ),
                "Cash Behavior": entry.get("cash_behavior", "n/a"),
            }
        )
    return pd.DataFrame(rows).to_markdown(index=False) + "\n"


def _sl_period_markdown(period_df: pd.DataFrame) -> str:
    if period_df.empty:
        return "No SL period breakdown.\n"
    lines = []
    for period in sorted(period_df["period"].unique()):
        sub = period_df[period_df["period"] == period]
        rows = []
        for _, row in sub.iterrows():
            rows.append(
                {
                    "Seed": row.get("seed"),
                    "Total Return": _fmt(row.get("total_return", 0.0), "pct"),
                    "Max Drawdown": _fmt(row.get("max_drawdown", 0.0), "pct"),
                    "Sortino": _fmt(row.get("sortino", 0.0), "number"),
                    "Turnover": _fmt(row.get("turnover", 0.0), "pct"),
                    "Avg Cash": _fmt(row.get("avg_cash_weight", 0.0), "pct"),
                }
            )
        lines.append(f"### {period}\n\n")
        lines.append(pd.DataFrame(rows).to_markdown(index=False))
        lines.append("\n\n")
    return "".join(lines)


def _overall_dataframe(records):
    rows = []
    for record in records:
        overall = record.get("overall", {})
        row = {
            "algo": record.get("algo", "unknown"),
            "cash_mode": record.get("cash_mode", "unknown"),
            "variant": record.get("variant", "base"),
            "seed": record.get("seed", "unknown"),
            "path": record.get("path", ""),
        }
        for key, _, _ in METRICS:
            row[key] = float(overall.get(key, 0.0))
        rows.append(row)
    return pd.DataFrame(rows)


def _period_dataframe(records):
    rows = []
    for record in records:
        for period_name, metrics in record.get("periods", {}).items():
            row = {
                "algo": record.get("algo", "unknown"),
                "cash_mode": record.get("cash_mode", "unknown"),
                "variant": record.get("variant", "base"),
                "seed": record.get("seed", "unknown"),
                "period": period_name,
                "test_start": metrics.get("test_start"),
                "test_end": metrics.get("test_end"),
                "was_clamped": metrics.get("was_clamped", False),
            }
            for key, _, _ in METRICS:
                row[key] = float(metrics.get(key, 0.0))
            rows.append(row)
    return pd.DataFrame(rows)


def classify_cash_behavior(raw):
    if raw["cash_mode"] != "enabled":
        return "cash disabled"
    if raw.get("avg_cash_weight_mean", 0.0) < 0.01:
        return "weak cash usage"
    if raw.get("cash_weight_std_mean", 0.0) < 0.01:
        return "static cash"
    return "active cash"


def _summary_tables(overall_df, period_df):
    grouped = overall_df.groupby(["algo", "cash_mode", "variant"])
    summary_rows = []
    raw_summary = []
    for (algo, cash_mode, variant), group in grouped:
        entry = {
            "Algo": algo.upper(),
            "Cash": cash_mode,
            "Variant": variant,
            "Seeds": int(group["seed"].nunique()),
        }
        raw = {
            "algo": algo,
            "cash_mode": cash_mode,
            "variant": variant,
            "seeds": sorted([int(s) for s in group["seed"].unique()]),
        }
        for key, label, kind in METRICS:
            mean = float(group[key].mean())
            std = float(group[key].std(ddof=0))
            entry[label] = _fmt_mean_std(mean, std, kind)
            raw[f"{key}_mean"] = mean
            raw[f"{key}_std"] = std
        raw["cash_behavior"] = classify_cash_behavior(raw)
        raw_summary.append(raw)
        summary_rows.append(entry)

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        # O3: base variant is the main ranking; with_features is a risk overlay and
        # must never outrank base. Sort base-first, then by Sortino/MDD/return. When
        # only with_features results exist, the best overlay still falls through.
        raw_df = pd.DataFrame(raw_summary)
        raw_df["_variant_rank"] = (raw_df["variant"] != "base").astype(int)
        raw_df = raw_df.sort_values(
            by=["_variant_rank", "sortino_mean", "max_drawdown_mean", "total_return_mean"],
            ascending=[True, False, True, False],
        ).drop(columns=["_variant_rank"])
        sorted_pairs = [
            (row["algo"].upper(), row["cash_mode"], row["variant"])
            for _, row in raw_df.iterrows()
        ]
        summary_df["_order"] = summary_df.apply(
            lambda row: sorted_pairs.index((row["Algo"], row["Cash"], row["Variant"])),
            axis=1,
        )
        summary_df = summary_df.sort_values("_order").drop(columns=["_order"])
        raw_summary = raw_df.to_dict(orient="records")

    period_summary = []
    if not period_df.empty:
        for (period, algo, cash_mode, variant), group in period_df.groupby(
            ["period", "algo", "cash_mode", "variant"]
        ):
            row = {
                "Period": period,
                "Algo": algo.upper(),
                "Cash": cash_mode,
                "Variant": variant,
                "Seeds": int(group["seed"].nunique()),
                "Test End": ", ".join(sorted(set(group["test_end"].dropna().astype(str)))) or "n/a",
                "Clamped": bool(group["was_clamped"].any()),
            }
            for key, label, kind in [
                ("total_return", "Total Return", "pct"),
                ("max_drawdown", "Max Drawdown", "pct"),
                ("avg_cash_weight", "Avg Cash", "pct"),
                ("win_rate", "Win Rate", "pct"),
            ]:
                row[label] = _fmt_mean_std(
                    float(group[key].mean()), float(group[key].std(ddof=0)), kind
                )
            period_summary.append(row)

    return summary_df, raw_summary, pd.DataFrame(period_summary)


def _make_conclusions(raw_summary, period_df):
    lines = []
    if not raw_summary:
        return ["No usable walk-forward metrics found."]

    best = raw_summary[0]
    lines.append(
        f"Best ranked model: {best['algo'].upper()} / cash={best['cash_mode']} "
        f"/ variant={best.get('variant', 'base')} "
        f"(Sortino {_fmt(best['sortino_mean'], 'number')}, "
        f"MDD {_fmt(best['max_drawdown_mean'], 'pct')})."
    )

    cash_enabled = [r for r in raw_summary if r["cash_mode"] == "enabled"]
    if cash_enabled:
        active = [r for r in cash_enabled if r["cash_behavior"] == "active cash"]
        if active:
            lines.append(
                "Dynamic cash behavior: at least one cash-enabled model shows active cash adjustment."
            )
        else:
            lines.append(
                "Dynamic cash behavior: not confirmed; cash usage is weak or too static."
            )
    else:
        lines.append("Dynamic cash behavior: no cash-enabled result set found.")

    if not period_df.empty:
        period_returns = period_df.groupby("period")["total_return"].mean().sort_values()
        worst_period = period_returns.index[0]
        worst_value = float(period_returns.iloc[0])
        lines.append(
            f"Main risk period: {worst_period} with average return {_fmt(worst_value, 'pct')}."
        )

        negative_periods = period_returns[period_returns < 0]
        if len(negative_periods) > 0:
            names = ", ".join(negative_periods.index.tolist())
            lines.append(
                f"Regime sensitivity warning: negative average OOS periods detected in {names}."
            )

        contribution = period_returns.abs() / max(period_returns.abs().sum(), 1e-8)
        dominant = contribution[contribution > 0.5]
        if len(dominant) > 0:
            lines.append(
                "Concentration warning: one period dominates absolute performance contribution "
                f"({dominant.index[0]})."
            )

    if best.get("seeds") and len(best["seeds"]) < 3:
        lines.append(
            "Evidence level: insufficient; fewer than 3 seeds are available for the best group."
        )
    elif len(raw_summary) < 4:
        lines.append(
            "Evidence level: incomplete; PPO/SAC x cash/no-cash matrix is not fully populated."
        )
    else:
        lines.append(
            "Evidence level: baseline matrix is populated; compare seed dispersion before live use."
        )

    lines.append(
        "Do not promote to live trading unless Sortino, drawdown, cash behavior, and turnover all remain acceptable across seeds."
    )
    return lines


def generate_report(
    results_dir="results_dir",
    output_md=None,
    output_json=None,
    env_config_hash: str | None = None,
    env_config_version: str | None = None,
    current_env_only: bool = True,
):
    output_md = output_md or str(SETTINGS.paths.experiment_report_md)
    output_json = output_json or str(SETTINGS.paths.experiment_summary_json)
    env_config_hash = env_config_hash or SETTINGS.research.env_config_hash
    env_config_version = env_config_version or SETTINGS.research.env_config_version
    current_env_config = build_env_config_snapshot(SETTINGS)

    print(f"=== Generate experiment report from {results_dir} ===")
    all_records = _read_metric_files(results_dir)
    sl_records = read_sl_metric_files(results_dir)
    if not all_records and not sl_records:
        print("No RL or SL metrics files found.")
        return

    records: list[dict] = []
    filter_notes: list[str] = []
    if all_records:
        records, filter_notes = _filter_records_by_env_config(
            all_records,
            env_config_hash=env_config_hash,
            env_config_version=env_config_version,
            current_env_only=current_env_only,
        )
        for note in filter_notes:
            print(f"[env_config] {note}")
        if not records:
            print("No RL metrics remain after env_config filter.")

    overall_df = _overall_dataframe(records) if records else pd.DataFrame()
    period_df = _period_dataframe(records) if records else pd.DataFrame()
    summary_df, raw_summary, period_summary_df = (
        _summary_tables(overall_df, period_df) if records else (pd.DataFrame(), [], pd.DataFrame())
    )
    conclusions = _make_conclusions(raw_summary, period_df) if records else []
    baseline_summary = _read_optional_json(os.path.join(results_dir, "baseline_summary.json"))
    ablation_summary = _read_optional_json(os.path.join(results_dir, "ablation_summary.json"))
    stress_summary = _read_optional_json(os.path.join(results_dir, "stress_summary.json"))

    sl_promotion_result = None
    sl_raw_summary: list[dict] = []
    sl_period_df = pd.DataFrame()
    sl_notes: list[str] = []
    if sl_records:
        sl_horizons = {int(r.get("horizon", 0)) for r in sl_records}
        sl_target_horizon = 10 if 10 in sl_horizons else None
        sl_promotion_result, sl_raw_summary, sl_period_df = run_sl_promotion_gate(
            results_dir=results_dir,
            target_horizon=sl_target_horizon,
            target_candidate_id=CURRENT_CANDIDATE_ID,
        )
        if not sl_raw_summary:
            sl_notes.append(
                f"No SL metrics tagged candidate_id={CURRENT_CANDIDATE_ID}; "
                "falling back to horizon-only legacy metrics for diagnostics."
            )
            sl_promotion_result, sl_raw_summary, sl_period_df = run_sl_promotion_gate(
                results_dir=results_dir,
                target_horizon=sl_target_horizon,
            )
        if sl_promotion_result and sl_raw_summary:
            from sl_pipeline.gate import save_sl_gate_result
            save_horizon = sl_target_horizon or int(sl_raw_summary[0].get("horizon", 0) or 0)
            save_sl_gate_result(
                sl_promotion_result,
                sl_raw_summary,
                results_dir=Path(results_dir),
                horizon=save_horizon,
                allocator="rule",
                seed=None
            )

    sl_vs_rl: dict = {}
    promotion_result = None
    if raw_summary:
        promotion_result = run_promotion_gate(
            raw_summary=raw_summary,
            period_df=period_df if not period_df.empty else None,
            baseline_summary=baseline_summary,
            ablation_summary=ablation_summary,
            stress_summary=stress_summary,
            min_seeds=getattr(SETTINGS.research, "promotion_min_seeds", 3),
            sortino_threshold=getattr(SETTINGS.research, "promotion_sortino_threshold", 0.8),
            max_drawdown_limit=getattr(SETTINGS.research, "promotion_max_drawdown", 0.35),
            turnover_limit=getattr(SETTINGS.research, "promotion_turnover_limit", 0.10),
        )

    if raw_summary and sl_raw_summary:
        sl_vs_rl = build_sl_vs_rl_comparison(
            raw_summary=raw_summary,
            sl_raw_summary=sl_raw_summary,
            period_df=period_df,
            sl_period_df=sl_period_df,
            rl_promotion=promotion_result,
            sl_promotion=sl_promotion_result,
        )
        if records:
            best_rl_record = next(
                (
                    r
                    for r in records
                    if r.get("algo") == sl_vs_rl["rl_candidate"].get("algo")
                    and r.get("cash_mode") == sl_vs_rl["rl_candidate"].get("cash_mode")
                    and r.get("variant") == sl_vs_rl["rl_candidate"].get("variant")
                ),
                records[0],
            )
            sl_vs_rl["rl_candidate"]["env_config_version"] = best_rl_record.get(
                "env_config_version"
            )

    md = "# Experiment Report\n\n"
    md += "> Generated by `experiment_report.py`.\n"
    md += f"> Generated at: `{utc_now_iso()}`.\n"
    md += f"> Active SL candidate: `{CURRENT_CANDIDATE_ID}`.\n"
    md += "> Live/RPA remains blocked unless the active SL candidate gate is approved.\n"
    md += "> Ranking priority: `OOS Sortino` first, then lower `Max Drawdown`, then higher `Total Return`.\n"
    md += "> Metrics are grouped by `algo`, `cash_mode`, and `variant` so feature experiments do not mix with base runs.\n"
    md += (
        f"> Env config filter: version=`{ENV_CONFIG_VERSION}`, "
        f"hash=`{current_env_config['hash']}` "
        f"({len(records)}/{len(all_records)} metric file(s) included).\n"
    )
    for note in filter_notes:
        md += f"> {note}\n"
    md += "\n"

    md += "## 0. Promotion Decision (RL)\n\n"
    if promotion_result is not None:
        if promotion_result.core_gate_approved:
            md += "### ✓ MODEL ELIGIBLE FOR PROMOTION\n\n"
        else:
            md += "### ✗ MODEL NOT ELIGIBLE FOR PROMOTION\n\n"
        md += f"Risk Level: **{promotion_result.risk_level.upper()}**\n\n"
        for gate in promotion_result.gates:
            status = "✓" if gate.passed else "✗"
            md += f"- {status} **{gate.name}**: {gate.message}\n"
        md += f"\n**Summary**: {promotion_result.summary}\n\n"
    else:
        md += "_No RL walk-forward metrics in scope; see §8 SL Baseline._\n\n"

    md += "## 1. Conclusions\n\n"
    if conclusions:
        for line in conclusions:
            md += f"- {line}\n"
    else:
        md += "- RL conclusions skipped (no RL metrics after filter).\n"

    md += "\n## 2. Promotion Checklist\n\n"
    md += "- At least 3 seeds should show stable Sortino before promotion.\n"
    md += "- Max drawdown must stay within the accepted risk budget across periods, not only on average.\n"
    md += "- Cash-enabled models must show meaningful and adaptive cash behavior, not static cash exposure.\n"
    md += "- Turnover must remain low enough after realistic fees, tax, and slippage.\n"
    md += "- Capital-flow overnight features must beat the baseline in ablation before becoming a default input.\n"

    md += "\n## 3. Walk-Forward Summary\n\n"
    if summary_df.empty:
        md += "No summary data."
    else:
        base_df = summary_df[summary_df["Variant"] == "base"].drop(columns=["Variant"])
        overlay_df = summary_df[summary_df["Variant"] == "with_features"].drop(
            columns=["Variant"]
        )
        md += "### 3a. Main Ranking (base features)\n\n"
        md += base_df.to_markdown(index=False) if not base_df.empty else "No base-feature runs."
        md += "\n\n### 3b. Risk Overlay (with_features — not in main ranking)\n\n"
        md += (
            "> overnight features are a risk-suppression overlay (R5/O6). They reduce drawdown "
            "and turnover but historically hurt Sortino/return, so they are excluded from the "
            "main promotion ranking.\n\n"
        )
        md += (
            overlay_df.to_markdown(index=False)
            if not overlay_df.empty
            else "No with_features overlay runs."
        )
    md += "\n\n## 4. Period Breakdown\n\n"
    if not period_summary_df.empty:
        for period in sorted(period_summary_df["Period"].unique()):
            md += f"### {period}\n\n"
            table = period_summary_df[period_summary_df["Period"] == period].drop(
                columns=["Period"]
            )
            md += table.to_markdown(index=False)
            md += "\n\n"
    else:
        md += "No period data.\n"

    md += "## 5. Cash Behavior Rules\n\n"
    md += "- Avg cash below 1% means the model is effectively full-stock most of the time.\n"
    md += "- Cash std below 1% means cash exposure is too static to prove dynamic risk control.\n"
    md += "- Cash/next-return correlation is only a diagnostic, not a promotion criterion by itself.\n"
    md += "- Evaluate cash behavior together with Sortino, max drawdown, turnover, and weak-period performance.\n"

    best_seed_count = len(raw_summary[0].get("seeds", [])) if raw_summary else 0
    md += "\n## 6. Next Steps\n\n"
    if best_seed_count < 3:
        md += "1. Add at least one more seed for the best-ranked group.\n"
    else:
        md += "1. Add comparable cash-disabled or SAC runs before promoting the model family.\n"
    md += "2. Investigate weak OOS periods before treating the model as robust.\n"
    md += "3. Compare cash-enabled and cash-disabled models using the same seed set.\n"
    md += "4. Verify that RPA and pending-buy logic correctly handle cash and guard states.\n"

    md += "\n## 7. Baselines And Ablations\n\n"
    if baseline_summary:
        md += "- Baseline summary loaded and included in promotion review.\n"
    else:
        md += "- Baseline summary missing; add buy-and-hold, `Semi_2x`, and `0050` comparisons before promotion.\n"
    if ablation_summary:
        md += "- Feature ablation summary loaded and included in promotion review.\n"
    else:
        md += "- Feature ablation summary missing; compare overnight features on/off before promotion.\n"
    if stress_summary:
        md += "- Stress test summary loaded and included in promotion review.\n"
    else:
        md += "- Stress test summary missing; add fee and slippage sensitivity before promotion.\n"

    md += "\n## 8. SL Baseline (Supervised Learning)\n\n"
    md += (
        "> Isolated from RL main ranking. Run: "
        "`python -m sl_pipeline.walk_forward_sl --allocator rule --gate`.\n\n"
    )
    for note in sl_notes:
        md += f"> Warning: {note}\n"
    if sl_notes:
        md += "\n"
    if sl_raw_summary:
        best_sl = sl_raw_summary[0]
        md += (
            f"Best SL config: **{best_sl.get('variant', 'sl_rule')}** "
            f"/ candidate `{best_sl.get('candidate_id', 'legacy')}` "
            f"(Sortino {_fmt(best_sl.get('sortino_mean', 0.0), 'number')}, "
            f"MDD {_fmt(best_sl.get('max_drawdown_mean', 0.0), 'pct')}).\n\n"
        )
        md += "### 8a. SL Summary\n\n"
        md += _sl_summary_markdown(sl_raw_summary)
        md += "\n### 8b. SL Period Breakdown\n\n"
        md += _sl_period_markdown(sl_period_df)
        if sl_promotion_result is not None:
            md += "### 8c. SL Promotion Gate\n\n"
            status = "ELIGIBLE" if sl_promotion_result.core_gate_approved else "BLOCKED"
            md += f"**{status}** (risk: {sl_promotion_result.risk_level.upper()})\n\n"
            if not sl_promotion_result.core_gate_approved:
                md += "**Live/RPA status: BLOCKED. Dry-run observation only.**\n\n"
            for gate in sl_promotion_result.gates:
                mark = "✓" if gate.passed else "✗"
                md += f"- {mark} **{gate.name}**: {gate.message}\n"
            md += f"\n{sl_promotion_result.summary}\n"
        if sl_vs_rl:
            md += "\n### 8d. SL vs RL Comparison (R6)\n\n"
            md += (
                "> Best RL = top of §3a main ranking (current env). "
                "Best SL = top of §8a. Same walk-forward periods and Gate thresholds.\n\n"
            )
            if sl_vs_rl.get("rl_candidate", {}).get("label"):
                md += f"- **RL candidate**: {sl_vs_rl['rl_candidate']['label']}"
                env_ver = sl_vs_rl["rl_candidate"].get("env_config_version")
                if env_ver:
                    md += f" (env `{env_ver}`)"
                md += "\n"
            if sl_vs_rl.get("sl_candidate", {}).get("label"):
                md += f"- **SL candidate**: {sl_vs_rl['sl_candidate']['label']}\n"
            md += "\n#### Overall Metrics\n\n"
            md += overall_comparison_markdown(sl_vs_rl)
            md += "\n#### Per-Period\n\n"
            md += period_comparison_markdown(sl_vs_rl)
            md += "\n#### Gate Status\n\n"
            md += gate_comparison_markdown(sl_vs_rl)
            md += "\n#### Verdict\n\n"
            for line in sl_vs_rl.get("verdict", []):
                md += f"- {line}\n"
    else:
        md += (
            "No `metrics_sl_*.json` found. Generate with "
            "`python -m sl_pipeline.walk_forward_sl --allocator rule --gate`.\n"
        )

    with open(output_md, "w", encoding="utf-8") as f:
        f.write(md)

    # Add promotion result to JSON
    summary_dict = {
        "generated_at": utc_now_iso(),
        "active_sl_candidate_id": CURRENT_CANDIDATE_ID,
        "ranking": ["sortino desc", "max_drawdown asc", "total_return desc"],
        "env_config_filter": {
            "version": ENV_CONFIG_VERSION,
            "hash": current_env_config["hash"],
            "snapshot": current_env_config,
            "included_files": len(records),
            "total_files": len(all_records),
            "notes": filter_notes,
        },
        "conclusions": conclusions,
        "summary": raw_summary,
        "baselines": baseline_summary,
        "ablations": ablation_summary,
        "stress_tests": stress_summary,
        "source_files": [r["path"] for r in records],
        # New: Promotion gate result
        "promotion_gate": (
            {
                "core_gate_approved": promotion_result.core_gate_approved,
                "full_gate_approved": promotion_result.full_gate_approved,
                "risk_level": promotion_result.risk_level,
                "summary": promotion_result.summary,
                "gates": [
                    {
                        "name": g.name,
                        "passed": g.passed,
                        "message": g.message,
                        "details": g.details,
                    }
                    for g in promotion_result.gates
                ],
            }
            if promotion_result is not None
            else None
        ),
        "sl_baseline": {
            "summary": sl_raw_summary,
            "notes": sl_notes,
            "source_files": [r["path"] for r in sl_records],
            "promotion_gate": (
                promotion_result_to_dict(sl_promotion_result)
                if sl_promotion_result is not None
                else None
            ),
        },
        "sl_vs_rl_comparison": sl_vs_rl or None,
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=4, ensure_ascii=False)

    print(f"[OK] wrote {output_md}")
    print(f"[OK] wrote {output_json}")
    if not summary_df.empty:
        print(summary_df.to_markdown(index=False))
    if promotion_result is not None:
        print(f"\n{promotion_result}")
    if sl_promotion_result is not None:
        print(f"\nSL Gate: {sl_promotion_result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate experiment report from metrics")
    parser.add_argument("--dir", type=str, default="results_dir", help="Results directory")
    parser.add_argument("--output-md", type=str, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument(
        "--env-config-hash",
        type=str,
        default=None,
        help="Only include metrics with this 8-char env_config_hash",
    )
    parser.add_argument(
        "--env-config-version",
        type=str,
        default=None,
        help="Only include metrics with this env_config_version label (e.g. r4)",
    )
    parser.add_argument(
        "--include-all-env-configs",
        action="store_true",
        help="Disable current-env-only filter and include every metrics_*.json",
    )
    args = parser.parse_args()
    generate_report(
        results_dir=args.dir,
        output_md=args.output_md,
        output_json=args.output_json,
        env_config_hash=args.env_config_hash,
        env_config_version=args.env_config_version,
        current_env_only=not args.include_all_env_configs,
    )
