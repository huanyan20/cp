import argparse
import glob
import json
import os
import re

import pandas as pd

from promotion_gate import run_promotion_gate
from settings import load_settings

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
    "^TWII",
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
        raw_df = pd.DataFrame(raw_summary).sort_values(
            by=["sortino_mean", "max_drawdown_mean", "total_return_mean"],
            ascending=[False, True, False],
        )
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
):
    output_md = output_md or str(SETTINGS.paths.experiment_report_md)
    output_json = output_json or str(SETTINGS.paths.experiment_summary_json)
    print(f"=== Generate experiment report from {results_dir} ===")
    records = _read_metric_files(results_dir)
    if not records:
        print("No metrics_*.json files found.")
        return

    overall_df = _overall_dataframe(records)
    period_df = _period_dataframe(records)
    summary_df, raw_summary, period_summary_df = _summary_tables(overall_df, period_df)
    conclusions = _make_conclusions(raw_summary, period_df)
    baseline_summary = _read_optional_json(os.path.join(results_dir, "baseline_summary.json"))
    ablation_summary = _read_optional_json(os.path.join(results_dir, "ablation_summary.json"))
    stress_summary = _read_optional_json(os.path.join(results_dir, "stress_summary.json"))

    # Run promotion gate analysis
    promotion_result = run_promotion_gate(
        raw_summary=raw_summary,
        period_df=period_df if not period_df.empty else None,
        baseline_summary=baseline_summary,
        ablation_summary=ablation_summary,
        stress_summary=stress_summary,
        # Use configurable thresholds from settings if available
        min_seeds=getattr(SETTINGS.research, "promotion_min_seeds", 3),
        sortino_threshold=getattr(SETTINGS.research, "promotion_sortino_threshold", 0.8),
        max_drawdown_limit=getattr(SETTINGS.research, "promotion_max_drawdown", 0.35),
        turnover_limit=getattr(SETTINGS.research, "promotion_turnover_limit", 0.10),
    )

    md = "# Experiment Report\n\n"
    md += "> Generated by `experiment_report.py`.\n"
    md += "> Ranking priority: `OOS Sortino` first, then lower `Max Drawdown`, then higher `Total Return`.\n"
    md += "> Metrics are grouped by `algo`, `cash_mode`, and `variant` so feature experiments do not mix with base runs.\n\n"

    # Add promotion gate result prominently
    md += "## 0. Promotion Decision\n\n"
    if promotion_result.can_promote:
        md += "### ✓ MODEL ELIGIBLE FOR PROMOTION\n\n"
    else:
        md += "### ✗ MODEL NOT ELIGIBLE FOR PROMOTION\n\n"
    md += f"Risk Level: **{promotion_result.risk_level.upper()}**\n\n"
    for gate in promotion_result.gates:
        status = "✓" if gate.passed else "✗"
        md += f"- {status} **{gate.name}**: {gate.message}\n"
    md += f"\n**Summary**: {promotion_result.summary}\n\n"

    md += "## 1. Conclusions\n\n"
    for line in conclusions:
        md += f"- {line}\n"

    md += "\n## 2. Promotion Checklist\n\n"
    md += "- At least 3 seeds should show stable Sortino before promotion.\n"
    md += "- Max drawdown must stay within the accepted risk budget across periods, not only on average.\n"
    md += "- Cash-enabled models must show meaningful and adaptive cash behavior, not static cash exposure.\n"
    md += "- Turnover must remain low enough after realistic fees, tax, and slippage.\n"
    md += "- Capital-flow overnight features must beat the baseline in ablation before becoming a default input.\n"

    md += "\n## 3. Walk-Forward Summary\n\n"
    md += summary_df.to_markdown(index=False) if not summary_df.empty else "No summary data."
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
        md += "- Baseline summary missing; add buy-and-hold, `^TWII`, and `0050` comparisons before promotion.\n"
    if ablation_summary:
        md += "- Feature ablation summary loaded and included in promotion review.\n"
    else:
        md += "- Feature ablation summary missing; compare overnight features on/off before promotion.\n"
    if stress_summary:
        md += "- Stress test summary loaded and included in promotion review.\n"
    else:
        md += "- Stress test summary missing; add fee and slippage sensitivity before promotion.\n"

    with open(output_md, "w", encoding="utf-8") as f:
        f.write(md)

    # Add promotion result to JSON
    summary_dict = {
        "ranking": ["sortino desc", "max_drawdown asc", "total_return desc"],
        "conclusions": conclusions,
        "summary": raw_summary,
        "baselines": baseline_summary,
        "ablations": ablation_summary,
        "stress_tests": stress_summary,
        "source_files": [r["path"] for r in records],
        # New: Promotion gate result
        "promotion_gate": {
            "can_promote": promotion_result.can_promote,
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
        },
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=4, ensure_ascii=False)

    print(f"[OK] wrote {output_md}")
    print(f"[OK] wrote {output_json}")
    if not summary_df.empty:
        print(summary_df.to_markdown(index=False))
    print(f"\n{promotion_result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate experiment report from metrics")
    parser.add_argument("--dir", type=str, default="results_dir", help="Results directory")
    parser.add_argument("--output-md", type=str, default="experiment_report.md")
    parser.add_argument("--output-json", type=str, default="experiment_summary.json")
    args = parser.parse_args()
    generate_report(
        results_dir=args.dir,
        output_md=args.output_md,
        output_json=args.output_json,
    )
