"""SL vs RL walk-forward comparison for experiment_report (S4)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from promotion_gate import PromotionResult

COMPARE_METRICS = (
    ("sortino", "OOS Sortino", "number", "higher"),
    ("max_drawdown", "Max Drawdown", "pct", "lower"),
    ("total_return", "Total Return", "pct", "higher"),
    ("turnover", "Turnover", "pct", "lower"),
    ("avg_cash_weight", "Avg Cash", "pct", "neutral"),
    ("win_rate", "Win Rate", "pct", "higher"),
)

GATE_NAMES = (
    "Sortino Stability",
    "Drawdown Gate",
    "Cash Behavior",
    "Turnover Gate",
    "Period Consistency",
)


def _fmt_value(value: float, kind: str) -> str:
    if kind == "pct":
        return f"{value * 100:.2f}%"
    return f"{value:.2f}"


def _winner(
    rl_value: float,
    sl_value: float,
    direction: str,
) -> str:
    if direction == "neutral":
        return "—"
    if direction == "higher":
        if sl_value > rl_value + 1e-9:
            return "SL"
        if rl_value > sl_value + 1e-9:
            return "RL"
        return "tie"
    # lower is better (drawdown stored positive in metrics)
    if sl_value < rl_value - 1e-9:
        return "SL"
    if rl_value < sl_value - 1e-9:
        return "RL"
    return "tie"


def _delta_text(rl_value: float, sl_value: float, kind: str) -> str:
    delta = sl_value - rl_value
    if kind == "pct":
        return f"{delta * 100:+.2f}pp"
    return f"{delta:+.2f}"


def _rl_label(best_rl: dict) -> str:
    return (
        f"{str(best_rl.get('algo', 'rl')).upper()} / "
        f"cash={best_rl.get('cash_mode', 'n/a')} / "
        f"{best_rl.get('variant', 'base')}"
    )


def _sl_label(best_sl: dict) -> str:
    return f"SL {best_sl.get('variant', 'sl_rule')}"


def build_overall_comparison(
    best_rl: dict | None,
    best_sl: dict | None,
) -> list[dict[str, Any]]:
    """Row-wise SL vs best RL overall metrics."""
    if not best_rl or not best_sl:
        return []

    rows: list[dict[str, Any]] = []
    for key, label, kind, direction in COMPARE_METRICS:
        rl_val = float(best_rl.get(f"{key}_mean", 0.0))
        sl_val = float(best_sl.get(f"{key}_mean", 0.0))
        rows.append(
            {
                "metric": key,
                "label": label,
                "rl_value": rl_val,
                "sl_value": sl_val,
                "rl_display": _fmt_value(rl_val, kind),
                "sl_display": _fmt_value(sl_val, kind),
                "delta": sl_val - rl_val,
                "delta_display": _delta_text(rl_val, sl_val, kind),
                "winner": _winner(rl_val, sl_val, direction),
                "better": direction,
            }
        )
    return rows


def build_period_comparison(
    period_df: pd.DataFrame,
    sl_period_df: pd.DataFrame,
    *,
    rl_group: tuple[str, str, str] | None = None,
) -> list[dict[str, Any]]:
    """Per-period SL vs RL (best RL group means by default)."""
    if period_df.empty or sl_period_df.empty:
        return []

    rl_df = period_df.copy()
    if rl_group is not None:
        algo, cash_mode, variant = rl_group
        rl_df = rl_df[
            (rl_df["algo"] == algo)
            & (rl_df["cash_mode"] == cash_mode)
            & (rl_df["variant"] == variant)
        ]
    if rl_df.empty:
        rl_df = period_df.copy()

    sl_df = sl_period_df.copy()
    rows: list[dict[str, Any]] = []
    for period in sorted(set(rl_df["period"].unique()) & set(sl_df["period"].unique())):
        rl_sub = rl_df[rl_df["period"] == period]
        sl_sub = sl_df[sl_df["period"] == period]
        entry: dict[str, Any] = {"period": period}
        for key, label, kind, direction in COMPARE_METRICS[:4]:
            rl_val = float(rl_sub[key].mean())
            sl_val = float(sl_sub[key].mean())
            entry[f"rl_{key}"] = rl_val
            entry[f"sl_{key}"] = sl_val
            entry[f"{key}_winner"] = _winner(rl_val, sl_val, direction)
        rows.append(entry)
    return rows


def build_gate_comparison(
    rl_gate: PromotionResult | None,
    sl_gate: PromotionResult | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rl_by_name = {g.name: g for g in (rl_gate.gates if rl_gate else [])}
    sl_by_name = {g.name: g for g in (sl_gate.gates if sl_gate else [])}
    for name in GATE_NAMES:
        rl_g = rl_by_name.get(name)
        sl_g = sl_by_name.get(name)
        rows.append(
            {
                "gate": name,
                "rl_passed": rl_g.passed if rl_g else None,
                "sl_passed": sl_g.passed if sl_g else None,
                "rl_message": rl_g.message if rl_g else "n/a",
                "sl_message": sl_g.message if sl_g else "n/a",
            }
        )
    if rl_gate and sl_gate:
        rows.append(
            {
                "gate": "Overall",
                "rl_passed": rl_gate.can_promote,
                "sl_passed": sl_gate.can_promote,
                "rl_message": rl_gate.summary,
                "sl_message": sl_gate.summary,
            }
        )
    return rows


def build_sl_vs_rl_verdict(
    overall_rows: list[dict[str, Any]],
    *,
    rl_gate: PromotionResult | None,
    sl_gate: PromotionResult | None,
    sortino_ratio_target: float = 0.8,
) -> list[str]:
    """Narrative bullets for experiment_report §8d."""
    lines: list[str] = []
    if not overall_rows:
        return ["SL vs RL comparison unavailable (missing one side)."]

    by_metric = {row["metric"]: row for row in overall_rows}
    sortino_row = by_metric.get("sortino", {})
    mdd_row = by_metric.get("max_drawdown", {})
    turnover_row = by_metric.get("turnover", {})

    rl_sortino = float(sortino_row.get("rl_value", 0.0))
    sl_sortino = float(sortino_row.get("sl_value", 0.0))
    rl_mdd = float(mdd_row.get("rl_value", 0.0))
    sl_mdd = float(mdd_row.get("sl_value", 0.0))

    if rl_sortino > 0:
        ratio = sl_sortino / rl_sortino
        target = sortino_ratio_target * rl_sortino
        if sl_sortino >= target:
            lines.append(
                f"Sortino: SL ({sl_sortino:.2f}) reaches ≥{sortino_ratio_target:.0%} of RL "
                f"({rl_sortino:.2f}); meets SL success criterion."
            )
        else:
            lines.append(
                f"Sortino: SL ({sl_sortino:.2f}) is below {sortino_ratio_target:.0%} of RL "
                f"({rl_sortino:.2f}); ratio={ratio:.0%}."
            )
    else:
        lines.append(f"Sortino: SL={sl_sortino:.2f}, RL={rl_sortino:.2f}.")

    if sl_mdd < rl_mdd - 1e-6:
        lines.append(
            f"MDD: SL ({sl_mdd * 100:.2f}%) beats RL ({rl_mdd * 100:.2f}%) — "
            "vol-target + tiered breaker working as intended."
        )
    elif sl_mdd > rl_mdd + 1e-6:
        lines.append(
            f"MDD: SL ({sl_mdd * 100:.2f}%) worse than RL ({rl_mdd * 100:.2f}%)."
        )
    else:
        lines.append(f"MDD: SL and RL comparable ({sl_mdd * 100:.2f}%).")

    t_winner = turnover_row.get("winner", "tie")
    if t_winner == "SL":
        lines.append("Turnover: SL lower than RL (favorable for net alpha).")
    elif t_winner == "RL":
        lines.append("Turnover: RL lower than SL.")
    else:
        lines.append("Turnover: SL and RL comparable.")

    if rl_gate and sl_gate:
        if sl_gate.can_promote and not rl_gate.can_promote:
            lines.append(
                "Promotion: SL clears Gate while RL is BLOCKED — prioritize SL as risk baseline."
            )
        elif rl_gate.can_promote and not sl_gate.can_promote:
            lines.append(
                "Promotion: RL clears Gate while SL is BLOCKED — RL remains production candidate."
            )
        elif rl_gate.can_promote and sl_gate.can_promote:
            lines.append("Promotion: both SL and RL clear core gates — compare Sortino/MDD trade-off.")
        else:
            lines.append("Promotion: both SL and RL blocked — continue R6/SL iteration.")

    return lines


def build_sl_vs_rl_comparison(
    *,
    raw_summary: list[dict],
    sl_raw_summary: list[dict],
    period_df: pd.DataFrame,
    sl_period_df: pd.DataFrame,
    rl_promotion: PromotionResult | None,
    sl_promotion: PromotionResult | None,
) -> dict[str, Any]:
    """Full S4 comparison payload for experiment_report JSON + markdown."""
    best_rl = raw_summary[0] if raw_summary else None
    best_sl = sl_raw_summary[0] if sl_raw_summary else None

    rl_group = None
    if best_rl:
        rl_group = (
            best_rl.get("algo"),
            best_rl.get("cash_mode"),
            best_rl.get("variant"),
        )

    overall = build_overall_comparison(best_rl, best_sl)
    periods = build_period_comparison(period_df, sl_period_df, rl_group=rl_group)
    gates = build_gate_comparison(rl_promotion, sl_promotion)
    verdict = build_sl_vs_rl_verdict(
        overall,
        rl_gate=rl_promotion,
        sl_gate=sl_promotion,
    )

    return {
        "rl_candidate": {
            "label": _rl_label(best_rl) if best_rl else None,
            "algo": best_rl.get("algo") if best_rl else None,
            "cash_mode": best_rl.get("cash_mode") if best_rl else None,
            "variant": best_rl.get("variant") if best_rl else None,
            "seeds": best_rl.get("seeds", []) if best_rl else [],
            "env_config_version": None,
        },
        "sl_candidate": {
            "label": _sl_label(best_sl) if best_sl else None,
            "variant": best_sl.get("variant") if best_sl else None,
            "horizon": best_sl.get("horizon") if best_sl else None,
            "seeds": best_sl.get("seeds", []) if best_sl else [],
        },
        "overall": overall,
        "periods": periods,
        "gates": gates,
        "verdict": verdict,
        "rl_gate_approved": rl_promotion.can_promote if rl_promotion else None,
        "sl_gate_approved": sl_promotion.can_promote if sl_promotion else None,
    }


def overall_comparison_markdown(
    comparison: dict[str, Any],
    *,
    rl_label: str | None = None,
    sl_label: str | None = None,
) -> str:
    overall = comparison.get("overall", [])
    if not overall:
        return "_No SL vs RL overall comparison (need both metrics families)._\n"

    rl_label = rl_label or comparison.get("rl_candidate", {}).get("label", "Best RL")
    sl_label = sl_label or comparison.get("sl_candidate", {}).get("label", "Best SL")

    rows = []
    for row in overall:
        rows.append(
            {
                "Metric": row["label"],
                rl_label: row["rl_display"],
                sl_label: row["sl_display"],
                "Δ (SL−RL)": row["delta_display"],
                "Better": row["winner"],
            }
        )
    return pd.DataFrame(rows).to_markdown(index=False) + "\n"


def period_comparison_markdown(comparison: dict[str, Any]) -> str:
    periods = comparison.get("periods", [])
    if not periods:
        return "_No overlapping walk-forward periods for SL vs RL._\n"

    rows = []
    for entry in periods:
        rows.append(
            {
                "Period": entry["period"],
                "RL Sortino": f"{entry.get('rl_sortino', 0.0):.2f}",
                "SL Sortino": f"{entry.get('sl_sortino', 0.0):.2f}",
                "Sortino": entry.get("sortino_winner", "—"),
                "RL MDD": f"{entry.get('rl_max_drawdown', 0.0) * 100:.2f}%",
                "SL MDD": f"{entry.get('sl_max_drawdown', 0.0) * 100:.2f}%",
                "MDD": entry.get("max_drawdown_winner", "—"),
                "RL Return": f"{entry.get('rl_total_return', 0.0) * 100:.2f}%",
                "SL Return": f"{entry.get('sl_total_return', 0.0) * 100:.2f}%",
                "Return": entry.get("total_return_winner", "—"),
            }
        )
    return pd.DataFrame(rows).to_markdown(index=False) + "\n"


def gate_comparison_markdown(comparison: dict[str, Any]) -> str:
    gates = comparison.get("gates", [])
    if not gates:
        return "_No gate comparison available._\n"

    rows = []
    for entry in gates:
        rl_passed = entry.get("rl_passed")
        sl_passed = entry.get("sl_passed")
        rl_status = "n/a" if rl_passed is None else ("PASS" if rl_passed else "FAIL")
        sl_status = "n/a" if sl_passed is None else ("PASS" if sl_passed else "FAIL")
        rows.append(
            {
                "Gate": entry["gate"],
                "RL": rl_status,
                "SL": sl_status,
            }
        )
    return pd.DataFrame(rows).to_markdown(index=False) + "\n"
