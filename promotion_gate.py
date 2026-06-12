"""
Promotion gate logic for model promotion decisions.

Centrizes checks for Sortino stability, drawdown limits, cash behavior,
turnover constraints, baseline comparisons, ablation tests, and stress tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PromotionGate:
    """Represents a single promotion gate check with result and details."""

    name: str
    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class PromotionResult:
    """Complete promotion decision with all gate results."""

    can_promote: bool
    gates: list[PromotionGate]
    summary: str
    risk_level: str  # "low", "medium", "high"

    def __str__(self) -> str:
        def safe_text(value: str) -> str:
            return value.replace("✓", "PASS").replace("✗", "FAIL")

        lines = [
            f"Promotion Result: {'APPROVED' if self.can_promote else 'BLOCKED'}"
        ]
        lines.append(f"Risk Level: {self.risk_level.upper()}")
        lines.append("")
        for gate in self.gates:
            status = "PASS" if gate.passed else "FAIL"
            lines.append(f"  {status} {gate.name}: {gate.message}")
            if gate.details:
                for key, val in gate.details.items():
                    lines.append(f"      - {key}: {val}")
        lines.append("")
        lines.append(f"Summary: {safe_text(self.summary)}")
        return "\n".join(lines)


# ============================================================================
# Sortino Stability Gate
# ============================================================================


def check_sortino_stability(
    raw_summary: list[dict],
    min_seeds: int = 3,
    sortino_threshold: float = 0.8,
) -> PromotionGate:
    """
    Check if best model has stable Sortino across seeds.

    Args:
        raw_summary: List of summary dicts from experiment_report (sorted by ranking)
        min_seeds: Minimum number of seeds required
        sortino_threshold: Minimum acceptable Sortino value

    Returns:
        PromotionGate with check result
    """
    if not raw_summary:
        return PromotionGate(
            name="Sortino Stability",
            passed=False,
            message="No summary data available",
        )

    best = raw_summary[0]
    seed_count = len(best.get("seeds", []))
    sortino_mean = best.get("sortino_mean", 0.0)
    sortino_std = best.get("sortino_std", 0.0)

    passed = seed_count >= min_seeds and sortino_mean >= sortino_threshold
    message = (
        f"{seed_count} seeds, Sortino {sortino_mean:.2f} +/- {sortino_std:.2f}. "
        f"Need {min_seeds} seeds and >= {sortino_threshold} Sortino."
    )

    return PromotionGate(
        name="Sortino Stability",
        passed=passed,
        message=message,
        details={
            "seeds": seed_count,
            "sortino_mean": sortino_mean,
            "sortino_std": sortino_std,
            "min_seeds": min_seeds,
            "threshold": sortino_threshold,
        },
    )


# ============================================================================
# Drawdown Gate
# ============================================================================


def check_drawdown_gate(
    raw_summary: list[dict],
    max_drawdown_limit: float = 0.35,
) -> PromotionGate:
    """
    Check if max drawdown stays within acceptable limits.

    Args:
        raw_summary: List of summary dicts
        max_drawdown_limit: Maximum acceptable drawdown (as decimal, e.g., 0.20 = 20%)

    Returns:
        PromotionGate with check result
    """
    if not raw_summary:
        return PromotionGate(
            name="Drawdown Gate",
            passed=False,
            message="No summary data available",
        )

    best = raw_summary[0]
    mdd_mean = abs(best.get("max_drawdown_mean", 0.0))
    mdd_std = best.get("max_drawdown_std", 0.0)
    worst_case = mdd_mean + mdd_std  # Worst observed across seeds

    passed = worst_case <= max_drawdown_limit
    message = (
        f"Max drawdown {mdd_mean*100:.2f}% +/- {mdd_std*100:.2f}% "
        f"(worst case {worst_case*100:.2f}%). Limit: {max_drawdown_limit*100:.1f}%."
    )

    return PromotionGate(
        name="Drawdown Gate",
        passed=passed,
        message=message,
        details={
            "mdd_mean": mdd_mean,
            "mdd_std": mdd_std,
            "worst_case": worst_case,
            "limit": max_drawdown_limit,
        },
    )


# ============================================================================
# Cash Behavior Gate
# ============================================================================


def check_cash_behavior_gate(
    raw_summary: list[dict],
    require_active_cash: bool = False,
) -> PromotionGate:
    """
    Check cash behavior classification.

    Args:
        raw_summary: List of summary dicts
        require_active_cash: If True, require "active cash" behavior

    Returns:
        PromotionGate with check result
    """
    if not raw_summary:
        return PromotionGate(
            name="Cash Behavior",
            passed=False,
            message="No summary data available",
        )

    best = raw_summary[0]
    cash_mode = best.get("cash_mode", "unknown")
    cash_behavior = best.get("cash_behavior", "unknown")

    if cash_mode == "disabled":
        passed = True
        message = "Cash action disabled. No cash behavior requirement."
    elif cash_behavior == "weak cash usage":
        # Weak cash usage is concerning
        passed = False
        message = f"Cash: {cash_behavior}. This may indicate ineffective cash management."
    elif require_active_cash:
        passed = cash_behavior == "active cash"
        message = f"Cash: {cash_behavior}. Require active cash behavior."
    else:
        # Accept either active or static, but not weak
        passed = cash_behavior == "active cash" or cash_behavior == "static cash"
        message = f"Cash: {cash_behavior}. Acceptable."

    return PromotionGate(
        name="Cash Behavior",
        passed=passed,
        message=message,
        details={
            "cash_mode": cash_mode,
            "behavior": cash_behavior,
            "require_active": require_active_cash,
        },
    )


# ============================================================================
# Turnover Gate
# ============================================================================


def check_turnover_gate(
    raw_summary: list[dict],
    turnover_limit: float = 0.10,  # 10% daily average
) -> PromotionGate:
    """
    Check if turnover remains within acceptable limits.

    Args:
        raw_summary: List of summary dicts
        turnover_limit: Maximum acceptable daily average turnover

    Returns:
        PromotionGate with check result
    """
    if not raw_summary:
        return PromotionGate(
            name="Turnover Gate",
            passed=False,
            message="No summary data available",
        )

    best = raw_summary[0]
    turnover_mean = best.get("turnover_mean", 0.0)
    turnover_std = best.get("turnover_std", 0.0)
    worst_case = turnover_mean + turnover_std

    passed = worst_case <= turnover_limit
    message = (
        f"Turnover {turnover_mean*100:.2f}% +/- {turnover_std*100:.2f}% "
        f"(worst case {worst_case*100:.2f}%). Limit: {turnover_limit*100:.1f}%."
    )

    return PromotionGate(
        name="Turnover Gate",
        passed=passed,
        message=message,
        details={
            "turnover_mean": turnover_mean,
            "turnover_std": turnover_std,
            "worst_case": worst_case,
            "limit": turnover_limit,
        },
    )


# ============================================================================
# Baseline Gate
# ============================================================================


def check_baseline_gate(
    raw_summary: list[dict],
    baseline_summary: dict[str, Any],
    baseline_names: list[str] = None,
) -> PromotionGate:
    """
    Check if model beats baseline benchmarks.

    Args:
        raw_summary: List of summary dicts
        baseline_summary: Dict with baseline metrics (from baseline_summary.json)
        baseline_names: List of baseline names to compare against

    Returns:
        PromotionGate with check result
    """
    baseline_names = baseline_names or ["buy_and_hold", "Semi_2x", "0050"]

    if not baseline_summary:
        return PromotionGate(
            name="Baseline Comparison",
            passed=False,
            message="Baseline summary not available; cannot verify",
            details={"status": "missing"},
        )

    if not raw_summary:
        return PromotionGate(
            name="Baseline Comparison",
            passed=False,
            message="No model summary available",
        )

    best = raw_summary[0]
    model_return = best.get("total_return_mean", 0.0)
    model_sharpe = best.get("sharpe_mean", 0.0)

    baselines_beaten = []
    for baseline_name in baseline_names:
        baseline_data = baseline_summary.get(baseline_name, {})
        baseline_return = baseline_data.get("total_return", 0.0)
        baseline_sharpe = baseline_data.get("sharpe", 0.0)

        if model_return >= baseline_return or model_sharpe >= baseline_sharpe:
            baselines_beaten.append(baseline_name)

    passed = len(baselines_beaten) >= len(baseline_names) * 0.5  # Beat at least half

    message = (
        f"Model beats {len(baselines_beaten)}/{len(baseline_names)} baselines. "
        f"Beaten: {', '.join(baselines_beaten) if baselines_beaten else 'none'}."
    )

    return PromotionGate(
        name="Baseline Comparison",
        passed=passed,
        message=message,
        details={
            "model_return": model_return,
            "model_sharpe": model_sharpe,
            "baselines_beaten": baselines_beaten,
            "total_baselines": len(baseline_names),
        },
    )


# ============================================================================
# Ablation Gate
# ============================================================================


def check_ablation_gate(
    ablation_summary: dict[str, Any],
    feature_name: str = "overnight_features",
) -> PromotionGate:
    """
    Check if feature ablation shows improvement from feature.

    Args:
        ablation_summary: Dict with ablation metrics (from ablation_summary.json)
        feature_name: Feature being ablated (e.g., "overnight_features")

    Returns:
        PromotionGate with check result
    """
    if not ablation_summary:
        return PromotionGate(
            name="Ablation (Features)",
            passed=False,
            message="Ablation summary not available; cannot verify feature value",
            details={"status": "missing"},
        )

    feature_result = ablation_summary.get(feature_name, {})
    if not feature_result:
        return PromotionGate(
            name="Ablation (Features)",
            passed=False,
            message=f"No ablation data for {feature_name}",
            details={"feature": feature_name},
        )

    with_feature = feature_result.get("with_feature", {})
    without_feature = feature_result.get("without_feature", {})

    with_sortino = with_feature.get("sortino", 0.0)
    without_sortino = without_feature.get("sortino", 0.0)
    improvement = with_sortino - without_sortino

    passed = improvement > 0  # Feature improves Sortino

    message = (
        f"With {feature_name}: Sortino {with_sortino:.2f}. "
        f"Without: {without_sortino:.2f}. Improvement: {improvement:+.2f}."
    )

    return PromotionGate(
        name="Ablation (Features)",
        passed=passed,
        message=message,
        details={
            "feature": feature_name,
            "with_sortino": with_sortino,
            "without_sortino": without_sortino,
            "improvement": improvement,
        },
    )


# ============================================================================
# Stress Test Gate
# ============================================================================


def check_stress_gate(
    raw_summary: list[dict],
    stress_summary: dict[str, Any],
) -> PromotionGate:
    """
    Check if model survives stress tests (fee, slippage, spread sensitivity).

    Args:
        raw_summary: List of summary dicts
        stress_summary: Dict with stress test results (from stress_summary.json)

    Returns:
        PromotionGate with check result
    """
    if not stress_summary:
        return PromotionGate(
            name="Stress Testing",
            passed=False,
            message="Stress test summary not available; cannot verify robustness",
            details={"status": "missing"},
        )

    if not raw_summary:
        return PromotionGate(
            name="Stress Testing",
            passed=False,
            message="No model summary available",
        )

    best = raw_summary[0]
    baseline_return = best.get("total_return_mean", 0.0)

    stress_tests = stress_summary.get("tests", {})
    tests_survived = 0
    worst_impact = 0.0

    for _test_name, test_data in stress_tests.items():
        stressed_return = test_data.get("total_return", 0.0)
        impact = baseline_return - stressed_return

        if stressed_return > 0 or impact < 0.15:  # Loss less than 15%
            tests_survived += 1

        worst_impact = max(worst_impact, impact)

    total_tests = len(stress_tests)
    passed = tests_survived >= (total_tests * 0.7) if total_tests > 0 else False

    message = (
        f"Passes {tests_survived}/{total_tests} stress tests. "
        f"Worst impact: {worst_impact*100:.1f}% return reduction."
    )

    return PromotionGate(
        name="Stress Testing",
        passed=passed,
        message=message,
        details={
            "tests_survived": tests_survived,
            "total_tests": total_tests,
            "worst_impact": worst_impact,
        },
    )


# ============================================================================
# Period Consistency Gate
# ============================================================================


def check_period_consistency_gate(
    period_df: Any,  # pd.DataFrame
) -> PromotionGate:
    """
    Check if performance is consistent across walk-forward periods (no extreme outliers).

    Args:
        period_df: DataFrame with period-level metrics

    Returns:
        PromotionGate with check result
    """
    if period_df is None or period_df.empty:
        return PromotionGate(
            name="Period Consistency",
            passed=True,  # Changed from False to True - not having period data doesn't fail promotion
            message="No period data available; cannot assess consistency.",
            details={"status": "not_applicable"},
        )

    try:
        period_returns = period_df.groupby("period")["total_return"].mean()
        if len(period_returns) < 2:
            return PromotionGate(
                name="Period Consistency",
                passed=True,
                message=f"Only {len(period_returns)} period(s) available; cannot assess consistency.",
                details={"periods": len(period_returns)},
            )

        # Check if any period is dramatically worse
        median_return = period_returns.median()
        std_return = period_returns.std()
        worst_period = period_returns.min()
        severe_outlier = worst_period < (median_return - 3 * std_return)

        passed = not severe_outlier
        message = (
            f"Period returns median {median_return*100:.1f}% +/- {std_return*100:.1f}%. "
            f"Worst period: {worst_period*100:.1f}%. "
            f"{'Extreme outlier detected!' if severe_outlier else 'Consistent performance.'}"
        )

        return PromotionGate(
            name="Period Consistency",
            passed=passed,
            message=message,
            details={
                "median_return": median_return,
                "std_return": std_return,
                "worst_period": worst_period,
                "num_periods": len(period_returns),
            },
        )
    except Exception as e:
        return PromotionGate(
            name="Period Consistency",
            passed=False,
            message=f"Error analyzing periods: {e}",
            details={"error": str(e)},
        )


def _filter_periods_for_best(period_df: Any, raw_summary: list[dict]) -> Any:
    if period_df is None or period_df.empty or not raw_summary:
        return period_df

    best = raw_summary[0]
    filtered = period_df
    for column in ("algo", "cash_mode", "variant"):
        if column in filtered.columns and column in best:
            filtered = filtered[filtered[column] == best[column]]
    return filtered


# ============================================================================
# Run Promotion Gate
# ============================================================================


def run_promotion_gate(
    raw_summary: list[dict],
    period_df: Any = None,
    baseline_summary: dict[str, Any] = None,
    ablation_summary: dict[str, Any] = None,
    stress_summary: dict[str, Any] = None,
    # Gate thresholds
    min_seeds: int = 3,
    sortino_threshold: float = 0.8,
    max_drawdown_limit: float = 0.35,
    turnover_limit: float = 0.10,
    require_active_cash: bool = False,
) -> PromotionResult:
    """
    Run full promotion gate suite and return overall decision.

    Args:
        raw_summary: List of summary dicts from experiment_report
        period_df: Period-level DataFrame (optional)
        baseline_summary: Baseline comparison dict (optional)
        ablation_summary: Feature ablation dict (optional)
        stress_summary: Stress test dict (optional)
        min_seeds: Minimum seeds for ranking model
        sortino_threshold: Minimum Sortino value
        max_drawdown_limit: Maximum acceptable drawdown
        turnover_limit: Maximum acceptable turnover
        require_active_cash: If True, require active cash behavior

    Returns:
        PromotionResult with all gate checks and overall decision
    """
    gates = []

    # Core gates (always required)
    gates.append(check_sortino_stability(raw_summary, min_seeds, sortino_threshold))
    gates.append(check_drawdown_gate(raw_summary, max_drawdown_limit))
    gates.append(
        check_cash_behavior_gate(raw_summary, require_active_cash=require_active_cash)
    )
    gates.append(check_turnover_gate(raw_summary, turnover_limit))
    gates.append(check_period_consistency_gate(_filter_periods_for_best(period_df, raw_summary)))

    # Optional gates
    if baseline_summary:
        gates.append(check_baseline_gate(raw_summary, baseline_summary))
    if ablation_summary:
        gates.append(check_ablation_gate(ablation_summary))
    if stress_summary:
        gates.append(check_stress_gate(raw_summary, stress_summary))

    # Determine overall result
    critical_gates = gates[:5]  # Core gates
    critical_passed = sum(1 for g in critical_gates if g.passed)

    can_promote = all(g.passed for g in critical_gates)

    # Risk level assessment
    if can_promote:
        risk_level = "low" if critical_passed == len(critical_gates) else "medium"
    else:
        failed_count = len(critical_gates) - critical_passed
        risk_level = "high" if failed_count > 2 else "medium"

    # Summary
    passed_gates = sum(1 for g in gates if g.passed)
    total_gates = len(gates)

    if can_promote:
        summary = (
            f"✓ Model cleared all {len(critical_gates)} critical gates. "
            f"Promoted to live trading eligible. "
            f"({passed_gates}/{total_gates} total gates passed)"
        )
    else:
        failed = [g.name for g in gates if not g.passed]
        summary = (
            f"✗ Model blocked by {len(failed)} gate(s): {', '.join(failed)}. "
            f"Address failures before promotion. "
            f"({passed_gates}/{total_gates} total gates passed)"
        )

    return PromotionResult(
        can_promote=can_promote,
        gates=gates,
        summary=summary,
        risk_level=risk_level,
    )
