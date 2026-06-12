"""
Integration test demonstrating promotion_gate workflow.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from promotion_gate import run_promotion_gate


def test_promotion_gate_workflow():
    """Demonstrate typical promotion gate workflow."""

    # Scenario 1: Strong model passes promotion
    print("=" * 70)
    print("SCENARIO 1: Strong Model (PPO + Cash Enabled)")
    print("=" * 70)

    strong_model = [
        {
            "algo": "ppo",
            "cash_mode": "enabled",
            "seeds": [42, 43, 44],
            "sortino_mean": 1.35,
            "sortino_std": 0.12,
            "max_drawdown_mean": -0.15,
            "max_drawdown_std": 0.025,
            "total_return_mean": 0.32,
            "total_return_std": 0.04,
            "turnover_mean": 0.07,
            "turnover_std": 0.015,
            "sharpe_mean": 1.05,
            "avg_cash_weight": 0.08,
            "cash_weight_std": 0.09,
            "cash_behavior": "active cash",
        }
    ]

    result1 = run_promotion_gate(raw_summary=strong_model, period_df=None)
    print(result1)
    assert result1.can_promote, "Strong model should be approved"
    print()

    # Scenario 2: Weak model fails promotion
    print("=" * 70)
    print("SCENARIO 2: Weak Model (SAC + Cash Disabled)")
    print("=" * 70)

    weak_model = [
        {
            "algo": "sac",
            "cash_mode": "disabled",
            "seeds": [42],
            "sortino_mean": 0.45,
            "sortino_std": 0.2,
            "max_drawdown_mean": -0.42,
            "max_drawdown_std": 0.08,
            "total_return_mean": 0.02,
            "total_return_std": 0.15,
            "turnover_mean": 0.22,
            "turnover_std": 0.05,
            "sharpe_mean": -0.1,
            "avg_cash_weight": 0.0,
            "cash_weight_std": 0.0,
            "cash_behavior": "cash disabled",
        }
    ]

    result2 = run_promotion_gate(raw_summary=weak_model, period_df=None)
    print(result2)
    assert not result2.can_promote, "Weak model should be blocked"
    print()

    # Scenario 3: Medium model with optional gates
    print("=" * 70)
    print("SCENARIO 3: Medium Model (With Optional Gates)")
    print("=" * 70)

    medium_model = [
        {
            "algo": "ppo",
            "cash_mode": "enabled",
            "seeds": [42, 43],
            "sortino_mean": 0.92,
            "sortino_std": 0.18,
            "max_drawdown_mean": -0.18,
            "max_drawdown_std": 0.035,
            "total_return_mean": 0.18,
            "total_return_std": 0.08,
            "turnover_mean": 0.09,
            "turnover_std": 0.025,
            "sharpe_mean": 0.75,
            "avg_cash_weight": 0.03,
            "cash_weight_std": 0.05,
            "cash_behavior": "weak cash usage",
        }
    ]

    baseline_summary = {
        "buy_and_hold": {"total_return": 0.12, "sharpe": 0.5},
        "Semi_2x": {"total_return": 0.10, "sharpe": 0.45},
        "0050": {"total_return": 0.14, "sharpe": 0.55},
    }

    ablation_summary = {
        "overnight_features": {
            "with_feature": {"sortino": 0.92},
            "without_feature": {"sortino": 0.78},
        }
    }

    stress_summary = {
        "tests": {
            "fee_1bp": {"total_return": 0.17},
            "slippage_1bp": {"total_return": 0.16},
            "spread_2bp": {"total_return": 0.15},
        }
    }

    result3 = run_promotion_gate(
        raw_summary=medium_model,
        period_df=None,
        baseline_summary=baseline_summary,
        ablation_summary=ablation_summary,
        stress_summary=stress_summary,
        min_seeds=2,  # Lower threshold for this scenario
    )
    print(result3)
    print()

    print("=" * 70)
    print("✅ Promotion gate workflow demonstration complete!")
    print("=" * 70)


if __name__ == "__main__":
    test_promotion_gate_workflow()
