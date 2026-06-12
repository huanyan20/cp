"""
Unit tests for promotion_gate module.
"""

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from promotion_gate import (
    check_ablation_gate,
    check_baseline_gate,
    check_cash_behavior_gate,
    check_drawdown_gate,
    check_period_consistency_gate,
    check_sortino_stability,
    check_stress_gate,
    check_turnover_gate,
    run_promotion_gate,
)


class PromotionGateTests(unittest.TestCase):
    """Test promotion gate logic."""

    def setUp(self):
        """Set up test data."""
        self.good_summary = [
            {
                "algo": "ppo",
                "cash_mode": "enabled",
                "seeds": [42, 43, 44],
                "sortino_mean": 1.2,
                "sortino_std": 0.15,
                "max_drawdown_mean": -0.12,
                "max_drawdown_std": 0.03,
                "total_return_mean": 0.25,
                "total_return_std": 0.05,
                "turnover_mean": 0.08,
                "turnover_std": 0.02,
                "sharpe_mean": 0.95,
                "avg_cash_weight": 0.05,
                "cash_weight_std": 0.08,
                "cash_behavior": "active cash",
            }
        ]

        self.poor_summary = [
            {
                "algo": "sac",
                "cash_mode": "disabled",
                "seeds": [42],
                "sortino_mean": 0.5,
                "sortino_std": 0.1,
                "max_drawdown_mean": -0.35,
                "max_drawdown_std": 0.05,
                "total_return_mean": 0.05,
                "total_return_std": 0.08,
                "turnover_mean": 0.15,
                "turnover_std": 0.03,
                "sharpe_mean": 0.2,
                "avg_cash_weight": 0.0,
                "cash_weight_std": 0.0,
                "cash_behavior": "cash disabled",
            }
        ]

    def test_sortino_stability_pass(self):
        """Test Sortino stability with good results."""
        gate = check_sortino_stability(
            self.good_summary, min_seeds=3, sortino_threshold=0.8
        )
        self.assertTrue(gate.passed)
        self.assertIn("3 seeds", gate.message)

    def test_sortino_stability_fail_insufficient_seeds(self):
        """Test Sortino stability fails with too few seeds."""
        gate = check_sortino_stability(
            self.poor_summary, min_seeds=3, sortino_threshold=0.8
        )
        self.assertFalse(gate.passed)
        self.assertIn("1 seeds", gate.message)

    def test_drawdown_gate_pass(self):
        """Test drawdown gate with acceptable drawdown."""
        gate = check_drawdown_gate(self.good_summary, max_drawdown_limit=0.20)
        self.assertTrue(gate.passed)

    def test_drawdown_gate_fail(self):
        """Test drawdown gate fails with excessive drawdown."""
        gate = check_drawdown_gate(self.poor_summary, max_drawdown_limit=0.20)
        self.assertFalse(gate.passed)
        self.assertIn("35.00%", gate.message)

    def test_cash_behavior_gate_active(self):
        """Test cash behavior gate with active cash."""
        gate = check_cash_behavior_gate(self.good_summary, require_active_cash=True)
        self.assertTrue(gate.passed)

    def test_cash_behavior_gate_disabled(self):
        """Test cash behavior gate with disabled cash."""
        gate = check_cash_behavior_gate(self.poor_summary, require_active_cash=False)
        self.assertTrue(gate.passed)  # Disabled is OK when not required

    def test_cash_behavior_gate_weak(self):
        """Test cash behavior gate rejects weak cash behavior."""
        weak_cash_summary = [
            {
                "algo": "ppo",
                "cash_mode": "enabled",
                "seeds": [42, 43, 44],
                "cash_behavior": "weak cash usage",
            }
        ]
        gate = check_cash_behavior_gate(weak_cash_summary, require_active_cash=False)
        self.assertFalse(gate.passed)  # Weak cash is not acceptable

    def test_turnover_gate_pass(self):
        """Test turnover gate with acceptable turnover."""
        gate = check_turnover_gate(self.good_summary, turnover_limit=0.15)
        self.assertTrue(gate.passed)

    def test_turnover_gate_fail(self):
        """Test turnover gate fails with high turnover."""
        gate = check_turnover_gate(self.poor_summary, turnover_limit=0.10)
        self.assertFalse(gate.passed)
        self.assertIn("18.00%", gate.message)

    def test_baseline_gate_no_data(self):
        """Test baseline gate with missing data."""
        gate = check_baseline_gate(self.good_summary, {})
        self.assertFalse(gate.passed)
        self.assertEqual(gate.details["status"], "missing")

    def test_baseline_gate_beats_baselines(self):
        """Test baseline gate when model beats baselines."""
        baseline_summary = {
            "buy_and_hold": {"total_return": 0.10, "sharpe": 0.5},
            "Semi_2x": {"total_return": 0.08, "sharpe": 0.4},
            "0050": {"total_return": 0.12, "sharpe": 0.6},
        }
        gate = check_baseline_gate(self.good_summary, baseline_summary)
        self.assertTrue(gate.passed)

    def test_ablation_gate_feature_improves(self):
        """Test ablation gate when feature improves Sortino."""
        ablation_summary = {
            "overnight_features": {
                "with_feature": {"sortino": 1.2},
                "without_feature": {"sortino": 0.9},
            }
        }
        gate = check_ablation_gate(ablation_summary)
        self.assertTrue(gate.passed)
        self.assertIn("+0.30", gate.message)

    def test_ablation_gate_feature_hurts(self):
        """Test ablation gate when feature hurts Sortino."""
        ablation_summary = {
            "overnight_features": {
                "with_feature": {"sortino": 0.8},
                "without_feature": {"sortino": 1.0},
            }
        }
        gate = check_ablation_gate(ablation_summary)
        self.assertFalse(gate.passed)
        self.assertIn("-0.20", gate.message)

    def test_stress_gate_passes_tests(self):
        """Test stress gate when model survives stress tests."""
        stress_summary = {
            "tests": {
                "fee_1bp": {"total_return": 0.24},
                "slippage_1bp": {"total_return": 0.23},
                "spread_2bp": {"total_return": 0.22},
            }
        }
        gate = check_stress_gate(self.good_summary, stress_summary)
        self.assertTrue(gate.passed)

    def test_stress_gate_fails_tests(self):
        """Test stress gate when model fails stress tests."""
        stress_summary = {
            "tests": {
                "fee_1bp": {"total_return": 0.0},
                "slippage_1bp": {"total_return": -0.05},
                "spread_2bp": {"total_return": -0.10},
            }
        }
        gate = check_stress_gate(self.good_summary, stress_summary)
        self.assertFalse(gate.passed)

    def test_period_consistency_gate_with_period_data(self):
        period_df = pd.DataFrame(
            [
                {"period": "2024H2", "total_return": 0.10},
                {"period": "2025H1", "total_return": 0.08},
                {"period": "2025H2", "total_return": 0.09},
            ]
        )
        gate = check_period_consistency_gate(period_df)
        self.assertTrue(gate.passed)

    def test_run_promotion_gate_approve(self):
        """Test full promotion gate suite approves good model."""
        result = run_promotion_gate(
            raw_summary=self.good_summary,
            period_df=None,
            baseline_summary={},
            ablation_summary={},
            stress_summary={},
        )
        self.assertTrue(result.can_promote)
        self.assertEqual(result.risk_level, "low")
        self.assertIn("✓", result.summary)

    def test_run_promotion_gate_block(self):
        """Test full promotion gate suite blocks poor model."""
        result = run_promotion_gate(
            raw_summary=self.poor_summary,
            period_df=None,
            baseline_summary={},
            ablation_summary={},
            stress_summary={},
        )
        self.assertFalse(result.can_promote)
        self.assertIn("✗", result.summary)

    def test_promotion_gate_result_format(self):
        """Test promotion result string formatting."""
        result = run_promotion_gate(
            raw_summary=self.good_summary,
            period_df=None,
            baseline_summary={},
            ablation_summary={},
            stress_summary={},
        )
        result_str = str(result)
        self.assertIn("Promotion Result", result_str)
        self.assertIn("Risk Level", result_str)
        self.assertIn("Summary", result_str)


if __name__ == "__main__":
    unittest.main()
