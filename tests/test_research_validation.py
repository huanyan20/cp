import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.experiment_report import classify_cash_behavior
from walk_forward import cash_modes_from_arg, clamp_periods, parse_seeds


class ResearchValidationTests(unittest.TestCase):
    def test_clamp_periods_limits_future_test_end(self):
        periods = [
            {
                "name": "future_end",
                "train_start": "2020-01-01",
                "train_end": "2025-12-31",
                "test_start": "2026-01-01",
                "test_end": "2026-06-30",
            }
        ]
        clamped = clamp_periods(periods, today="2026-06-06")[0]
        self.assertEqual(clamped["effective_test_end"], "2026-06-06")
        self.assertTrue(clamped["was_clamped"])

    def test_clamp_periods_skips_not_started_period(self):
        periods = [
            {
                "name": "not_started",
                "train_start": "2020-01-01",
                "train_end": "2026-12-31",
                "test_start": "2027-01-01",
                "test_end": "2027-06-30",
            }
        ]
        clamped = clamp_periods(periods, today="2026-06-06")[0]
        self.assertIn("skip_reason", clamped)

    def test_cash_mode_and_seed_parsing(self):
        self.assertEqual(cash_modes_from_arg("both"), [True, False])
        self.assertEqual(parse_seeds("42, 43,44"), [42, 43, 44])

    def test_cash_behavior_classification(self):
        self.assertEqual(
            classify_cash_behavior(
                {
                    "cash_mode": "enabled",
                    "avg_cash_weight_mean": 0.005,
                    "cash_weight_std_mean": 0.02,
                }
            ),
            "weak cash usage",
        )
        self.assertEqual(
            classify_cash_behavior(
                {
                    "cash_mode": "enabled",
                    "avg_cash_weight_mean": 0.05,
                    "cash_weight_std_mean": 0.005,
                }
            ),
            "static cash",
        )
        self.assertEqual(
            classify_cash_behavior(
                {
                    "cash_mode": "enabled",
                    "avg_cash_weight_mean": 0.05,
                    "cash_weight_std_mean": 0.02,
                }
            ),
            "active cash",
        )


if __name__ == "__main__":
    unittest.main()
