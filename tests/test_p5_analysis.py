"""Tests for p5_analysis — baseline / ablation / stress generators.

Coverage
--------
- ablation: with/without feature file pairing, seed matching, delta direction
- ablation: graceful handling when no counterpart exists
- stress: cost drag calculation, scenario ordering, settings injection
- stress: worst_case drag > high_slippage > high_fee > base
- baseline: OOS period inference from metrics files
- baseline: yfinance-independent (buy_and_hold format validated via mock)
- integration: all three outputs consumed by promotion gate without "missing"
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is on path when running directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.p5_analysis as p5_analysis
from scripts.p5_analysis import apply_cost_drag, run_ablation, run_stress
from settings import AppSettings, StressSettings

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_metrics(
    tmp: Path,
    filename: str,
    algo: str = "ppo",
    cash_mode: str = "enabled",
    seed: int = 42,
    sortino: float = 1.5,
    total_return: float = 0.80,
    max_drawdown: float = 0.20,
    turnover: float = 0.06,
    periods: dict | None = None,
) -> Path:
    """Write a canonical metrics_*.json file to *tmp*."""
    periods = periods or {
        "2024H2": {
            "total_return": 0.05, "max_drawdown": 0.10,
            "sortino": 0.8, "sharpe": 0.6,
            "win_rate": 0.52, "avg_cash_weight": 0.05,
            "long_exposure": 0.95, "short_exposure": 0.0,
            "cash_weight_std": 0.01, "cash_corr_next_return": 0.0,
            "short_corr_next_return": 0.0, "turnover": 0.05,
            "top_holdings": {},
            "test_start": "2024-07-01", "test_end": "2024-12-31",
            "was_clamped": False,
        },
        "2025H1": {
            "total_return": 0.10, "max_drawdown": 0.15,
            "sortino": 1.2, "sharpe": 0.9,
            "win_rate": 0.54, "avg_cash_weight": 0.08,
            "long_exposure": 0.92, "short_exposure": 0.0,
            "cash_weight_std": 0.02, "cash_corr_next_return": 0.01,
            "short_corr_next_return": 0.0, "turnover": 0.07,
            "top_holdings": {},
            "test_start": "2025-01-01", "test_end": "2025-06-30",
            "was_clamped": False,
        },
    }
    data = {
        "algo": algo,
        "seed": seed,
        "cash_mode": cash_mode,
        "enable_cash_action": cash_mode == "enabled",
        "enable_margin_short": False,
        "train_test_period": "Walk-Forward",
        "timesteps": 10000,
        "overall": {
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "sharpe": 1.0,
            "sortino": sortino,
            "win_rate": 0.53,
            "avg_cash_weight": 0.07,
            "long_exposure": 0.93,
            "short_exposure": 0.0,
            "cash_weight_std": 0.02,
            "cash_corr_next_return": 0.01,
            "short_corr_next_return": 0.0,
            "turnover": turnover,
            "top_holdings": {},
        },
        "periods": periods,
        "skipped_periods": {},
    }
    path = tmp / filename
    path.write_text(json.dumps(data, indent=4), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Ablation tests
# ---------------------------------------------------------------------------

class AblationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_ablation_reads_with_vs_without_features_metrics(self):
        """Ablation correctly pairs with/without-features files for the same seed."""
        _make_metrics(
            self.tmp,
            "metrics_ppo_enabled_wf_seed42.json",
            sortino=1.5, total_return=0.80, max_drawdown=0.20,
        )
        _make_metrics(
            self.tmp,
            "metrics_ppo_enabled_with_features_wf_seed42.json",
            sortino=1.6, total_return=0.82, max_drawdown=0.18,
        )
        out = self.tmp / "ablation_summary.json"
        run_ablation(str(self.tmp), str(out))

        result = json.loads(out.read_text(encoding="utf-8"))
        feat = result["overnight_features"]

        self.assertEqual(feat["matched_seeds"], [42])
        self.assertIn("with_feature", feat)
        self.assertIn("without_feature", feat)
        self.assertIn("delta", feat)
        # with_feature has higher sortino
        self.assertGreater(feat["delta"]["sortino"], 0)

    def test_ablation_delta_direction_correct(self):
        """Delta = with_feature - without_feature for each metric."""
        _make_metrics(self.tmp, "metrics_ppo_enabled_wf_seed42.json", sortino=1.0)
        _make_metrics(
            self.tmp,
            "metrics_ppo_enabled_with_features_wf_seed42.json",
            sortino=1.3,
        )
        out = self.tmp / "ablation_summary.json"
        run_ablation(str(self.tmp), str(out))
        feat = json.loads(out.read_text())["overnight_features"]
        self.assertAlmostEqual(
            feat["delta"]["sortino"],
            feat["with_feature"]["sortino"] - feat["without_feature"]["sortino"],
            places=5,
        )

    def test_ablation_handles_missing_counterpart(self):
        """When no _with_features_ file exists, ablation writes empty matched_seeds."""
        _make_metrics(self.tmp, "metrics_ppo_enabled_wf_seed42.json")
        # no with_features counterpart
        out = self.tmp / "ablation_summary.json"
        run_ablation(str(self.tmp), str(out))

        result = json.loads(out.read_text(encoding="utf-8"))
        feat = result["overnight_features"]
        self.assertEqual(feat["matched_seeds"], [])
        self.assertIn("no matched seeds", feat["verdict"])

    def test_ablation_verdict_reflects_improvement(self):
        """Verdict text should reflect whether sortino improved."""
        _make_metrics(self.tmp, "metrics_ppo_enabled_wf_seed42.json", sortino=1.0)
        _make_metrics(
            self.tmp,
            "metrics_ppo_enabled_with_features_wf_seed42.json",
            sortino=1.2,
        )
        out = self.tmp / "ablation_summary.json"
        run_ablation(str(self.tmp), str(out))
        feat = json.loads(out.read_text())["overnight_features"]
        self.assertIn("improves", feat["verdict"])


# ---------------------------------------------------------------------------
# Stress tests
# ---------------------------------------------------------------------------

class StressTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _write_one_metrics(self, turnover: float = 0.06, total_return: float = 0.80):
        _make_metrics(
            self.tmp,
            "metrics_ppo_enabled_wf_seed42.json",
            turnover=turnover,
            total_return=total_return,
        )

    def test_stress_applies_fee_drag_and_reduces_return(self):
        """Every scenario's stressed return must be <= base_return (positive turnover)."""
        self._write_one_metrics(turnover=0.06, total_return=0.80)
        out = self.tmp / "stress_summary.json"
        run_stress(str(self.tmp), str(out))

        result = json.loads(out.read_text(encoding="utf-8"))
        base_return = result["baseline_return"]
        for name, data in result["tests"].items():
            self.assertLessEqual(
                data["total_return"], base_return,
                msg=f"Scenario '{name}': stressed return should be <= baseline",
            )

    def test_stress_worst_case_has_largest_drag(self):
        """worst_case drag > high_slippage > high_fee > base."""
        self._write_one_metrics(turnover=0.06, total_return=0.80)
        out = self.tmp / "stress_summary.json"
        run_stress(str(self.tmp), str(out))

        tests = json.loads(out.read_text())["tests"]
        drags = {name: tests[name]["cost_drag"] for name in tests}
        self.assertGreater(drags["worst_case"], drags["high_slippage"])
        self.assertGreater(drags["high_slippage"], drags["high_fee"])
        self.assertGreater(drags["high_fee"], drags["base"])

    def test_stress_settings_injection_overrides_fee_rates(self):
        """Custom StressSettings should be reflected in output descriptions/rates."""
        self._write_one_metrics()
        custom_stress = StressSettings(
            base_fee_rate=0.0,
            base_tax_rate=0.0,
            high_fee_rate=0.002,
            high_fee_tax_rate=0.002,
            high_slippage_fee_rate=0.004,
            high_slippage_tax_rate=0.003,
            worst_case_fee_rate=0.006,
            worst_case_tax_rate=0.005,
        )
        custom_settings = AppSettings(stress=custom_stress)
        out = self.tmp / "stress_custom.json"
        run_stress(str(self.tmp), str(out), settings=custom_settings)

        result = json.loads(out.read_text())
        # base scenario with 0% fee → zero cost drag
        self.assertAlmostEqual(result["tests"]["base"]["cost_drag"], 0.0, places=6)
        # worst_case has the highest rate → largest drag
        self.assertGreater(
            result["tests"]["worst_case"]["cost_drag"],
            result["tests"]["high_fee"]["cost_drag"],
        )

    def test_stress_zero_turnover_means_zero_drag(self):
        """Zero turnover should produce zero cost drag in all scenarios."""
        self._write_one_metrics(turnover=0.0, total_return=0.50)
        out = self.tmp / "stress_summary.json"
        run_stress(str(self.tmp), str(out))

        result = json.loads(out.read_text())
        for name, data in result["tests"].items():
            self.assertAlmostEqual(
                data["cost_drag"], 0.0, places=6,
                msg=f"Scenario '{name}': cost_drag should be 0 with zero turnover",
            )

    def test_apply_cost_drag_formula(self):
        """Unit-test the cost drag formula directly."""
        result = apply_cost_drag(
            total_return=1.0,
            avg_daily_turnover=0.10,
            trading_days=100,
            fee_rate=0.001,
            tax_rate=0.002,
        )
        # round_trip_cost = 2*0.001 + 0.002 = 0.004
        # total_drag = 0.10 * 0.004 * 100 = 0.04
        # stressed = 2.0 * (1 - 0.04) - 1 = 0.92
        self.assertAlmostEqual(result["cost_drag"], 0.04, places=6)
        self.assertAlmostEqual(result["total_return"], 0.92, places=5)


# ---------------------------------------------------------------------------
# Baseline tests (no network — yfinance mocked)
# ---------------------------------------------------------------------------

class BaselineTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _write_metrics_with_periods(self):
        _make_metrics(
            self.tmp,
            "metrics_ppo_enabled_wf_seed42.json",
            periods={
                "2024H2": {
                    "total_return": 0.05, "max_drawdown": 0.10,
                    "sortino": 0.8, "sharpe": 0.6,
                    "win_rate": 0.52, "avg_cash_weight": 0.05,
                    "long_exposure": 0.95, "short_exposure": 0.0,
                    "cash_weight_std": 0.01, "cash_corr_next_return": 0.0,
                    "short_corr_next_return": 0.0, "turnover": 0.05,
                    "top_holdings": {},
                    "test_start": "2024-07-01", "test_end": "2024-12-31",
                    "was_clamped": False,
                },
                "2025H1": {
                    "total_return": 0.10, "max_drawdown": 0.15,
                    "sortino": 1.2, "sharpe": 0.9,
                    "win_rate": 0.54, "avg_cash_weight": 0.08,
                    "long_exposure": 0.92, "short_exposure": 0.0,
                    "cash_weight_std": 0.02, "cash_corr_next_return": 0.01,
                    "short_corr_next_return": 0.0, "turnover": 0.07,
                    "top_holdings": {},
                    "test_start": "2025-01-01", "test_end": "2025-06-30",
                    "was_clamped": False,
                },
            },
        )

    def _mock_yf_download(self, total_return: float = 0.15):
        """Return a mock yf.download that yields a fake Close price series."""
        import pandas as pd

        prices = pd.Series(
            [100.0 * (1.0 + total_return * i / 251) for i in range(252)],
            name="Close",
        )

        mock_yf = MagicMock()
        mock_df = MagicMock()
        mock_df.__getitem__ = MagicMock(return_value=prices)
        mock_df.__getitem__.return_value.squeeze = MagicMock(return_value=prices)
        mock_yf.download = MagicMock(return_value=mock_df)
        return mock_yf

    def test_baseline_period_is_inferred_from_metrics(self):
        """OOS period boundaries come from period test_start / test_end fields."""
        self._write_metrics_with_periods()
        records = p5_analysis._read_metrics_files(str(self.tmp))
        start, end = p5_analysis._oos_date_range(records)
        self.assertEqual(start, "2024-07-01")
        self.assertEqual(end, "2025-06-30")

    def test_baseline_output_has_required_keys(self):
        """Generated baseline_summary.json must contain all three benchmark keys."""
        self._write_metrics_with_periods()
        out = self.tmp / "baseline_summary.json"

        mock_yf = self._mock_yf_download(total_return=0.12)
        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            p5_analysis.run_baseline(str(self.tmp), str(out))

        result = json.loads(out.read_text(encoding="utf-8"))
        for key in ("buy_and_hold", "^TWII", "0050"):
            self.assertIn(key, result, msg=f"Missing key: {key}")
            self.assertIn("total_return", result[key])
            self.assertIn("description", result[key])

    def test_baseline_buy_and_hold_is_non_zero(self):
        """buy_and_hold total_return must reflect price movement (not stuck at 0)."""
        self._write_metrics_with_periods()
        out = self.tmp / "baseline_summary.json"

        mock_yf = self._mock_yf_download(total_return=0.15)
        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            p5_analysis.run_baseline(str(self.tmp), str(out))

        result = json.loads(out.read_text(encoding="utf-8"))
        self.assertNotEqual(result["buy_and_hold"]["total_return"], 0.0)


# ---------------------------------------------------------------------------
# Integration: all three outputs consumed by promotion gate
# ---------------------------------------------------------------------------

class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _write_full_metrics_set(self):
        """Write both with- and without-features metrics so ablation has pairs."""
        _make_metrics(
            self.tmp,
            "metrics_ppo_enabled_wf_seed42.json",
            sortino=1.5, total_return=0.80, max_drawdown=0.20, turnover=0.06,
        )
        _make_metrics(
            self.tmp,
            "metrics_ppo_enabled_with_features_wf_seed42.json",
            sortino=1.6, total_return=0.82, max_drawdown=0.18, turnover=0.06,
        )

    def test_ablation_and_stress_outputs_are_valid_json_consumed_by_gate(self):
        """After generating ablation + stress, promotion gate should load them."""
        from promotion_gate import check_ablation_gate, check_stress_gate

        self._write_full_metrics_set()
        ablation_out = self.tmp / "ablation_summary.json"
        stress_out = self.tmp / "stress_summary.json"

        run_ablation(str(self.tmp), str(ablation_out))
        run_stress(str(self.tmp), str(stress_out))

        ablation_data = json.loads(ablation_out.read_text(encoding="utf-8"))
        stress_data = json.loads(stress_out.read_text(encoding="utf-8"))

        # Ablation gate should not say "missing"
        ablation_gate = check_ablation_gate(ablation_data)
        self.assertNotIn("not available", ablation_gate.message)

        # Stress gate should not say "missing"
        raw_summary = [{"total_return_mean": 0.80, "algo": "ppo", "cash_mode": "enabled"}]
        stress_gate = check_stress_gate(raw_summary, stress_data)
        self.assertNotIn("not available", stress_gate.message)

    def test_stress_summary_has_four_scenarios(self):
        """Stress output must contain all four scenario keys."""
        self._write_full_metrics_set()
        out = self.tmp / "stress_summary.json"
        run_stress(str(self.tmp), str(out))

        result = json.loads(out.read_text())
        tests = result["tests"]
        for name in ("base", "high_fee", "high_slippage", "worst_case"):
            self.assertIn(name, tests, msg=f"Missing stress scenario: {name}")

    def test_settings_stress_paths_point_to_results_dir(self):
        """PathSettings should expose all three summary paths under results_dir."""
        from settings import load_settings

        s = load_settings()
        self.assertTrue(str(s.paths.baseline_summary_path).endswith("baseline_summary.json"))
        self.assertTrue(str(s.paths.ablation_summary_path).endswith("ablation_summary.json"))
        self.assertTrue(str(s.paths.stress_summary_path).endswith("stress_summary.json"))
        self.assertEqual(s.paths.baseline_summary_path.parent, s.paths.results_dir)


if __name__ == "__main__":
    unittest.main()
