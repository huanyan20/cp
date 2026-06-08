"""
Unit tests for train/eval orchestration functions extracted to research_pipeline.py
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_pipeline import (
    build_eval_env,
    build_train_env,
    persist_period_metrics,
    run_eval_loop,
    train_and_save_model,
)


class TrainEvalSplitTests(unittest.TestCase):
    """Test train/eval orchestration function signatures and behaviors."""

    def test_build_train_env_function_signature(self):
        """Verify build_train_env accepts expected parameters."""
        # This is a smoke test - just verify the function exists and has correct signature
        sig_params = [
            "tickers",
            "train_start",
            "train_end",
            "window_size",
            "macro_tickers",
            "settings",
            "enable_cash_action",
            "enable_margin_short",
            "overnight_feature_path",
        ]
        import inspect

        sig = inspect.signature(build_train_env)
        for param in sig_params:
            self.assertIn(param, sig.parameters, f"Missing parameter {param}")

    def test_train_and_save_model_function_signature(self):
        """Verify train_and_save_model accepts expected parameters."""
        sig_params = [
            "algo",
            "train_env",
            "timesteps",
            "model_path",
            "temporal_extractor",
        ]
        import inspect

        sig = inspect.signature(train_and_save_model)
        for param in sig_params:
            self.assertIn(param, sig.parameters, f"Missing parameter {param}")

    def test_build_eval_env_function_signature(self):
        """Verify build_eval_env accepts expected parameters."""
        sig_params = [
            "tickers",
            "test_start",
            "test_end",
            "window_size",
            "macro_tickers",
            "settings",
            "enable_cash_action",
            "enable_margin_short",
            "overnight_feature_path",
        ]
        import inspect

        sig = inspect.signature(build_eval_env)
        for param in sig_params:
            self.assertIn(param, sig.parameters, f"Missing parameter {param}")

    def test_run_eval_loop_function_signature(self):
        """Verify run_eval_loop accepts expected parameters."""
        sig_params = ["model", "test_env", "seed"]
        import inspect

        sig = inspect.signature(run_eval_loop)
        for param in sig_params:
            self.assertIn(param, sig.parameters, f"Missing parameter {param}")

    def test_persist_period_metrics_function_signature(self):
        """Verify persist_period_metrics accepts expected parameters."""
        sig_params = [
            "algo",
            "cash_mode",
            "seed",
            "feature_suffix",
            "tickers",
            "test_start",
            "test_end",
            "eval_results",
            "period_name",
            "results_dir",
        ]
        import inspect

        sig = inspect.signature(persist_period_metrics)
        for param in sig_params:
            self.assertIn(param, sig.parameters, f"Missing parameter {param}")

    def test_run_eval_loop_returns_expected_structure(self):
        """Verify run_eval_loop returns dict with expected keys."""
        # Mock the model and environment
        mock_model = MagicMock()
        mock_env = MagicMock()

        # Setup mock environment behavior
        mock_env.reset.return_value = (MagicMock(), {})
        mock_env.initial_balance = 1_000_000

        # Setup mock step behavior - one step then done
        mock_action = 0
        mock_model.predict.return_value = (mock_action, None)

        mock_obs = MagicMock()
        mock_info = {
            "portfolio_value": 1_000_100,
            "positions": [0.5, 0.3, 0.2],
            "cash_weight": 0.0,
            "turnover": 0.02,
        }
        mock_env.step.return_value = (mock_obs, 0, True, False, mock_info)

        # Run eval loop
        result = run_eval_loop(mock_model, mock_env, seed=42)

        # Verify return structure
        self.assertIsInstance(result, dict)
        expected_keys = {
            "daily_returns",
            "positions",
            "cash_weights",
            "turnover",
            "portfolio_hist",
        }
        self.assertEqual(set(result.keys()), expected_keys)

        # Verify data types
        self.assertIsInstance(result["daily_returns"], list)
        self.assertIsInstance(result["positions"], list)
        self.assertIsInstance(result["cash_weights"], list)
        self.assertIsInstance(result["turnover"], list)
        self.assertIsInstance(result["portfolio_hist"], list)

    def test_persist_period_metrics_returns_metrics_dict(self):
        """Verify persist_period_metrics returns dict with expected keys."""
        eval_results = {
            "daily_returns": [0.001, -0.002, 0.0015],
            "positions": [[0.5, 0.3, 0.2], [0.4, 0.35, 0.25], [0.5, 0.3, 0.2]],
            "cash_weights": [0.0, 0.0, 0.0],
            "turnover": [0.02, 0.01, 0.01],
            "portfolio_hist": [1_000_000, 1_001_000, 999_000, 1_000_500],
        }

        tickers = ["2330", "2454", "3711"]

        result = persist_period_metrics(
            algo="ppo",
            cash_mode="enabled",
            seed=42,
            feature_suffix="_with_features",
            tickers=tickers,
            test_start="2024-07-01",
            test_end="2024-12-31",
            eval_results=eval_results,
            period_name="2024H2",
            results_dir="results_dir",
        )

        # Verify return is dict
        self.assertIsInstance(result, dict)

        # Verify test_start and test_end are present
        self.assertEqual(result["test_start"], "2024-07-01")
        self.assertEqual(result["test_end"], "2024-12-31")

        # Verify metrics keys are present (from calculate_metrics)
        expected_metric_keys = {
            "total_return",
            "max_drawdown",
            "sharpe",
            "sortino",
            "avg_cash_weight",
        }
        actual_keys = set(result.keys())
        self.assertTrue(
            expected_metric_keys.issubset(actual_keys),
            f"Missing keys: {expected_metric_keys - actual_keys}",
        )


if __name__ == "__main__":
    unittest.main()
