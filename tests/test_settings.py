import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.evaluate_portfolio as evaluate_portfolio
import settings
import train_portfolio
import walk_forward


class SettingsTests(unittest.TestCase):
    def test_default_settings_preserve_existing_research_defaults(self):
        app_settings = settings.load_settings()

        self.assertEqual(app_settings.research.train_start, "2020-01-01")
        self.assertEqual(app_settings.research.train_end, "2023-12-31")
        self.assertEqual(app_settings.research.timesteps, 300_000)
        self.assertEqual(app_settings.research.walk_forward_timesteps, 300_000)
        self.assertEqual(app_settings.research.walk_forward_cash_mode, "enabled")
        self.assertEqual(app_settings.research.default_topk, 5)
        self.assertEqual(app_settings.research.default_softmax_temp, 1.0)

    def test_environment_overrides_settings(self):
        env = {
            "RESEARCH_TRAIN_START": "2021-01-01",
            "RESEARCH_TIMESTEPS": "12345",
            "RESEARCH_SEED": "7",
            "RESEARCH_TOPK": "3",
            "RESEARCH_SOFTMAX_TEMP": "0.25",
            "WALK_FORWARD_TIMESTEPS": "54321",
            "WALK_FORWARD_CASH_MODE": "both",
            "EVALUATION_MODEL_NAME": "custom_model.zip",
            "MAX_SINGLE_WEIGHT": "0.2",
        }

        with patch.dict(os.environ, env, clear=False):
            app_settings = settings.load_settings()

        self.assertEqual(app_settings.research.train_start, "2021-01-01")
        self.assertEqual(app_settings.research.timesteps, 12345)
        self.assertEqual(app_settings.research.default_seed, 7)
        self.assertEqual(app_settings.research.default_topk, 3)
        self.assertEqual(app_settings.research.default_softmax_temp, 0.25)
        self.assertEqual(app_settings.research.walk_forward_timesteps, 54321)
        self.assertEqual(app_settings.research.walk_forward_cash_mode, "both")
        self.assertEqual(app_settings.evaluation.model_name, "custom_model.zip")
        self.assertEqual(app_settings.risk_limits.max_single_weight, 0.2)

    def test_cli_defaults_are_loaded_from_settings(self):
        self.assertEqual(train_portfolio.TIMESTEPS, train_portfolio.SETTINGS.research.timesteps)
        self.assertEqual(walk_forward.DEFAULT_TIMESTEPS, walk_forward.SETTINGS.research.walk_forward_timesteps)
        self.assertEqual(
            evaluate_portfolio.TEST_START,
            evaluate_portfolio.SETTINGS.evaluation.test_start,
        )


class TrainingTierTests(unittest.TestCase):
    def test_research_tier_defaults_to_opt_out(self):
        app_settings = settings.load_settings()
        self.assertEqual(app_settings.research.research_tier, "")

    def test_resolve_tier_maps_timesteps_and_truncates_seeds(self):
        base_seeds = [42, 43, 44]
        self.assertEqual(settings.resolve_tier("smoke", base_seeds), (500_000, [42]))
        self.assertEqual(settings.resolve_tier("candidate", base_seeds), (500_000, [42, 43]))
        self.assertEqual(settings.resolve_tier("promotion", base_seeds), (500_000, [42, 43, 44]))

    def test_resolve_tier_is_case_insensitive(self):
        self.assertEqual(settings.resolve_tier("PROMOTION", [42, 43, 44])[0], 500_000)

    def test_resolve_tier_rejects_unknown_tier(self):
        with self.assertRaises(ValueError):
            settings.resolve_tier("turbo", [42])


class TorchDeviceTests(unittest.TestCase):
    def test_resolve_torch_device_auto_prefers_cuda_when_available(self):
        import torch

        expected = "cuda" if torch.cuda.is_available() else "cpu"
        self.assertEqual(settings.resolve_torch_device("auto"), expected)

    def test_resolve_torch_device_cpu(self):
        self.assertEqual(settings.resolve_torch_device("cpu"), "cpu")

    def test_resolve_torch_device_rejects_unknown_value(self):
        with self.assertRaises(ValueError):
            settings.resolve_torch_device("tpu")

    def test_research_torch_device_defaults_to_auto(self):
        app_settings = settings.load_settings()
        self.assertEqual(app_settings.research.torch_device, "auto")


if __name__ == "__main__":
    unittest.main()
