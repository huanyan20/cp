import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_pipeline import (
    build_artifact_paths,
    build_pending_walk_forward_tasks,
    build_period_plan,
    build_seed_metrics,
    feature_suffix_from_path,
    should_skip_artifact,
    write_metrics_json,
)


class WalkForwardRefactorTests(unittest.TestCase):
    def test_build_artifact_paths_uses_existing_layout(self):
        paths = build_artifact_paths(
            algo="ppo",
            cash_mode="enabled",
            seed=42,
            feature_suffix="_with_features",
            results_dir="results_dir",
        )

        self.assertEqual(paths["metrics"], "results_dir/metrics_ppo_enabled_with_features_wf_seed42.json")
        self.assertEqual(paths["model"], "results_dir/wf_ppo_enabled_with_features_model_2024H2_seed42")
        self.assertEqual(paths["chart"], "results_dir/walk_forward_ppo_enabled_with_features_seed42.png")

    def test_build_period_plan_keeps_default_periods(self):
        periods = build_period_plan(today="2026-06-06")
        self.assertEqual(periods[0]["name"], "2024H2")
        self.assertTrue(any("effective_test_end" in p for p in periods))

    def test_feature_suffix_matches_existing_result_names(self):
        self.assertEqual(feature_suffix_from_path("features.csv"), "_with_features")
        self.assertEqual(feature_suffix_from_path(None), "")

    def test_pending_tasks_respects_existing_metrics_and_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            metrics = Path(tmp) / "metrics_ppo_enabled_wf_seed42.json"
            metrics.write_text("{}", encoding="utf-8")

            pending = build_pending_walk_forward_tasks(
                algos=["ppo"],
                cash_modes=[True],
                seeds=[42, 43],
                results_dir=tmp,
                overwrite=False,
            )
            self.assertEqual(pending, [("ppo", True, 43)])

            pending_overwrite = build_pending_walk_forward_tasks(
                algos=["ppo"],
                cash_modes=[True],
                seeds=[42],
                results_dir=tmp,
                overwrite=True,
            )
            self.assertEqual(pending_overwrite, [("ppo", True, 42)])
            self.assertTrue(should_skip_artifact(str(metrics), overwrite=False))
            self.assertFalse(should_skip_artifact(str(metrics), overwrite=True))

    def test_seed_metrics_and_persistence_are_centralized(self):
        metrics = build_seed_metrics(
            algo="ppo",
            seed=42,
            cash_mode="enabled",
            enable_cash_action=True,
            enable_margin_short=False,
            timesteps=150_000,
        )
        self.assertEqual(metrics["train_test_period"], "Walk-Forward")
        self.assertEqual(metrics["periods"], {})

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "metrics.json"
            write_metrics_json(metrics, str(path))
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
