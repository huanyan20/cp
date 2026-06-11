import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import env_config
import trading_env
from env_config import build_env_config_snapshot, compute_env_config_hash


class EnvConfigTests(unittest.TestCase):
    def test_snapshot_contains_reward_constants(self):
        snap = build_env_config_snapshot()
        self.assertEqual(snap["version"], env_config.ENV_CONFIG_VERSION)
        self.assertEqual(snap["lambda_drawdown"], trading_env.LAMBDA_DRAWDOWN)
        self.assertEqual(snap["reward_ref_dd"], trading_env.REWARD_REF_DD)
        self.assertEqual(snap["regime_dd_threshold"], trading_env.REGIME_DD_THRESHOLD)
        self.assertEqual(snap["num_account_features"], trading_env.NUM_ACCOUNT_FEATURES)
        self.assertEqual(snap["softmax_temp"], 1.0)
        self.assertEqual(snap["min_top_k_weight"], trading_env.MIN_TOP_K_WEIGHT)
        self.assertEqual(len(snap["hash"]), 8)

    def test_hash_is_stable_for_same_config(self):
        snap_a = build_env_config_snapshot()
        snap_b = build_env_config_snapshot()
        self.assertEqual(snap_a["hash"], snap_b["hash"])
        self.assertEqual(snap_a["hash"], compute_env_config_hash(snap_a))

    def test_hash_changes_when_drawdown_lambda_changes(self):
        snap = build_env_config_snapshot()
        mutated = dict(snap)
        mutated["lambda_drawdown"] = snap["lambda_drawdown"] + 0.1
        self.assertNotEqual(snap["hash"], compute_env_config_hash(mutated))

    def test_sl_features_tag_is_optional(self):
        base = build_env_config_snapshot()
        tagged = build_env_config_snapshot(sl_features="v1")
        self.assertNotIn("sl_features", base)
        self.assertEqual(tagged.get("sl_features"), "v1")
        self.assertNotEqual(base["hash"], tagged["hash"])


if __name__ == "__main__":
    unittest.main()
