import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trading_env import TaiwanStockEnv


def make_df(rows=80):
    idx = pd.RangeIndex(rows)
    return pd.DataFrame(
        {
            "feature_a": np.linspace(-0.1, 0.1, rows, dtype=np.float32),
            "feature_b": np.linspace(0.2, -0.2, rows, dtype=np.float32),
            "log_return": np.sin(np.arange(rows) / 10.0).astype(np.float32) / 100.0,
        },
        index=idx,
    )


def make_env(enable_cash_action=False, topk=2):
    return TaiwanStockEnv(
        {
            "2330.TW": make_df(),
            "2317.TW": make_df(),
            "2454.TW": make_df(),
        },
        window_size=5,
        topk=topk,
        use_benchmark_reward=False,
        enable_cash_action=enable_cash_action,
    )


class DynamicCashEnvTests(unittest.TestCase):
    def test_legacy_shape_is_unchanged(self):
        env = make_env(enable_cash_action=False)
        self.assertEqual(env.action_space.shape, (3,))
        self.assertEqual(env.observation_space.shape, (3, 5 * 3 + 6))

    def test_cash_action_adds_one_dimension(self):
        env = make_env(enable_cash_action=True)
        self.assertEqual(env.action_space.shape, (4,))

    def test_topk_does_not_mask_cash_dimension(self):
        env = make_env(enable_cash_action=True, topk=1)
        action = np.array([5.0, 4.0, 3.0, 5.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(obs.shape, env.observation_space.shape)
        self.assertIsInstance(reward, float)
        self.assertEqual(np.count_nonzero(info["positions"] > 1e-6), 1)
        self.assertGreater(info["cash_weight"], 0.0)

    def test_stock_plus_cash_weights_sum_to_one(self):
        env = make_env(enable_cash_action=True, topk=2)
        _, _, _, _, info = env.step(np.array([1.0, 2.0, 3.0, 0.5], dtype=np.float32))
        total = float(np.sum(info["positions"]) + info["cash_weight"])
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_bypass_action_transform_stays_stock_only_compatible(self):
        env = make_env(enable_cash_action=True, topk=1)
        _, _, _, _, info = env.step(
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            bypass_action_transform=True,
        )

        self.assertAlmostEqual(float(info["positions"][0]), 1.0, places=5)
        self.assertAlmostEqual(float(info["cash_weight"]), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
