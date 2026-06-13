"""M1d/M1c: action decode — softmax temperature + top-k entropy floor."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import trading_env
from trading_env import MIN_TOP_K_WEIGHT, TaiwanStockEnv


def make_df(rows: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_a": np.zeros(rows, dtype=np.float32),
            "feature_b": np.zeros(rows, dtype=np.float32),
            "log_return": np.zeros(rows, dtype=np.float32),
        }
    )


def make_env(softmax_temp: float = 1.0, num_stocks: int = 6, **kwargs) -> TaiwanStockEnv:
    tickers = [f"S{i}" for i in range(num_stocks)]
    df_dict = {t: make_df() for t in tickers}
    return TaiwanStockEnv(
        df_dict,
        window_size=5,
        topk=5,
        softmax_temp=softmax_temp,
        use_benchmark_reward=False,
        **kwargs,
    )


def _dominant_action(n: int, cash: bool = False) -> np.ndarray:
    action = np.zeros(n + (1 if cash else 0), dtype=np.float32)
    action[0] = 3.0
    return action


class ActionDecodeM1dTests(unittest.TestCase):
    def test_default_softmax_temp_is_one(self):
        env = make_env()
        self.assertEqual(env._softmax_temp, 1.0)

    def test_higher_temp_reduces_peak_weight(self):
        action = _dominant_action(6)
        hot = make_env(softmax_temp=0.5)._raw_transform_action(action)
        mild = make_env(softmax_temp=1.0)._raw_transform_action(action)
        self.assertGreater(float(np.max(hot)), float(np.max(mild)))

    def test_r51_decode_less_concentrated_than_legacy_temp_half(self):
        action = _dominant_action(45, cash=True)
        legacy = make_env(softmax_temp=0.5, num_stocks=45, enable_cash_action=True)
        with patch.object(trading_env, "MIN_TOP_K_WEIGHT", 0.0):
            legacy_w = legacy._raw_transform_action(action)
        current = make_env(softmax_temp=1.0, num_stocks=45, enable_cash_action=True)
        current_w = current._raw_transform_action(action)
        self.assertGreater(float(np.max(legacy_w)), 0.9)
        self.assertLess(float(np.max(current_w)), float(np.max(legacy_w)))

    def test_temp_one_with_cash_beats_temp_half(self):
        env_hot = make_env(softmax_temp=0.5, num_stocks=6, enable_cash_action=True)
        env_mild = make_env(softmax_temp=1.0, num_stocks=6, enable_cash_action=True)
        action = _dominant_action(6, cash=True)
        hot = env_hot._raw_transform_action(action)
        mild = env_mild._raw_transform_action(action)
        self.assertGreater(float(np.max(hot)), float(np.max(mild)))


class ActionDecodeM1cTests(unittest.TestCase):
    def test_entropy_floor_reduces_peak_vs_no_floor(self):
        env = make_env(softmax_temp=1.0, num_stocks=6)
        action = np.array([5.0, -5.0, -5.0, -5.0, -5.0, -5.0], dtype=np.float32)
        with_floor = env._raw_transform_action(action)
        with patch.object(trading_env, "MIN_TOP_K_WEIGHT", 0.0):
            no_floor = make_env(softmax_temp=1.0, num_stocks=6)._raw_transform_action(action)
        self.assertLess(float(np.max(with_floor)), float(np.max(no_floor)))

    def test_top_k_floor_enforces_minimum_mass_before_renormalize(self):
        env = make_env(softmax_temp=1.0, num_stocks=6)
        action = np.array([5.0, 4.9, 4.8, 0.0, 0.0, 0.0], dtype=np.float32)
        shifted = action - np.max(action)
        exp_a = np.exp(shifted / env._softmax_temp)
        soft_weights = exp_a / (np.sum(exp_a) + 1e-8)
        topk_indices = np.argsort(soft_weights)[-env._topk :]
        clipped = soft_weights.copy()
        clipped[topk_indices] = np.maximum(clipped[topk_indices], MIN_TOP_K_WEIGHT)
        self.assertGreaterEqual(float(np.min(clipped[topk_indices])), MIN_TOP_K_WEIGHT)

    def test_step_preserves_weight_sum(self):
        env = make_env(enable_cash_action=True, softmax_temp=1.0, num_stocks=6)
        env.reset()
        action = np.array([4.0, 3.0, 2.0, 1.0, 0.0, -1.0, 0.5], dtype=np.float32)
        _, _, _, _, info = env.step(action)
        total = float(np.sum(info["positions"]) + info["cash_weight"])
        self.assertAlmostEqual(total, 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
