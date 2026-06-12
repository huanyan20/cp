"""M1a: POMDP observation features align with reward drivers."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trading_env import (
    IDX_CURRENT_DRAWDOWN,
    IDX_MAX_DRAWDOWN,
    IDX_ROLLING_SORTINO,
    IDX_ROLLING_VOL,
    NUM_ACCOUNT_FEATURES,
    SHARPE_WINDOW,
    TaiwanStockEnv,
    _softsign,
)


def make_df(rows: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "feature_a": rng.normal(size=rows),
            "feature_b": rng.normal(size=rows),
            "log_return": rng.normal(scale=0.02, size=rows),
        }
    )


def make_env(**kwargs) -> TaiwanStockEnv:
    tickers = ["2330.TW", "2317.TW", "2454.TW"]
    df_dict = {t: make_df(seed=i) for i, t in enumerate(tickers)}
    return TaiwanStockEnv(df_dict, window_size=10, topk=2, **kwargs)


class M1aPomdpObsTests(unittest.TestCase):
    def test_observation_shape_includes_pomdp_features(self):
        env = make_env(use_benchmark_reward=False)
        obs, _ = env.reset()
        market_dim = env.window_size * env.num_market_features
        self.assertEqual(env._NUM_ACCOUNT_FEATURES, NUM_ACCOUNT_FEATURES)
        self.assertEqual(obs.shape, (env.num_stocks, market_dim + NUM_ACCOUNT_FEATURES))

    def test_pomdp_features_broadcast_to_all_stocks(self):
        env = make_env(use_benchmark_reward=False)
        env.reset()
        rng = np.random.default_rng(0)
        for _ in range(25):
            obs, _, _, _, _ = env.step(rng.normal(size=env.action_space.shape).astype(np.float32))

        market_dim = env.window_size * env.num_market_features
        account = obs[:, market_dim:]
        np.testing.assert_allclose(account[:, IDX_ROLLING_VOL], account[0, IDX_ROLLING_VOL])
        np.testing.assert_allclose(account[:, IDX_ROLLING_SORTINO], account[0, IDX_ROLLING_SORTINO])
        np.testing.assert_allclose(account[:, IDX_CURRENT_DRAWDOWN], account[0, IDX_CURRENT_DRAWDOWN])

    def test_pomdp_features_match_internal_computation(self):
        env = make_env(use_benchmark_reward=True, enable_cash_action=True)
        env.reset()
        rng = np.random.default_rng(11)
        for _ in range(30):
            obs, _, _, _, _ = env.step(rng.normal(size=env.action_space.shape).astype(np.float32))

        market_dim = env.window_size * env.num_market_features
        rolling_vol, sortino_proxy, current_dd = env._reward_calculator.compute_pomdp_features(env._current_drawdown())
        np.testing.assert_allclose(obs[0, market_dim + IDX_ROLLING_VOL], rolling_vol, atol=1e-7)
        np.testing.assert_allclose(obs[0, market_dim + IDX_ROLLING_SORTINO], sortino_proxy, atol=1e-7)
        np.testing.assert_allclose(obs[0, market_dim + IDX_CURRENT_DRAWDOWN], current_dd, atol=1e-7)

    def test_sortino_proxy_matches_reward_formula_after_warmup(self):
        env = make_env(use_benchmark_reward=False)
        env.reset()
        rng = np.random.default_rng(3)
        for _ in range(SHARPE_WINDOW + 5):
            env.step(rng.normal(size=env.action_space.shape).astype(np.float32))

        arr = np.array(env._reward_calculator._return_history, dtype=np.float64)
        mean_r = float(np.mean(arr))
        neg_returns = arr[arr < 0]
        downside_std = (
            float(np.std(neg_returns) + 1e-8)
            if len(neg_returns) >= 2
            else float(np.std(arr) + 1e-8)
        )
        expected = float(_softsign(mean_r / downside_std))
        _, sortino_proxy, _ = env._reward_calculator.compute_pomdp_features(env._current_drawdown())
        self.assertAlmostEqual(sortino_proxy, expected, places=7)

    def test_current_drawdown_can_differ_from_max_drawdown(self):
        env = make_env(use_benchmark_reward=False)
        env.reset()
        env._peak_value = env.initial_balance * 1.2
        env._portfolio_value = env.initial_balance * 1.1
        env._max_drawdown = 0.15
        env._reward_calculator._pomdp_cache = None

        obs = env._get_observation()
        market_dim = env.window_size * env.num_market_features
        current_dd = float(obs[0, market_dim + IDX_CURRENT_DRAWDOWN])
        max_dd = float(obs[0, market_dim + IDX_MAX_DRAWDOWN])

        self.assertAlmostEqual(current_dd, 1.0 / 12.0, places=5)
        self.assertAlmostEqual(max_dd, 0.15, places=5)
        self.assertLess(current_dd, max_dd)


if __name__ == "__main__":
    unittest.main()
