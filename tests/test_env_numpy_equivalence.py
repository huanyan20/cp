"""Equivalence tests for the NumPy-backed TaiwanStockEnv hot paths.

The 2026-06 optimization replaced per-step Pandas .iloc reads in step() and
_get_observation() with pre-stacked NumPy arrays. These tests recompute the
same quantities straight from the Pandas DataFrames (the old code path) and
assert the env produces identical values.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trading_env import _BENCHMARK_LOOKBACK, TaiwanStockEnv

WINDOW = 5
NUM_FEATURES = 3
TICKERS = ["2330.TW", "2317.TW", "2454.TW", "3008.TW"]


def make_df(rows: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "feature_a": rng.normal(size=rows),
            "feature_b": rng.normal(size=rows),
            "log_return": rng.normal(scale=0.01, size=rows),
        }
    )


def make_env(**kwargs) -> TaiwanStockEnv:
    df_dict = {t: make_df(seed=i) for i, t in enumerate(TICKERS)}
    return TaiwanStockEnv(df_dict, window_size=WINDOW, topk=2, **kwargs)


def reference_market_block(env: TaiwanStockEnv) -> np.ndarray:
    """Old _get_observation market window: per-ticker .iloc slice + flatten."""
    start = env._current_step - env.window_size
    return np.stack(
        [
            env.dfs[t].iloc[start : env._current_step].values.flatten().astype(np.float32)
            for t in env.tickers
        ]
    )


class NumpyEquivalenceTests(unittest.TestCase):
    def test_observation_market_block_matches_pandas(self):
        env = make_env(use_benchmark_reward=False)
        obs, _ = env.reset()
        market_dim = WINDOW * NUM_FEATURES

        rng = np.random.default_rng(42)
        for _ in range(30):
            np.testing.assert_array_equal(
                obs[:, :market_dim], reference_market_block(env)
            )
            obs, _, terminated, _, _ = env.step(
                rng.normal(size=env.action_space.shape).astype(np.float32)
            )
            if terminated:
                break

    def test_log_returns_match_pandas_cells(self):
        env = make_env(use_benchmark_reward=False)
        expected = np.stack(
            [env.dfs[t]["log_return"].to_numpy(dtype=np.float64) for t in env.tickers],
            axis=1,
        )
        np.testing.assert_array_equal(env._log_returns, expected)

    def test_benchmark_scores_match_pandas_rolling_sum(self):
        env = make_env(use_benchmark_reward=True)
        env.reset()
        step = max(env._current_step, _BENCHMARK_LOOKBACK + 3)
        env._current_step = step

        new_scores = env._log_returns[step - _BENCHMARK_LOOKBACK : step].sum(axis=0)
        old_scores = np.array(
            [
                env.dfs[t]["log_return"].iloc[step - _BENCHMARK_LOOKBACK : step].sum()
                for t in env.tickers
            ]
        )
        np.testing.assert_allclose(new_scores, old_scores, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(
            np.argsort(new_scores)[-3:], np.argsort(old_scores)[-3:]
        )

    def test_account_features_match_old_layout(self):
        env = make_env(use_benchmark_reward=False, enable_cash_action=True)
        env.reset()
        rng = np.random.default_rng(7)
        for _ in range(10):
            obs, _, _, _, _ = env.step(
                rng.normal(size=env.action_space.shape).astype(np.float32)
            )
        market_dim = WINDOW * NUM_FEATURES
        account = obs[:, market_dim : market_dim + 6]

        total_ret = (env._portfolio_value - env.initial_balance) / env.initial_balance
        for i in range(env.num_stocks):
            expected = np.array(
                [
                    float(env._cash_weight),
                    float(np.clip(total_ret, -1.0, 1.0)),
                    float(np.clip(env._max_drawdown, 0.0, 1.0)),
                    float(env._positions[i]),
                    float(np.clip(env._trade_returns[i], -1.0, 1.0)),
                    float(np.clip(env._holding_periods[i] / 100.0, 0.0, 1.0)),
                ],
                dtype=np.float32,
            )
            np.testing.assert_allclose(account[i], expected, rtol=0, atol=1e-7)

    def test_sl_features_padding_and_lookup(self):
        rows = 120
        df_dict = {t: make_df(rows=rows, seed=i) for i, t in enumerate(TICKERS)}
        sl = {
            # Full-length array.
            TICKERS[0]: np.arange(rows * 3, dtype=np.float32).reshape(rows, 3),
            # Short array: steps beyond its length must read as zeros.
            TICKERS[1]: np.ones((WINDOW + 2, 3), dtype=np.float32),
            # TICKERS[2] / TICKERS[3] missing: always zeros.
        }
        env = TaiwanStockEnv(
            df_dict,
            window_size=WINDOW,
            topk=2,
            use_benchmark_reward=False,
            enable_sl_features=True,
            sl_features_by_ticker=sl,
        )
        obs, _ = env.reset()
        sl_block = obs[:, -3:]
        step = env._current_step
        np.testing.assert_array_equal(sl_block[0], sl[TICKERS[0]][step])
        np.testing.assert_array_equal(sl_block[1], sl[TICKERS[1]][step])
        np.testing.assert_array_equal(sl_block[2], np.zeros(3, dtype=np.float32))

        # Advance past the short array's length.
        for _ in range(WINDOW + 5):
            obs, _, _, _, _ = env.step(
                np.zeros(env.action_space.shape, dtype=np.float32)
            )
        self.assertGreaterEqual(env._current_step, len(sl[TICKERS[1]]))
        np.testing.assert_array_equal(obs[1, -3:], np.zeros(3, dtype=np.float32))
        np.testing.assert_array_equal(
            obs[0, -3:], sl[TICKERS[0]][env._current_step]
        )

    def test_full_episode_runs_and_obs_dtype_stable(self):
        env = make_env(use_benchmark_reward=True, enable_cash_action=True)
        obs, _ = env.reset()
        self.assertEqual(obs.dtype, np.float32)
        rng = np.random.default_rng(1)
        terminated = False
        while not terminated:
            obs, reward, terminated, _, _ = env.step(
                rng.normal(size=env.action_space.shape).astype(np.float32)
            )
            self.assertEqual(obs.dtype, np.float32)
            self.assertTrue(np.isfinite(reward))


if __name__ == "__main__":
    unittest.main()
