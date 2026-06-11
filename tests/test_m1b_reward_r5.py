"""M1b: reward r5 constants + ENV_CONFIG_VERSION bump."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import env_config
import trading_env
from env_config import build_env_config_snapshot, compute_env_config_hash
from trading_env import TaiwanStockEnv

# Frozen r4 knobs (pre-M1b) for regression comparison.
R4_REWARD_KNOBS = {
    "lambda_drawdown": 0.8,
    "reward_ref_dd": 0.03,
    "regime_dd_threshold": 0.08,
    "regime_penalty_coef": 1.0,
    "lambda_cash_defensive": 0.2,
}


def _risk_penalties(
    raw_dd: float,
    stock_exposure: float,
    cash_ratio: float,
    *,
    lambda_drawdown: float,
    reward_ref_dd: float,
    regime_dd_threshold: float,
    regime_penalty_coef: float,
    lambda_cash_defensive: float,
) -> tuple[float, float, float]:
    drawdown_p = lambda_drawdown * max(0.0, raw_dd - reward_ref_dd)
    regime_penalty = 0.0
    if raw_dd > regime_dd_threshold:
        regime_penalty = (
            regime_penalty_coef
            * stock_exposure
            * (raw_dd - regime_dd_threshold)
        )
    cash_bonus = 0.0
    if raw_dd > regime_dd_threshold:
        cash_bonus = (
            lambda_cash_defensive
            * cash_ratio
            * (raw_dd - regime_dd_threshold)
        )
    return drawdown_p, regime_penalty, cash_bonus


def make_df(rows: int = 80) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_a": np.linspace(-0.1, 0.1, rows, dtype=np.float32),
            "feature_b": np.linspace(0.2, -0.2, rows, dtype=np.float32),
            "log_return": np.zeros(rows, dtype=np.float32),
        }
    )


class M1bRewardR5Tests(unittest.TestCase):
    def test_env_config_version_is_r5(self):
        self.assertEqual(env_config.ENV_CONFIG_VERSION, "r5.1")
        snap = build_env_config_snapshot()
        self.assertEqual(snap["version"], "r5.1")
        self.assertEqual(snap["lambda_drawdown"], trading_env.LAMBDA_DRAWDOWN)
        self.assertEqual(snap["reward_ref_dd"], trading_env.REWARD_REF_DD)
        self.assertEqual(snap["regime_dd_threshold"], trading_env.REGIME_DD_THRESHOLD)
        self.assertEqual(snap["regime_penalty_coef"], trading_env.REGIME_PENALTY_COEF)
        self.assertEqual(snap["lambda_cash_defensive"], trading_env.LAMBDA_CASH_DEFENSIVE)

    def test_r5_hash_differs_from_r4_snapshot(self):
        r5 = build_env_config_snapshot()
        r4 = dict(r5)
        r4["version"] = "r4"
        for key, value in R4_REWARD_KNOBS.items():
            r4[key] = value
        self.assertNotEqual(r5["hash"], compute_env_config_hash(r4))

    def test_r5_penalties_stronger_than_r4_at_high_drawdown(self):
        raw_dd = 0.34
        exposure = 0.93
        cash = 0.07

        r4_dd, r4_regime, r4_cash = _risk_penalties(
            raw_dd, exposure, cash, **R4_REWARD_KNOBS
        )
        r5_dd, r5_regime, r5_cash = _risk_penalties(
            raw_dd,
            exposure,
            cash,
            lambda_drawdown=trading_env.LAMBDA_DRAWDOWN,
            reward_ref_dd=trading_env.REWARD_REF_DD,
            regime_dd_threshold=trading_env.REGIME_DD_THRESHOLD,
            regime_penalty_coef=trading_env.REGIME_PENALTY_COEF,
            lambda_cash_defensive=trading_env.LAMBDA_CASH_DEFENSIVE,
        )

        self.assertGreater(r5_dd, r4_dd)
        self.assertGreater(r5_regime, r4_regime)
        self.assertGreater(r5_cash, r4_cash)
        self.assertGreater(
            (r5_dd + r5_regime - r5_cash),
            (r4_dd + r4_regime - r4_cash),
        )

    def test_high_drawdown_reward_near_zero_with_small_daily_gain(self):
        env = TaiwanStockEnv(
            {"2330.TW": make_df(), "2317.TW": make_df()},
            window_size=5,
            use_benchmark_reward=True,
            enable_cash_action=True,
        )
        env.reset()
        env._peak_value = env.initial_balance * 1.5
        env._portfolio_value = env.initial_balance
        env._positions = np.array([0.465, 0.465], dtype=np.float32)
        env._cash_weight = 0.07
        env._return_history.extend([0.005] * 20)

        prev = env._portfolio_value
        env._portfolio_value *= 1.005
        reward = env._compute_reward(prev, env._portfolio_value, trade_cost=0.0)

        self.assertLess(reward, 0.05)


if __name__ == "__main__":
    unittest.main()
