import numpy as np
from typing import Tuple, List, Optional
from collections import deque

try:
    from trading_env_kernels import (
        compute_reward_kernel,
        _softsign,
    )
    _NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    _NUMBA_AVAILABLE = False
    
    def _softsign(x: float) -> float:
        return float(x / (1.0 + abs(x)))

# Constants copied from trading_env.py for isolated testing and modularity
LAMBDA_COST = 5.0
LAMBDA_TURNOVER = 1.0
LAMBDA_CASH = 0.0
LAMBDA_DRAWDOWN = 1.2
LAMBDA_CASH_DEFENSIVE = 0.35
REWARD_REF_DD = 0.02
REGIME_DD_THRESHOLD = 0.06
REGIME_PENALTY_COEF = 1.5
LAMBDA_WHIPSAW = 0.05

class RewardCalculator:
    def __init__(
        self,
        sharpe_window: int = 20,
        use_benchmark_reward: bool = True,
        enable_cash_action: bool = False,
        enable_margin_short: bool = False,
    ):
        self.sharpe_window = sharpe_window
        self.use_benchmark_reward = use_benchmark_reward
        self.enable_cash_action = enable_cash_action
        self.enable_margin_short = enable_margin_short

        self._return_history: deque = deque(maxlen=self.sharpe_window)
        self._pomdp_cache: Optional[Tuple[float, float, float]] = None

    def reset_state(self):
        self._return_history.clear()
        self._pomdp_cache = None

    def compute_pomdp_features(self, current_dd: float) -> Tuple[float, float, float]:
        """Rolling vol / Sortino proxy / current DD — shared by obs and reward.
        
        This should be called ONCE per step after appending to _return_history.
        """
        current_dd = float(np.clip(current_dd, 0.0, 1.0))
        n = len(self._return_history)
        if n < 2:
            self._pomdp_cache = (0.0, 0.0, current_dd)
            return self._pomdp_cache

        arr = np.array(self._return_history, dtype=np.float64)
        rolling_vol = float(np.clip(np.std(arr) * 100.0, 0.0, 1.0))
        
        if n < 5:
            self._pomdp_cache = (rolling_vol, 0.0, current_dd)
            return self._pomdp_cache

        mean_r = float(np.mean(arr))
        neg_returns = arr[arr < 0]
        downside_std = (
            float(np.std(neg_returns) + 1e-8)
            if len(neg_returns) >= 2
            else float(np.std(arr) + 1e-8)
        )
        sortino_proxy = float(_softsign(mean_r / downside_std))
        self._pomdp_cache = (rolling_vol, sortino_proxy, current_dd)
        return self._pomdp_cache

    def get_pomdp_cache(self, current_dd: float) -> Tuple[float, float, float]:
        """Get the cached pomdp features or compute them if missing."""
        if self._pomdp_cache is not None:
            return self._pomdp_cache
        return self.compute_pomdp_features(current_dd)

    def compute_reward(
        self,
        prev_value: float,
        curr_value: float,
        trade_cost: float,
        benchmark_log_r: float,
        positions: np.ndarray,
        cash_weight: float,
        last_turnover: float,
        current_dd: float,
        whipsaw_penalty: float = 0.0,
    ) -> float:
        log_r = float(np.log(max(curr_value, 1e-8) / max(prev_value, 1e-8)))
        
        # Cache invalidation and new calculation happens here
        self._pomdp_cache = None
        self._return_history.append(log_r)
        
        _, sortino_proxy, raw_dd = self.compute_pomdp_features(current_dd)

        capital_util = float(np.sum(np.abs(positions)))
        cash_ratio = (
            float(cash_weight)
            if self.enable_cash_action
            else max(0.0, 1.0 - capital_util)
        )

        if _NUMBA_AVAILABLE:
            return compute_reward_kernel(
                float(log_r),
                float(benchmark_log_r),
                float(sortino_proxy),
                float(capital_util),
                float(trade_cost),
                float(last_turnover),
                float(cash_ratio),
                float(raw_dd),
                bool(self.use_benchmark_reward),
                bool(self.enable_margin_short),
                bool(self.enable_cash_action),
                len(self._return_history),
                LAMBDA_COST,
                LAMBDA_TURNOVER,
                LAMBDA_CASH,
                LAMBDA_DRAWDOWN,
                REWARD_REF_DD,
                REGIME_DD_THRESHOLD,
                REGIME_PENALTY_COEF,
                LAMBDA_CASH_DEFENSIVE,
                float(whipsaw_penalty),
                LAMBDA_WHIPSAW,
            )

        # Fallback pure-Python path
        return_component = _softsign(log_r * 100)
        if len(self._return_history) >= 5:
            sortino_component = sortino_proxy * capital_util
            if self.use_benchmark_reward:
                benchmark_component = _softsign((log_r - benchmark_log_r) * 100)
                hybrid_reward = (
                    0.4 * return_component
                    + 0.3 * sortino_component
                    + 0.3 * benchmark_component
                )
            else:
                hybrid_reward = 0.5 * return_component + 0.5 * sortino_component
        else:
            if self.use_benchmark_reward:
                benchmark_component = _softsign((log_r - benchmark_log_r) * 100)
                hybrid_reward = 0.6 * return_component + 0.4 * benchmark_component
            else:
                hybrid_reward = return_component

        cost_p = LAMBDA_COST * trade_cost
        
        if last_turnover <= 0.3:
            turnover_p = LAMBDA_TURNOVER * 2.0 * (last_turnover ** 3)
        else:
            turnover_p = LAMBDA_TURNOVER * (0.054 + 0.54 * (last_turnover - 0.3))
            
        cash_p = LAMBDA_CASH * cash_ratio
        drawdown_p = LAMBDA_DRAWDOWN * max(0.0, raw_dd - REWARD_REF_DD)
        whipsaw_p = LAMBDA_WHIPSAW * whipsaw_penalty
        
        regime_penalty = 0.0
        if raw_dd > REGIME_DD_THRESHOLD and not self.enable_margin_short:
            regime_penalty = (
                REGIME_PENALTY_COEF
                * capital_util
                * (raw_dd - REGIME_DD_THRESHOLD)
            )
            
        cash_defensive_bonus = 0.0
        if raw_dd > REGIME_DD_THRESHOLD and self.enable_cash_action:
            cash_defensive_bonus = (
                LAMBDA_CASH_DEFENSIVE
                * cash_ratio
                * (raw_dd - REGIME_DD_THRESHOLD)
            )
            
        return float(np.clip(
            hybrid_reward - cost_p - turnover_p - cash_p
            - drawdown_p - regime_penalty - whipsaw_p + cash_defensive_bonus,
            -10.0, 10.0,
        ))
