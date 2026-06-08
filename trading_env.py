"""Portfolio trading environment for Taiwan stock allocation experiments."""

from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces

COMMISSION_RATE = 0.001425
TAX_RATE_SELL = 0.003
SLIPPAGE_RATE = 0.001
BORROW_RATE_DAILY = 0.015 / 252

LAMBDA_COST = 5.0      # Increased from 0.5 to strongly penalize friction
LAMBDA_TURNOVER = 1.0  # New penalty for portfolio churn
LAMBDA_CASH = 0.0
LAMBDA_DRAWDOWN = 0.3
SHARPE_WINDOW = 20
_BENCHMARK_TOPK = 3
_BENCHMARK_LOOKBACK = 20


def _softsign(x: float) -> float:
    return float(x / (1.0 + abs(x)))


class TaiwanStockEnv(gym.Env):
    """Long-only portfolio environment with optional dynamic cash allocation.

    Legacy mode keeps the old action shape, ``(num_stocks,)``. Cash-aware mode
    uses ``(num_stocks + 1,)`` where the last logit controls cash. TopK filtering
    is applied only to stock logits.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df_dict: dict,
        window_size: int = 20,
        initial_balance: float = 1_000_000.0,
        topk: int = 5,
        softmax_temp: float = 0.5,
        use_benchmark_reward: bool = True,
        enable_cash_action: bool = False,
        enable_margin_short: bool = False,
        max_leverage: float = 2.0,
        record_trades: bool = False,
    ):
        super().__init__()

        self.tickers = list(df_dict.keys())
        self.num_stocks = len(self.tickers)
        self._topk = min(topk, self.num_stocks)
        self._softmax_temp = max(softmax_temp, 1e-3)
        self.use_benchmark_reward = use_benchmark_reward
        self.enable_cash_action = enable_cash_action
        self.enable_margin_short = enable_margin_short
        self.max_leverage = max_leverage
        self.record_trades = record_trades

        self.MARGIN_RATE_DAILY = 0.06 / 252  # 6% annual margin loan interest
        self.SHORT_RATE_DAILY = 0.015 / 252  # 1.5% annual stock borrow fee

        self.dfs = {ticker: df.reset_index(drop=True) for ticker, df in df_dict.items()}
        self.max_steps = len(self.dfs[self.tickers[0]])
        self.num_market_features = self.dfs[self.tickers[0]].shape[1]
        self.window_size = window_size
        self.initial_balance = initial_balance

        self._NUM_ACCOUNT_FEATURES = 6
        action_dim = self.num_stocks + 1 if self.enable_cash_action else self.num_stocks
        self.action_space = spaces.Box(
            low=-5.0, high=5.0, shape=(action_dim,), dtype=np.float32
        )

        self._obs_dim_per_stock = (
            window_size * self.num_market_features + self._NUM_ACCOUNT_FEATURES
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_stocks, self._obs_dim_per_stock),
            dtype=np.float32,
        )

        self._reset_state()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_observation(), self._get_info()

    def step(self, action: np.ndarray, bypass_action_transform: bool = False):
        target_positions = self._transform_action(action, bypass_action_transform)
        trade_cost = self._execute_trades(target_positions)

        benchmark_top3_idx = None
        if self.use_benchmark_reward and self._current_step >= _BENCHMARK_LOOKBACK:
            bm_scores = np.array(
                [
                    self.dfs[t]["log_return"]
                    .iloc[self._current_step - _BENCHMARK_LOOKBACK : self._current_step]
                    .sum()
                    for t in self.tickers
                ]
            )
            benchmark_top3_idx = np.argsort(bm_scores)[-_BENCHMARK_TOPK:]

        self._current_step += 1
        terminated = self._current_step >= self.max_steps - 1

        log_returns = np.array(
            [
                self.dfs[t].iloc[self._current_step]["log_return"]
                if not terminated
                else 0.0
                for t in self.tickers
            ]
        )

        self._benchmark_log_r = (
            float(np.mean(log_returns[benchmark_top3_idx]))
            if benchmark_top3_idx is not None
            else 0.0
        )

        prev_value = self._portfolio_value
        self._update_portfolio(log_returns)
        reward = self._compute_reward(prev_value, self._portfolio_value, trade_cost)

        self._holding_periods = np.where(
            np.abs(self._positions) > 0.01, self._holding_periods + 1, 0
        )

        return (
            self._get_observation(),
            float(reward),
            terminated,
            False,
            self._get_info(),
        )

    def render(self, mode="human"):
        top_pos = sorted(
            zip(self.tickers, self._positions, strict=True),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:3]
        top_str = " | ".join([f"{t}:{p:+.2f}" for t, p in top_pos])
        print(
            f"Step {self._current_step:4d} | PV: {self._portfolio_value:,.0f} "
            f"| Cash: {self._cash_weight:.2f} | Top: {top_str}"
        )

    def _reset_state(self):
        self._current_step = self.window_size
        self._portfolio_value = self.initial_balance
        self._peak_value = self.initial_balance
        self._max_drawdown = 0.0
        self._positions = np.zeros(self.num_stocks, dtype=np.float32)
        self._trade_returns = np.zeros(self.num_stocks, dtype=np.float32)
        self._holding_periods = np.zeros(self.num_stocks, dtype=np.float32)
        self._return_history: deque = deque(maxlen=SHARPE_WINDOW)
        self._benchmark_log_r = 0.0
        self._cash_weight = 1.0
        self._last_turnover = 0.0
        self.trades_history = []

    def _transform_action(
        self, action: np.ndarray, bypass_action_transform: bool = False
    ) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(-1)

        if bypass_action_transform:
            raw = np.clip(action, 0.0, None)
            if self.enable_cash_action and raw.shape[0] == self.num_stocks + 1:
                stock_raw = raw[: self.num_stocks]
                cash_raw = float(raw[-1])
                total = float(np.sum(stock_raw) + cash_raw)
                if total > 1e-6:
                    self._cash_weight = cash_raw / (total + 1e-8)
                    return (stock_raw / (total + 1e-8)).astype(np.float32)
                self._cash_weight = 1.0
                return np.zeros(self.num_stocks, dtype=np.float32)

            stock_raw = raw[: self.num_stocks]
            total = float(np.sum(stock_raw))
            if total > 1e-6:
                weights = stock_raw / (total + 1e-8)
                self._cash_weight = max(0.0, 1.0 - float(np.sum(weights)))
                return weights.astype(np.float32)
            self._cash_weight = 1.0
            return np.zeros(self.num_stocks, dtype=np.float32)

        expected_dim = self.num_stocks + 1 if self.enable_cash_action else self.num_stocks
        if action.shape[0] != expected_dim:
            raise ValueError(f"Expected action shape ({expected_dim},), got {action.shape}")

        if self.enable_margin_short:
            # -- Tanh Mode (Long/Short & Margin) --
            if self.enable_cash_action:
                stock_weights = np.tanh(action[: self.num_stocks])
                # In margin mode, cash is naturally a residual, but if a cash action is provided,
                # we can use it to scale down overall exposure if it's very large. 
                # For simplicity, we just ignore the cash logit in margin mode and compute cash residually,
                # or we just rely on the max_leverage normalization.
            else:
                stock_weights = np.tanh(action)
                
            if self._topk < self.num_stocks:
                abs_weights = np.abs(stock_weights)
                topk_indices = np.argsort(abs_weights)[-self._topk :]
                mask = np.zeros(self.num_stocks, dtype=np.float32)
                mask[topk_indices] = 1.0
                stock_weights = stock_weights * mask
                
            total_abs_exposure = float(np.sum(np.abs(stock_weights)))
            if total_abs_exposure > self.max_leverage:
                stock_weights = stock_weights * (self.max_leverage / total_abs_exposure)
                
            self._cash_weight = 1.0 - float(np.sum(stock_weights))
            return stock_weights.astype(np.float32)

        # -- Legacy Softmax Mode (Long-Only) --
        shifted = action - np.max(action)
        exp_a = np.exp(shifted / self._softmax_temp)
        soft_weights = exp_a / (np.sum(exp_a) + 1e-8)

        if self.enable_cash_action:
            stock_weights = soft_weights[: self.num_stocks]
            cash_weight = float(soft_weights[-1])
        else:
            stock_weights = soft_weights
            cash_weight = 0.0

        if self._topk < self.num_stocks:
            topk_indices = np.argsort(stock_weights)[-self._topk :]
            mask = np.zeros(self.num_stocks, dtype=np.float32)
            mask[topk_indices] = 1.0
            stock_weights = stock_weights * mask

        total = float(np.sum(stock_weights) + cash_weight)
        if total > 1e-6:
            stock_weights = stock_weights / (total + 1e-8)
            cash_weight = cash_weight / (total + 1e-8)

        self._cash_weight = float(cash_weight) if self.enable_cash_action else 0.0
        return stock_weights.astype(np.float32)

    def _execute_trades(self, target_positions: np.ndarray) -> float:
        self._last_turnover = float(np.sum(np.abs(target_positions - self._positions)))
        total_cost = 0.0
        for i in range(self.num_stocks):
            target = float(target_positions[i])
            current = float(self._positions[i])
            if abs(target - current) < 1e-4:
                continue
            trade_amount = abs(target - current) * self._portfolio_value
            cost = trade_amount * (COMMISSION_RATE + SLIPPAGE_RATE)
            if target < current:
                cost += trade_amount * TAX_RATE_SELL
            total_cost += cost
            
            if self.record_trades:
                date = str(self.dfs[self.tickers[i]].index[self._current_step])[:10]
                self.trades_history.append({
                    "step": self._current_step,
                    "date": date,
                    "ticker": self.tickers[i],
                    "target_weight": target,
                    "prev_weight": current,
                    "trade_amount_twd": trade_amount,
                    "cost": cost,
                    "holding_period_days": float(self._holding_periods[i]) if target < current else 0.0,
                    "trade_type": "BUY" if target > current else "SELL"
                })

            if abs(target) < 1e-4 or np.sign(target) != np.sign(current):
                self._trade_returns[i] = 0.0
            self._positions[i] = target
        self._portfolio_value -= total_cost
        return total_cost / max(self._portfolio_value, 1e-8)

    def _update_portfolio(self, log_returns: np.ndarray):
        daily_returns = np.exp(log_returns) - 1.0
        daily_pnl = np.sum(self._portfolio_value * self._positions * daily_returns)
        self._portfolio_value += daily_pnl

        short_mask = self._positions < 0
        borrow_cost = np.sum(
            self._portfolio_value
            * np.abs(self._positions[short_mask])
            * self.SHORT_RATE_DAILY
        )
        self._portfolio_value -= borrow_cost
        
        # Calculate margin loan interest for negative cash
        if self._cash_weight < 0:
            margin_loan_amount = self._portfolio_value * abs(self._cash_weight)
            margin_interest = margin_loan_amount * self.MARGIN_RATE_DAILY
            self._portfolio_value -= margin_interest

        for i in range(self.num_stocks):
            if abs(self._positions[i]) > 1e-4:
                d = np.sign(self._positions[i]) * daily_returns[i]
                self._trade_returns[i] = (1 + self._trade_returns[i]) * (1 + d) - 1

        if self._portfolio_value > self._peak_value:
            self._peak_value = self._portfolio_value
        dd = (self._peak_value - self._portfolio_value) / max(self._peak_value, 1e-8)
        self._max_drawdown = max(self._max_drawdown, dd)

    def _compute_reward(
        self, prev_value: float, curr_value: float, trade_cost: float
    ) -> float:
        log_r = np.log(max(curr_value, 1e-8) / max(prev_value, 1e-8))
        self._return_history.append(log_r)

        capital_util = np.sum(np.abs(self._positions))
        return_component = _softsign(log_r * 100)

        if len(self._return_history) >= 5:
            arr = np.array(self._return_history)
            mean_r = np.mean(arr)
            neg_returns = arr[arr < 0]
            downside_std = (
                np.std(neg_returns) + 1e-8
                if len(neg_returns) >= 2
                else np.std(arr) + 1e-8
            )
            sortino_component = _softsign(mean_r / downside_std) * capital_util

            if self.use_benchmark_reward:
                benchmark_component = _softsign((log_r - self._benchmark_log_r) * 100)
                hybrid_reward = (
                    0.4 * return_component
                    + 0.3 * sortino_component
                    + 0.3 * benchmark_component
                )
            else:
                hybrid_reward = 0.5 * return_component + 0.5 * sortino_component
        else:
            if self.use_benchmark_reward:
                benchmark_component = _softsign((log_r - self._benchmark_log_r) * 100)
                hybrid_reward = 0.6 * return_component + 0.4 * benchmark_component
            else:
                hybrid_reward = return_component

        cost_p = LAMBDA_COST * trade_cost
        turnover_p = LAMBDA_TURNOVER * (self._last_turnover / 2.0) # Normalized turnover penalty
        cash_ratio = (
            float(self._cash_weight)
            if self.enable_cash_action
            else max(0.0, 1.0 - np.sum(np.abs(self._positions)))
        )
        cash_p = LAMBDA_CASH * cash_ratio

        reward_ref_dd = 0.05
        raw_dd = (self._peak_value - self._portfolio_value) / max(
            self._peak_value, 1e-8
        )
        drawdown_p = LAMBDA_DRAWDOWN * max(0.0, raw_dd - reward_ref_dd)

        # Dynamic Regime Penalty: If drawdown is severe, penalize high stock exposure
        regime_penalty = 0.0
        if raw_dd > 0.10 and not self.enable_margin_short:
            stock_exposure = np.sum(np.abs(self._positions))
            regime_penalty = 0.5 * stock_exposure * (raw_dd - 0.10) # Linearly penalize exposure during drawdown

        return float(np.clip(hybrid_reward - cost_p - turnover_p - cash_p - drawdown_p - regime_penalty, -1.0, 1.0))

    def _get_observation(self) -> np.ndarray:
        obs = np.zeros((self.num_stocks, self._obs_dim_per_stock), dtype=np.float32)
        start = self._current_step - self.window_size
        stock_exposure = float(np.sum(np.abs(self._positions)))
        if self.enable_margin_short or self.enable_cash_action:
            cash_ratio = float(self._cash_weight)
        else:
            cash_ratio = max(0.0, 1.0 - stock_exposure)
            
        total_ret = (self._portfolio_value - self.initial_balance) / max(
            self.initial_balance, 1e-8
        )
        dd_norm = float(np.clip(self._max_drawdown, 0.0, 1.0))
        for i, ticker in enumerate(self.tickers):
            market_window = (
                self.dfs[ticker].iloc[start : self._current_step].values.flatten()
            )
            account = np.array(
                [
                    cash_ratio,
                    float(np.clip(total_ret, -1.0, 1.0)),
                    dd_norm,
                    float(self._positions[i]),
                    float(np.clip(self._trade_returns[i], -1.0, 1.0)),
                    float(np.clip(self._holding_periods[i] / 100.0, 0.0, 1.0)),
                ],
                dtype=np.float32,
            )
            obs[i] = np.concatenate([market_window.astype(np.float32), account])
        return obs

    def _get_info(self) -> dict:
        stock_exposure = float(np.sum(np.abs(self._positions)))
        if self.enable_margin_short or self.enable_cash_action:
            cash_weight = float(self._cash_weight)
        else:
            cash_weight = max(0.0, 1.0 - stock_exposure)
            
        return {
            "portfolio_value": self._portfolio_value,
            "max_drawdown": self._max_drawdown,
            "positions": self._positions.copy(),
            "cash_weight": cash_weight,
            "stock_exposure": stock_exposure,
            "turnover": float(self._last_turnover),
            "tickers": self.tickers,
            "step": self._current_step,
            "trades_history": self.trades_history
        }

    @property
    def total_return(self) -> float:
        return (self._portfolio_value - self.initial_balance) / max(
            self.initial_balance, 1e-8
        )
