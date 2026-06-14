"""Portfolio trading environment for Taiwan stock allocation experiments."""

from collections import deque

import gymnasium as gym
import numpy as np
import os
from gymnasium import spaces

from env_core.observation_builder import ObservationBuilder
from env_core.reward_calculator import RewardCalculator

try:
    from trading_env_kernels import (
        execute_trades_kernel,
        update_portfolio_kernel,
        compute_reward_kernel,
    )
    _NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    _NUMBA_AVAILABLE = False

COMMISSION_RATE = 0.001425
TAX_RATE_SELL = 0.003
SLIPPAGE_RATE = 0.001
SLIPPAGE_MULTIPLIER = 0.05
BORROW_RATE_DAILY = 0.015 / 252

LAMBDA_COST = 5.0           # Strongly penalize transaction friction
LAMBDA_TURNOVER = 1.0       # Penalize portfolio churn
LAMBDA_CASH = 0.0
LAMBDA_DRAWDOWN = 20.0      # R6.2: Quadratic penalty scaling (was 4.0 linear)
LAMBDA_CASH_DEFENSIVE = 0.80  # R6.1: 0.60 -> 0.80
REWARD_REF_DD = 0.03        # R6.2: 2% → 3% buffer — tolerate small noise
REGIME_DD_THRESHOLD = 0.05  # R6.1: 6% → 5% — regime exposure penalty earlier
REGIME_PENALTY_COEF = 2.0   # R6.1: 1.5 → 2.0 — heavier stock exposure during DD
LAMBDA_WHIPSAW = 0.05
ACTION_DEADBAND = 0.02
SHARPE_WINDOW = 20
MIN_TOP_K_WEIGHT = 0.05  # M1c: floor per active top-k stock before re-normalize
_BENCHMARK_TOPK = 3
_BENCHMARK_LOOKBACK = 20

# Account feature layout (per stock; portfolio-level tail features broadcast).
NUM_ACCOUNT_FEATURES = 9
IDX_CASH = 0
IDX_TOTAL_RETURN = 1
IDX_MAX_DRAWDOWN = 2
IDX_POSITION = 3
IDX_TRADE_RETURN = 4
IDX_HOLDING_PERIOD = 5
IDX_ROLLING_VOL = 6
IDX_ROLLING_SORTINO = 7
IDX_CURRENT_DRAWDOWN = 8


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
        df_dict: dict | None = None,
        npz_path: str | None = None,
        window_size: int = 20,
        initial_balance: float = 1_000_000.0,
        topk: int = 5,
        softmax_temp: float = 1.0,
        use_benchmark_reward: bool = True,
        enable_cash_action: bool = False,
        enable_margin_short: bool = False,
        max_leverage: float = 2.0,
        record_trades: bool = False,
        enable_sl_features: bool = False,
        sl_features_by_ticker: dict | None = None,
        algo: str = "ppo",
    ):
        super().__init__()

        self.algo = algo.lower()

        if df_dict is None and npz_path is None:
            raise ValueError("Must provide either df_dict or npz_path.")

        self.use_benchmark_reward = use_benchmark_reward
        self.enable_cash_action = enable_cash_action
        self.enable_margin_short = enable_margin_short
        self.max_leverage = max_leverage
        self.record_trades = record_trades
        self.window_size = window_size
        self.initial_balance = initial_balance

        self.MARGIN_RATE_DAILY = 0.06 / 252  # 6% annual margin loan interest
        self.SHORT_RATE_DAILY = 0.015 / 252  # 1.5% annual stock borrow fee

        if npz_path and os.path.exists(npz_path):
            data = np.load(npz_path, allow_pickle=True)
            self._full_market_data = data["market_data"]
            self._full_log_returns = data["log_returns"]
            self._full_open_returns = data["open_returns"] if "open_returns" in data else data["log_returns"]
            self.tickers = list(data["tickers"])
            self._full_dates = data["dates"]
            self.num_stocks = self._full_market_data.shape[1]
            self.num_market_features = self._full_market_data.shape[2]
            self.dfs = None
        else:
            self.tickers = list(df_dict.keys())
            self.num_stocks = len(self.tickers)
            self.dfs = {ticker: df.reset_index(drop=True) for ticker, df in df_dict.items()}
            self.num_market_features = self.dfs[self.tickers[0]].shape[1]
            first_ticker = self.tickers[0]
            try:
                self._full_dates = df_dict[first_ticker].index.strftime("%Y-%m-%d").to_numpy()
            except AttributeError:
                self._full_dates = df_dict[first_ticker].index.to_numpy().astype(str)

            self._full_market_data = np.stack(
                [self.dfs[t].to_numpy(dtype=np.float32) for t in self.tickers], axis=1
            )
            self._full_log_returns = np.stack(
                [self.dfs[t]["log_return"].to_numpy(dtype=np.float64) for t in self.tickers],
                axis=1,
            )
            self._full_open_returns = np.stack(
                [self.dfs[t].get("open_return", self.dfs[t]["log_return"]).to_numpy(dtype=np.float64) for t in self.tickers],
                axis=1,
            )

        self._topk = min(topk, self.num_stocks)
        self._softmax_temp = max(softmax_temp, 1e-3)

        self._NUM_ACCOUNT_FEATURES = NUM_ACCOUNT_FEATURES
        self.enable_sl_features = bool(enable_sl_features)
        self._sl_features_by_ticker = sl_features_by_ticker or {}
        self._NUM_SL_FEATURES = 3 if self.enable_sl_features else 0
        if self.enable_sl_features and not self._sl_features_by_ticker:
            raise ValueError("enable_sl_features=True requires sl_features_by_ticker.")
            
        self._full_sl_data = None
        if self.enable_sl_features:
            full_max_steps = self._full_market_data.shape[0]
            self._full_sl_data = np.zeros(
                (full_max_steps, self.num_stocks, self._NUM_SL_FEATURES),
                dtype=np.float32,
            )
            for i, ticker in enumerate(self.tickers):
                sl_row = self._sl_features_by_ticker.get(ticker)
                if sl_row is None:
                    continue
                sl_arr = np.asarray(sl_row, dtype=np.float32)
                n = min(len(sl_arr), full_max_steps)
                self._full_sl_data[:n, i, :] = sl_arr[:n]

        # Initialize the active view to the entire dataset
        self.set_time_window(None, None)

        self._reward_calculator = RewardCalculator(
            sharpe_window=SHARPE_WINDOW,
            use_benchmark_reward=self.use_benchmark_reward,
            enable_cash_action=self.enable_cash_action,
            enable_margin_short=self.enable_margin_short,
        )

        self._observation_builder = ObservationBuilder(
            num_stocks=self.num_stocks,
            window_size=self.window_size,
            num_market_features=self.num_market_features,
            enable_cash_action=self.enable_cash_action,
            enable_margin_short=self.enable_margin_short,
            enable_sl_features=self.enable_sl_features,
            num_sl_features=self._NUM_SL_FEATURES,
        )

        self._obs_dim_per_stock = self._observation_builder.obs_dim_per_stock

        action_dim = self.num_stocks + 1 if self.enable_cash_action else self.num_stocks
        self.action_space = spaces.Box(
            low=-5.0, high=5.0, shape=(action_dim,), dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_stocks, self._obs_dim_per_stock),
            dtype=np.float32,
        )

        self._reset_state()
        
    def set_time_window(self, start_date: str | None, end_date: str | None):
        """
        Dynamically slice the environment's active dataset view.
        If start_date/end_date are None, no bound is applied.
        Date formats should be 'YYYY-MM-DD'.
        """
        start_idx = 0
        end_idx = len(self._full_dates)

        if start_date is not None:
            matches = np.where(self._full_dates >= start_date)[0]
            if len(matches) > 0:
                start_idx = matches[0]

        if end_date is not None:
            matches = np.where(self._full_dates <= end_date)[0]
            if len(matches) > 0:
                end_idx = matches[-1] + 1  # Exclusive bound

        if start_idx >= end_idx:
            raise ValueError(f"Invalid time window: {start_date} to {end_date} yields empty data.")

        self._market_data = self._full_market_data[start_idx:end_idx]
        self._log_returns = self._full_log_returns[start_idx:end_idx]
        self._open_returns = self._full_open_returns[start_idx:end_idx]
        self.dates = self._full_dates[start_idx:end_idx]
        self.max_steps = self._market_data.shape[0]

        if self._full_sl_data is not None:
            self._sl_data = self._full_sl_data[start_idx:end_idx]
        else:
            self._sl_data = None

        print(f"[TradingEnv] Time window updated: {self.dates[0]} ~ {self.dates[-1]} ({self.max_steps} steps)")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_observation(), self._get_info()

    def step(self, action: np.ndarray, bypass_action_transform: bool = False):
        raw_target_positions = self._raw_transform_action(action, bypass_action_transform)

        self._current_step += 1
        terminated = self._current_step >= self.max_steps - 1

        if terminated:
            open_returns = np.zeros(self.num_stocks)
            log_returns = np.zeros(self.num_stocks)
        else:
            open_returns = self._open_returns[self._current_step]
            log_returns = self._log_returns[self._current_step]

        # 1. Apply overnight return to current positions (PV drifts to Open price)
        daily_open_rets = np.exp(open_returns) - 1.0
        overnight_pnl = np.sum(self._portfolio_value * self._positions * daily_open_rets)
        
        prev_pv = self._portfolio_value
        self._portfolio_value += overnight_pnl
        
        if self._portfolio_value > 1e-8:
            self._positions = (self._positions * prev_pv * (1.0 + daily_open_rets)) / self._portfolio_value
            self._cash_weight = max(0.0, 1.0 - float(np.sum(np.abs(self._positions))))
        else:
            self._positions = np.zeros(self.num_stocks, dtype=np.float32)
            self._cash_weight = 1.0

        active_mask = np.abs(self._positions) > 1e-4
        if np.any(active_mask):
            d_open = np.where(self._positions > 0, daily_open_rets, -daily_open_rets)
            self._trade_returns[active_mask] = (1.0 + self._trade_returns[active_mask]) * (1.0 + d_open[active_mask]) - 1.0

        # 2. Action deadband and Limit Up/Down Blocking at Open price
        weight_diff = np.abs(raw_target_positions - self._positions)
        target_positions = np.where(weight_diff < ACTION_DEADBAND, self._positions, raw_target_positions)
        
        is_limit_up = open_returns >= 0.095
        is_limit_down = open_returns <= -0.095
        
        target_positions = np.where(is_limit_up & (target_positions > self._positions), self._positions, target_positions)
        target_positions = np.where(is_limit_down & (target_positions < self._positions), self._positions, target_positions)
        
        if self.enable_margin_short:
            self._cash_weight = 1.0 - float(np.sum(target_positions))
        elif self.enable_cash_action:
            self._cash_weight = max(0.0, 1.0 - float(np.sum(target_positions)))
        else:
            self._cash_weight = 0.0
            
        target_positions = target_positions.astype(np.float32)

        # 3. Calculate penalties and execute trades
        whipsaw_penalty = 0.0
        for i in range(self.num_stocks):
            if target_positions[i] < self._positions[i]:
                if self._holding_periods[i] > 0:
                    days_held = float(self._holding_periods[i])
                    whipsaw_penalty += float(self._positions[i] - target_positions[i]) * max(0.0, 3.0 - days_held)
                    
        trade_cost = self._execute_trades(target_positions)

        # 4. Intraday execution and return
        intraday_log_returns = log_returns - open_returns
        
        self._benchmark_log_r = (
            float(np.mean(log_returns))
            if self.use_benchmark_reward
            else 0.0
        )

        self._update_portfolio(intraday_log_returns)
        
        reward = self._compute_reward(
            prev_pv, self._portfolio_value, trade_cost, whipsaw_penalty
        )

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
        self._benchmark_log_r = 0.0
        self._cash_weight = 1.0
        self._last_turnover = 0.0
        self.trades_history = []
        self._reward_calculator.reset_state()

    def _raw_transform_action(
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
        
        if self.algo == "sac":
            # Soft Top-K for SAC: avoid non-differentiable hard masking
            exp_a = np.exp(shifted / (self._softmax_temp * 0.1)) # Lower temp for sharper concentration natively
        else:
            exp_a = np.exp(shifted / self._softmax_temp)
            
        soft_weights = exp_a / (np.sum(exp_a) + 1e-8)

        if self.enable_cash_action:
            stock_weights = soft_weights[: self.num_stocks]
            cash_weight = float(soft_weights[-1])
        else:
            stock_weights = soft_weights
            cash_weight = 0.0

        if self._topk < self.num_stocks and self.algo != "sac":
            # Hard Top-K Masking (PPO only, SAC fails due to zero gradients at step edge)
            topk_indices = np.argsort(stock_weights)[-self._topk :]
            mask = np.zeros(self.num_stocks, dtype=np.float32)
            mask[topk_indices] = 1.0
            stock_weights = stock_weights * mask
            stock_weights[topk_indices] = np.maximum(
                stock_weights[topk_indices], MIN_TOP_K_WEIGHT
            )

        total = float(np.sum(stock_weights) + cash_weight)
        if total > 1e-6:
            stock_weights = stock_weights / (total + 1e-8)
            cash_weight = cash_weight / (total + 1e-8)

        self._cash_weight = float(cash_weight) if self.enable_cash_action else 0.0
        return stock_weights.astype(np.float32)

    def _execute_trades(self, target_positions: np.ndarray) -> float:
        if _NUMBA_AVAILABLE:
            cost_ratio, new_pv, turnover, new_pos, new_tr = execute_trades_kernel(
                target_positions.astype(np.float32),
                self._positions,
                self._trade_returns,
                self._portfolio_value,
                COMMISSION_RATE,
                TAX_RATE_SELL,
                SLIPPAGE_RATE,
                SLIPPAGE_MULTIPLIER,
            )
            self._last_turnover = float(turnover)
            self._positions = new_pos
            self._trade_returns = new_tr
            self._portfolio_value = float(new_pv)
            return float(cost_ratio)

        # Fallback pure-Python path (record_trades mode or Numba unavailable)
        self._last_turnover = float(np.sum(np.abs(target_positions - self._positions)))
        total_cost = 0.0
        for i in range(self.num_stocks):
            target = float(target_positions[i])
            current = float(self._positions[i])
            if abs(target - current) < 1e-4:
                continue
            trade_weight = abs(target - current)
            trade_amount = trade_weight * self._portfolio_value
            dynamic_slippage = SLIPPAGE_RATE + SLIPPAGE_MULTIPLIER * (trade_weight ** 2)
            cost = trade_amount * (COMMISSION_RATE + dynamic_slippage)
            if target < current:
                cost += trade_amount * TAX_RATE_SELL
            total_cost += cost
            if self.record_trades:
                date = str(self.dates[self._current_step])[:10]
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
        if _NUMBA_AVAILABLE:
            new_pv, new_peak, new_mdd, new_tr = update_portfolio_kernel(
                self._positions,
                log_returns,
                self._trade_returns,
                self._portfolio_value,
                self._peak_value,
                self._max_drawdown,
                self._cash_weight,
                self.SHORT_RATE_DAILY,
                self.MARGIN_RATE_DAILY,
            )
            self._portfolio_value = float(new_pv)
            self._peak_value = float(new_peak)
            self._max_drawdown = float(new_mdd)
            self._trade_returns = new_tr.astype(np.float32)
            return

        # Fallback path
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


    def _current_drawdown(self) -> float:
        return (self._peak_value - self._portfolio_value) / max(
            self._peak_value, 1e-8
        )

    def _compute_reward(
        self, prev_value: float, curr_value: float, trade_cost: float, whipsaw_penalty: float
    ) -> float:
        return self._reward_calculator.compute_reward(
            prev_value=prev_value,
            curr_value=curr_value,
            trade_cost=trade_cost,
            whipsaw_penalty=whipsaw_penalty,
            benchmark_log_r=self._benchmark_log_r,
            positions=self._positions,
            cash_weight=self._cash_weight,
            last_turnover=self._last_turnover,
            current_dd=float(np.clip(self._current_drawdown(), 0.0, 1.0)),
        )

    def _get_observation(self) -> np.ndarray:
        return self._observation_builder.build(
            current_step=self._current_step,
            positions=self._positions,
            cash_weight=self._cash_weight,
            portfolio_value=self._portfolio_value,
            initial_balance=self.initial_balance,
            max_drawdown=self._max_drawdown,
            trade_returns=self._trade_returns,
            holding_periods=self._holding_periods,
            pomdp_features=self._reward_calculator.get_pomdp_cache(
                current_dd=float(np.clip(self._current_drawdown(), 0.0, 1.0))
            ),
            market_data=self._market_data,
            sl_data=self._sl_data if self.enable_sl_features else None,
        )

    def portfolio_state_snapshot(self):
        """PortfolioState-compatible dict for SL allocators (S5)."""
        from sl_pipeline.allocator import PortfolioState

        stock_exposure = float(np.sum(np.abs(self._positions)))
        if self.enable_margin_short or self.enable_cash_action:
            cash_weight = float(self._cash_weight)
        else:
            cash_weight = max(0.0, 1.0 - stock_exposure)
        return PortfolioState(
            positions={
                self.tickers[i]: float(self._positions[i])
                for i in range(self.num_stocks)
                if abs(self._positions[i]) > 1e-6
            },
            cash_weight=cash_weight,
            portfolio_value=float(self._portfolio_value),
            peak_value=float(self._peak_value),
            rolling_mdd=float(self._max_drawdown),
        )

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
