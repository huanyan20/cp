"""
Numba-JIT compiled hot-path kernels for TaiwanStockEnv.

These pure-NumPy functions are extracted from trading_env.py so that Numba
can compile them to native machine code.  All functions must avoid Python
objects (dicts, lists, deque) — only scalars and NumPy arrays are allowed.

First call triggers JIT compilation (a few seconds); subsequent calls are
near-C speed.  cache=True stores the compiled binary to __pycache__ so
restarted processes skip recompilation.
"""

import numpy as np
import numba as nb


# ---------------------------------------------------------------------------
# Trade execution kernel
# ---------------------------------------------------------------------------
@nb.jit(nopython=True, cache=True)
def execute_trades_kernel(
    target_positions: np.ndarray,   # (num_stocks,) float32
    positions: np.ndarray,          # (num_stocks,) float32  [in/out]
    trade_returns: np.ndarray,      # (num_stocks,) float32  [in/out]
    portfolio_value: float,
    commission_rate: float,
    tax_rate_sell: float,
    slippage_rate: float,
) -> tuple:
    """
    Returns (total_cost_ratio, new_portfolio_value, turnover, new_positions, new_trade_returns).
    """
    turnover = 0.0
    total_cost = 0.0
    n = target_positions.shape[0]
    new_positions = positions.copy()
    new_trade_returns = trade_returns.copy()

    for i in range(n):
        target = float(target_positions[i])
        current = float(positions[i])
        delta = abs(target - current)
        turnover += delta
        if delta < 1e-4:
            continue
        trade_amount = delta * portfolio_value
        cost = trade_amount * (commission_rate + slippage_rate)
        if target < current:
            cost += trade_amount * tax_rate_sell
        total_cost += cost

        if abs(target) < 1e-4 or (target > 0) != (current > 0):
            new_trade_returns[i] = 0.0
        new_positions[i] = target

    new_pv = portfolio_value - total_cost
    cost_ratio = total_cost / max(new_pv, 1e-8)
    return cost_ratio, new_pv, turnover, new_positions, new_trade_returns


# ---------------------------------------------------------------------------
# Portfolio update kernel
# ---------------------------------------------------------------------------
@nb.jit(nopython=True, cache=True)
def update_portfolio_kernel(
    positions: np.ndarray,          # (num_stocks,) float32
    log_returns: np.ndarray,        # (num_stocks,) float64
    trade_returns: np.ndarray,      # (num_stocks,) float32  [in/out]
    portfolio_value: float,
    peak_value: float,
    max_drawdown: float,
    cash_weight: float,
    short_rate_daily: float,
    margin_rate_daily: float,
) -> tuple:
    """
    Returns (new_portfolio_value, new_peak_value, new_max_drawdown, new_trade_returns).
    """
    daily_returns = np.exp(log_returns) - 1.0

    daily_pnl = 0.0
    for i in range(positions.shape[0]):
        daily_pnl += portfolio_value * positions[i] * daily_returns[i]
    portfolio_value += daily_pnl

    # Short borrow cost
    for i in range(positions.shape[0]):
        if positions[i] < 0:
            portfolio_value -= portfolio_value * abs(positions[i]) * short_rate_daily

    # Margin interest on negative cash
    if cash_weight < 0:
        portfolio_value -= portfolio_value * abs(cash_weight) * margin_rate_daily

    # Trade returns (compounding)
    new_trade_returns = trade_returns.copy()
    for i in range(positions.shape[0]):
        if abs(positions[i]) > 1e-4:
            d = (1.0 if positions[i] > 0 else -1.0) * daily_returns[i]
            new_trade_returns[i] = (1.0 + trade_returns[i]) * (1.0 + d) - 1.0

    # Peak & MDD
    if portfolio_value > peak_value:
        peak_value = portfolio_value
    dd = (peak_value - portfolio_value) / max(peak_value, 1e-8)
    if dd > max_drawdown:
        max_drawdown = dd

    return portfolio_value, peak_value, max_drawdown, new_trade_returns


# ---------------------------------------------------------------------------
# Reward kernel  (scalar maths only — deque history stays in Python)
# ---------------------------------------------------------------------------
@nb.jit(nopython=True, cache=True)
def compute_reward_kernel(
    log_r: float,
    benchmark_log_r: float,
    sortino_proxy: float,
    capital_util: float,
    trade_cost: float,
    last_turnover: float,
    cash_ratio: float,
    raw_dd: float,
    use_benchmark_reward: bool,
    enable_margin_short: bool,
    enable_cash_action: bool,
    history_len: int,           # len(return_history) — passed from Python
    # reward lambdas
    lambda_cost: float,
    lambda_turnover: float,
    lambda_cash: float,
    lambda_drawdown: float,
    reward_ref_dd: float,
    regime_dd_threshold: float,
    regime_penalty_coef: float,
    lambda_cash_defensive: float,
) -> float:
    def softsign(x: float) -> float:
        return x / (1.0 + abs(x))

    return_component = softsign(log_r * 100.0)

    if history_len >= 5:
        sortino_component = sortino_proxy * capital_util
        if use_benchmark_reward:
            benchmark_component = softsign((log_r - benchmark_log_r) * 100.0)
            hybrid_reward = (
                0.4 * return_component
                + 0.3 * sortino_component
                + 0.3 * benchmark_component
            )
        else:
            hybrid_reward = 0.5 * return_component + 0.5 * sortino_component
    else:
        if use_benchmark_reward:
            benchmark_component = softsign((log_r - benchmark_log_r) * 100.0)
            hybrid_reward = 0.6 * return_component + 0.4 * benchmark_component
        else:
            hybrid_reward = return_component

    cost_p = lambda_cost * trade_cost
    turnover_p = lambda_turnover * (last_turnover / 2.0)
    cash_p = lambda_cash * cash_ratio
    drawdown_p = lambda_drawdown * max(0.0, raw_dd - reward_ref_dd)

    regime_penalty = 0.0
    if raw_dd > regime_dd_threshold and not enable_margin_short:
        regime_penalty = (
            regime_penalty_coef
            * capital_util
            * (raw_dd - regime_dd_threshold)
        )

    cash_defensive_bonus = 0.0
    if raw_dd > regime_dd_threshold and enable_cash_action:
        cash_defensive_bonus = (
            lambda_cash_defensive * cash_ratio * (raw_dd - regime_dd_threshold)
        )

    raw = (
        hybrid_reward
        - cost_p - turnover_p - cash_p
        - drawdown_p - regime_penalty
        + cash_defensive_bonus
    )
    return max(-1.0, min(1.0, raw))
