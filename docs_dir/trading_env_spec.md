# TaiwanStockEnv Specification

Updated: 2026-06-07

`TaiwanStockEnv` is a Gymnasium environment for Taiwan stock portfolio allocation
experiments. It supports long-only allocation, optional cash allocation, and an
experimental margin/short mode.

## Inputs

```python
TaiwanStockEnv(
    df_dict: dict[str, pandas.DataFrame],
    window_size=20,
    initial_balance=1_000_000.0,
    topk=5,
    softmax_temp=0.5,
    use_benchmark_reward=True,
    enable_cash_action=False,
    enable_margin_short=False,
    max_leverage=2.0,
    record_trades=False,
)
```

Each DataFrame in `df_dict` must be aligned by row and include `log_return`.
The loader used by training and evaluation is `data_loader.fetch_multi_asset_data()`.

## Observation

Observation shape:

```text
(num_stocks, window_size * num_market_features + 6)
```

The six account features appended per stock are:

| Feature | Meaning |
| --- | --- |
| `cash_ratio` | Current cash or residual cash weight |
| `portfolio_total_return` | Total return since reset, clipped to `[-1, 1]` |
| `drawdown` | Max drawdown, clipped to `[0, 1]` |
| `position_i` | Current weight for this stock |
| `trade_return_i` | Return since current trade entry, clipped to `[-1, 1]` |
| `holding_period_i` | Holding period divided by 100, clipped to `[0, 1]` |

Market features include normalized OHLCV, technical indicators, momentum,
peer/sector features, RL macro features, and optional overnight capital-flow
features when supplied during data loading.

## Action Modes

### Legacy long-only mode

When `enable_cash_action=False` and `enable_margin_short=False`:

```text
action shape = (num_stocks,)
```

The action is transformed by softmax, stock Top-K masking, and normalization.
This mode is effectively fully invested in selected stocks.

### Cash-aware long-only mode

When `enable_cash_action=True`:

```text
action shape = (num_stocks + 1,)
last action element = cash logit
```

Top-K filtering applies only to stock logits. Stock weights plus cash weight are
normalized to sum to one.

### Margin/short experimental mode

When `enable_margin_short=True`, stock logits are transformed with `tanh`.
Positions may be positive or negative, Top-K uses absolute weight, and total
absolute exposure is capped by `max_leverage`. Cash is residual:

```text
cash_weight = 1 - sum(stock_weights)
```

The cash logit is not a primary control in this mode.

## Trading Costs

Current constants in `trading_env.py`:

```python
COMMISSION_RATE = 0.001425
TAX_RATE_SELL = 0.003
SLIPPAGE_RATE = 0.001
BORROW_RATE_DAILY = 0.015 / 252
MARGIN_RATE_DAILY = 0.06 / 252
SHORT_RATE_DAILY = 0.015 / 252
```

Trade cost is charged on turnover. Sell tax is added when reducing a position.
Short borrow cost and margin loan interest are charged during portfolio updates.

## Reward

Current reward constants:

```python
LAMBDA_COST = 5.0
LAMBDA_TURNOVER = 1.0
LAMBDA_CASH = 0.0
LAMBDA_DRAWDOWN = 0.3
SHARPE_WINDOW = 20
```

The environment combines:

- daily log-return component
- rolling Sortino-like component
- optional benchmark-relative component against top-3 recent momentum stocks
- transaction cost penalty
- turnover penalty
- drawdown penalty after a 5% drawdown buffer
- regime penalty for high long-only exposure when drawdown exceeds 10%

The final reward is clipped to `[-1, 1]`.

## Benchmark Reward

If `use_benchmark_reward=True` and enough history is available:

```text
lookback = 20 trading days
benchmark_topk = 3 stocks
```

The benchmark component compares the portfolio log return to the average return
of the top-3 stocks by recent cumulative log return.

## Trade Recording

When `record_trades=True`, `info["trades_history"]` includes step, date, ticker,
previous weight, target weight, notional trade amount, cost, holding period, and
trade type.

## Known Cautions

- Observation shape changes when feature columns change. Saved models must be
  evaluated with the same feature set used during training.
- The margin/short mode is experimental and should not be treated as live-ready.
- Any live trading flow should validate `signal.json` separately before order
  placement.
