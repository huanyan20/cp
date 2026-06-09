# TaiwanStockEnv Specification

Updated: 2026-06-08 (R4 reward tuning + O1 env_config versioning)

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

Current reward constants (updated R4, 2026-06-08):

```python
LAMBDA_COST = 5.0            # transaction friction penalty
LAMBDA_TURNOVER = 1.0        # portfolio churn penalty
LAMBDA_CASH = 0.0            # static cash penalty (disabled)
LAMBDA_DRAWDOWN = 0.8        # R4: raised from 0.3 → 0.8
LAMBDA_CASH_DEFENSIVE = 0.2  # R4: new — rewards holding cash during drawdown
SHARPE_WINDOW = 20
```

The environment combines:

- daily log-return component
- rolling Sortino-like component
- optional benchmark-relative component against top-3 recent momentum stocks
- transaction cost penalty
- turnover penalty
- drawdown penalty after a **3% buffer** (R4: reduced from 5%)
- regime penalty for high stock exposure when drawdown exceeds **8%** (R4: threshold lowered from 10%, coefficient raised from 0.5 → 1.0)
- defensive cash bonus when drawdown exceeds 8% and `enable_cash_action=True` (R4: new)

The final reward is clipped to `[-1, 1]`.

### R4 Motivation

Experiment data showed MDD accumulated almost entirely in 2025H1 (bear market), with
SAC/enabled averaging only 7.38% cash during that period. The original `LAMBDA_DRAWDOWN=0.3`
let a small daily gain outweigh the drawdown penalty, leaving no strong training signal
to cut equity exposure. The R4 adjustments make it unprofitable to remain fully invested
during sustained drawdowns and provide a positive incentive for holding cash defensively.

### Experiment versioning (O1)

Walk-forward metrics include an `env_config` snapshot from `env_config.py`:

- `env_config_version`: human label (currently `r4`); bump when reward/regime changes.
- `env_config_hash`: 8-char SHA-256 fingerprint of reward + topk knobs.

`experiment_report.py` defaults to **current-env-only** so pre-R4 metrics are not mixed
into Promotion Gate after R6 retraining.

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
