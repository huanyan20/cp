# Capital Flow Analysis

`capital_flow_analysis` provides overnight macro and capital-flow features for
the Taiwan stock RL project. It is intentionally separate from the core RL stock
universe so that research features can be tested, ablated, and monitored before
being promoted into the live signal flow.

## Purpose

The package has two jobs:

- Build overnight features from ADR, SOX, Nasdaq, VIX, FX, crypto, and related
  market data.
- Run a pre-open macro guard that can reduce or block pending buy orders before
  the Taiwan market opens.

The current RL default uses a small selected feature set:

- `tsm_adr_premium_chg`
- `tsm_adr_premium`
- `TSM_ret`

Broader macro features are available for research and reporting, but should be
promoted only after ablation shows they improve out-of-sample behavior.

## Daily Operations

Generate overnight features:

```bash
python capital_flow_analysis/src/data_pipeline/overnight_gap_features.py --report
```

Outputs:

- `capital_flow_analysis/data/overnight_gap_features_1d.csv`
- `capital_flow_analysis/reports/overnight_gap_feature_report.md`

Run the pre-open guard:

```bash
python capital_flow_analysis/src/monitoring/preopen_macro_check.py
```

Output:

- `capital_flow_analysis/data/preopen_macro_check.json`

Guard levels:

| Level | Trading behavior |
| --- | --- |
| `OK` | Normal signal flow |
| `WARN` | Pending buys are reduced |
| `CRITICAL` | Pending buys are skipped; sell-only flow may continue |

Generate an RL signal with overnight features:

```bash
python evaluate_portfolio.py --model-path ppo_portfolio_full_stock_seed42.zip --overnight-feature-path capital_flow_analysis/data/overnight_gap_features_1d.csv
```

The evaluator writes `signal.json`, which is later consumed by the CMoney RPA
flow. Live order placement is controlled separately by `ENABLE_LIVE_TRADING`.

## Research

Evaluate the gap models:

```bash
python capital_flow_analysis/src/modeling/evaluate_gap_model.py --target open_gap
python capital_flow_analysis/src/modeling/evaluate_gap_model.py --target intraday_return
python capital_flow_analysis/src/modeling/evaluate_gap_model.py --target gap_fade_return
```

The evaluation code uses time-aware splits and excludes target columns from
feature ranking to reduce leakage risk.

## Tests

Run the project tests from the repository root:

```bash
python -m unittest discover -s tests
```

After installing the development dependencies, pytest can also discover the same
tests:

```bash
python -m pytest
```

## Notes

- `MACRO_TICKERS_RL` is the smaller macro universe used by the RL portfolio.
- `MACRO_TICKERS_FLOW` is the broader macro universe used by flow analysis.
- Data caches and generated reports are runtime artifacts and should not be
  committed by default.
