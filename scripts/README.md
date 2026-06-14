# Scripts README

> Strategy snapshot: 2026-06-14
> Use scripts to support SL-first research and gated production only.

## Main Commands

```powershell
# SL mainline
.\env\Scripts\python.exe -m sl_pipeline.walk_forward_sl --allocator rule --gate --seed 42

# Report and gate review
.\env\Scripts\python.exe experiment_report.py

# SAC research smoke only
.\env\Scripts\python.exe walk_forward.py --tier smoke --algo sac --cash-mode enabled
```

## Current Policy

- SL h10 experiments are the production-candidate path; h5 is retired from the mainline.
- SAC scripts are for low-cost ablation and failure detection.
- Do not run full SAC promotion matrices unless a smoke result has already passed cash, drawdown, concentration, turnover, and weak-period checks.
- Any live-facing script must consume a gate-approved candidate and pass trade_guard validation.

## Useful Script Groups

| Group | Files |
|---|---|
| Evaluation | `evaluate_portfolio.py`, `model_test_report.py`, `validate_period.py` |
| Research diagnostics | `error_analysis.py`, `friction_analysis.py`, `sector_analysis.py`, `shap_analysis.py` |
| SAC/RL utilities | `smoke_r6.py`, `test_models.py`, `verify_models.py` |
| Production guard | `rpa_pipeline/*`, `trade_guard.py` |

Historical scripts may still mention SAC promotion. Treat those comments as legacy unless updated after 2026-06-14.
