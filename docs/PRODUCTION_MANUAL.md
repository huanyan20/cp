# Production Manual

> Strategy snapshot: 2026-06-16
> Live trading remains blocked until promotion gate explicitly approves a current SL-first candidate.

## 1. Production Policy

No SAC model is currently eligible for production. SAC results show unstable seed behavior, drawdown over budget, performance concentration, and cash-collapse risk. A high single-period or single-seed SAC result must not be promoted.

The production candidate path is now:

```text
SL signal -> rule allocator -> promotion gate -> trade_guard dry-run -> CMoney RPA
```

## 2. Live Preconditions

Before enabling live trading:

- `promotion_gate.py` must return APPROVED for the exact candidate being deployed.
- `experiment_report.md` must show acceptable OOS Sortino, MDD, turnover, period consistency, and cash behavior.
- `trade_guard.py` dry-run diff must pass.
- Signal TTL, account id, and position limits must validate.
- The candidate must not rely on all-cash behavior or a single strong period to pass risk checks.

## 3. Current Status

| Area | Status |
|---|---|
| SAC | Not promotable; research-only. |
| SL | **BLOCKED / high risk**: active candidate `sl_rule_h10_top20_equal_no_voltarget` has not passed the current multiseed gate. |
| Live | `trade_guard` dry-run observation only. |
| RPA | Blocked unless the active candidate gate reports `core` or `full` approval and dry-run validation passes. |

## 4. Deployment Checklist

```powershell
.\env\Scripts\python.exe -m sl_pipeline.walk_forward_sl --allocator rule --gate --seed 42
.\env\Scripts\python.exe scripts\run_multi_seed_sl.py --horizon 10 --seeds 42,43,44 --allow-blocked-exit-zero
.\env\Scripts\python.exe scripts\experiment_report.py
.\env\Scripts\python.exe scripts\sl_status_check.py
.\env\Scripts\python.exe rpa_pipeline\signal_validator.py
```

Only proceed to CMoney automation when the active candidate is gate-approved and the dry-run diff is acceptable.

## 5. Risk Notes

- SAC cash-enabled models can look safe because they stop trading; this is not production risk control.
- Drawdown must be judged by worst case, not only mean MDD.
- Weak periods must be inspected directly, especially 2024H2, 2025H1, and 2025H2.
- Archive plans before 2026-06-14 are historical and do not authorize live deployment.
