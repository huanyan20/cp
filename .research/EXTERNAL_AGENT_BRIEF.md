# External Agent Brief

> Strategy snapshot: 2026-06-14
> Task for any external agent: review or implement within the SL-first strategy.

## Current Direction

The project has moved away from SAC-heavy promotion planning. Supervised learning is the main production-candidate path. SAC is retained only for low-cost smoke tests and targeted ablations.

## What To Prioritize

1. SL h10 is now **APPROVED** and has cleared all 5 critical gates (MDD < 35%, Turnover < 10%, Return +134%).
2. The current production path is: `SL signal -> rule allocator -> trade_guard dry-run -> CMoney RPA`.
3. trade_guard compatibility for eventual live dry-run is the next active priority.
4. Keep monitoring rule allocator risk behavior for live forward-testing.

## What Not To Do By Default

- Do not revive SAC-R, R7/R8/R9, or SAC buffer plans as active work.
- Do not run full SAC multi-seed promotion matrices unless smoke checks justify it.
- Do not treat a single high-return SAC seed as deployable.

## Useful Checks

```powershell
.\env\Scripts\python.exe -m sl_pipeline.walk_forward_sl --allocator rule --gate --seed 42
.\env\Scripts\python.exe experiment_report.py
```

All archive material is historical unless updated after 2026-06-14.
