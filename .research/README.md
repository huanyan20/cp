# CP Research Hub

> Strategy snapshot: 2026-06-14
> Active decision: SL-first, SAC research-only.

## Current State

The active research queue is now centered on supervised learning: signal quality, allocator risk control, promotion gate reliability, and trade_guard integration. SAC remains useful for low-cost ablation, but it is no longer the default promotion path.

## Active Documents

| File | Role |
|---|---|
| [decisions/decision_algorithm_review.md](decisions/decision_algorithm_review.md) | Current algorithm decision and rationale. |
| [decisions/README.md](decisions/README.md) | Decision index. |
| [runs/README.md](runs/README.md) | Experiment run notes. |
| [baselines/README.md](baselines/README.md) | Historical baselines. |
| [archive/README.md](archive/README.md) | Superseded SAC-heavy plans and reviews. |

## Current Queue

1. Abandon the h5 SL line for production; repair h10 risk behavior first.
2. Re-run SL walk-forward and promotion gate after each material change.
3. Use SAC only for smoke-sized reward/action ablations.
4. Promote nothing to live until the exact candidate is gate-approved and trade_guard dry-run passes.

## Superseded Work

Older R7/R8/R9/SAC-R/SAC-buffer plans are retained for technical context only. They were superseded on 2026-06-14 after SAC showed unstable seed behavior, high drawdown, performance concentration, and cash-collapse risk.
