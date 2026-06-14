# Experiment Report

> Manual strategy snapshot: 2026-06-14
> This file was manually synchronized with the SL-first strategy. Running `experiment_report.py` may overwrite this narrative.

## 0. Current Promotion Decision

### MODEL NOT ELIGIBLE FOR SAC PROMOTION

SAC is not currently suitable for live promotion. The best SAC results show useful research signal but not enough production robustness:

- `SAC / cash=disabled / r5.1 / seed42`: high total return, but max drawdown is above the 35% risk budget and evidence is only one seed.
- `SAC / cash=enabled / vr5.1 / seed43`: low drawdown mostly comes from staying near all-cash after 2022_BEAR, so the result is a cash-collapse warning rather than deployable risk control.
- Older cash-enabled SAC runs have large seed dispersion and weak-period instability.
- Standalone SAC eval remains poor, with negative return, high drawdown, and high turnover.

## 1. Strategy Decision

The project direction is now **SL-first / SAC research-only**.

```text
SL signal -> rule allocator -> promotion gate -> trade_guard -> live only after approval
```

SAC remains useful for low-cost reward/action/allocator ablations, but full 300K multi-seed SAC promotion is no longer the default iteration loop.

## 2. SAC Findings

| Candidate | Reading | Decision |
|---|---|---|
| SAC cash-disabled r5.1 seed42 | High return, high drawdown, one seed, strong 2026H1 concentration. | Research signal only; not promotable. |
| SAC cash-enabled vr5.1 seed43 | Very low drawdown because most OOS periods are 100% cash. | Cash collapse; not promotable. |
| SAC cash-enabled r4 seeds42/43/44 | Average metrics look plausible but seed dispersion is too large. | Historical baseline only. |
| SAC standalone eval | Negative return and high turnover/drawdown. | Confirms poor generalization risk. |

## 3. SL Priority

SL is now the production-candidate track because it is faster to iterate, easier to diagnose, and better suited to allocator/risk-gate improvements. Immediate work should focus on:

1. Abandon h5 as a production line; focus repair work on h10.
2. Reduce h10 MDD below 35% with allocator risk controls.
3. MDD and turnover reduction.
4. Period consistency, especially 2024H2 and 2025H1.
5. trade_guard dry-run compatibility after gate approval.

## 4. Promotion Checklist

Any candidate must pass the gate for the exact artifact being deployed:

- Sortino stability.
- Worst-case max drawdown within budget.
- Turnover after realistic costs.
- No cash-collapse behavior.
- No single-period performance dependence.
- Baseline and stress comparison where available.

## 5. Historical Metrics

Detailed historical metrics remain in `experiment_summary.json` and `results_dir/metrics_*.json`. Those artifacts should be treated as data sources; the current strategy interpretation is this 2026-06-14 snapshot.
