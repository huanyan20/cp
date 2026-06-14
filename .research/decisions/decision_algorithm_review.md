# Algorithm Decision Review

> Strategy snapshot: 2026-06-14
> Decision: SL-first for production candidates; SAC is research-only.

## 1. Decision

The project will prioritize supervised learning for the production candidate path. SAC and PPO are no longer default promotion candidates. SAC remains in the toolbox for low-cost research smoke tests and targeted ablations.

## 2. Why SAC Was Demoted

Recent SAC results are not robust enough for live promotion:

- `SAC cash=disabled r5.1 seed42` produced high total return but exceeded the drawdown budget and only has one seed.
- `SAC cash-enabled vr5.1 seed43` reduced drawdown mostly by holding cash across most OOS periods, which is cash collapse rather than deployable risk control.
- Older `r4 cash-enabled` runs had high dispersion across seeds.
- A standalone SAC eval showed negative return, high drawdown, and high turnover, indicating poor generalization.

The failure mode is visible early, so full multi-seed SAC promotion runs are not an efficient default.

## 3. Current Mainline

```text
SL signal -> rule allocator -> promotion gate -> trade_guard -> live only after approval
```

SL is preferred because it is faster to iterate, more interpretable, easier to diagnose by period and feature, and better aligned with allocator/risk-gate work.


## 3. SL Horizon Decision

The h5 SL line is no longer a production candidate. The latest completed h5 walk-forward across seeds 42/43/44 produced negative total returns, max drawdown near 49% to 54%, and turnover around 23% to 25%. Its good 2026H1 behavior does not offset persistent losses in 2022_BEAR and 2024H2.

The active SL repair target is h10. The current h10 reference has strong return and Sortino with acceptable turnover, but MDD remains above the 35% gate. Next work should reduce h10 drawdown through lower vol target, weak-period cash/risk gates, and allocator constraints.

## 4. SAC Research Rules

SAC experiments should use 5K/20K/50K smoke budgets by default. Escalate to 300K only after the smoke result avoids these failures:

- all-cash or near all-cash behavior;
- single-stock concentration;
- weak-period drawdown above budget;
- turnover that fails cost realism;
- total return dominated by one period.

Passing smoke only permits a larger ablation. It does not imply promotion.

## 5. Promotion Standard

Any production candidate, including SL, must pass the gate for the exact artifact being deployed. The gate must reject cash-collapse models and must judge drawdown by worst-case behavior, not only mean metrics.

## 6. Superseded Decisions

Earlier SAC-heavy decisions, SAC-R plans, and MDD remediation queues are historical. They are kept in archive for context and are superseded by this 2026-06-14 decision.
