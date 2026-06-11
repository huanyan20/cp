---
name: cp-ppo-efficiency
description: >-
  P10 PPO VecEnv ablation — ARCHIVED. VecEnv not merged (acceptance FAIL).
  Do not use as active queue item under v3.
---
# CP PPO Efficiency (P10) — ARCHIVED

**Status**: Done · **Not merged** · frozen in `cp-p10-ppo` worktree.

## Result summary (2025H1 seed42)

- A1 fps 1.20× vs A0 (need ≥1.5×) — FAIL
- A3 wall 59% vs A0 (need ≤50%) — FAIL
- |ΔMDD| ~14pp — FAIL

Review: `.research/archive/reviews/P10-reviewed-by-external.md`  
Handoff: `.research/archive/handoffs/P10.json`

## v3 rule

Do **not** re-run or merge VecEnv defaults. PPO stays R6 configuration on `main`.

## If human requests re-test

Worktree: `../cp-p10-ppo` · `scripts/ppo_efficiency_ablation.py`
