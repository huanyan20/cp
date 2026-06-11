---
name: cp-sac-buffer
description: >-
  P8 IndexedReplayBuffer reference and v3 RL-rebuild guardrails. Use when
  touching indexed_replay_buffer.py or SAC train path — not for R7b/R8/R9/SAC-R.
---
# CP SAC Buffer (v3 — infrastructure only)

**Active plan**: `docs/RESEARCH_STRATEGY_V3.md`  
**Archived spec**: `docs/archive/SAC_BUFFER_PLAN_v2.md` (stub: `docs/SAC_BUFFER_PLAN.md`)

## v3 rule

Only **MDD-aligned** changes (r5 obs/reward/concentration) go in queue.  
Do **not** add efficiency ablations, PER, SL-only obs, gradient_steps sweeps, or SAC-R work.

## Keep (merged infrastructure)

| Item | Why keep |
|------|----------|
| `indexed_replay_buffer.py` | 300K replay on Windows RAM |
| float16 account + vectorized `_reconstruct_obs` | training feasible |
| `SAC_BUFFER_RAM_GB` | R6 reproduction only (`=1` → legacy 2,805 cap) |

SAC uses `gradient_steps=1` (fixed). No `SAC_GRADIENT_STEPS` env.

## Cut / frozen (do not resume without v3 amendment)

- R7b `sac_gradient_ablation.py` — deleted
- R7 WF for P8→MDD proof — cancelled
- R8 SL-only obs — cancelled
- R9 PER — cancelled (M3 only)
- SAC-R line — frozen (`cp-sac-r` worktree)

## Validation (buffer edits only)

```powershell
.\env\Scripts\python.exe -m pytest tests/test_indexed_replay_buffer.py -q
```

## Do not

- Block buffer fixes on R6 ΔMDD
- Re-introduce R7b/R8/R9 queue items
- Mix SAC-R into main `train_portfolio` defaults
