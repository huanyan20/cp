---
name: cp-research-loop
description: >-
  CP v3 RL-rebuild orchestrator: read research_state.json, M1/M2 queue,
  worktree rules. Use when planning r5 changes, M2 tiered WF, or triaging Gate.
---
# CP Research Loop Orchestrator (v3)

## Read first (every turn)

1. `docs/RESEARCH_STRATEGY_V3.md` — **active roadmap**
2. `.research/research_state.json`
3. Last 5 lines of `.research/experiment_ledger.jsonl`
4. `docs/RESEARCH_LOOP.md` if queue logic unclear

Do **not** use `docs/SAC_BUFFER_PLAN.md` v2 as active schedule.

## Phase

| `phase` | Meaning |
|---------|---------|
| `rl_rebuild_v3` | M1 code changes or M2 tiered WF |

## Active queue (v3)

1. **M1a** — obs POMDP fix
2. **M1b** — reward r5 + `ENV_CONFIG_VERSION` bump
3. **M1c** — concentration guard (optional)
4. **M2-smoke** → **M2-candidate** → **M2-promotion** → Gate retry

## Cancelled / frozen (do not resume)

- R7, R7b, R8, R9
- SAC-R R-S2b+ (line frozen)
- P10 VecEnv merge
- `scripts/sac_gradient_ablation.py` (deleted)

## Cross-tool agents (v3)

| Runtime | Current task |
|---------|--------------|
| **External** | v3 strategy review → `V3-reviewed-by-*.md` |
| **Cursor** | M1 implement, orchestrator |

Brief: `.research/reviews/V3-STRATEGY-REVIEW-BRIEF.md`

## Worktree

- **Active**: `cp` main only
- **Frozen**: `cp-p10-ppo`, `cp-sac-r`
- **Merged**: `cp-p8-buffer`

## Forbidden

- Restore R7b/R8/R9 or unfreeze SAC-R without v3 plan amendment
- Auto-run 300K `walk_forward` without human confirmation (`train_slot`)
- Multiple simultaneous MDP changes (one variable per M1 round)

## Orchestrator

```powershell
.\env\Scripts\python.exe scripts\research_orchestrator.py
```

## After M2-promotion

1. `.\env\Scripts\python.exe experiment_report.py`
2. Update `gate.*` in `research_state.json`
3. If worst MDD > 38% → RL path pause per v3 §6
