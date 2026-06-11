---
name: cp-research-loop
description: >-
  CP post-R6 research orchestrator: read .research/research_state.json, pick
  next queue task, worktree rules, forbidden actions during R6. Use when
  planning P8/R7/R10, arming automation loops, or triaging after experiment_report.
---
# CP Research Loop Orchestrator

## Read first (every turn)

1. `.research/research_state.json`
2. Last 5 lines of `.research/experiment_ledger.jsonl`
3. `docs/RESEARCH_LOOP.md` if queue logic unclear
4. `.cursor/skills/cp-handoff/SKILL.md` if handoff / cross_review

Do **not** re-derive tier timesteps, buffer sizes, or Gate thresholds from memory.

## Cross-tool agents

| Runtime | Owns |
|---------|------|
| **Cursor** (you) | P10, review P8, R7, orchestrator |
| **External** (other software) | P8, review P10 |

External agent reads worktree `AGENTS.md` + `.research/EXTERNAL_AGENT_BRIEF.md`.
Communication: `handoffs/*.json` + `reviews/*.md` + git branches only.

## Phase guards

| `phase` | Agent may |
|---------|-----------|
| `r6_ppo_running` | Monitor `walk_forward` only; **no** code changes to `train_portfolio.py` buffer/PPO defaults; **no** R7 training |
| `post_r6` | Execute queue per priority; one `in_progress` training at a time |

Check walk_forward running:

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -match 'walk_forward' }
```

## Queue priority (after `freeze_r6` done)

1. `P8` (external) and `P10` (cursor) **parallel implement** on separate worktrees
2. Both write `handoffs/{P8,P10}.json` → `cross_review` swap
3. `R7` only after `P8` merge-ready + `cross_review` memos + `train_slot.status == free`
4. `R8` / `R9` only if R7 worst-case MDD still > 35% (see `cp-promotion-gate` skill)

## Worktree convention

```powershell
git worktree add ..\cp-p8-buffer -b feat/p8-indexed-replay-buffer
git worktree add ..\cp-p10-ppo -b feat/p10-ppo-vecenv
```

One branch per agent; merge only after `pytest` + `ruff check .`.

## Forbidden (until R6 frozen)

- Change `SAC_BUFFER_RAM_GB` during active R6 walk_forward
- Merge P8 before copying metrics to `.research/baselines/`
- Multiple simultaneous edits to `train_portfolio.py` on different branches without coordination
- Auto-run 300K `walk_forward` without human confirmation (`train_slot`)

## Dry-run orchestrator

```powershell
.\env\Scripts\python.exe scripts\research_orchestrator.py
```

## After R6 complete

1. `.\env\Scripts\python.exe experiment_report.py`
2. Copy `results_dir/metrics_*_wf_*.json` → `.research/baselines/`
3. Set `phase` → `post_r6`, `freeze_r6` → `done`, `train_slot` → `free`
4. Append ledger; pick `P8` or `P10`

## Ledger append format

```json
{"ts":"ISO8601","agent":"...","action":"...","task_id":"P8","decision":"...","metrics":{"mdd":0.38,"sortino":2.1}}
```

Keep metrics summary ≤ 5 numbers; full paths only.
