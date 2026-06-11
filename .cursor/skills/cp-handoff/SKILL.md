---
name: cp-handoff
description: >-
  Cross-tool handoff between Cursor agent and external AI coding software.
  Use when P8/P10 implement completes, cross-review swap, or reading
  .research/handoffs/*.json from a non-Cursor agent.
---
# CP Cross-Tool Handoff

## Bus (only channel)

Git branch + `.research/handoffs/*.json` + `.research/reviews/*.md`

No real-time agent chat. External tool reads `.research/EXTERNAL_AGENT_BRIEF.md`.

## Roles (from research_state.json agents)

| Agent | Tasks |
|-------|-------|
| **external** (Antigravity IDE) | P8 implement, review P10 |
| **cursor** | P10 implement, review P8, R7, orchestrator |

## Cursor: after P10 implement

1. `pytest` + `ruff check .` in `../cp-p10-ppo`
2. Write `.research/handoffs/P10.json` (same schema as P8 in EXTERNAL_AGENT_BRIEF §4)
3. Append `experiment_ledger.jsonl`
4. Wait for `.research/handoffs/P8.json` before `cross_review` task

## Cursor: cross-review P8

1. Read `.research/handoffs/P8.json`
2. `git diff feat/p8-indexed-replay-buffer` (or worktree `../cp-p8-buffer`)
3. Run pytest in p8 worktree if patches needed
4. Write `.research/reviews/P8-reviewed-by-cursor.md`
5. Optional: ≤3 file fixes on **p8 branch only**

## When both handoffs exist

Orchestrator next task = `cross_review`. External reviews P10; Cursor reviews P8.

## Forbidden

- Cursor editing `feat/p8-indexed-replay-buffer` during external **implement**
- External editing `feat/p10-ppo-vecenv` during Cursor **implement**
- Merge before both review memos exist
