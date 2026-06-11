---
name: cp-handoff
description: >-
  Cross-tool handoff protocol. P8/P10 implement done; v3 uses strategy review
  brief. Use when reading handoffs or writing V3 review verdict.
---
# CP Cross-Tool Handoff

## Bus

Git + `.research/handoffs/*.json` + `.research/reviews/*.md`

## v3 (current)

External agent reviews **strategy v3**, not P8/P10 implement.

| Input | Output |
|-------|--------|
| `reviews/V3-STRATEGY-REVIEW-BRIEF.md` | `reviews/V3-reviewed-by-<agent>.md` |

Setup: `.research/EXTERNAL_AGENT_BRIEF.md`

## Historical (P8/P10 — complete)

| File | Status |
|------|--------|
| `archive/handoffs/P8.json` | done · merged |
| `archive/handoffs/P10.json` | done · VecEnv not merged |
| `archive/reviews/P8-reviewed-by-cursor.md` | done |
| `archive/reviews/P10-reviewed-by-external.md` | done |

## Historical (SAC-R — frozen)

| File | Status |
|------|--------|
| `archive/handoffs/SAC-R.json` | frozen · R-S0~S2 only |

## Forbidden

- Resume P8/P10 implement loops unless human requests
- Merge P10 VecEnv without new acceptance run
