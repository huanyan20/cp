---
name: cp-promotion-gate
description: >-
  CP Promotion Gate triage after experiment_report. Use after M2-promotion
  or when interpreting metrics JSON under v3 RL rebuild.
---
# CP Promotion Gate (v3)

Active plan: `docs/RESEARCH_STRATEGY_V3.md`

## Run report

```powershell
.\env\Scripts\python.exe experiment_report.py
```

Default: `--current-env-only`

## Key thresholds

| Gate | Default |
|------|---------|
| Sortino stability | ≥ 0.8 across ≥ 3 seeds |
| Max drawdown | **worst-case ≤ 35%** |
| Turnover | ≤ 0.10 |

## v3 decision rule (replaces R7/R8/R9)

**R6/r4 baseline**: worst MDD **44.41%** — Drawdown FAIL only.

| After M2 tier | Action |
|---------------|--------|
| promotion worst MDD ≤ 35% | Gate retry · live prep |
| candidate worst MDD 35–38% | Another M1 single-variable round |
| smoke 2025H1 MDD ≥ r4 same seed | Revert change; do not advance tier |
| promotion worst MDD > 38% | **Pause RL live path** → SL hybrid (M3) |

**Do not** prioritize P8→R7 buffer proof or R8/R9 — cancelled in v3.

## Update state

Write `gate.status`, `gate.failed_checks` into `.research/research_state.json`.
