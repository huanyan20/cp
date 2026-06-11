---
name: cp-promotion-gate
description: >-
  CP Promotion Gate triage after experiment_report. Use when interpreting
  metrics JSON, deciding if R7/R8/R9 is warranted, or summarizing BLOCKED checks.
---
# CP Promotion Gate

## Run report

```powershell
.\env\Scripts\python.exe experiment_report.py
```

Default: `--current-env-only` (do not mix pre-R4 metrics into Gate).

## Eight checks (see `promotion_gate.py` / `docs/RESEARCH_PLAYBOOK.md`)

Key thresholds (`settings.py`, overridable via env):

| Gate | Default |
|------|---------|
| Sortino stability | ≥ 0.8 across ≥ 3 seeds |
| Max drawdown | **worst-case ≤ 35%** |
| Turnover | ≤ 0.10 |

## R7 decision rule

After R6 SAC baseline (buffer=2,805):

- If **worst-case MDD still > 35%** after R4 reward → prioritize **P8→R7** (buffer hypothesis)
- If R7 MDD drops materially toward 35% → buffer was main constraint; consider R8 for throughput
- If R7 MDD unchanged → reward / sim-to-real / CVaR path (`docs/ALGORITHM_REVIEW.md` §四); downgrade R9 PER priority

## Update state

Write `gate.status`, `gate.failed_checks`, `gate.last_report_path` into `.research/research_state.json` after each report run.
