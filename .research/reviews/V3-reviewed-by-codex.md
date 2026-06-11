# V3 Strategy Review - Codex

reviewed_at: 2026-06-11
agent: codex
verdict: block
m1_recommendation: BLOCK

## Summary

The v3 strategy itself is directionally sound: `docs/RESEARCH_STRATEGY_V3.md`,
`.research/research_state.json`, `專案總覽.md`, and `docs/RESEARCH_LOOP.md`
all point the active path to M1a/M1b/M2 and align the north star to the
Drawdown Gate (`worst MDD <= 35%`). The queue dependency shape is workable:
M1a obs completion before M1b reward r5, M2 tiers blocked until M1 is ready,
and RL pause if promotion remains above the stop-loss threshold.

However, the repository does not satisfy the v3 cleanup contract. R7b is still
present in code and still running, while multiple docs/state files say it was
deleted/stopped. That makes the strategy unsafe to hand to the next agent as-is.

## Blockers

1. `scripts/sac_gradient_ablation.py` still exists, but v3 requires it to be
   deleted.
   - Evidence: file exists at `scripts/sac_gradient_ablation.py`, 7429 bytes,
     last modified 2026-06-11 17:50:46.

2. `train_portfolio.py` still honors `SAC_GRADIENT_STEPS`, so SAC is not fixed
   to `gradient_steps=1`.
   - Evidence: `train_portfolio.py:132` reads
     `os.environ.get("SAC_GRADIENT_STEPS", "1")`; `train_portfolio.py:160`
     passes `gradient_steps=gradient_steps`.

3. An R7b ablation process is still running despite `train_slot.status=free`.
   - Evidence: PID 11816 (`env\\Scripts\\python.exe`) and PID 17968
     (`python3.12.exe`) were running
     `scripts\\sac_gradient_ablation.py --period 2025H1 --seed 42 --model-dir results_dir\\sac_gradient_models`.
   - Started: 2026-06-11 17:50:50 local time.

4. Machine-readable state contradicts runtime/code reality.
   - `.research/research_state.json` says `train_slot.status=free` and the R7b
     note says "script deleted, SAC_GRADIENT_STEPS removed", but both are false
     while the process is running and the code path remains.

## Nits

1. `docs/ALGORITHM_REVIEW.md` has a v3 header, but its short-term/final
   next-step text still says to validate R4/R6 through O2. That should be
   rewritten to M1a obs POMDP -> M1b reward r5 -> M2 smoke.

2. `.research/handoffs/SAC-R.json` is correctly marked `status=frozen` and
   `superseded_by=docs/RESEARCH_STRATEGY_V3.md`, but it still has
   `next_phase=R-S2b` and open R-S2b questions. Because it is clearly frozen,
   this is not a blocker, but renaming those fields to historical/frozen
   context would reduce agent ambiguity.

3. `docs/SAC_BUFFER_PLAN.md` is clearly archived at the top and in section 6,
   but it still contains a long v2 SAC-R route section. Acceptable as history,
   but fragile for agents that skim headings only.

## Stale Docs Found

- No active roadmap doc was found that makes R7/R7b/R8/R9 or active SAC-R the
  next step. The active next step is consistently M1a/M1b.
- Stale cleanup assertions were found:
  - `docs/RESEARCH_STRATEGY_V3.md:36-37`, `docs/RESEARCH_STRATEGY_V3.md:57`,
    and `docs/RESEARCH_STRATEGY_V3.md:172` claim the R7b script and
    `SAC_GRADIENT_STEPS` were removed.
  - `.research/research_state.json:47` and `.research/research_state.json:52`
    claim R7b was stopped and cleanup completed.
  - `scripts/README.md:6`, `docs/SAC_BUFFER_PLAN.md:49`,
    `docs/SAC_BUFFER_PLAN.md:109`, `.cursor/skills/cp-sac-buffer/SKILL.md:25`,
    `.cursor/skills/cp-sac-buffer/SKILL.md:29`, and
    `.cursor/skills/cp-research-loop/SKILL.md:36` repeat the false deletion or
    fixed-gradient assertion.
  - `docs/ALGORITHM_REVIEW.md:132` and `docs/ALGORITHM_REVIEW.md:152` still
    describe the pre-v3 R4/R6 validation next step.

## Verification

- `pytest tests/test_indexed_replay_buffer.py -q`: pass (`5 passed`).
- `train_slot.status`: reported `free` in `.research/research_state.json`, but
  contradicted by the live R7b process.
- Stale active-plan search: no current M1/M2 planning doc treats R7/R7b/R8/R9
  or active SAC-R as the next step; hits were archived/history/brief text except
  for the false cleanup assertions listed above.

## Required Changes Before M1

1. Stop the live `scripts\\sac_gradient_ablation.py` process and confirm no
   R7b/WF process remains.
2. Delete `scripts/sac_gradient_ablation.py`.
3. Remove `SAC_GRADIENT_STEPS` support from `train_portfolio.py` and hard-code
   SAC `gradient_steps=1`.
4. Update the stale cleanup assertions after the code/runtime state is true.
5. Optionally update `docs/ALGORITHM_REVIEW.md` to remove the R4/R6 next-step
   text.

---

## Remediation (2026-06-11, Cursor)

Applied after review:

| Blocker | Fix | Verified |
|---------|-----|----------|
| R7b process running | Stopped PID 11816 | 0 matching processes |
| `sac_gradient_ablation.py` exists | Deleted | `Test-Path` → False |
| `SAC_GRADIENT_STEPS` in code | Removed; `gradient_steps=1` fixed | no rg match |
| `ALGORITHM_REVIEW` stale next steps | Updated to M1a/M1b/M2 | done |
| `SAC-R.json` ambiguity | `next_phase=null` + `next_phase_if_unfrozen` | done |

`pytest tests/test_indexed_replay_buffer.py -q`: **5 passed**

**Post-remediation status**: strategy docs consistent with code; **M1a GO**.

