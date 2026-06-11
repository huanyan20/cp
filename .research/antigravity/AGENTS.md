# CP P8 Agent — Antigravity IDE

You implement **P8 IndexedReplayBuffer** on branch `feat/p8-indexed-replay-buffer`.
Cursor (another IDE) owns **P10** on a separate branch — do not edit PPO/VecEnv code.

## Read first

1. `docs/SAC_BUFFER_PLAN.md` — full P8 spec
2. `../cp/.research/EXTERNAL_AGENT_BRIEF.md` — handoff + cross-review protocol
3. `../cp/.research/research_state.json` — queue status
4. `../cp/.research/baselines/metrics_sac_enabled_wf_seed*.json` — R6 baseline

## Scope (P8 only)

- Add `IndexedReplayBuffer`: store `(t, account_block, action, reward, done)`; rebuild obs from `trading_env._market_data[t]`
- Wire into SAC path in `train_portfolio.py`
- Unit tests for sample / wrap boundaries
- **Do not** change PPO defaults, VecEnv, or walk_forward tier logic

## Commands (Windows)

Use the main repo venv from this worktree:

```powershell
..\cp\env\Scripts\python.exe -m pytest tests/ -q --tb=short
..\cp\env\Scripts\ruff.exe check .
```

## Handoff (required when done)

Write JSON to the **main repo** (not only this worktree copy if paths differ):

`C:\Users\ggini\Desktop\cp\.research\handoffs\P8.json`

Set `"tool": "antigravity-ide"`, `"pytest": "pass"`, `"ready_for_cross_review": true`.
Schema: `../cp/.research/handoffs/README.md`

## Cross-review P10 (after Cursor handoff)

When `../cp/.research/handoffs/P10.json` exists:

1. Read it + `docs/SAC_BUFFER_PLAN.md` §4
2. Review branch `feat/p10-ppo-vecenv` (read-only; max 3 file edits)
3. Write `../cp/.research/reviews/P10-reviewed-by-external.md`

## Communication

No chat with Cursor. **Only** git commits + `.research/handoffs/` + `.research/reviews/`.
