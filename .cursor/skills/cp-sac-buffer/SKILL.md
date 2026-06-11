---
name: cp-sac-buffer
description: >-
  P8 IndexedReplayBuffer and R7 SAC retrain. Use when implementing or reviewing
  SAC replay buffer changes, R7 controlled experiment, or SAC_BUFFER_RAM_GB.
---
# CP SAC Buffer (P8 / R7)

Full spec: `docs/SAC_BUFFER_PLAN.md`

## Problem

R6 SAC uses buffer **2,805** (~0.9% of 300K steps) → quasi on-policy; bear-market transitions evicted quickly.

## P8 — IndexedReplayBuffer

- Store per transition: `t` + account block + action/reward/done (~2.4KB)
- Rebuild market window from env `_market_data` on `sample()` — bit-identical obs
- Inject via `SAC(replay_buffer_class=..., replay_buffer_kwargs=...)`
- Tests: `tests/test_indexed_replay_buffer.py` (add when implementing)

## R7 — controlled retrain

- **Sole variable**: buffer capacity (2,805 → full 300K history)
- Do **not** change reward, obs, network, or timesteps vs R6
- Do **not** set `SAC_BUFFER_RAM_GB=1` for R7 (use default 4GB or P8 indexed buffer)
- Compare against `.research/baselines/r6_sac_*.json`

## R6 continuity (if resuming interrupted R6 SAC)

```powershell
$env:SAC_BUFFER_RAM_GB='1'
.\env\Scripts\python.exe -u walk_forward.py --candidates --tier promotion
```
