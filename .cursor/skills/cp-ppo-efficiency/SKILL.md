---
name: cp-ppo-efficiency
description: >-
  P10 PPO training throughput ablation (VecEnv, n_steps, n_epochs). Use after R6
  when optimizing PPO wall-clock without changing algorithm.
---
# CP PPO Efficiency (P10)

Full spec: `docs/SAC_BUFFER_PLAN.md` §4

## When

After R6 `ppo/disabled` metrics frozen; parallel with P8 code work (separate worktree).

## Ablation matrix (one variable at a time)

Fixed: `ppo`, `cash=disabled`, seed=42, 300K, period **2025H1**

| Stage | Change |
|-------|--------|
| A0 | Baseline DummyVecEnv |
| A1 | `SubprocVecEnv` n_envs=4 (or 2 if VRAM tight) |
| A2 | A1 + `n_epochs=5` |
| A3 | A2 + `n_steps=512` |

## Acceptance

- A1 fps ≥ 1.5× A0
- A3 wall time ≤ 50% A0
- OOS |ΔMDD| < 5pp vs A0

## Script (R6后新增)

```powershell
.\env\Scripts\python.exe scripts\ppo_efficiency_ablation.py --period 2025H1 --seed 42
```

Worktree: `../cp-p10-ppo` branch `feat/p10-ppo-vecenv`
