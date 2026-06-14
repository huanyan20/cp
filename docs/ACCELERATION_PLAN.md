# 訓練效能加速計畫

> Strategy snapshot: 2026-06-14
> Goal: maximize research iteration speed under the SL-first strategy.

## 1. Current Priority

訓練效率的主要目的已經改變：不是讓 SAC 更快跑完整 promotion matrix，而是避免 SAC 消耗主要迭代資源，並把時間投入 SL label、allocator、risk gate 的快速循環。

## 2. Default Iteration Budget

| Track | Default budget | Escalation rule |
|---|---:|---|
| SL h10 | Full walk-forward acceptable | 主線；集中救 h10。h5 已淘汰，不再優先多 seed。 |
| SAC smoke | 5K / 20K / 50K steps | 只用來淘汰 reward/action 設計。 |
| SAC ablation | 300K only when justified | 必須先通過 cash collapse、MDD、concentration checks。 |
| PPO | Minimal baseline | 不投入主要調參。 |

## 3. SAC Stop Rules

停止 SAC 方向，不進完整多 seed 訓練：

- 大多數 period 全現金或 0 報酬。
- 弱期 MDD 明顯高於 gate budget。
- 績效集中在單一強期。
- 單股/少數持股過度集中。
- Turnover 在成本後不可交易。

## 4. Efficiency Work That Still Matters

| Area | Keep? | Reason |
|---|---|---|
| Replay buffer memory controls | Yes | 讓 SAC smoke/ablation 不因 RAM 中斷。 |
| Numba env kernels | Yes | 降低 RL smoke 成本。 |
| Multi-worker promotion matrix | No default | 不再預設 SAC 多 seed 大矩陣。 |
| SL scoring cache | Yes | 支援主線快速迭代。 |
| Report/gate automation | Yes | 避免人工挑單一好結果。 |

## 5. Recommended Commands

```powershell
# Mainline SL iteration
.\env\Scripts\python.exe -m sl_pipeline.walk_forward_sl --allocator rule --gate --seed 42
.\env\Scripts\python.exe experiment_report.py

# SAC research-only smoke
.\env\Scripts\python.exe walk_forward.py --tier smoke --algo sac --cash-mode enabled
```

## 6. Superseded Items

Older acceleration notes about making SAC promotion matrices faster are retained only as implementation context. They are superseded by the 2026-06-14 SL-first strategy.
