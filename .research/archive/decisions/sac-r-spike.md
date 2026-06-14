> Superseded by 2026-06-14 SL-first strategy.
> This document is retained as historical context only. It must not be treated as an active implementation queue unless explicitly updated after 2026-06-14.
# Decision: SAC-R R-S0 Recurrent Spike

> **Line status**（2026-06-11）：SAC-R **frozen** — handoff 保留，不進 v3 active queue。
> **Date**: 2026-06-11
> **Phase**: R-S0
> **Worktree**: `../cp-sac-r` · branch `feat/sac-r-recurrent`
> **Artifact**: `results_dir/sac_r_spike.json` (`../../results_dir/sac_r_spike.json`)

---

## 1. 目的

驗證 Recurrent（LSTM）路線在 Windows + CUDA 上可跑通，並量測相對 Line A 參考实现的 fps 差距。
**不**以 R6 metrics 為判準。

---

## 2. Spike 結果（mini 2-stock env · 1000 steps · seed 42 · GTX 1060）

| Variant | Line | fps | vs PPO+GNN |
|---------|------|-----|------------|
| PPO + GNN（2D 視窗 obs） | sac_classic_ref | **194.5** | 1.00× |
| PPO + Mlp flat | ablation | 287.4 | 1.48× |
| **RecurrentPPO + MlpLstmPolicy** | sac_r_candidate | **89.6** | **0.46×** |
| SAC + Mlp flat | sac_classic_ref | 66.9 | 0.34× |

RecurrentPPO（sb3-contrib）在相同 flat obs 下 **約慢 2.2×** vs PPO+GNN mini env；**可跑通**，無 crash。

---

## 3. 解讀

### 3.1 可行性 ✅

- `sb3-contrib` 已安裝；`RecurrentPPO` + `MlpLstmPolicy` + `DummyVecEnv` + Windows **smoke 通過**。
- LSTM hidden state 由 sb3-contrib 管理；尚未驗證 episode 邊界 burn-in（留 R-S2）。

### 3.2 效能 ⚠️

- Mini env 的 flat obs 僅 32 維，**不能**外推全尺寸 45×2126 視窗。
- 全尺寸若 flatten → LSTM 輸入 ~96K 維，**不可行**。
- **結論**：SAC-R 必須走 **R-S1 `obs_mode=daily`**（日截面 ~500 維）再疊 LSTM，而非 flatten 現行視窗 obs。

### 3.3 Recurrent SAC

- SB3 **仍無** Recurrent SAC；R-S2 需自研 `LstmSacPolicy` + `SequenceReplayBuffer` 或適配 CleanRL 模式。
- RecurrentPPO 僅作 **工程參考**（LSTM + VecEnv 整合），不是最終算法。

---

## 4. 決策

| 項 | 決定 |
|----|------|
| R7 WF 重訓 | **停止**（已停 process）；不追 R6 ΔMDD |
| R-S0 | **完成** — spike JSON + 本 memo |
| R-S1 | **GO** — 實作 `obs_mode=daily`（env 契約 + 測試） |
| R-S2 | **GO（條件）** — daily obs smoke 通過後，SequenceBuffer + LSTM-SAC prototype |
| 合併 main | **禁止** 直至 R-S2 smoke + handoff |

---

## 5. R-S1 驗收草案

1. `TaiwanStockEnv(obs_mode="daily")` — market 段只輸出當日 `[N, F_day]`，account 不變
2. obs 維度 ≤ 600（45×~13 量級）
3. reward / step 邏輯與 full-window **同一套**（只改 obs 組裝）
4. 測試：daily vs window 在同 action 序列下 **reward 逐位一致**

---

## 6. 指令

```powershell
cd C:\Users\ggini\Desktop\cp-sac-r
..\cp\env\Scripts\python.exe scripts\sac_r_spike.py --timesteps 1000
```

---

## 7. Verdict

```
R-S0 PASS → proceed R-S1
```
