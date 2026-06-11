# SAC 基礎設施與研究線計畫（v2 · 已封存）

> **狀態**：**已封存**（2026-06-11）— 活躍路線圖改為 [`../RESEARCH_STRATEGY_V3.md`](../RESEARCH_STRATEGY_V3.md)  
> **保留用途**：P8 IndexedReplayBuffer 實作規格與歷史決策紀錄；**不再**作為排程或研究 queue 來源
> **範圍**：`indexed_replay_buffer.py`、`train_portfolio.py`、`trading_env.py`、未來 `sac_r/`
> **對應**：[`專案總覽.md`](../專案總覽.md) · [`RESEARCH_LOOP.md`](RESEARCH_LOOP.md) · [`ALGORITHM_REVIEW.md`](ALGORITHM_REVIEW.md)

---

## 0. 戰略轉向（2026-06-11）

**舊模式**：P8→R7 重訓 → 與 R6 metrics 比 ΔMDD → 決定 R8/R9。  
**新模式**：先把 **效率、驗證、維護** 做對；算法實驗各自有基線，**不綁 R6 Gate**。

| 項目 | 舊 | 新 |
|------|----|----|
| 工程合併條件 | R7 MDD 優於 R6 | 測試 + ablation JSON + review memo |
| R6 metrics | 唯一對照 | **歷史快照**（`.research/baselines/r6/`），僅供參考 |
| Recurrent SAC | ❌ 否決（破壞可比性） | ✅ **獨立研究線 SAC-R**（刻意不可與 R6 比） |
| Promotion Gate | 擋 live | **仍擋 live**；研究線各自累積 evidence 後再送 Gate |

```text
                    ┌─────────────────────────────────────┐
                    │  三支柱（持續，不等待 R7）              │
                    │  效率 · 驗證 · 維護                   │
                    └─────────────────┬───────────────────┘
                                      │
              ┌───────────────────────┴───────────────────────┐
              ▼                                               ▼
   ┌──────────────────────┐                    ┌──────────────────────┐
   │  Line A: SAC-Classic │                    │  Line B: SAC-R       │
   │  現行 GNN + 視窗 obs   │                    │  Recurrent / 日截面   │
   │  IndexedReplayBuffer  │                    │  新 MDP · 新基線      │
   └──────────────────────┘                    └──────────────────────┘
```

---

## 1. 三支柱（根本改動，持續進行）

### 1.1 效率（Efficiency）

| 項 | 狀態 | 說明 |
|----|------|------|
| P7 env NumPy 化 | ✅ | `_market_data [T,N,F]`，rollout 25× |
| P8 IndexedReplayBuffer | ✅ merged | 374KB→~0.7KB/transition；300K 全歷史 |
| float16 account 儲存 | ✅ | sample 時 cast float32；RAM ~0.2GB @ 300K |
| 向量化 `_reconstruct_obs` | ✅ | batch fancy-index，取代 Python loop |
| R7b `SAC_GRADIENT_STEPS` | ❌ 已砍（v3.1） | 腳本已刪；SAC 固定 `gradient_steps=1` |
| P10 PPO VecEnv | ✅ 不合併 | 驗收 FAIL；PPO 維持 R6 預設 |
| 待做 | ⬜ | sample 端 batch 重建 profiling；parquet 資料快取 |

**原則**：吞吐與 RAM 優化 **先合併**，不以 OOS MDD 為門檻。

### 1.2 驗證（Validation）

每項工程改動必備：

1. **單元測試** — buffer 等價（loop vs vectorized）、float16 rtol、wrap-around
2. **Smoke** — `1K steps` 可跑通（Windows spawn / CUDA）
3. **Ablation JSON** — `results_dir/*_ablation.json`（fps、elapsed、OOS 摘要）
4. **研究線基線目錄** — 不與 R6 混用：

```text
.research/baselines/
  r6/              # 凍結，只讀參考
  sac_classic/     # Line A 當前 canonical（P8 merge 後 smoke/WF）
  sac_r/           # Line B 專用（R-S0 起建立）
```

5. **Gate** — 僅在 **送 live / promotion 合併** 時跑 `experiment_report.py`；日常工程不阻塞。

### 1.3 維護（Maintainability）

| 規則 | 實作 |
|------|------|
| Buffer 與 SB3 解耦 | `indexed_replay_buffer.py` 單檔；`replay_buffer_class=` 注入 |
| 超參環境變數 | `SAC_BUFFER_RAM_GB`（R6 重現用 only） |
| 不 fork SB3 核心 | PER / Recurrent 用 **新模組或新研究線**，不 patch site-packages |
| 一研究線一分支/worktree | SAC-R → `feat/sac-r-recurrent`（尚未建立） |
| 文件即契約 | 每線記載 obs shape、reward 版本、buffer 語意 |

---

## 2. 問題背景（仍成立，但 R7 不是唯一出口）

SAC off-policy 優勢曾被 ~2,805 transition buffer 壓成準 on-policy。**P8 已從結構上解決**（300K @ ~0.2GB RAM）。

R6 worst MDD 44.41% 是 **歷史觀測**，不是「IndexedReplayBuffer 無效」的判准。  
Line A 後續實驗以 **sac_classic 基線** 為 Δ，不以 R6 為 hard gate。

---

## 3. 研究線 A — SAC-Classic（現行主線）

**不變**：GNN extractor、20 日視窗 obs、env r4 reward、cash=enabled SAC。

### 已完成工程

- P8 IndexedReplayBuffer + review merge
- P10 ablation（VecEnv 不進預設）
- cross_review 雙向 memos

### 進行中 / 可選（v2 · 已廢止 → 見 v3）

| ID | v3 處置 |
|----|---------|
| R7 | **已取消** |
| R7b | **已砍**（腳本已刪） |
| R8 | **已取消** |
| R9 | **已取消** |

活躍 queue：`docs/RESEARCH_STRATEGY_V3.md` M1/M2。

### Line A 驗收（取代「R7 vs R6」）

1. `pytest tests/test_indexed_replay_buffer.py` 全過
2. Ablation / smoke JSON 入 `results_dir/`
3. 若跑 WF：metrics 寫入 `.research/baselines/sac_classic/`
4. cross_review memo 無 **blocker**（nits 可後續修）

---

## 4. 研究線 B — SAC-R（Recurrent SAC，全新）

**刻意與 R6 / Line A 脫鉤**：改 MDP 資訊結構、改網路、改 buffer 抽樣語意。

### 4.1 動機

| 痛點 | Line A（視窗 obs） | SAC-R |
|------|-------------------|-------|
| obs 維度 | 45×2126 ≈ 96K | 目標 45×(F_day+account) ≈ 數百 |
| 視窗冗餘 | IndexedReplayBuffer 壓 RAM，GPU 仍大 forward | 日截面 + RNN 狀態，buffer 存 (s, a, h) |
| SB3 現成 | SAC + custom buffer ✅ | **無 recurrent SAC** → 自研或適配 |

### 4.2 技術路線

```text
R-S0  Spike — ✅ decision: .research/decisions/sac-r-spike.md
R-S1  obs_mode="daily" — ✅ cp-sac-r · tests/test_obs_mode_daily.py (reward 一致)
R-S2  SequenceReplayBuffer + LSTM encoder smoke — ✅ handoff `.research/handoffs/SAC-R.json`
R-S2b LSTM-SAC train loop — blocked by plan review
R-S3  WF + sac_r 基線 — blocked by R-S2b
```

### 4.3 SAC-R 否決條件（仍適用）

- 需 fork SB3 且無測試覆蓋 → 先 spike 再決定
- spike fps 低於 Line A 50% 且 MDD 無改善 → 暫緩 R-S2
- 與 Line A 共用 `train_portfolio.build_model` 未隔離 → **禁止**；用 `sac_r/train.py` 或 worktree

### 4.4 工作區

```powershell
git worktree add ..\cp-sac-r -b feat/sac-r-recurrent
# 規則：不修改 main 的 SAC-Classic 預設；handoff → .research/handoffs/SAC-R.json
```

---

## 5. 提案速查（原 §1 精簡）

| 提案 | Line A | SAC-R | 備註 |
|------|--------|-------|------|
| IndexedReplayBuffer | ✅ 採用 | 可能改 SequenceBuffer | P8 done |
| 兩階段 SL-only obs | R8 | 可合併進 daily obs | 吞吐 ↑ |
| float16 buffer | ✅ account | 同 | 取樣 float32 |
| PER | R9 條件式 | 低優先 | 需 subclass SAC |
| Recurrent | ❌ Line A | ✅ **SAC-R 主軸** | 新研究線 |
| PPO VecEnv | P10 不合併 | — | 品質 FAIL |

---

## 6. 執行排程（v2 · 已廢止）

> **取代**：[`RESEARCH_STRATEGY_V3.md`](RESEARCH_STRATEGY_V3.md) — M1/M2 queue。  
> 以下排程 **勿執行**：R7b 已刪、R7/R8/R9 已砍、SAC-R 已 frozen。

| 優先 | ID | v3 處置 |
|------|-----|---------|
| R7b | gradient_steps | **已砍** |
| R7 | WF sac_classic | **已取消** |
| R-S2b+ | LSTM-SAC | **frozen** |
| R8/R9 | SL-only / PER | **已取消** |

---

## 7. P10 摘要（不變，見原 §4）

PPO VecEnv ablation 已完成；**不 merge 預設**（A1 fps 1.20×、|ΔMDD| 14pp）。  
腳本留 `cp-p10-ppo` worktree 供日後硬體升級重測。

---

## 8. 指令速查

```powershell
# Line A — buffer 測試
.\env\Scripts\python.exe -m pytest tests/test_indexed_replay_buffer.py -q

# v3 M2 smoke（取代 R7/R7b）
.\env\Scripts\python.exe walk_forward.py --tier smoke --algo sac --cash-mode enabled
# 見 docs/RESEARCH_STRATEGY_V3.md

# Line B — R-S0（待建 worktree）
# git worktree add ..\cp-sac-r -b feat/sac-r-recurrent
```

---

## 9. 修訂紀錄

| 日期 | 變更 |
|------|------|
| 2026-06-10 | v1：P8/R7/R8/R9 排程；Recurrent 否決 |
| 2026-06-11 | v2：三支柱；R6 降級為參考；SAC-R 新研究線；R7 可選；P8/float16/向量化已落地 |
| 2026-06-11 | **封存**：v3 取代排程；R7b/R8/R9 已砍；見 `RESEARCH_STRATEGY_V3.md` |
