# 訓練計畫 v4（2026-06-11 重新規劃）

> **[更新 2026-06-11 23:55] 本計畫已完結。**
> 依據決策樹，S5 測試（MDD 41.84%）失敗，但 SL 驗證（seed 43）以 MDD 33.14% 成功通過 35% 門檻（T0b）。
> RL 研發路徑正式暫停，系統全速轉向 SL 實盤（Live Ops）部署。
> 請參閱 `專案總覽.md` 與 `docs/LIVE_OPS.md` 獲取後續實盤準備資訊。

> **前提**：基於系統性全局評估（`docs/RISK_REGISTER.md`）與現行程式碼（r5.1）重新規劃。  
> **硬體**：GTX 1060 3GB VRAM、RAM 有限（SAC buffer 上限 1.5GB）、Windows 單進程。  
> **目前狀態**：Promotion Gate BLOCKED（RL worst MDD 44.41%；SL 38.55% 舊跑 / 33.14% 最新）  
> **核心問題已解決**：`walk_forward.py` 每期結束後未釋放記憶體（OOM at period 3），已加入 `del + gc.collect()`。

---

## 一、現況盤點（重新評估）

| 項目 | 數值 | 來源 | 可信度 |
|------|------|------|--------|
| RL r4 baseline worst MDD | 44.41% | `.research/baselines/` | ✅ 凍結基線 |
| SL 舊跑（2026-06-09）MDD | 38.55% | `experiment_report.md` | ✅ 穩定 |
| SL 最新重跑 MDD | 33.14% | `task.md` 記錄 | ⚠️ 需確認 `metrics_sl_rule_h5_seed42.json` |
| r5.1 改動 | M1a+M1b+M1c+M1d 全部完成 | 程式碼 | ✅ |
| S5 短測（5K 步）| 第 1、2 期正常，第 3 期 OOM | task-510 log | OOM 已修 |
| Obs 維度（含 SL）| 45 × 2132 = 95,940 | 計算確認 | ✅ |

**關鍵發現**：obs_dim_per_stock = 20 × 106 + 9 + 3 = **2132**（106 = 24 base + 7 cross + 75 macro）。  
每個 period 訓練結束後，PyTorch model 和 numpy market_data 未釋放，是 OOM 根本原因。

---

## 二、訓練軌道設計

**原則**：嚴格單軌順序執行（硬體不允許並行）。先驗證再推進，有停損不強行。

```
軌道 T0 → 軌道 T1 → 軌道 T2（視 T1 結果決定是否執行）
```

---

## 軌道 T0：立即確認 SL Gate 狀態（今天，< 5 分鐘）

**目的**：確認最新 SL 跑是否真的 MDD 33.14%（若是，SL 可能已達標）

```powershell
# 重新產生 experiment_report，確認最新 SL metrics
.\env\Scripts\python.exe experiment_report.py
```

**判斷**：
- SL MDD < 35% → **SL Gate PASS！直接進入 T0b 準備推送 SL**
- SL MDD ≥ 35% → 繼續 T1

### T0b（若 SL 通過）：多 seed SL 驗證

```powershell
# 額外 seed 43（SL 只有 1 seed，需要至少在 report 說明）
.\env\Scripts\python.exe -m sl_pipeline.walk_forward_sl --allocator rule --gate --seed 43
```

---

## 軌道 T1：S5 全程 300K Walk-Forward（主線）

**目的**：RL（SAC）以 SL 5d 分數作為 obs 特徵，觀察是否協同降低 MDD

**前提**：OOM 修復已就緒（gc.collect 已加入 walk_forward.py）

```powershell
$env:SAC_BUFFER_RAM_GB="1.5"
.\env\Scripts\python.exe walk_forward.py `
    --timesteps 300000 `
    --algo sac `
    --cash-mode disabled `
    --sl-scores-dir results_dir `
    --overwrite
```

**估計時間**：每期 ≈ 2 小時 × 4 期 = **≈ 8–10 小時**

**停損條件**：
- 任一期訓練中 RAM OOM → 降低 `SAC_BUFFER_RAM_GB` 至 `1.0` 後重跑
- 第 1 期（2024H2）完成後 MDD > 45%（遠劣於 r4）→ 停止，改用 r5.1 pure RL

**成功條件（進入 T2）**：
- 4 期全跑完
- Overall worst-case MDD < 40%（優於 r4 的 44.41%）

---

## 軌道 T2：r5.1 純 RL Smoke（T1 結果為基準的對照組）

**目的**：確認 M1a/M1b/M1c/M1d 本身（無 SL 特徵）對 MDD 的改善幅度

```powershell
$env:SAC_BUFFER_RAM_GB="1.5"
.\env\Scripts\python.exe walk_forward.py `
    --tier smoke `
    --algo sac `
    --cash-mode enabled `
    --overwrite
```

**停損條件**（v3 定義，維持）：
- 2025H1 MDD ≥ 37.7%（r4 seed42）→ 停止，不進 candidate

**決策**：
- smoke MDD < 37.7% → 進 candidate（seeds 42,43）→ 若 worst < 38% 進 promotion
- smoke 失敗 → r5.1 RL 路徑停損，轉全力推 S5 / SL hybrid

---

## 三、決策樹

```
T0: SL Gate check
├── SL MDD < 35% → [SL PASS] 優先 SL 上線路徑
│   └── T0b: seed 43 SL → 若 worst MDD < 35% → Promotion 準備
│   └── 同時跑 T1 S5 → 若 S5 更優 → 以 S5 RL 作為主推薦
└── SL MDD ≥ 35%
    └── T1: S5 全程 300K
        ├── 通過（MDD < 40%）→ 比較 SL vs S5 → 取優者進推廣
        └── 失敗 → T2: r5.1 pure RL smoke
            ├── smoke 通過 → r5.1 candidate → promotion
            └── smoke 失敗 → [暫停 RL 路徑] 聚焦 SL 調優
```

---

## 四、實驗命名規範（防止 results_dir 汙染）

| 實驗 | metrics 檔前綴 | 說明 |
|------|----------------|------|
| T1 S5 run | `metrics_sac_disabled_s5_wf_seed42` | `--cash-mode disabled --sl-scores-dir` |
| T2 r5.1 smoke | `metrics_sac_enabled_wf_seed42` | 標準名稱（覆蓋 r4，所以需要 `--overwrite`） |
| SL seed43 | `metrics_sl_rule_h5_seed43` | 新 seed |

> ⚠️ **注意**：T2 會覆蓋 r4 的 `metrics_sac_enabled_wf_seed42.json`。確保 r4 已備份在 `.research/baselines/`（已確認）。

---

## 五、監控重點

每期完成後立即確認：

```powershell
# 查看最新期的 MDD
.\env\Scripts\python.exe experiment_report.py 2>&1 | Select-String "MDD|Drawdown|period"
```

**關注**：
1. 任一期的 `max_drawdown` — 若 > 0.40 視為警示
2. `top_holdings` — 單檔 > 50% 視為 decode 失效（M1c 應已解決）
3. `avg_cash_weight` — 若 > 90% 持續出現，LAMBDA_CASH_DEFENSIVE 需調降

---

## 六、若全部軌道失敗的保底方案

> 觸發條件：T0 SL MDD ≥ 35%，T1 S5 worse than SL，T2 r5.1 smoke 失敗

按 `ALGORITHM_REVIEW.md §七` 升級至 M3：

1. **M3-1**：固定 126 日 episode + 隨機起點（降低長 episode credit assignment 問題）
2. **M3-SL 調優**：調整 SL 標籤至 10d、調降 vol-target 至 15%、調整熊市現金機制
3. **M3-4**：CVaR 約束（計算成本最高，最後評估）

---

## 七、版本紀錄

| 日期 | 變更 |
|------|------|
| 2026-06-11 | v4 建立：基於 RISK_REGISTER 全局評估，重新設計三軌道計畫；加入 OOM 根因分析與 obs_dim 計算 |
