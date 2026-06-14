> Superseded by 2026-06-14 SL-first strategy.
> This document is retained as historical context only. It must not be treated as an active implementation queue unless explicitly updated after 2026-06-14.
# 研究戰略 v3 — RL 重構（對齊 Gate）

> **日期**：2026-06-11 · **文件精簡**：2026-06-11 cleanup
> **決策**：RL rebuild · SAC-R 封存 · R7/R7b/R8/R9 已砍

## 閱讀路徑（審核 agent / 新進）

```text
1. 本文件（路線圖 + queue）
2. .research/research_state.json（機器狀態）
3. docs/RESEARCH_PLAYBOOK.md §1（O2 分層：M2 用）
4. docs/ALGORITHM_REVIEW.md §二（MDD 根因）
```

歷史（P8/P10/SAC-R/v2 排程）：`.research/archive/` (`../.research/archive/README.md`) · `docs/archive/SAC_BUFFER_PLAN_v2.md` (`archive/SAC_BUFFER_PLAN_v2.md`)

---

## 0. 為何重畫

v2 把 R6 Gate 降級為「僅參考」，同時平行推 SAC-Classic 工程（R7b）與 SAC-R（R-S2b）。
實證顯示：

- R6（env r4）三 seed **個別** MDD 皆 > 35%（37.7% / 46.0% / 38.2%）
- Reward 優化日內 proxy，Gate 檢查 OOS worst-case MDD — **目標錯位**（見 `ALGORITHM_REVIEW.md`）
- R7b gradient_steps、SAC-R LSTM 對 MDD **無直接槓桿**

**結論**：在現行 MDP 上繼續堆工程，不太可能「自然就過 Gate」。
v3 北極星：**用一輪結構性 RL 重構（r5）對準 Drawdown Gate**，其餘實驗封存。

---

## 1. 北極星與停損

| 項目 | 定義 |
|------|------|
| **主目標** | SAC enabled · cash=enabled · 3 seeds · worst-case MDD ≤ 35% |
| **次目標** | Sortino mean ≥ 0.8（維持現有水準） |
| **停損** | r5 smoke（1 seed）若 2025H1 MDD ≥ r4 同 seed → **放棄該變更**，不進 candidate |
| **停損（全局）** | candidate（2 seeds）若 worst MDD 仍 > 38% → **暫停 RL 上線路徑**，轉評估 SL hybrid |

### 1.1 SAC 工程：砍 vs 留

| 類別 | 項目 | 處置 | 理由 |
|------|------|------|------|
| **砍** | R7b `sac_gradient_ablation.py` | 已刪 | 只測 fps / gradient_steps，與 MDD 無關 |
| **砍** | `SAC_GRADIENT_STEPS` env | 已移除 | R7b 專用；SAC 固定 `gradient_steps=1` |
| **砍** | R7 WF（證明 P8→MDD） | 取消 | P8 未實測 MDD 改善；併入 r5 smoke 即可 |
| **砍** | R8 SL-only obs | 取消 | 吞吐實驗，非 MDD 槓桿 |
| **砍** | R9 PER | 取消 | 需 fork SAC.train；MDD 證據不足 |
| **砍** | SAC-R（R-S2b+） | 封存 | 新 MDP，短期對 Gate 無幫助 |
| **砍** | P10 VecEnv 合併 | 封存 | 驗收 FAIL + MDD 惡化 |
| **砍** | `SAC_BUFFER_PLAN.md` v2 排程 | 封存 | 由本文件取代 |
| **留** | P8 IndexedReplayBuffer | 維持 | 非 MDD 槓桿，但 **300K 訓練可行**（Windows RAM） |
| **留** | P7 env NumPy | 維持 | WF 可跑；語意不變 |
| **留** | `SAC_BUFFER_RAM_GB` | 維持 | 僅 R6 重現（legacy 2,805 cap） |

**原則**：v3 只允許 **直接對準 MDD / POMDP / 集中度** 的改動進 queue；其餘 SAC 工程一律砍或封存。

---

## 2. 封存項目（不佔 queue / train_slot）

| 項目 | 處置 | 資產保留 |
|------|------|----------|
| **SAC-R** | 封存 | `cp-sac-r` worktree、`.research/archive/handoffs/SAC-R.json`、R-S0~S2 測試 |
| **R7b** | 已砍 | 腳本已刪；`SAC_GRADIENT_STEPS` 已移除 |
| **R7** WF 重訓 | 已取消 | 不獨立開 R7；P8 效果在 r5 smoke 一併觀察 |
| **P10 VecEnv** | 封存 | `cp-p10-ppo` worktree |
| **R8/R9** | 已取消 | 不進 v3 queue；M3 再評估 PER / 兩階段 SL |

---

## 3. r5 重構範圍（按優先序）

### Phase M1 — 對齊 Gate 的診斷（1–2 天，無 WF）

1. **觀測補全（POMDP）**
   - 將驅動 reward 的狀態納入 obs：rolling vol、rolling Sortino proxy、當前 drawdown 深度
   - 驗收：單元測試 obs shape + 與 reward 計算一致

2. **Reward r5 草案**（`ENV_CONFIG_VERSION` → `r5`）— **done**
   - 提高 drawdown / regime 懲罰權重（相對 r4）

3. **Action decode（M1d + M1c）** — **done → r5.1**
   - **M1d**：`softmax_temp` 0.5 → 1.0（`decision_algorithm_review.md` (`../.research/decisions/decision_algorithm_review.md`) P0）
   - **M1c**：top-k entropy floor `MIN_TOP_K_WEIGHT=0.05`（P1）
   - smoke 驗收加：**單檔 top_holdings < 50%**

4. **集中度護欄（進階）** — 若 smoke 仍集中，再評估 soft cap / CVaR（M3）

### Phase M2 — 分層重訓（嚴格 O2）

```text
smoke     → 300K · seed 42 · 看 2025H1 MDD 方向
candidate → 300K · seed 42,43 · worst MDD 趨勢
promotion → 300K · seed 42,43,44 · 送 Gate
```

```powershell
# M2 smoke（r5 落地後）
.\env\Scripts\python.exe walk_forward.py --tier smoke --algo sac --cash-mode enabled

# 通過後
.\env\Scripts\python.exe walk_forward.py --tier candidate --algo sac --cash-mode enabled
.\env\Scripts\python.exe walk_forward.py --tier promotion --algo sac --cash-mode enabled
.\env\Scripts\python.exe experiment_report.py
```

**規則**：smoke 不過 → 不跑 candidate；**禁止**跳過 tier 直接 promotion。

### Phase M3 — 若 r5 仍失敗（預案，先不執行）

按 `ALGORITHM_REVIEW.md` 中期建議，依序評估：

1. 固定長 episode（126 日）+ 隨機起點
2. 環境 T+1 執行延遲
3. 兩階段：SL 打分 → RL 只做 top-k 權重（S5 整合）
4. CVaR 約束取代純 reward shaping

---

## 4. 架構不變部分

```text
GNN extractor + 視窗 obs + Top-K decode + IndexedReplayBuffer (P8)
算法：SAC · cash=enabled · 300K timesteps
驗證：Walk-Forward · experiment_report · Promotion Gate 8 項
```

**不變**：PPO 預設、VecEnv、SAC-R、PER、SL-only obs（除非 M3 解凍）。

---

## 5. Queue（v3）

| ID | 狀態 | 內容 |
|----|------|------|
| P8 | done | 已 merge，作為 r5 基礎設施 |
| **M1a** | done | obs POMDP（NUM_ACCOUNT_FEATURES 9） |
| **M1b** | done | reward r5 |
| **M1d** | done | softmax_temp 1.0 |
| **M1c** | done | top-k entropy floor 5% |
| **M2-smoke** | cancelled | 中止，轉 S5 Option B |
| **S5 Integration** | **done** | `--sl-scores-dir` CLI + `build_sl_feature_arrays` in `build_train_env`/`build_eval_env`; 短測 5K 步驗證通過 |
| **S5 Full 300K WF** | pending | 等待執行：`walk_forward.py --timesteps 300000 --sl-scores-dir results_dir` |
| **M2-candidate** | blocked | blocked_by S5 Full WF 結果（評估是否需要 candidate 對比） |
| **M2-promotion** | blocked | blocked_by M2-candidate |
| SAC-R | frozen | 封存，不進 active queue |
| R7b | cancelled | 已停止 |

`train_slot`：M1 階段 **free**（無背景 WF）；M2 每次僅一個 tier 訓練。

---

## 6. 成功 / 失敗定義

| 結果 | 動作 |
|------|------|
| M2-promotion worst MDD ≤ 35% | 更新 baselines → Gate retry → 進 live 準備清單 |
| M2-candidate worst MDD 35–38% | 再一輪 M1 調參（單變因），或進 M3-1 episode |
| M2-smoke MDD ≥ r4 | 回滾該變更；換下一個 M1 假設 |
| M2-smoke 單檔 top_holdings ≥ 50% | 視為 decode 失敗；不進 candidate |
| M2-promotion worst MDD > 38% | **RL 上線路徑暫停**；主線改 SL + 風控層（見 `專案總覽.md` §4.4） |

---

## 7. 文件對照

| 文件 | v3 角色 |
|------|---------|
| **本文件** | 唯一活躍路線圖 |
| `.research/reviews/V3-STRATEGY-REVIEW-BRIEF.md` | **外部審核 agent 入口** |
| `.research/EXTERNAL_AGENT_BRIEF.md` | 外部 agent 設定 |
| `SAC_BUFFER_PLAN.md` | P8 規格（排程已封存） |
| `RESEARCH_PLAYBOOK.md` | 分層協議 O2（M2 強制遵守） |
| `ALGORITHM_REVIEW.md` | M1/M3 技術依據 |
| `.research/research_state.json` | 機器可讀 queue |
| `.research/decisions/decision_algorithm_review.md` | action decode 修訂依據 |

---

## 8. 修訂紀錄

| 日期 | 變更 |
|------|------|
| 2026-06-11 | v3：RL rebuild；SAC-R freeze；R7b stop；queue → M1/M2 |
| 2026-06-11 | v3.1：砍 R7b 腳本、SAC_GRADIENT_STEPS、R8/R9；封存 SAC_BUFFER_PLAN 排程 |
| 2026-06-11 | v3.2：全 repo 文件對齊 v3；外部審核 brief `V3-STRATEGY-REVIEW-BRIEF.md` |
| 2026-06-11 | v3.3：M2-smoke 中止；M1d+M1c action decode；`ENV_CONFIG_VERSION=r5.1` |
| 2026-06-11 | v3.4：S5 Option B 整合完成；`--sl-scores-dir` CLI；`build_sl_feature_arrays` 接通 `TaiwanStockEnv`；短測 5K 步驗證通過 |
