# Phase 2 計畫：研究營運效率與維護成本優化

> **封存文件** · 狀態以 [`../../專案總覽.md`](../../專案總覽.md) 為準  
> **前提**：P0–P6 已完成（見 [`重構計畫_P0-P6.md`](重構計畫_P0-P6.md)）  
> **動機**：訓練耗時大（完整矩陣 ≈ 18M timesteps）、reward 變更後難以比較、markdown 分散

---

## Summary

Phase 2 **不開新 repo、不重拆模組**。目標是降低每次研究迭代的時間與認知成本，讓 R4/R6 這類 reward 調整可以**安全比較、精準重訓**，而不是覆寫後無法對照。

排序依據：**實驗可比性 > 訓練成本 > 文件收斂 > 遺留清理**

---

## 背景：為什麼需要 Phase 2


| 痛點    | 現象                                          | 影響                            |
| ----- | ------------------------------------------- | ----------------------------- |
| 無實驗版本 | R4 重訓 `--overwrite` 蓋掉 R3 metrics           | Promotion Gate 混用不同 reward 世代 |
| 矩陣過大  | PPO/SAC × cash × 3 seeds × 4 periods × 300K | 單次改參數成本 ≈ 數天 GPU/CPU          |
| 文件分散  | 11 份 markdown，部分矛盾                          | 每次改動要同步多份                     |
| 遺留程式  | `train.py`、`files/` 副本                      | 入口模糊、維護負擔                     |


P0–P6 解決了**程式結構**；Phase 2 解決**研究營運**。

---

## Priority Order

### O1 — 實驗版本與 artifact 治理 ✅ 已實作

**目的**：reward / env 變更後，metrics 可並排比較，report 不混用舊世代。

**實作**：


| 元件                     | 變更                                                              |
| ---------------------- | --------------------------------------------------------------- |
| `env_config.py`        | `ENV_CONFIG_VERSION`（人工標籤，目前 `r4`）+ 8 字元 `hash`                 |
| `trading_env.py`       | `REWARD_REF_DD` / `REGIME_`* 提升為模組常數                            |
| `research_pipeline.py` | `build_seed_metrics()` 寫入 `env_config` / `env_config_hash`      |
| `experiment_report.py` | 預設 `--current-env-only`：只讀當前 hash；legacy 無 tag 的檔案排除並警告         |
| `settings.py`          | `RESEARCH_ENV_CONFIG_HASH` / `RESEARCH_ENV_CONFIG_VERSION` 環境變數 |


**metrics JSON 新增欄位**：

```json
{
  "env_config_version": "r4",
  "env_config_hash": "a1b2c3d4",
  "env_config": { "...": "..." }
}
```

**使用方式**：

```bash
# 預設：只報告當前 env（r4）的 metrics；排除無 tag 的舊檔
python experiment_report.py

# 明確指定版本（R3 vs R4 對照）
python experiment_report.py --env-config-version r3
python experiment_report.py --env-config-hash abcdef12

# 包含所有世代（不建議用於 Promotion Gate）
python experiment_report.py --include-all-env-configs
```

**版本升級規則**：改 reward / regime / topk 等影響訓練語意的參數時：

1. 更新 `env_config.ENV_CONFIG_VERSION`（例如 `r4` → `r5`）
2. 重訓必要組合（見 O2 分層協議）
3. `experiment_report.py` 自動只讀新版本

**驗收**：

```bash
python -m unittest tests.test_env_config tests.test_experiment_report -v
```

---

### O2 — 分層訓練協議 ✅ 已實作

**目的**：砍掉 80% 無效 full-matrix 重訓。


| 層級        | timesteps | seeds | 通過條件             | 用途                |
| --------- | --------- | ----- | ---------------- | ----------------- |
| Smoke     | 30K       | 1     | 能跑完、MDD 方向不惡化    | reward / env 改動初篩 |
| Candidate | 150K      | 2     | 2025H1 MDD 改善趨勢  | 確認是否值得 full run   |
| Promotion | 300K      | 3     | 通過 Drawdown Gate | 僅最佳候選 + 1 對照      |


**規則**：Smoke 不過 → 不進 Candidate；Candidate 方向錯 → 不進 Promotion。

**實作**：


| 元件                       | 變更                                                                                                   |
| ------------------------ | ---------------------------------------------------------------------------------------------------- |
| `settings.py`            | `TIER_PRESETS` + `resolve_tier()`；`ResearchSettings.research_tier`（`RESEARCH_TIER` 環境變數，預設空＝opt-out） |
| `walk_forward.py`        | CLI `--tier smoke|candidate|promotion`，覆寫 `--timesteps` / `--seeds`                                  |
| `tests/test_settings.py` | `TrainingTierTests` 覆蓋 tier 對應、case-insensitive、未知 tier fail-fast                                    |


`seeds` 取自 `--seeds` 前 N 個。不帶 `--tier` 時行為與舊版相同。用法見 `docs/RESEARCH_PLAYBOOK.md`。

---

### O3 — 縮小預設研究矩陣 ✅ 已實作

**目的**：預設不全跑 PPO/SAC × cash on/off。

**實作**：


| 元件                     | 變更                                                                                        |
| ---------------------- | ----------------------------------------------------------------------------------------- |
| `walk_forward.py`      | `CANDIDATE_PAIRS=[(sac,enabled),(ppo,disabled)]` + `run_candidate_set()` + `--candidates` |
| `walk_forward.py`      | `--matrix` help 標註成本（≈18M steps）；`--overnight-feature-path` 預設改 None（base 為預設）            |
| `experiment_report.py` | 排序 base-first（with_features 永不超越 base）；report 拆「Main Ranking」/「Risk Overlay」兩表            |
| tests                  | `CandidateSetTests` + base 永遠優先於 with_features 的排名測試                                      |


- 預設候選：**SAC enabled**（#1）+ **PPO disabled**（對照），以 `--candidates` 觸發。
- `--matrix` 為 opt-in，CLI help 標註成本。
- `with_features` 為獨立 risk-overlay 線，不進主 ranking、不作 promotion 候選。

---

### O4 — 文件收斂 ✅ 已實作（核心三份）

**目標結構**：

```
docs/
  ARCHITECTURE.md       ← 從 教學文件.md 精簡 ✅ 已建立
  RESEARCH_PLAYBOOK.md  ← 訓練分層 + gate + CLI + O6 overlay ✅ 已建立
  LIVE_OPS.md           ← daily_trade_runner + guard + RPA ✅ 已建立

重構計畫.md              → 封存（P0–P6 完成）
重構計畫_Phase2.md       → 本文件（進行中）
重構計畫_現況評估.md      → 持續更新研究狀態
模型算法評估.md          → 算法合理性評估 ✅ 已建立
experiment_report.md     → 自動產出
```

**選做（未阻擋）**：合併 `VERIFICATION_SUMMARY.md`、`RL_Trading_DQN_PPO_Guide.md` 過時段落；
`教學文件.md` 保留為權威逐模組教學來源。

---

### O5 — 遺留程式清理 ✅ 已實作


| 動作            | 檔案                                  | 狀態                                                                      |
| ------------- | ----------------------------------- | ----------------------------------------------------------------------- |
| 移入 `archive/` | `train.py`, `evaluate.py`, `files/` | ✅ 已封存                                                                   |
| 收斂 import     | `capital_flow_analysis/` 根目錄重複模組    | ✅ 已是 backward-compat shim（re-export `src.data_pipeline.`*），保留供測試 import |
| 移入 `scripts/` | 13 個診斷/臨時腳本                         | ✅ 已搬移                                                                   |


**搬入 `scripts/` 的檔案**：`analyze_gap.py`、`analyze_model_data.py`、`error_analysis.py`、
`friction_analysis.py`、`sector_analysis.py`、`shap_analysis.py`、`model_test_report.py`、
`verify_models.py`、`validation_p4_p5.py`、`test_load_from_zip.py`、`test_margin_short.py`、
`test_models.py`、`test_single_model.py`。

**配套處理**：

- 引用 first-party 模組的腳本加上 `sys.path` bootstrap（`Path(__file__).resolve().parent.parent`），
以 `ROOT_DIR.parent.parent` 修正 repo 根推導；一律從 repo 根執行（見 `scripts/README.md`）。
- `pyproject.toml`：`ruff` 與 `mypy` 排除 `archive/`、`scripts/`（避免重複模組名與工具雜訊）。
- 保留在根目錄：`auto_login.py`（live RPA 登入，讀取根目錄 `.env`）、`optuna_tune.py`（研究調參入口）。

**驗收**：`scripts/` 全數 `compileall` 通過；`scripts/test_margin_short.py` 由 repo 根執行成功；
`tests/` 123 tests OK（不受影響）。低風險，與 O2 並行完成。

---

### O6 — Risk Overlay 取代 Feature Ablation 主線 ✅ 已文件化 + 落地

- RL 主線維持 base features（O3 已將 `--overnight-feature-path` 預設改 None）
- `trade_guard.py` / macro guard 作 risk overlay
- overnight features 不進預設 observation（R5 決策）
- `experiment_report.py` 將 with_features 拆為獨立 Risk Overlay 表，不進主排名
- 完整決策與 ablation 依據見 `docs/RESEARCH_PLAYBOOK.md` 第 7 節

---

### SL — 監督式學習混合架構（方案 B，設計拍板）

**文件**：`docs/SUPERVISED_LEARNING_PLAN.md` · **打勾總表**：[`專案總覽.md`](專案總覽.md) §4.4

**摘要**：`SignalGenerator`（LightGBM 5d 截面超額）+ `PortfolioAllocator` 抽象；第一版 `RuleBasedAllocator`（Top-5、反波動、18% vol-target、10%/15% 階梯 MDD、換手遲滯），預留 `RLAllocator`。與 RL 共用 walk-forward / Gate，**不動 live path**。


| 階段  | 內容                                           | 狀態        |
| --- | -------------------------------------------- | --------- |
| S1  | `sl_pipeline/` 骨架 + labels + SignalGenerator | ✅         |
| S2  | RuleBasedAllocator + backtest                | ✅         |
| S3  | `walk_forward_sl` CLI + metrics + Gate 整合   | ✅ 程式      |
| S3 執行 | 全 4 期 OOS 實跑                               | ⬜ 待研究執行  |
| S4  | `experiment_report` SL 區塊 + RL 對照          | ✅         |
| S5  | RLAllocator spike（`rl_spike` / `sl_features`） | ✅ spike   |
| S5 整合 | 分數併入 `TaiwanStockEnv` 正式訓練               | ⬜ 第二階段  |


---

## 與 R 系列研究的關係


| 項目                | Phase 2                                    | R 系列                 |
| ----------------- | ------------------------------------------ | -------------------- |
| R4 reward 調整      | O1 讓 R4/R3 可分離比較                           | ✅ 程式完成               |
| R6 重訓驗證           | O1 自動標記 `env_config_version=r4`；O2/O3 降低成本 | ⏳ 待執行（env 已修復為 3.12） |
| R5 暫停 features 預設 | 對齊 O6（已落地：base 為預設、overlay 拆分）             | ✅                    |
| Drawdown Gate     | O2/O3 減少重試成本                               | 待 R6 結果              |


---

## 建議執行順序

```text
程式/文件項目（全部完成）
  O1 ✅ 實驗版本治理
  O2 ✅ 分層訓練 CLI + settings
  O3 ✅ 候選集 + with_features overlay 拆分
  O4 ✅ docs/ARCHITECTURE / RESEARCH_PLAYBOOK / LIVE_OPS
  O5 ✅ archive + scripts/ 清理
  O6 ✅ risk overlay 決策文件化 + 落地

唯一剩餘（研究執行）
  R6 walk-forward 重訓（建議：.\env\Scripts\python.exe walk_forward.py --candidates --tier promotion）
     產出帶 r4 tag 的 metrics
  R6 完成 → .\env\Scripts\python.exe experiment_report.py（自動只讀 r4）→ 檢查 Drawdown Gate
```

> 環境（2026-06-09 已修復）：`env/` 重建為 **Python 3.12.10** venv（`--system-site-packages` 共用既有
> torch/sb3 等重量級套件，另補裝 ruff/pytest/mypy/FinMind 等）。
> 驗證：`pytest` **128 passed**、`ruff check .` **All checks passed**。
>
> `cmoney_rpa.py` lint 清理（2026-06-09）：移除 dead `datetime` import；P6 backward-compat re-export
> 統一移至檔首並以 `__all__` 標示（消除 F401/E402）；移除方法內冗餘 re-import（消除 F811）。
> 行為不變，P6 re-export 合約由 `test_p6_refactor` 覆蓋。

---

## Test Plan

```bash
# O1 驗收
python -m unittest tests.test_env_config tests.test_experiment_report -v

# 全量回歸
python -m unittest discover -s tests -p "test_*.py"

# R6 後報告
python experiment_report.py
# 確認 report 標頭含 env_config_version=r4, hash=xxxxxxxx
```

---

## Assumptions

- 不改 `daily_trade_runner.py` 生產路徑行為
- 不改 `results_dir` 檔名格式（版本資訊在 JSON 內，避免破壞既有 resume）
- `ENV_CONFIG_VERSION` 由開發者手動 bump，hash 自動計算
- Legacy metrics 無 tag 時，report  fallback 並警告（向後相容）

