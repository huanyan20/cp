# Research Playbook

> **狀態**：v3 進行中 · **路線圖**：[`RESEARCH_STRATEGY_V3.md`](RESEARCH_STRATEGY_V3.md) · **入口**：[`../專案總覽.md`](../專案總覽.md)  
> **目的**：r5 改動 → O2 分層重訓（smoke/candidate/promotion）→ Gate。R7/R7b/R8/R9 已砍。

---

## 1. 分層訓練協議（O2）

砍掉無效的 full-matrix 重訓。每一層通過後才進下一層。

| Tier | timesteps | seeds | 通過條件 | 用途 |
|------|-----------|-------|----------|------|
| `smoke` | 300K | 1 | 能跑完、MDD 方向不惡化 | reward / env 改動初篩 |
| `candidate` | 300K | 2 | 2025H1 MDD 改善趨勢 | 確認是否值得 full run |
| `promotion` | 300K | 3 | 通過 Drawdown Gate | 僅最佳候選 + 1 對照 |

**規則**：

- Smoke 不過 → 不進 Candidate
- Candidate 方向錯 → 不進 Promotion
- Promotion 只跑最佳候選與 1 個對照，不跑全矩陣

`seeds` 取自 `--seeds`（預設 `42,43,44`）的前 N 個。tier 定義集中於 `settings.TIER_PRESETS`。

---

## 2. CLI 用法

```bash
# Smoke：改 reward 後健全性檢查（300K, seed 42）
python walk_forward.py --tier smoke --algo sac --cash-mode enabled

# Candidate：確認改善趨勢（300K, seed 42,43）
python walk_forward.py --tier candidate --algo sac --cash-mode enabled

# Promotion：最佳候選 + 對照（300K, seed 42,43,44）
python walk_forward.py --tier promotion --algo sac --cash-mode enabled
python walk_forward.py --tier promotion --algo ppo --cash-mode disabled

# 不帶 --tier 時沿用 walk_forward_timesteps 預設（300K）
python walk_forward.py --seeds 42,43,44
```

`--tier` 會覆寫 `--timesteps` 與 `--seeds`。也可用環境變數 `RESEARCH_TIER=smoke` 設預設。

### 預設候選集 vs 全矩陣（O3）

```bash
# 候選集（推薦）：只訓練 SAC enabled + PPO disabled（歷史前二），非全矩陣
python walk_forward.py --candidates --tier promotion

# 全矩陣（opt-in，昂貴）：PPO/SAC x cash on/off x seeds x 4 periods（~18M steps@300K x3）
python walk_forward.py --matrix --tier promotion
```

overnight features 為 opt-in（預設 base）。要產生 risk-overlay 對照才加 `--overnight-feature-path <csv>`。

---

## 3. 實驗版本治理（O1，依賴）

改 reward / regime / topk 等影響訓練語意的參數時：

1. 更新 `env_config.ENV_CONFIG_VERSION`（例如 `r4` → `r5`）
2. 依分層協議重訓必要組合
3. metrics JSON 自動帶 `env_config_version` / `env_config_hash`
4. `experiment_report.py` 預設 `--current-env-only`，只讀當前版本，不混用舊世代

```bash
python experiment_report.py                     # 只讀當前 env
python experiment_report.py --env-config-version r3   # R3/R4 對照
python experiment_report.py --include-all-env-configs # 全世代（不建議用於 Gate）
```

---

## 4. 完整研究迭代流程

```text
改 reward / env（bump ENV_CONFIG_VERSION）
  ↓
smoke（300K/1 seed）      ── 不過則回頭改參數
  ↓ 方向正確
candidate（300K/2 seeds） ── 趨勢錯則放棄此方向
  ↓ MDD 改善趨勢
promotion（300K/3 seeds，最佳候選 + 對照）

（已拍板：timesteps 全 tier 統一 300K，tier 僅差 seed 數）
  ↓
python experiment_report.py   （自動只讀當前 env）
  ↓
Promotion Gate（8 checks）
  ↓ APPROVED 才考慮上線
```

---

## 5. Promotion Gate（驗收標準）

最佳候選需同時滿足（門檻見 `settings.ResearchSettings`，可由環境變數覆寫）：

| Gate | 預設門檻 |
|------|----------|
| Sortino 穩定性 | ≥ 0.8（跨 ≥ 3 seeds） |
| Max Drawdown | worst-case ≤ 35% |
| Turnover | ≤ 0.10 |
| Cash behavior | cash-enabled 須有實際調整 |
| Baseline / Ablation / Stress / Period consistency | 見 `promotion_gate.py` |

目前狀態：`BLOCKED`，卡 Drawdown Gate（worst-case 38.71%）與 Ablation (Features)。詳見 [`../專案總覽.md`](../專案總覽.md) §5。

---

## 6. 測試與驗收

```bash
# O1 / O2 單元測試
python -m unittest tests.test_env_config tests.test_settings tests.test_experiment_report -v

# 全量回歸
python -m unittest discover -s tests -p "test_*.py"

# 報告（確認標頭含 env_config_version / hash）
python experiment_report.py
```

---

## 7. Risk Overlay 策略（O6 / R5 決策）

**決策**：RL 主線維持 base features；隔夜/宏觀特徵作為「風險疊加層」，不進預設 observation。

依據（ablation，見 [`../專案總覽.md`](../專案總覽.md) §5）：

| 指標 | With Features | Without Features |
|------|--------------:|-----------------:|
| Sortino | 0.88 | 2.20 |
| Total Return | 107.98% | 159.26% |
| Max Drawdown | 25.46% | 36.12% |
| Turnover | 0.45% | 3.15% |

overnight features 改善回撤與換手，但明顯犧牲 Sortino 與總報酬 → 只能視為**風險抑制訊號**，
不能當作提升 alpha 的預設輸入。

**落地方式**：

1. RL 訓練/評估預設 base features（`walk_forward.py --overnight-feature-path` 預設 None）。
2. 宏觀風險改由 overlay 承接：盤前 `preopen_macro_check.py`（WARN 減半 / CRITICAL 跳過買單）
   與 `trade_guard.py` 風險限額，而非塞進 state。
3. `experiment_report.py` 將 `with_features` 拆為獨立「Risk Overlay」表，**不進主排名、不作 promotion 候選**。
4. 要重新評估特徵時，以 opt-in 方式重訓：`walk_forward.py --overnight-feature-path <csv>`，
   產出 `*_with_features` metrics，於 overlay 區比較。

## 相關文件

- [`../專案總覽.md`](../專案總覽.md)：計畫打勾與 Gate 狀態
- [`ALGORITHM_REVIEW.md`](ALGORITHM_REVIEW.md)：算法合理性評估
- [`archive/重構計畫_Phase2.md`](archive/重構計畫_Phase2.md)：O1–O6 歷史實作細節（封存）
- [`../教學文件.md`](../教學文件.md)：系統架構與模組說明
