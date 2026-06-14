> Horizon update: 2026-06-14 latest SL walk-forward retires h5 from the production path. h10 is the active SL repair target.
> Superseded by 2026-06-14 SL-first strategy.
> This document is retained as historical context only. It must not be treated as an active implementation queue unless explicitly updated after 2026-06-14.
# 整體決策算法合理性評估（完整版）

> **評估日期**：2026-06-11 · **評估者**：Antigravity
> **評估基準**：v3 戰略（r5 RL rebuild）· 硬體限制已知（GTX 1060，~36–52 fps；Windows RAM）
> **評估範圍**：MDP → 特徵提取器 → Action 解碼 → Reward → 訓練超參 → 驗證體系 → 戰略決策

---

## 一、全局評分表

| 面向 | 評分 | 合理性判斷 |
|------|------|-----------|
| MDP 問題建模 | ★★★★☆ | 合理；連續動作投資組合是 RL 正確場景 |
| 特徵提取器（GNN） | ★★★★☆ | 合理；Self-Attention 捕捉跨股依賴 |
| Action 解碼管線 | ★★☆☆☆ | **有結構性問題**；softmax(0.5) + top-k 梯度死區 |
| Reward 設計 | ★★★☆☆ | 方向對，但 proxy 與 Gate 目標錯位 |
| 訓練超參（SAC） | ★★★★☆ | 合理；buffer_size / gradient_steps 決定正確 |
| 驗證體系 | ★★★★★ | 業界高標準；WF + 多 seed + 8 項 Gate |
| v3 戰略轉向 | ★★★★☆ | 根因診斷正確；焦點收斂合理 |
| SL hybrid 預案 | ★★★★☆ | 設計完整；是 RL 失敗時最可行的保底 |

---

## 二、實驗數據彙整（r4 基線）

> 來源：`.research/baselines/metrics_sac_enabled_wf_seed{42,43,44}.json`

### SAC enabled（主線）

| Seed | Overall MDD | Overall Sortino | Avg Cash | 2025H1 MDD | 備註 |
|------|------------|----------------|----------|------------|------|
| 42 | **37.73%** | 1.30 | 30.2% | 37.73% | 2024H2 幾乎全現金 |
| 43 | **46.0%** | 0.54 | 25.0% | 0.22%（全現金） | 2024H2 單股 99%（8046） |
| 44 | **38.2%** | **2.31** | 38.3% | 38.2% | 2024H2 單股 54%（3529）；最佳 Sortino |

- **Worst-case MDD：46.0%（seed43）**，遠超 Gate 門檻 35%
- 三 seed MDD 全部超標，不是邊際失敗
- Seed 間差距極大（Sortino 0.54 vs 2.31），顯示 policy 對初始化高度敏感

### PPO disabled（對照組）

| Seed | Overall MDD | Sortino | 現金行為 | 問題 |
|------|------------|---------|---------|------|
| 42 | **54.96%** | 0.06 | ~0% | 幾乎全倉、高換手 12.9% |
| 43 | **50.36%** | -0.23 | ~0% | 負 Sortino |
| 44（SAC44 作對比）| 38.2% | 2.31 | 38.3% | — |

PPO disabled 的問題最清楚：**avg_cash ≈ 0，cash=disabled，換手 13–21%**，顯示 SAC + cash=enabled 確實對風控有正面作用，但仍不夠。

---

## 三、合理的部分（保留，不動）

### 3.1 MDP 設計 ★★★★☆

**投資組合配置 = 連續動作空間 RL 的正確場景**。

- State：`(45 stocks, 20 days × market_features + 9 account_features)` — 足夠描述局部市場狀態
- Action：`Box(46)` logits → decode → target_weights，讓 agent 直接輸出連續權重，比 rule-based 更靈活
- Termination：一次 episode 跑整段歷史（缺陷詳見 §4.5）

M1a 已將帳戶特徵從 6 → 9（加入 rolling_vol、rolling_sortino_proxy、current_drawdown）。這是正確的 POMDP 修復：reward 的 Sortino 分量依賴 `_return_history`，原本 obs 沒有這個狀態，agent 學習信號不完整。

### 3.2 特徵提取器（GNN）★★★★☆

```python
# gnn_extractor.py 架構
input_norm → node_embedder(Linear→Softsign→Linear→Softsign)
           → MultiheadAttention(embed_dim=64, num_heads=4)
           → residual + LayerNorm
           → output_net(Linear→Softsign)  → features_dim=256
```

**設計亮點**：
- Self-Attention 讓 45 支股票互相交換資訊，適合台股科技股的板塊輪動特性
- Softsign 取代 ReLU：有界 (-1,1)，消除 Dead Neuron，長訓練穩定性更好
- 殘差連接 + LayerNorm：防梯度消失，是 Transformer 標準做法
- input_norm（LayerNorm）對 raw feature 正規化，降低 obs 尺度差異影響

**潛在問題**：embed_dim=64、num_heads=4、net_arch=[256,256] 在 GTX 1060 下是合理的輕量架構，不是瓶頸。

### 3.3 SAC 訓練超參 ★★★★☆

```python
# train_portfolio.py SAC config
learning_rate = 3e-4     # SAC 標準
learning_starts = 1_000  # warm-up 合理
batch_size = 256         # off-policy 標準
tau = 0.005              # soft update，穩定
gamma = 0.99             # 長期視野
train_freq = 10          # 每 10 step 更新一次，降低更新頻率避免過擬合
gradient_steps = 1       # 已正確固定（R7b 已證明無槓桿）
ent_coef = "auto"        # 自動調整熵係數，對探索現金比例有優勢
buffer_size = min(300K, RAM_cap)  # P8 已解決 300K 全歷史
```

`train_freq=10` + `gradient_steps=1` 的組合是保守但穩定的設定，避免 off-policy 在早期過度擬合到少量 transitions。

`ent_coef="auto"` 是 SAC 的關鍵優勢：在 cash=enabled 情境下，agent 需要探索「保持現金」這個行為，自動熵係數有助於在訓練早期保持探索，這在 PPO 的固定 entropy 下更難做到。

### 3.4 驗證體系 ★★★★★

```
Walk-Forward 四期（2024H2 → 2026H1）
× 3 seeds（42 / 43 / 44）
× Promotion Gate 8 項
```

特別好的設計：
- **Worst-case MDD** 而非 mean MDD 作為 Gate 條件，真實反映實盤最壞情境
- 分層協議（O2）：smoke → candidate → promotion，禁止跳層，節省訓練資源
- `experiment_report.py --current-env-only`：不混用不同 reward 版本的 metrics，版本治理清晰

**Promotion Gate 卡住（worst MDD 44.41%）是正確的決策**，不是過度嚴格。

### 3.5 v3 戰略轉向 ★★★★☆

v2 的錯誤：在 reward proxy 與 MDD 目標錯位還未解決前，平行推進 SAC 工程（R7b）和 SAC-R（LSTM），兩者都不能直接降 MDD。

v3 正確識別的根因：

```
MDD 超標根因樹
├── reward proxy 與 OOS worst-case MDD 錯位（最根本）
├── POMDP：agent 看不到驅動 reward 的 rolling 序列（M1a 已修）
└── Action 解碼非線性 → 結構性集中度失控（新診斷，未完全處置）
```

砍掉 R7b、R8（SL-only obs）、R9（PER）、SAC-R 是正確的焦點收斂，這些改動都不能直接改善 OOS worst-case MDD。

---

## 四、有問題的部分（待解）

### 4.1 🔴 Action 解碼管線的結構性非線性（最被低估的問題）

#### 完整 decode 流程（SAC enabled，主要路徑）

```
網路輸出 logits ∈ [-5, 5]^46    ← action_space bounds
         ↓
① shifted = logit - max(logit)   ← 數值穩定
         ↓
② exp(shifted / 0.5)             ← 溫度 0.5 = 梯度放大 2×
         ↓
③ softmax → soft_weights^46      ← 機率分佈，和=1
         ↓
④ split: stock_weights^45 + cash_weight^1
         ↓
⑤ top-k(5) 遮罩                  ← 40 支股票梯度=0（不可微邊界）
         ↓
⑥ re-normalize（stock+cash 和=1） ← 條件性除法
         ↓
最終 target_positions ∈ [0,1]^45 + cash
```

#### 問題 A：temperature=0.5 的數學後果

| temperature | logit 差 1.0 的權重比 | logit 差 3.0 的最大股票占比 |
|-------------|----------------------|----------------------------|
| 2.0（寬鬆）  | e^0.5 ≈ 1.65×        | ~30% |
| 1.0（中性） | e^1 ≈ 2.72×          | ~73% |
| **0.5（現行）** | **e^2 ≈ 7.39×**  | **~99%** |
| 0.1         | e^10 ≈ 22,026×       | ≈ 100% |

**實驗驗證**：seed44 / 2024H2，3529.TWO = 53.8%；seed42 / 2025H1，多檔超過 15%；seed43 / 2024H2，**8046.TW = 98.9%**。

seed43 的 98.9% 集中是 softmax(0.5) 的必然結果：只要有一個 logit 比其他高 3，即使沒有任何 reward 鼓勵，softmax 就會輸出接近全倉的結果。這不是 reward 的問題，是**解碼函數的非線性性質**。

#### 問題 B：top-k 遮罩造成梯度死區

```python
topk_indices = np.argsort(stock_weights)[-5:]
mask = np.zeros(45)
mask[topk_indices] = 1.0
stock_weights = stock_weights * mask  # 40 支股票從此梯度=0
```

每個 step，45 支股票中只有 5 支有有效梯度，其他 40 支的 logit 即使很離譜也不會被修正。這導致：
- 學習效率低：90% 的輸出維度大多數時候沒有訓練訊號
- Policy 的 logit 空間在訓練中大量未探索
- Seed 間差異大（Sortino 0.54 vs 2.31）的部分原因：40 支「沉默」logit 的初始值決定了 top-k 的競爭格局

#### 問題 C：sorting 邊界不可微

top-k 的邊界（第 5 名 vs 第 6 名）是離散操作，無法透過 backprop 平滑。當兩支股票的 logit 接近時，投資組合有跳躍式切換風險，policy gradient 在邊界附近估計不穩定。

#### 對 r5 的影響

即便 M1b 將 LAMBDA_DRAWDOWN 提高到 1.2，只要 softmax_temp 維持 0.5，**reward shaping 無法阻止 network 在某些初始化下輸出 99% 集中的 logit**。r5 的 MDD 改善潛力因此受到 action 解碼的非線性上限制約。

---

### 4.2 🔴 Reward 與 Gate 目標錯位

```python
# 每步 reward（trading_env.py _compute_reward）
hybrid = 0.4 * softsign(log_r * 100)          # 日報酬
       + 0.3 * sortino_proxy * capital_util    # 滾動 Sortino
       + 0.3 * softsign((log_r - bm) * 100)   # 超越基準

penalty = LAMBDA_COST * trade_cost             # 5.0 × 成本
        + LAMBDA_TURNOVER * turnover / 2       # 換手
        + LAMBDA_DRAWDOWN * max(0, dd - 0.02)  # 1.2 × 超過 2% 的回撤
        + regime_penalty                       # 1.5 × 曝險 × (dd - 0.06)

reward = clip(hybrid - penalty + cash_defensive_bonus, -1, 1)
```

**Gate 要求的是**：OOS 3 seeds 的 **worst-case MDD ≤ 35%**

這兩個目標之間有根本落差：
- Agent 每步優化**日複合 proxy**，不優化**多期最壞情境**
- `drawdown_p = 1.2 × max(0, dd - 0.02)` 是**線性懲罰**，在 dd=0.35 時懲罰值 = 1.2×0.33 = 0.396，被 hybrid_reward（最大 1.0）輕易覆蓋
- reward clip 到 [-1, 1] 進一步削弱了回撤懲罰的相對比重

**r4 的失敗說明了什麼**：三個 seed 的 MDD 37.7% / 46.0% / 38.2% 說明，reward shaping 即使加強，也只能「減少發生頻率」，不能「保證上界」。

r5 的改進（LAMBDA_DRAWDOWN 0.8→1.2，REWARD_REF_DD 0.03→0.02）方向正確，但本質仍是 proxy shaping，沒有從數學上解決「保證 MDD ≤ 35%」的問題。

---

### 4.3 🟡 Seed 間不穩定性（穩健性不足）

```
SAC enabled Sortino：0.54（seed43）vs 2.31（seed44）→ 差距 4.3×
SAC enabled MDD：37.7%（seed42）vs 46.0%（seed43）→ 差距 8.3pp
PPO disabled Sortino：-0.23（seed43）vs 0.06（seed42）→ 雙雙不及格
```

這個差距規模暗示：**算法本身的隨機性超過了 policy 真正學到的訊號**。數種可能的根因（不互斥）：

1. **action decode 非線性（§4.1）**：初始 logit 決定哪 5 支股票在 top-k，形成路徑依賴
2. **Long episode 的 credit assignment 問題**：800–1200 步的單一 episode 讓梯度在早期 step 訊號稀疏
3. **Replay buffer 早期 sampling bias**：SAC 在 learning_starts 前不更新，前 1000 transitions 的行為完全隨機，不同 seed 的初始探索路徑差異很大

---

### 4.4 🟡 Cash 機制的現金囤積問題

```
seed43 / 2024H2：avg_cash = 99.5%（2025H1 期間）
seed44 / 2025H2：avg_cash = 99.9%（幾乎全現金，long_exposure = 0.001）
```

這不是合理的「熊市防禦」，而是 policy 在某些 seed 下進入了「全現金省力」的局部最優。Regime penalty 設計為鼓勵在 MDD > 6% 時增加現金，但這個機制可能過強：policy 學到「只要保持大量現金，就可以避免所有回撤懲罰」，導致 Sortino 接近 0（什麼都不賺）但也不虧的退化策略。

LAMBDA_CASH_DEFENSIVE（r5: 0.35）進一步強化了防禦現金的 reward，在某些情境下可能反而惡化「過度囤現金」問題。

---

### 4.5 🟡 Episode 設計：長 Episode 的 Credit Assignment

每個 episode 跨越整段歷史（~945 步），reward 是每日累積。台股 2020–2024 經歷了多個牛熊切換，市場是**非平穩的**。

長 episode 的問題：
- 訓練早期的 transition（牛市 2021 年）的梯度難以正確影響後期（熊市 2022 年）的行為
- Discount factor γ=0.99 下，945 步後的 reward 折現為 0.99^945 ≈ 0.000086，幾乎消失

M3 預案（固定 126 日 episode + 隨機起點）是正確的修復方向。

---

### 4.6 🟡 Sim-to-Real Gap

| 訓練環境假設 | CMoney 實盤現實 |
|------------|----------------|
| 收盤即時調倉 | T+1 賣（今收盤）→ T+2 早盤買 |
| 完全成交 | 滑價 + RPA 延遲 + 部分成交 |
| 同日計報酬 | 隔夜 gap 風險未建模 |
| 每日強制 step | 實盤可跳過 |

此問題在研究階段不影響 Gate 判斷，但上線後實盤 MDD 會高於回測。M3 的 T+1 延遲修復是中期必要項目。

---

## 五、各層次問題與修復成本矩陣

| 問題 | 嚴重度 | 對 MDD 的槓桿 | 修復成本 | 當前狀態 |
|------|--------|--------------|---------|---------|
| Action softmax temp=0.5 | 🔴 高 | 直接：移除集中度失控根因 | 極低（一行參數） | ❌ 未處置 |
| Action top-k 梯度死區 | 🟡 中高 | 間接：改善學習效率 | 低（需換架構） | ❌ 未處置 |
| Reward proxy 錯位 | 🔴 高 | 直接：r5 調強懲罰 | 中（M1b 已完成） | ✅ M1b 完成，待驗證 |
| POMDP 觀測不完整 | 🟡 中 | 間接：學習訊號更完整 | 低（M1a 已完成） | ✅ M1a 完成，待驗證 |
| 過度囤現金（退化策略） | 🟡 中 | 中：影響 Sortino | 低（調 CASH_DEFENSIVE） | ❌ 未評估 |
| Seed 間不穩定性 | 🟡 中 | 間接 | 多因素 | 部分 M1 改善 |
| Long episode | 🟡 中 | 間接：credit assignment | 中（M3-1） | M3 預案 |
| Sim-to-Real T+1 | 🟡 中 | 不影響研究 | 中（M3-2） | M3 預案 |

---

## 六、硬體限制下的可行性分析

> GTX 1060，~36–52 fps；Windows spawn 限制；RAM 有限

### 在當前硬體下完全可行（零額外成本）

| 修復 | 改動量 | 方式 |
|------|--------|------|
| `softmax_temp` 0.5 → 1.0 | 1 個參數 | `settings.py` 或 `--softmax-temp` flag |
| M1b r5 reward 已完成 | 已 done | 等 smoke 結果 |
| M1a POMDP obs 已完成 | 已 done | 等 smoke 結果 |
| 評估 LAMBDA_CASH_DEFENSIVE 過強問題 | 分析 + 可選 tune | 不需重訓即可分析 |

### 有硬體限制但仍可接受

| 修復 | 影響 | 建議 |
|------|------|------|
| M2-smoke（正在執行） | 每次 300K，數小時 | 不並行，嚴格停損紀律 |
| M1c 集中度護欄 | 輕量 action decode 修改 | 建議 smoke 完成後執行 |
| M3-1 固定 episode | 需重設訓練邏輯 | r5 promotion > 38% 才觸發 |

### 已正確封存（硬體資源不允許）

- **SAC-R（LSTM）**：fps 更低，encode 計算成本高
- **P10 VecEnv**：Windows spawn 限制 + 驗收 FAIL + MDD 惡化
- **全矩陣重訓**：已改為候選集策略（O3）

---

## 七、建議的修復優先序

### 立即（在等 M2-smoke 期間，零成本）

**P0：將 `softmax_temp` 從 0.5 改為 1.0**

```python
# settings.py 或 env_config.py 中
default_softmax_temp: float = 1.0   # 原 0.5

# 或在 env_config 版本化（推薦，與 r5 一起 bump）
# env_config_version = "r5"
# softmax_temp = 1.0
```

驗證：smoke 完成後觀察 `top_holdings`，seed43 類型不應再出現 >60% 單股集中。

---

### M2-smoke 通過後

**P1：M1c 集中度護欄（entropy floor）**

```python
# trading_env.py _transform_action()，top-k 遮罩後、re-normalize 前插入
MIN_WEIGHT_IN_TOPK = 0.05   # top-k 內每股最少 5%
stock_weights[topk_indices] = np.maximum(
    stock_weights[topk_indices], MIN_WEIGHT_IN_TOPK
)
# 後接原有 re-normalize
```

**P2：評估 LAMBDA_CASH_DEFENSIVE 是否造成過度囤現金**

檢查 r5 smoke 後各期 avg_cash_weight，若仍出現 >90%，可能需要降低 LAMBDA_CASH_DEFENSIVE 或加入最低曝險下限。

**P3：推進 M2-candidate（seeds 42,43）**

---

### 若 r5 M2-promotion 失敗（worst MDD > 38%）

**按成本從低到高**：

1. **M3-1**：固定 126 日 episode + 隨機起點（解決 credit assignment 問題）
2. **評估 top-k 替代方案**：直接輸出 top-5 logits（縮小 action space，消除梯度死區）
3. **SL hybrid（方案 B，SUPERVISED_LEARNING_PLAN.md 已設計完整）**：LightGBM 打分 + RuleBasedAllocator，可在數分鐘內訓練，MDD 由階梯式風控**顯式**控制，不依賴 reward shaping
4. **M3-4**：CVaR 約束取代 reward shaping（計算成本高，最後手段）

---

## 八、總體判斷

### 算法本身：合理

PPO / SAC + Self-Attention GNN + Walk-Forward + Promotion Gate 的整體架構設計合理，實驗數據（SAC44 Sortino 2.31）證明系統確實學到了有效的配置能力。

### 當前卡關根因：兩個相互疊加的問題

```
MDD 無法達標
    ├── 主因：reward proxy 與 OOS worst-case MDD 在數學上不等價
    │         → r5 M1b 方向對，但天花板未知
    └── 加劇因素：softmax(0.5) + top-k 允許結構性集中度失控
                  → 某些 seed 的 MDD 因單股 99% 而爆表
                  → 修復成本極低（一行），應立即處理
```

### 硬體限制的定位

硬體不足**不影響算法合理性**，但決定了可行的實驗範疇。SAC-R、VecEnv 的封存是正確決策。最主要的硬體影響是**迭代速度**（每次 smoke 數小時），因此停損紀律（smoke 失敗立即回滾，不推進 candidate）必須嚴格執行。

### 最終建議

> 在推進 M2-smoke 等結果的同時，立即將 `softmax_temp` 從 0.5 改為 1.0，並準備在 smoke 通過後執行 M1c。這是目前成本最低、潛在收益最大的改動，且不影響任何其他在進行中的計畫。

---

## 九、S5 整合進展（2026-06-11 更新）

### 完成
- ✅ `sl_pipeline/walk_forward_sl.py`：`run_walk_forward_sl` 現在在每個週期後自動把 `scores_{name}_h5.csv` 寫入 `output_dir`
- ✅ `research_pipeline.py`：`build_train_env` 和 `build_eval_env` 都接受 `enable_sl_features` + `sl_scores`；透過 `build_sl_feature_arrays` 轉換為 `(n_steps, 3)` 陣列接通 `TaiwanStockEnv`
- ✅ `walk_forward.py`：CLI `--sl-scores-dir`；每個 period 自動讀取對應 CSV；帶 SL 特徵的 5K 步短跑已確認無 SyntaxError、無 MemoryError、模型正常儲存
- ✅ `trading_env.py`：`enable_sl_features=False` 時維度不變（向後相容），`=True` 時 obs 維度正確擴張 `num_stocks × 3`

### 待執行
```powershell
# 完整 S5 300K Walk-Forward（下一步）
$env:SAC_BUFFER_RAM_GB="1.5"
.\env\Scripts\python.exe walk_forward.py --timesteps 300000 --algo sac --cash-mode disabled --sl-scores-dir results_dir --overwrite
```

---

*評估依據：`trading_env.py`、`gnn_extractor.py`、`train_portfolio.py`、`docs/ALGORITHM_REVIEW.md`、`docs/RESEARCH_STRATEGY_V3.md`、`docs/SUPERVISED_LEARNING_PLAN.md`、`.research/baselines/*.json`（seed 42/43/44）*
*更新：2026-06-11 · Antigravity*
