# 模型算法評估：強化學習投資組合策略

> **v3 行動**（2026-06-11）：依本文件 §二/§四 執行 **r5 RL 重構**（M1 obs/reward → M2 分層 WF）。詳見 [`RESEARCH_STRATEGY_V3.md`](RESEARCH_STRATEGY_V3.md)。  
> **狀態**：研究評估（2026-06-09）
> **範圍**：`trading_env.py`、`gnn_extractor.py`、`train_portfolio.py`、`research_pipeline.py`
> **對應計畫**：[`../專案總覽.md`](../專案總覽.md)

---

## 總結

**算法選擇合理，且高於一般研究專案水準，但尚不足以上線。**

主要問題不是「算法選錯」，而是：

1. reward proxy 與 Promotion Gate 的 MDD 目標錯位
2. 觀測不完整造成的 POMDP
3. 訓練模擬與 T+1 實盤節奏不一致

Promotion Gate 擋下（worst-case MDD 38.71% > 35%）是正確的決策。

| 維度 | 評分 | 說明 |
|------|------|------|
| 問題建模（MDP） | ★★★★☆ | 投資組合配置 → 連續動作 RL，合理 |
| 算法選擇（PPO/SAC） | ★★★★☆ | 適合連續權重空間；SAC + 動態現金與實驗一致 |
| 網路架構（Attention） | ★★★★☆ | 跨股資訊交換對輪動策略有意義 |
| Reward 設計 | ★★★☆☆ | 方向對，但多項人工權重，與真實目標有落差 |
| 驗證方法 | ★★★★★ | Walk-Forward、多 seed、Promotion Gate 扎實 |
| 實盤可遷移性 | ★★☆☆☆ | T+1 下單、MDD 超標顯示 sim-to-real 仍有距離 |

---

## 一、合理的設計（保留）

### 1. PPO / SAC 用於連續權重配置

投資組合配置本質是連續動作空間（各股目標權重）。PPO 穩定、SAC 自帶熵正則，皆為文獻與實務常見選擇。實驗結果也支持：SAC + `cash=enabled` Sortino 最佳（2.32），與「連續調倉 + 需要探索現金比例」設定相符。

- PPO：低 learning rate（3e-5）、entropy annealing（0.05 → 0.001）、`target_kl=0.08`
- SAC：`ent_coef="auto"`、依記憶體自動調整 `buffer_size`

### 2. 跨股 Self-Attention 特徵提取器（`gnn_extractor.py`）

觀測 `(45 stocks, features_per_stock)`，以 Multi-Head Attention 讓股票互相交換資訊，比「各股獨立 MLP 再拼接」更符合板塊輪動、相對強弱邏輯。名為 GNN，實為 stock-level Transformer，概念合理。Softsign 激活有界，緩解梯度爆炸與 dead neuron。

### 3. Top-K + Softmax 動作解碼

- Top-K（預設 5）：集中持倉、降低換手
- Softmax temperature（0.5）：控制權重分散
- 可選現金維度：熊市降曝險

比直接輸出 45 維自由權重更貼近實務，並降低有效動作維度。

### 4. Reward 多目標（大方向正確）

混合：日報酬（0.4）+ Sortino（0.3）+ 超越基準（0.3），扣除交易成本、換手、回撤懲罰，最後 `clip(-1, 1)`。`softsign` 有界化是金融 RL 常見技巧。R4 強化回撤懲罰顯示團隊對準真實痛點調整。

### 5. Walk-Forward + 多 Seed 驗證

擴展視窗 Walk-Forward（2024H2 → 2026H1）、3 seeds、`deterministic=True` 推論。此驗證標準高於多數 RL 交易專案。

### 6. 實驗結果證明有學到東西

最佳 SAC：Sortino ≈ 2.32、勝率 ≈ 55%、換手率 ≈ 0.8%。非隨機權重可達，代表算法與特徵確有配置能力。

---

## 二、有問題或存疑的設計

### 1. Reward 與真實目標錯位（最關鍵）

Promotion Gate 卡 worst-case MDD，但 reward 是每日混合 proxy。後果：

- Agent 優化「日報酬 + 滾動 Sortino + 懲罰項」，非「OOS 最大回撤」
- 多個懲罰權重（`LAMBDA_COST=5.0`、`LAMBDA_DRAWDOWN=0.8` 等）手調，易 reward hacking
- R4 加大回撤懲罰方向對，但本質仍為事後補救，不保證 MDD < 35%

### 2. 部分可觀測（POMDP）

Reward 的 Sortino 分量依賴 20 日滾動 `_return_history`，但觀測只含 `total_return`、`max_drawdown` 等摘要，**未含完整滾動序列**。Agent 看不到驅動 reward 的狀態 → POMDP，policy gradient 學習變慢、不穩定。

### 3. Softmax 動作解碼非線性過強

logits → softmax → top-k → normalize 為高度非線性、多對一映射，梯度敏感度隨 temperature 變化，可能是學習效率與穩定性瓶頸。連續控制常見做法為直接輸出權重 + projection 保證約束。

### 4. 訓練模擬 vs 實盤執行不一致

| 訓練環境 | 實盤（CMoney RPA） |
|----------|-------------------|
| 每日收盤即時調倉 | T+1 賣、隔日早盤買 |
| 假設完全成交 | RPA 有滑價、排程延遲 |
| 同日交易再計報酬 | 實際有隔夜 gap 風險 |

存在 sim-to-real gap，會壓縮實盤表現。

### 5. Episode 設計：整段歷史為單一 Episode

每個 episode 跨牛熊（約 800–1200 步）。市場非平穩、credit assignment 困難。Walk-Forward OOS 部分緩解，但訓練動力學非最佳。可改固定長度 episode（如 126 日）+ 隨機起點。

### 6. Overnight Features Ablation 結果負面

| 指標 | With Features | Without Features |
|------|---------------|------------------|
| Sortino | 0.88 | 2.20 |
| MDD | 25.5% | 36.1% |

特徵改善回撤卻傷害 alpha，較像「特徵與 RL 整合方式」問題（維度、尺度、共線），而非概念錯誤。當作獨立風控模組（`preopen_macro_check`）比塞進 state 合理——已對齊 O6 / R5 決策。

### 7. 穩健性不足

PPO disabled 總報酬 std 達 113%，seed 間分散大，顯示 policy 對輸入/初始化敏感。

---

## 三、與替代方案比較

| 方案 | 優點 | RL 相對位置 |
|------|------|-------------|
| 等權 / 動量 / 風險平價 | 簡單、可解釋 | baseline 有對照，RL 有超越 |
| 監督學習預測 + 優化器配置 | 目標清晰、易除錯 | RL 端到端但難解釋 |
| 截面排序 + Top-K 規則 | 貼近台股實務 | RL top-k 已部分吸收 |
| 本專案：RL 直接學權重 | 可學非線性輪動、含成本 | Sortino 2.3+ 有效，但 MDD 控制不足 |

RL 比簡單 baseline 有價值，但 MDD 未過關前，複雜度尚未完全被證明值得。

---

## 四、具體建議（對齊重構計畫）

### 短期（v3 M1 — 不改大架構）

1. **M1a**：觀測補 rolling vol / rolling Sortino proxy / drawdown 深度，緩解 POMDP
2. **M1b**：reward r5 + `ENV_CONFIG_VERSION` bump（單變因對準 MDD）
3. **M2-smoke**：O2 tier smoke（seed 42）— 2025H1 MDD 須優於 r4 同 seed
4. overnight features 維持風控層（O6），不進 RL 預設輸入（R5）

### 中期（若 r5 promotion 仍 MDD > 38% — M3）

4. 改固定長度 episode（如 126 交易日）+ 隨機起點
5. 兩階段：監督學習打分 → RL 只做 top-k 權重分配
6. 環境加入 T+1 執行延遲，縮小 sim-to-real gap

### 長期

7. MDD 仍無法達標時，評估改用風險約束優化（如 CVaR constraint）取代純 reward shaping

---

## 五、最終判斷

- **算法選擇合理嗎？** 合理，PPO/SAC、Attention、Top-K、成本建模、Walk-Forward 都站得住腳，且實驗證明學到可交易策略。
- **足夠上線嗎？** 還不行。瓶頸在 reward 與 MDD 目標錯位、POMDP 觀測不完整、模擬與實盤節奏不一致。
- **下一步**（v3）：M1a → M1b → M2 分層 WF；R6/r4 已驗證不足，不再重跑為主線。
