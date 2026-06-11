# 整體決策算法合理性評估

> 評估基準：v3 戰略（r5 RL rebuild）· 硬體限制已知（Windows RAM / GPU 受限）  
> 評估範圍：MDP 設計 → 算法選擇 → reward 架構 → 驗證體系 → 戰略決策邏輯 → Action 空間非線性  
> 評估日期：2026-06-11 · 評估者：Antigravity

---

## 一、核心結論

**算法選擇整體合理，戰略決策邏輯清晰，但存在三個結構性瓶頸，在硬體限制無法解除的前提下，部分問題需要更謹慎的期望管理。**

| 評估面向 | 合理性 | 說明 |
|----------|--------|------|
| MDP 問題建模 | ✅ 合理 | 連續動作投資組合是 RL 合適場景 |
| 算法選擇（SAC / PPO） | ✅ 合理 | SAC 熵正則對探索現金有優勢 |
| GNN / Attention 架構 | ✅ 合理 | 跨股資訊交換符合輪動直覺 |
| Reward 設計 | ⚠️ 有問題 | 目標錯位是當前卡關根因 |
| 驗證體系（Walk-Forward + Gate） | ✅ 業界高標準 | 多 seed、擴展 WF、8 項 Gate |
| v3 戰略轉向邏輯 | ✅ 合理 | 正確識別根因並收斂焦點 |
| **Action 空間非線性** | 🔴 結構性問題 | softmax(0.5) + top-k 梯度死區 + 集中度失控 |
| 硬體限制下的可行性 | ⚠️ 部分受限 | 迭代慢、OOM 已出現 |

---

## 二、合理的部分（值得保留）

### 2.1 問題建模（MDP 設計）★★★★☆

**合理**。台股科技/電子股約 45 支，連續權重配置是真實問題，連續動作空間 RL 是正確方向。

- Action 空間：`Box(num_stocks+1)` logits → softmax → top-k(5) → 權重  
- 引入現金維度（`cash=enabled`）是正確的熊市防禦機制  
- State 含 20 日視窗市場特徵 + 9 個帳戶特徵（M1a 已補全 rolling vol / sortino proxy / current DD）

**硬體限制影響**：obs 維度 ≈ 96K，P8 IndexedReplayBuffer 已壓縮 RAM 至 ~0.2GB@300K，但已觸發一次 ArrayMemoryError（experiment_ledger 第 24 條），已 fix。

### 2.2 算法選擇（SAC + enabled cash）★★★★☆

**合理**。

- SAC 的 **熵正則**（`ent_coef="auto"`）鼓勵探索，對現金比例決策有幫助  
- 實驗資料支持：SAC enabled 的 Sortino 達 2.32（R6 最佳），非隨機結果  
- `gradient_steps=1` 的決定正確（R7b 已證明 gradient_steps 對 MDD 無直接槓桿）
- P8 IndexedReplayBuffer 確保 300K off-policy 訓練可行（Windows RAM）

### 2.3 GNN / Self-Attention 特徵提取器 ★★★★☆

**合理**。Multi-Head Attention（名為 GNN，實為 stock-level Transformer）讓各股互相交換資訊，符合板塊輪動直覺。Softsign 激活有界，避免梯度爆炸。

### 2.4 驗證體系 ★★★★★

**業界高標準，無需更改**。

- **Walk-Forward**：擴展視窗（2024H2 → 2026H1），不是 in-sample 回測
- **多 seed（3 seeds）**：排除初始化運氣；worst-case MDD 取最差 seed
- **8 項 Promotion Gate**：Sortino、MDD、Turnover、Cash behavior、Baseline、Ablation、Stress、Period consistency
- **O2 分層**：smoke → candidate → promotion，禁止跳層

**Promotion Gate 擋下 worst-case MDD 38.71% > 35% 是正確的決策，不是過度嚴格。**

### 2.5 v3 戰略轉向邏輯 ✅

**決策邏輯清晰合理**。v2 同時推 SAC 工程（R7b）與 SAC-R（LSTM），但兩者都不能直接降低 MDD。v3 正確識別根因：

> **reward proxy 與 MDD Gate 目標錯位 + POMDP 觀測不完整**

砍掉 R7b、R8、R9、SAC-R，集中資源於 M1a（obs 補全）+ M1b（reward r5）是正確的焦點收斂。

---

## 三、有問題的部分

### 3.1 🔴 Action 空間非線性（新診斷）

#### 實際 Decode 管線（Long-Only / SAC enabled 路徑）

```
網路輸出 logits ∈ [-5, 5]^46
     ↓
① softmax(temp=0.5)      ← 指數函數，高度非線性
     ↓
② top-k(5) 遮罩          ← 非連續、多對一映射，40支梯度=0
     ↓
③ re-normalize           ← 條件性除法
     ↓
最終權重 ∈ [0,1]^45 + cash
```

#### 問題 A：softmax temperature=0.5 接近 argmax

```python
shifted = action - np.max(action)
exp_a = np.exp(shifted / 0.5)   # 等效於 exp(logit * 2)
```

| temperature | logit 差距 1.0 時的權重比 |
|-------------|--------------------------|
| 1.0（標準）  | e^1 ≈ 2.7x               |
| **0.5（現行）** | **e^2 ≈ 7.4x**        |
| 0.1          | e^10 ≈ 22,026x（幾乎 argmax）|

**只需要一個 logit 比其他高 3.0，對應股票就占約 99.4% 的權重。**

```
seed43 / 2024H2：8046.TW = 98.9% 集中度
→ 直接由 softmax(0.5) 的結構性放大造成，不是 reward 的問題
```

#### 問題 B：top-k 遮罩製造梯度死區

```python
topk_indices = np.argsort(stock_weights)[-self._topk:]
mask = np.zeros(...)
mask[topk_indices] = 1.0
stock_weights = stock_weights * mask   # 40支股票梯度強制=0
```

| 股票排名 | 梯度行為 |
|----------|---------|
| top-5 內 | 正常梯度回傳 |
| top-5 外（40支）| **梯度 = 0**，參數無法更新 |

網路 45 個輸出維度中，只有 5 個在大多數 step 有梯度訊號，學習效率低的根本原因之一。

#### 問題 C：top-k 邊界不可微（sorting 是離散操作）

第 5 名與第 6 名分數接近時，微小的 logit 擾動就可以讓**選中的 5 支股票完全改變**，在邊界附近 policy gradient 梯度方向不穩定，投資組合有跳躍式切換風險。

---

### 3.2 🔴 Reward 與目標錯位（最關鍵）

| Reward 實際優化的是 | Promotion Gate 要求的是 |
|---------------------|------------------------|
| 0.4·日報酬 + 0.3·Sortino + 0.3·超越基準 - 成本 - 換手 - 回撤懲罰 | worst-case OOS MDD ≤ 35% |

- R4 加大 `LAMBDA_DRAWDOWN=0.8`→r5 的 1.2 方向對，但仍是**事後補救型 shaping**，數學上無法保證 MDD 達標
- R6 三 seed worst MDD：37.7% / 46.0% / 38.2%，全部超標
- 多個手調權重存在 reward hacking 風險

### 3.3 🟡 POMDP 觀測不完整

Reward 的 Sortino 分量依賴 20 日滾動 `_return_history`，但 obs 只含摘要統計。**M1a 已完成修復**（NUM_ACCOUNT_FEATURES 6→9），需等 M2-smoke 確認效果。

### 3.4 🟡 集中度問題（seed43 警訊）

如上 §3.1 分析，seed43 / 2024H2 單股 98.9% 集中是 **softmax(0.5) 的結構性後果**，不是 reward shaping 能完全解決的問題。M1c 集中度護欄尚未執行。

### 3.5 🟡 Sim-to-Real Gap（中期問題）

| 訓練環境 | 台股實盤 |
|----------|---------|
| 收盤即時調倉 | T+1 賣、隔日早盤買 |
| 完全成交假設 | 滑價 + 排程延遲 |

M3 預案解決，不影響研究階段 Gate 判斷。

### 3.6 🟡 Episode 設計

整段歷史 800–1200 步為一個 episode，非平穩市場的 credit assignment 困難。M3-1 預案（固定 126 日 episode + 隨機起點）。

---

## 四、硬體限制下的可行路徑分析

> 前提：Windows、有限 GPU VRAM / RAM，暫時無法解決

### 可以做、硬體影響有限

| 問題 | 修復方案 | 硬體成本 |
|------|----------|---------|
| M1a obs 補全 | 改 env，不增訓練時間 | 無 |
| M1b reward r5 | 同等訓練時間 | 無 |
| **softmax temp 提高（0.5→1.0）** | **單一參數，改一行** | **無** |
| **entropy floor / 最小權重下界** | action decode 後加 clip | 無 |
| M1c 集中度護欄 | action decode 層修改 | 無 |
| OOM fix（ArrayMemoryError）| per-sample slice | 已 fix |

### 有硬體瓶頸、需期望管理

| 問題 | 硬體影響 |
|------|---------|
| 迭代速度慢 | 每次 smoke 需 300K steps，r5 多輪調參時間成本高 |
| SAC-R（LSTM）| fps 壓縮更嚴重，已正確封存 ✅ |
| P10 VecEnv | Windows spawn 限制，MDD 惡化，已正確封存 ✅ |

### 特別注意

1. **smoke 停損紀律必須嚴格**：`2025H1 MDD ≥ r4 seed42（37.7%）→ 立即回滾`
2. M2-smoke 目前 `running`（research_state.json），等結果前不推進 candidate
3. candidate 需要 2 seeds，訓練時間加倍，需提前預估時間

---

## 五、修復優先序建議

| 優先 | 項目 | 方案 | 成本 |
|------|------|------|------|
| **P0** | softmax temperature | `0.5 → 1.0`（一行改動，隨 M1b 一起驗證） | 極低 |
| **P0** | M2-smoke 結果等待 | 正在執行，不提前行動 | — |
| **P1** | M1c 集中度護欄 | entropy floor：top-k 內每股最低 5% 下限 | 低 |
| **P2** | M2-candidate | smoke 通過後 seeds 42,43 | 中（時間） |
| **P3** | M3-1（若 r5 失敗） | 固定 126 日 episode + 隨機起點 | 中 |
| **P4** | M3-4（若 M3-1 仍失敗） | CVaR 約束取代 reward shaping | 高 |

### softmax temperature 修復細節

```python
# trading_env.py 第 59 行，__init__ 參數預設值
# 現行
softmax_temp: float = 0.5

# 建議
softmax_temp: float = 1.0   # 或透過 env_config.py 版本化

# 驗證指標：smoke 後觀察 top_holdings 的集中度
# seed43 / 2024H2 不應再出現單股 > 50%
```

### entropy floor 修復細節

```python
# _transform_action() 第 287 行 top-k 遮罩後，re-normalize 前插入
MIN_TOP_K_WEIGHT = 0.05   # top-k 內每股至少 5%
stock_weights[topk_indices] = np.maximum(
    stock_weights[topk_indices], MIN_TOP_K_WEIGHT
)
# 原有 re-normalize 不變
total = float(np.sum(stock_weights) + cash_weight)
```

---

## 六、分項評分摘要

| 面向 | 問題 | 嚴重度 | 當前處置 | 是否足夠 |
|------|------|--------|----------|---------|
| **Action 空間非線性** | softmax(0.5) + top-k | 🔴 高 | 未處置 | ❌ 需處理 |
| Reward 目標錯位 | proxy ≠ MDD Gate | 🔴 高 | M1b r5 reward | 方向對，效果未知 |
| POMDP 觀測 | 不完整 | 🟡 中 | M1a ✅ 已完成 | ✅ 待驗證 |
| 集中度 seed43 | 單股 99% | 🟡 中 | M1c 待執行 | ❌ 未處置 |
| Episode 設計 | 長 episode | 🟡 中 | M3 預案 | 暫緩合理 |
| Sim-to-Real | T+1 gap | 🟡 中 | M3 預案 | 暫緩合理 |
| 硬體迭代速度 | smoke 耗時 | 🟡 中 | 無法解決 | 需期望管理 |
| SAC-R LSTM | 資源密集 | 🟡 中 | 已封存 ✅ | ✅ 正確 |
| P10 VecEnv | Windows 限制 | 🟡 中 | 已封存 ✅ | ✅ 正確 |

---

## 七、總體判斷

### 算法決策邏輯：✅ 合理（但 Action 空間有結構性問題）

- MDP → SAC → GNN → WF 整條鏈設計合理，Sortino 2.32+ 有學到策略
- v3 聚焦 MDD 根因是正確的焦點轉移
- 停損紀律（smoke 不過不進 candidate）設計嚴格合理

### 新增診斷：softmax(0.5) + top-k 是集中度問題的結構根因

seed43 的 98.9% 集中不是 reward shaping 能完全解決的問題。即便 M1b 加強了 LAMBDA_DRAWDOWN，只要 softmax temperature 維持 0.5，網路輸出稍大的 logit 就會被放大為接近 argmax 的權重。

**建議：在 M2-smoke 前，或與 M2-smoke 結果一起評估時，加入 `softmax_temp=1.0` 的對照實驗。這是成本最低、潛在收益最大的改動。**

### 最大風險：reward shaping 天花板 + action 非線性雙重疊加

若 r5 M2-promotion 仍失敗（worst MDD > 38%），需同時從兩個方向修復：
1. **Action 解碼**：提高 temperature + entropy floor（移除結構性集中度缺陷）
2. **目標函數**：M3 CVaR 約束取代 reward shaping（從數學上對準 MDD）

在硬體限制下，M3-1（固定 episode）+ Action 解碼修復應先於 CVaR 評估，因計算成本較低。

---

*文件生成：2026-06-11 · 評估者：Antigravity*  
*依據文件：`docs/ALGORITHM_REVIEW.md`、`docs/RESEARCH_STRATEGY_V3.md`、`trading_env.py`、`.research/baselines/*.json`*
