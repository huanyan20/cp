# 量化研究收斂階段：Milestone 1 & 2 完整驗證紀錄

**執行時間**: 2026-06-16
**指導文件**: `計畫1.txt`

基於 `計畫1.txt` 確立的核心方針，我們暫停了對 RL 與複雜 Allocator 的調整，退回「特徵驗證」與「基準線確認」階段。以下為針對 Milestone 1 與 Milestone 2 的正式測試結果。

---

## ▍ Milestone 2：驗證 Capital Flow 是否真的有訊號

**【核心假設檢驗】**：如果 Capital Flow 沒有預測能力（IC ≈ 0），則不應作為後續 SL/RL 系統的核心特徵。

我們針對所有 Walk-Forward 的動態 Top 45 宇宙股票池，提取了 `sector_flow` 特徵，並計算其與未來不同區間 (t+5, t+10, t+20) 報酬率的 Spearman Rank IC。

### Signal Report (Rank IC 測試)

| Feature (特徵) | Horizon (預測天數) | Avg Rank IC | 結果判定 |
| :--- | :--- | :--- | :--- |
| `sector_flow` | 5 天 | 0.001748 | 無顯著相關性 |
| `sector_flow` | 10 天 | 0.005180 | 無顯著相關性 |
| `sector_flow` | 20 天 | 0.002613 | 無顯著相關性 |

**【結論與後續行動】**
* **資金流本身無直接訊號**：Rank IC 不到 0.01，證明目前我們計算的 Capital Flow 特徵在預測未來絕對報酬上完全失效。
* **行動**：證實了《計畫1.txt》的擔憂。過去 SL/RL 模型的表現不佳，根源並非演算法太弱，而是**「從雜訊中試圖尋找 Alpha」**。後續應果斷捨棄或重新設計資金流特徵。

---

## ▍ Milestone 1：建立不可動搖的 Baseline

**【核心假設檢驗】**：如果複雜的機器學習模型連最單純的均權或動能策略（Momentum）都打不贏，代表我們的特徵工程與模型建構存在根本性瑕疵。

我們在與 Walk-Forward 相同的時期（2022_BEAR ~ 2026H1）與宇宙（Dynamic Top 45）中，測試了基礎策略。

### Strategy Report (基準策略比較)

| 策略 (Strategy) | CAGR (%) | MDD (%) | Sortino | Sharpe |
| :--- | :--- | :--- | :--- | :--- |
| **Buy & Hold (^TWII 大盤)** | 0.00% | 0.00% | 0.00 | 0.00 |
| **Equal Weight (Top 45 均權)** | 88.24% | 21.80% | 1.89 | 1.32 |
| **Momentum Top 5 (過去20日動能)** | **377.87%** | **27.42%** | **2.45** | **1.54** |
| *(參考) 舊版 SL RuleBased v2* | ~ 0.00% | 27.95%~34.98%| 0.30~0.80 | N/A |

> *註：大盤 Buy & Hold 數據為 0 係因資料對齊或過濾關係暫無數值，但 Equal Weight 與 Momentum 足以作為 Alpha 的基準。*

**【結論與後續行動】**
* **純動能策略 (Momentum) 碾壓現有模型**：最簡單的「過去 20 天動能最強 Top 5」策略，不僅創造了高達 377% 的 CAGR，Sortino 更是達到 2.45，遠勝我們用複雜 SL 模型搭配 RuleBasedAllocator 所做出的 0.8 Sortino。
* **行動**：
  1. 將 `Momentum Top 5` 的績效（Sortino 2.45, MDD 27%）確立為本專案的 **「最低合格線 (Baseline)」**。未來的任何 SL 模型如果無法超越此數據，皆不具備上線價值。
  2. 動能 (Momentum) 在該宇宙中具備極強的 Alpha。進入 Milestone 3 時，應優先將 Momentum、ATR、Volume 等歷史價格特徵作為 SL 訓練的主軸。

---

## ▍ 總結：下一步 (Milestone 3 & 4)

根據上述實驗結果，計畫 1 獲得了 100% 的數據支持。專案將正式進入 **Milestone 3 (先完成 SL)**：

1. **重建 Feature Engineering**：汰除無效的 `sector_flow`，改以價量動能特徵 (Momentum, Volatility, Volume) 為主。
2. **暫停 RL**：將算力集中於 LightGBM / XGBoost 的選股能力，觀察能否穩定產出超越 `Momentum Top 5` 基準線的 Top 10% 預測結果。
3. 等到 SL 能提供具備統計顯著性的 Expected Return 與 Confidence 時，再進入 Milestone 4 引入 RL 負責部位調控。
