# Live Operations

> **上線前提**：Promotion Gate APPROVED（見 [`../專案總覽.md`](../專案總覽.md)）。  
> 每日生產流程、風控與下單操作手冊。

## 1. 安全開關

| 項目 | 機制 | 說明 |
|------|------|------|
| 實盤總開關 | `ENABLE_LIVE_TRADING` | 非 `true` 一律 dry-run，不送真實委託 |
| 帳號比對 | `CMONEY_AID` | `signal.aid` 必須等於 `CMONEY_AID`，否則中止 |
| 訊號時效 | `SIGNAL_TTL_SECONDS`（預設 900s） | `signal.json` 過期即中止 |
| 宏觀守衛 | `preopen_macro_check.json` 的 `level` | 非 `OK` 影響買單（見下） |
| 風險限額 | `RiskLimits` | 單股權重 / 總曝險超標 → dry-run diff 失敗 → 中止 |

> 預設 fail-closed：盤前守衛失敗或例外時，視為 CRITICAL、跳過買單。

## 2. 每日流程（`daily_trade_runner.py main()`）

```text
Step -1  盤前宏觀守衛   preopen_macro_check.py → preopen_macro_check.json
            OK       → 正常
            WARN     → 買單減半（--half-buys）
            CRITICAL → 跳過 pending buys（fail-closed）
Step 0   自動登入       auto_login.py（更新 cookie）
            + 執行前一日 Pending BUY：cmoney_rpa.py --pending-buys [--half-buys] [--execute]
            （CRITICAL 時跳過）
Step 1   產生訊號       先刪除舊 signal.json → evaluate_portfolio.py
            --model-path ... --overnight-feature-path ... [--half-buys(WARN)]
            → 產生 signal.json（找不到即中止）
            → _require_dry_run_diff()：呼叫 trade_guard.py 產生並驗證 dry-run diff
Step 2   下單（T+1）    _require_live_execution_context() + _require_signal_guard()
            → cmoney_rpa.py --signal signal.json --sell-only [--execute]
            （T+1：當日只送 SELL；BUY 寫入 pending，隔日 Step 0 執行）
```

每個關鍵步驟都會經 `send_notification()` 發 Email（需 `.env` 設定 `EMAIL_*`）。

## 3. 風控驗證（`trade_guard.py`）

`generate_diff(signal, aid)`：

1. `load_signal()` 檢查 aid match 與 TTL。
2. 由 `CMoneyRPA.get_account_status()["inventory"]` 取得庫存。
3. `build_dry_run_diff()` 計算目標 vs 現況差異。
4. `evaluate_risk_limits()`：
   - 單股權重 ≤ `MAX_SINGLE_WEIGHT`（預設 0.35）
   - 總曝險 ≤ `MAX_TOTAL_EXPOSURE`（預設 1.0）
   - 任一超標 → 拋錯，dry-run diff 標記 `risk_checks.passed=false`。
5. 寫出 `trade_guard_diff.json`（`LiveSettings.dry_run_diff_path`）。

`daily_trade_runner._require_dry_run_diff()` 會在 Step 1 後重跑此檢查，失敗即 `sys.exit(1)`。

## 4. 盤前宏觀守衛

```bash
python capital_flow_analysis/src/monitoring/preopen_macro_check.py
# 產出 capital_flow_analysis/data/preopen_macro_check.json
```

| Level | 交易行為 |
|-------|----------|
| `OK` | 正常訊號流 |
| `WARN` | pending buys 與 evaluate 目標權重減半 |
| `CRITICAL` | 跳過 pending buys；僅 sell-only 可續行 |

## 5. 手動 dry-run（不送單）

```bash
# 1. 產生訊號
python evaluate_portfolio.py --model-path ppo_portfolio_full_stock_seed42.zip \
  --overnight-feature-path capital_flow_analysis/data/overnight_gap_features_1d.csv

# 2. 風控與 diff（不帶 --execute）
python trade_guard.py --signal signal.json --aid <CMONEY_AID>
#   檢視 trade_guard_diff.json

# 3. RPA dry-run（未設 ENABLE_LIVE_TRADING=true 時不會真實下單）
python cmoney_rpa.py --signal signal.json --sell-only
```

## 6. 啟用實盤前檢查清單

- [x] Promotion Gate 為 APPROVED（見 `docs/RESEARCH_PLAYBOOK.md`，SL 模型已達標）
- [ ] `.env` 已設 `CMONEY_AID`、`CMONEY_USERNAME/PASSWORD`、`EMAIL_*`
- [ ] dry-run diff `risk_checks.passed=true`
- [ ] 盤前守衛 `level=OK`
- [ ] 最後才設 `ENABLE_LIVE_TRADING=true`

## 7. 相關文件

- `docs/ARCHITECTURE.md`：系統架構
- `docs/RESEARCH_PLAYBOOK.md`：研究與 Promotion Gate
- `capital_flow_analysis/README.md`：宏觀特徵與盤前守衛細節
- `教學文件.md`：完整逐模組教學

## 8. 緊急應變處理 (Emergency SOP)

> 當系統發生嚴重異常（如收到 CRITICAL 通知或排程中斷）時的處置流程。

### 8.1 T+1 競態條件 (Race Condition)
- **情境**：收到 `CRITICAL ACTION REQUIRED: T+1 Pending BUYS were executed successfully today, but the model failed to generate SELL signals...`
- **問題**：系統已經在今天開盤買入了昨天的標的，但今天應該賣出的訊號沒有產生/沒有執行。這會導致資金過度集中或槓桿超標。
- **處置**：
  1. 登入 CMoney 大富翁 Web 介面。
  2. 檢查 `pending_buys` 執行的標的是否真的買到了。
  3. 根據最新的模型訊號或手動策略，將不需要的庫存手動掛單賣出，以平衡投資組合。
  4. 確認 `cmoney_rpa.py` 及網路連線正常後，明天讓系統自然恢復。

### 8.2 每日 P&L 斷路器觸發 (Circuit Breaker)
- **情境**：收到 `Dry-run diff validation failed: CIRCUIT BREAKER: Daily loss X% exceeds 5% threshold` 或 MDD 超標。
- **問題**：活體資金出現劇烈回撤，系統已主動上鎖，禁止任何新的買賣委託。
- **處置**：
  1. 檢查 CMoney 帳戶是否為真實虧損，或只是資料錯誤/除權息導致的帳面落差。
  2. 若為真實系統性崩盤，手動登入 CMoney 清倉或避險。
  3. 若為資料錯誤（需解除斷路器）：
     - 編輯 `capital_flow_analysis/data/live_equity_curve.json`，將錯誤的 PnL 資料修正或刪除。
     - 重新手動執行 `daily_trade_runner.py`。

### 8.3 RPA 單點故障無備援 (Single Point of Failure)
- **情境**：CMoney 網站改版或驗證碼阻擋，導致 `cmoney_rpa.py` 無法登入或下單。
- **問題**：系統完全無法送單。
- **處置**：
  1. 打開 `trade_guard_diff.json`，裡面會列出今日應買賣的精確張數與標的。
  2. 手動登入 CMoney 網頁版，依據 diff 內容掛單。
  3. 待開發者修復 RPA 後，再恢復排程。


# 全局風險登記冊（Risk Register）

> **建立日期**：2026-06-11 · **來源**：系統性評估（研發者 + Antigravity 聯合稽核）  
> **目的**：記錄演算法、資料品質、生產部署、市場微觀結構、研究方法論、SL 設計的已知風險，作為每次迭代前的強制確認項目。  
> **維護規則**：每次上線前、每次 M2-promotion 前必須重讀本文件。關閉某項風險前必須有實驗/程式碼依據。

---

## 一、資料品質風險

### R-D1 🔴 股息未調整（高優先）

**問題**：`data_pipeline/core.py` L22 呼叫 `yf.download(ticker, ...)` 時**沒有傳入 `auto_adjust=True`**。  
yfinance `0.2.x` 之後預設 `auto_adjust=True`，但舊版或明確不傳時 Close = unadjusted price，除息日出現人工跳空負報酬。

**影響**：
- `log_return = log(Close_t / Close_{t-1})` 在除息日記錄大額負報酬
- RL 模型學到「除息日 = 賣出訊號」，而非真實 alpha 訊號
- 台積電等重倉股每年多次除息，訓練樣本被系統性汙染

**狀態**：✅ **已確認安全**（2026-06-11 驗證）  
**驗證結果**：yfinance `1.4.1` 的 `yf.download()` 預設 `auto_adjust=True`（即 `Close` 欄位已是 adjusted close）。`log_return` 計算基準正確。  
**行動項**：在 `data_pipeline/core.py` L22 加入明確的 `auto_adjust=True` 參數（防禦性文件化），避免未來 yfinance 版本變更破壞行為。

---

### R-D2 🟡 SL 標籤的 universe-median 在熊市失效（中優先）

**問題**：`SUPERVISED_LEARNING_PLAN.md §3.1` 的截面去均值標籤：
```
Target_5d_Cross_Demean = raw_5d_return_i - universe_median_5d
```
45 檔全是台灣科技/電子股，相關係數在熊市可能超過 0.9。  
熊市時整體 universe 同向下跌，去均值後「跌最少」的股票拿到正標籤，模型在熊市期間傾向繼續持股而非轉現金，與 MDD Gate 目標方向衝突。

**影響**：SL 策略在熊市期間無法有效發出「轉現金」訊號，MDD 可能被低估。

**狀態**：❌ 未處理（已知設計取捨，需監控）  
**處置建議**：
- 監控各期 avg_cash_weight；熊市 OOS 期若 avg_cash < 10%，視為標籤失效徵兆
- 備選方案：在標籤計算後加入「全市場同步下跌時強制發放現金訊號」的規則層

---

### R-D3 🟢 滾動正規化的前瞻洩漏風險（低優先，需確認）

**問題**：`data_pipeline/core.py` 的滾動指標（rolling mean/std）使用 `.rolling(window)` 逐點計算，未見全局 StandardScaler fit on entire history，但未明確排除。

**狀態**：⚠️ 未確認（現有程式碼傾向安全，但需明確文件化）  
**處置建議**：確認所有特徵正規化均為純滾動（expanding 或 rolling），無全局 fit，並在文件中明確說明。

---

### R-D4 🟡 Walk-Forward OOS 視窗太短（中優先）

**問題**：4 個半年期（2024H2 → 2026H1）全落在 AI 熱潮牛市週期，不含 2022 年完整熊市（該段在訓練集，未在 OOS）。

**影響**：Promotion Gate 的 worst-case MDD 可能低估完整牛熊週期的真實下行風險。

**狀態**：❌ 結構性限制，暫無解法  
**備注**：需在研究報告中明確披露此侷限性。

---

## 二、生產部署風險

### R-P1 🔴 T+1 執行的競態條件（高優先）

**問題**：`LIVE_OPS.md §2` 流程：Step 0 執行前日 Pending BUY → Step 1 產生新訊號 → Step 2 執行當日 SELL。  
若 Step 0 成功執行 BUY，但 Step 1 失敗（evaluate_portfolio 異常），當日沒有新訊號，不執行 SELL。  
結果：持有昨日買入倉位，但今日沒有賣出訊號，倉位無法清掉。

**狀態**：❌ 無文件化處理流程  
**處置建議**：
- 在 `daily_trade_runner.py` 中加入 Step 1 失敗後的保護機制：若訊號無法產生，發出 CRITICAL alert，並選擇：(a) 重新嘗試 evaluate_portfolio、(b) 發出「維持現有倉位」的保守訊號、(c) 人工介入
- 文件化此競態條件的手動 SOP

---

### R-P2 🔴 每日 P&L 斷路器缺失（高優先）

**問題**：`trade_guard.py` 只檢查單股權重（≤35%）和總曝險（≤100%）。缺少：
- 單日虧損超過 X% 自動暫停
- 連續 N 日虧損暫停
- 帳戶淨值低於某水位停止

**影響**：策略嚴重失效時，系統會繼續自動下單直到人工介入。

**狀態**：❌ 未實作  
**處置建議**：在 `trade_guard.py` 的 `evaluate_risk_limits()` 中新增：
```python
# 建議增加
if daily_pnl_pct < -SETTINGS.risk_limits.max_daily_loss_pct:
    reasons.append("daily loss circuit breaker triggered")
if drawdown_from_peak > SETTINGS.risk_limits.max_rolling_drawdown:
    reasons.append("rolling drawdown limit exceeded")
```

---

### R-P3 🔴 RPA 單點故障無備援（高優先）

**問題**：整個下單系統完全依賴 `cmoney_rpa.py` 網頁 RPA，無備援機制：
- CMoney 網站維護/UI 改版/登入異常 → 所有單都不送
- Cookie 過期需手動操作
- `SIGNAL_TTL_SECONDS=900`：RPA 失敗重跑超時後，trade_guard 中止當日賣單

**狀態**：❌ 無 fallback 文件，無手動下單 SOP  
**處置建議**：
- 文件化「RPA 失敗時的手動下單 SOP」（最低限度：電話/人工確認當日持倉，再手動執行重要的賣單）
- 增加 RPA 重試機制（最多 3 次，每次間隔 30 秒）

---

### R-P4 🟡 通知機制單一（中優先）

**問題**：唯一通知管道為 Email（`send_notification()`）。Email 故障/垃圾桶/網路問題讓運維無感知。

**狀態**：❌ 無備用通知  
**處置建議**：增加 Line Notify 或 Telegram 作為第二通知管道（低成本實作）。

---

## 三、市場微觀結構風險

### R-M1 🟡 流動性未建模（中優先）

**問題**：`trading_env.py` 使用固定 `SLIPPAGE_RATE = 0.001`（0.1%），對流動性差的小型股過於樂觀。集中持倉時（如 seed43 曾出現單檔 99%）大單進出可能造成 1–3% 滑價。

**狀態**：❌ 已知取捨，需在上線前重新評估  
**處置建議**：針對 45 檔股票中日均成交量低於閾值的股票，套用更高的 SLIPPAGE_RATE（分層費率）。

---

### R-M2 🟡 台灣漲跌停限制未建模（中優先）

**問題**：台股每日 ±10% 漲跌停。`_execute_trades` 假設所有交易都能在目標價格完全成交，在 AI 題材股熱炒日（漲停）明顯不現實。

**狀態**：❌ 未建模（研究階段可接受，上線前必須評估）

---

### R-M3 🟡 股票宇宙相關性過高（中優先）

**問題**：45 檔全是台灣科技/電子股（半導體供應鏈），熊市相關係數可能超過 0.9。Top-k=5 的「多元化配置」在此宇宙缺乏真實分散效果，Walk-Forward MDD 可能低估系統性風險。

**狀態**：❌ 結構性限制

---

## 四、研究方法論風險

### R-R1 🟡 多重測試統計效度問題（中優先）

**問題**：Gate 門檻（MDD ≤ 35%、Sortino ≥ 0.8）在 R1 設定，已被測試超過 9 輪（R1–R9）加上 SL 多個變體。每次失敗後調整 reward 再測試，Gate 實際已「隱性 in-sample」。

**影響**：若 r5 以 34.9% MDD 通過 Gate，統計顯著性遠低於「第一次測試」，上線後真實 OOS MDD 可能超過 35%。

**狀態**：⚠️ 已知風險，必須在研究報告中明確披露

---

### R-R2 🟡 3 個 seed 的統計可信度有限（中優先）

**問題**：Promotion 要求 worst-case across 3 seeds，但 3 個 seed 的樣本量不足以高信心保證第 4、5 個 seed 也通過。

**狀態**：⚠️ 資源限制下的務實取捨，需在報告中披露

---

### R-R3 🟡 基準選擇偏差（中優先）

**問題**：Reward 的 `_benchmark_top3_idx` = 「過去 20 天報酬前三的股票」（動量基準）。這是個低門檻基準，且可能造成 agent 追漲。`experiment_report.py` 的 baseline 應包括等權月再平衡，不只自訓動量。

**狀態**：❌ 待改善（不影響 Gate 判斷，但影響報告可信度）

---

## 五、SL 設計風險

### R-S1 🟡 Pooled 模型假設可交換性（中優先）

**問題**：`SUPERVISED_LEARNING_PLAN.md §2`：pooled LightGBM 假設市場特徵 → alpha 的映射函數在所有股票間相同，但台積電（30%+ 市值佔比）與小型 IC 設計廠的特徵敏感度截然不同。

**狀態**：⚠️ 已知設計取捨（簡化首版）  
**處置建議**：二版考慮加入市值分組特徵或分群 LightGBM。

---

### R-S2 🟡 反波動率加權在高波動期的反向效應（中優先）

**問題**：`RuleBasedAllocator` 使用反波動率加權 + Vol-target 18%。台股科技股暴跌時，所有持股的短期 vol 同時飆升 → 全部持股被 vol-target 機制縮減 → 在最差時點產生被動賣壓（助跌）。

**狀態**：⚠️ vol-target 的已知特性，文件未討論此反饋迴圈

---

## 六、文件與維護風險

### R-Doc1 🟢 ARCHITECTURE.md 文件漂移（低優先）

**問題**：`ARCHITECTURE.md §4` 仍寫「6 個帳戶特徵」，但 r5 後 `NUM_ACCOUNT_FEATURES = 9`；同節 `softmax_temp=0.5`，但 M1d 後已改為 1.0；`ENV_CONFIG_VERSION` 標記仍寫 `r4`（現為 `r5.1`）。

**狀態**：❌ 文件漂移（已標注，待修正）

---

### R-Doc2 🔴 關鍵人風險（高優先）

**問題**：整個系統由單一開發者維護，所有決策無外部確認機制（除 AI agent review）。自動交易觸發邊界條件時，無備用人可介入。

**狀態**：⚠️ 結構性風險，需在上線前文件化緊急停損 SOP

---

## 七、優先級總表

| ID | 類別 | 問題 | 優先級 | 狀態 |
|----|------|------|--------|------|
| R-P1 | 生產 | T+1 競態條件 | 🔴 高 | ❌ |
| R-P2 | 生產 | 每日 P&L 斷路器缺失 | 🔴 高 | ❌ |
| R-P3 | 生產 | RPA 單點故障無備援 | 🔴 高 | ❌ |
| R-D1 | 資料 | 股息未調整（需確認） | 🔴 高 | ⚠️ 待確認 |
| R-Doc2 | 維護 | 關鍵人風險 | 🔴 高 | ⚠️ |
| R-M1 | 市場 | 流動性未建模 | 🟡 中 | ❌ |
| R-R1 | 研究 | 多重測試統計效度 | 🟡 中 | ⚠️ 需披露 |
| R-D2 | 資料 | SL 標籤熊市失效 | 🟡 中 | ❌ |
| R-M2 | 市場 | 漲跌停未建模 | 🟡 中 | ❌ |
| R-R2 | 研究 | 只有 3 seeds | 🟡 中 | ⚠️ |
| R-D4 | 資料 | OOS 視窗太短 | 🟡 中 | ❌ |
| R-P4 | 生產 | 通知機制單一 | 🟡 中 | ❌ |
| R-M3 | 市場 | 宇宙相關性過高 | 🟡 中 | ⚠️ |
| R-S1 | SL | Pooled 模型假設 | 🟡 中 | ⚠️ |
| R-S2 | SL | Vol-target 反向效應 | 🟡 中 | ⚠️ |
| R-R3 | 研究 | 基準選擇偏差 | 🟡 中 | ❌ |
| R-D3 | 資料 | 正規化洩漏（待確認） | 🟢 低 | ⚠️ |
| R-Doc1 | 文件 | ARCHITECTURE 漂移 | 🟢 低 | ❌ |

---

## 八、研究階段 vs. 上線前 分類

**研究階段可接受（暫不處理）**：R-D4、R-M2、R-M3、R-S1、R-S2、R-R1、R-R2、R-R3

**M2-promotion 前必須確認**：R-D1（股息確認）、R-Doc1（文件對齊）

**上線前必須處理**：R-P1、R-P2、R-P3、R-P4、R-Doc2

---

*建立：2026-06-11 · 來源：系統性評估（研發者主導 + Antigravity 稽核）*  
*下次更新觸發條件：M2-promotion 開始前 / 每次上線前確認*


