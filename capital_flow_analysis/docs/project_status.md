# Capital Project Status Overview

> [!IMPORTANT]
> Capital 目前是 CP 專案的輔助研究與風控層，不是單獨決定買賣的主系統。它提供資料、研究報告與盤前 guard，實際交易仍由主流程與 RPA 執行端控制。

## 1. Capability Matrix

### 已驗證 / 可作為目前流程依據

- **Preopen Guard**：使用 YFinance 等來源抓取 SOX、Nasdaq、VIX、DXY、JPY、BTC 等資料，並以 rolling z-score 產生 `OK`、`WARN`、`CRITICAL` 狀態。
- **Top 3 Overnight Features**：目前正式穩定支援 `tsm_adr_premium_chg`、`tsm_adr_premium`、`TSM_ret`。
- **Dry-run RPA**：在 `ENABLE_LIVE_TRADING=false` 時，可驗證 signal、sell-only 與 pending buy 流程，不會送出實單。
- **RL Signal Generation**：可由 PPO 模型產生 `signal.json`，供後續執行端讀取。

### 尚未驗證 / 僅作為研究方向

- **Top 8 Overnight Features**：broader macro / ADR feature set 尚未完成充分 walk-forward 與 ablation，不列為正式預設。
- **Capital Gap Alpha**：gap model 報告可用來觀察特徵關係，但目前不可直接宣稱穩定 alpha。
- **Live Trading**：真實委託需要額外人工確認與風控設定，本文件不將其視為已驗證能力。

## 2. Data Health Thresholds

`overnight_gap_features_1d.csv` 的樣本數會直接影響研究可信度：

- **Rows < 60**：只能視為 smoke test，用來確認管線可跑通，不可解讀模型表現。
- **Rows < 250**：可以做初步研究，但不可宣稱穩定 alpha，也不應用來推進正式模型。
- **High missing rate features**：高缺值或高度依賴補值的欄位不得進入正式 Top features。

目前報告若顯示資料不足、缺值過高或 stale data，應先修資料來源，再談模型提升。

## 3. Signal vs. Guard

Signal 與 Guard 是兩個不同層次：

- **Signal**：模型觀點，描述目標持股權重。
- **Guard**：交易執行閘門，決定是否允許買進或降低買進強度。

當觸發 `CRITICAL` 時，真實行為是：

1. PPO 仍可產生 `signal.json`。
2. `daily_trade_runner.py` 讀取 guard 狀態後，跳過 pending buy orders。
3. Sell-only 流程仍可執行，用於減碼或風險處理。

因此 `CRITICAL` 不等於 signal 主動清倉，也不代表 `signal.json` 會被改寫成空倉；它代表執行端暫停 buy-side。

## 4. Model And Cash Weight

目前日常流程使用的模型是：

```text
ppo_portfolio_full_stock_seed42.zip
```

這是 full-stock 模型，不是 cash-enabled 模型。因此：

- `signal.json` 中 `cash_weight = 0.0` 是預期行為。
- 模型本身不會主動保留現金避險。
- 若需要降低風險，現階段依賴 preopen guard 的 `WARN` / `CRITICAL` 執行規則。

## 5. Current Recommendation

Capital 目前最適合扮演「盤前風險與研究輔助層」：

- 用它檢查海外半導體與宏觀風險。
- 用它產出研究報告，輔助判斷 feature 是否值得接入。
- 不要把單次 gap model 或小樣本結果直接當成實盤買賣規則。
