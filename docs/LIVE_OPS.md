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

- [ ] Promotion Gate 為 APPROVED（見 `docs/RESEARCH_PLAYBOOK.md`，目前 BLOCKED）
- [ ] `.env` 已設 `CMONEY_AID`、`CMONEY_USERNAME/PASSWORD`、`EMAIL_*`
- [ ] dry-run diff `risk_checks.passed=true`
- [ ] 盤前守衛 `level=OK`
- [ ] 最後才設 `ENABLE_LIVE_TRADING=true`

## 7. 相關文件

- `docs/ARCHITECTURE.md`：系統架構
- `docs/RESEARCH_PLAYBOOK.md`：研究與 Promotion Gate
- `capital_flow_analysis/README.md`：宏觀特徵與盤前守衛細節
- `教學文件.md`：完整逐模組教學
