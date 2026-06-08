# Capital Flow Research Plan

## 1. Purpose

`capital_flow_analysis` 的目標是替 CP 專案提供輔助資料與風控訊號，而不是取代主交易系統。它負責整理 ADR、SOX、Nasdaq、VIX、匯率與其他海外資料，協助判斷台股開盤前的風險與研究方向。

目前正式接入的 overnight features 採 Top 3 口徑：

- `tsm_adr_premium_chg`
- `tsm_adr_premium`
- `TSM_ret`

Top 8 或更多 macro features 仍屬研究候選，必須通過資料品質、walk-forward 與 ablation 驗證後才可升級為正式預設。

## 2. Data Health Rules

資料品質優先於模型複雜度：

- `overnight_gap_features_1d.csv` 若少於 60 rows，只能用於 smoke test。
- 少於 250 rows 時，不可宣稱穩定 alpha。
- 高缺值、stale data 或 corporate-action flagged rows 不應用於模型 promotion。
- 任何 feature 若只能靠大量補值存在，應先排除在正式特徵集之外。

## 3. Signal And Guard Semantics

Signal 與 Guard 必須分開理解：

- **Signal**：模型輸出的目標權重，例如 `signal.json`。
- **Guard**：交易執行層的風控閘門，例如 preopen macro check。

目前日常模型 `ppo_portfolio_full_stock_seed42.zip` 是 full-stock 模型，因此 `cash_weight = 0.0` 是正常結果。模型不會主動保留現金。

Preopen guard 使用 rolling z-score：

- `OK`：照一般流程執行。
- `WARN`：pending buys 減半。
- `CRITICAL`：跳過 pending buy orders，但允許 sell-only 流程。

`CRITICAL` 不代表模型主動清倉，也不代表 `signal.json` 會被清空；它代表執行端暫停 buy-side。

## 4. Research Principles

新增 macro / ADR feature 時，請遵守以下原則：

1. **先證明資料可靠**：確認資料來源、日期對齊、缺值率與 stale data。
2. **先做 walk-forward**：不使用單次切分或 in-sample 結果判定 alpha。
3. **先做 ablation**：新增 features 必須證明比 Top 3 baseline 更好。
4. **先小後大**：穩定小特徵集優先於寬而不穩的 feature set。

若加入大量 macro features 後導致 out-of-sample 表現下降，應退回 Top 3 特徵集。

## 5. Recommended Next Experiments

- **Top 3 baseline 固化**：以目前三個正式特徵建立可重複的 baseline 報告。
- **Top 8 ablation**：逐步加入 SOX、Nasdaq、VIX、DXY、JPY 等特徵，觀察是否真正改善 Sortino、MDD 與 turnover。
- **Guard impact study**：比較 `WARN` 減半買進與完全跳過買進的長期結果。
- **Cash-enabled model**：若希望模型主動避險，需另外訓練 cash-enabled policy，而不是期待 full-stock 模型產生現金部位。
- **Intraday validation**：未來可加入更細粒度的期貨或盤前資料，但必須先解決資料穩定性與時區對齊。
