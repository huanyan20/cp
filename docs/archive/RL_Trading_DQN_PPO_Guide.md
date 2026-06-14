> Superseded by 2026-06-14 SL-first strategy.
> This document is retained as historical context only. It must not be treated as an active implementation queue unless explicitly updated after 2026-06-14.
# CP 台股科技股 RL Trading Guide

> 文件版本：v10.1
> 更新日期：2026-06-07
> 適用範圍：45 檔科技台股投資組合、PPO/SAC、動態現金避險、Capital Flow 輔助特徵、CMoney RPA

## 1. 專案定位

本專案是一套台股科技股投資組合研究與自動化交易系統。主線負責資料下載、特徵工程、強化學習訓練、walk-forward 驗證與訊號產生；`capital_flow_analysis/` 作為輔助模組，提供盤前全球資金流、ADR/SOX 特徵與 pre-open 風控。

系統分工：

```text
stock_universe.py
  -> 定義股池、板塊、RL macro universe 與 capital flow macro universe

data_loader.py
  -> 下載台股與 macro 資料，建立技術、動能、跨股、板塊與 overnight 特徵

trading_env.py
  -> TaiwanStockEnv，封裝 observation、action、reward、交易成本與現金節點

train_portfolio.py / walk_forward.py
  -> PPO/SAC 訓練、多 seed、walk-forward 驗證

evaluate_portfolio.py
  -> 評估模型或 momentum baseline，產出 signal.json

capital_flow_analysis/
  -> 產生 ADR/SOX/宏觀隔夜特徵與 preopen guard

daily_trade_runner.py / auto_login.py / cmoney_rpa.py
  -> 每日流程、登入、RPA 下單與 T+1 pending buy
```

## 2. 目前策略狀態

目前最值得追蹤的策略線：

- **PPO + cash enabled**：walk-forward 中 Sortino 排名最佳，但目前最佳組 seed 數不足，仍需擴大驗證。
- **Momentum 60D Top-1**：報酬高但集中度與最大回撤也高，適合作為 benchmark，不宜單獨視為穩健生產策略。
- **Capital Flow open gap model**：ADR 類特徵對 2330 開盤 gap 有明顯幫助，適合作為盤前輔助特徵與風控提示。

重要原則：

- 不只看 total return，排序優先看 OOS Sortino，再看 Max Drawdown，最後才看 Total Return。
- cash enabled 模型必須檢查現金比例是否真的動態調整，而不是學到固定持有少量現金。
- `capital_flow_analysis` 不直接下單，只提供特徵、報告與 guard 狀態。

## 3. 股池與 Macro Universe

主要股池定義於 `stock_universe.py`：

- `TICKERS_TECH_EXPANDED`：45 檔科技台股。
- `SECTOR_GROUPS`：6 個板塊。
- `MACRO_TICKERS_RL`：RL baseline 使用的宏觀資料。
- `MACRO_TICKERS_FLOW`：capital flow 輔助線使用的全球資金流資料。

分離原則：

- RL baseline 使用 `^TWII`、`^IXIC`、`USDTWD=X`，保持 observation space 可比較。
- VIX、TNX、BTC、ETH、NQ futures、ES futures、JPY、DXY 等資料留給 capital flow 或顯式新實驗組。
- 若要把 extended macro 接入 RL，應透過明確參數命名，例如 `overnight_feature_path` 或 `macro_mode=extended`。

## 4. Observation 設計

每檔股票 observation 由兩部分組成：

- 市場視窗特徵：技術指標、動能、跨股、板塊、macro、可選 overnight 特徵。
- 帳戶狀態特徵：現金比例、累計報酬、drawdown、持倉、未實現損益、持有天數。

主要特徵群：

| 特徵群 | 內容 |
|:---|:---|
| 個股基礎技術特徵 | OHLCV 正規化、RSI、MACD、BB、ADX、DMI、ATR、Stochastic、OBV、MFI、log return |
| 動能特徵 | `mom_60d`、`ma60_bias` |
| 跨股特徵 | peer log return、20 日 rolling correlation、relative strength |
| 板塊特徵 | `sector_flow` |
| RL macro | `^TWII`、`^IXIC`、`USDTWD=X` 的特徵 |
| optional overnight | ADR/SOX/VIX/DXY/JPY 等 capital flow 特徵 |

`data_loader.py` 會根據實際欄位數動態決定 observation 維度，因此新增特徵後舊模型通常無法直接載入，必須重新訓練。

## 5. Action 與動態現金避險

v9 起 action space 加入 `CASH` 節點，shape 為 `N_stocks + 1`。

處理流程：

```text
raw logits
  -> subtract max for numerical stability
  -> softmax(action / T)
  -> Top-K mask
  -> normalize to sum = 1
  -> target stock weights + cash weight
```

意義：

- 股票節點代表 long-only 權重。
- `CASH` 節點代表保留現金。
- 若模型在風險升高時提高 cash logit，Top-K 會自然降低股票曝險。

## 6. Reward 設計

核心 reward 使用 softsign 壓縮，保留排序性並避免硬截斷：

```python
def _softsign(x: float) -> float:
    return x / (1.0 + abs(x))
```

主要分量：

- return component：當步 portfolio log return。
- sortino component：下行風險調整後報酬。
- benchmark component：相對 momentum top-3 benchmark 的超額報酬。
- cost penalty：交易成本懲罰。
- drawdown penalty：超過緩衝後的回撤懲罰（R4: `LAMBDA_DRAWDOWN=0.8`，緩衝 3%）。
- regime penalty：DD>8% 時對高股票曝險額外懲罰（R4: 門檻 8%，係數 1.0）。
- defensive cash bonus：DD>8% 且 `enable_cash_action=True` 時，持有現金獲正向獎勵（R4: `LAMBDA_CASH_DEFENSIVE=0.2`）。

評估時需要同時看：

- Sortino 是否改善。
- Max Drawdown 是否可接受。
- turnover 是否過高。
- cash behavior 是否真的有動態避險效果。

## 7. 演算法與特徵提取器

支援演算法：

- PPO：目前主線，適合 on-policy 長訓練與 walk-forward。
- SAC：已加入支援，使用自動 buffer size 防止 Windows 記憶體溢出。

特徵提取器：

- `GnnFeatureExtractor`：股票節點 embedding + multi-head attention。
- `TemporalGnnFeatureExtractor`：加入時間序列處理，適合測試 overnight/capital flow 等跨日特徵。

## 8. Capital Flow 輔助模組

`capital_flow_analysis/` 不是獨立交易策略，而是盤前輔助層。

主要用途：

- 產生 `overnight_gap_features_1d.csv`。
- 評估 open gap、intraday、gap fade 特徵組。
- 提供 `preopen_macro_check.json` 給 `daily_trade_runner.py`。
- 將 Top 3 overnight 特徵接入 RL observation。

常用指令：

```bash
python capital_flow_analysis/src/data_pipeline/overnight_gap_features.py --start 2024-01-01 --report
python capital_flow_analysis/src/modeling/evaluate_gap_model.py --target open_gap
python capital_flow_analysis/src/modeling/evaluate_gap_model.py --target gap_fade
python capital_flow_analysis/src/monitoring/preopen_macro_check.py
```

目前判讀：

- ADR 類特徵對 `open_gap` 最有效。
- `gap_fade` 還偏弱，先當風險提示，不宜直接下單。
- preopen guard 應保持保守，資料缺失時至少要降級為 WARN。

## 9. 評估與 Walk-Forward

`walk_forward.py` 使用 expanding window：

```text
train: 2020-01-01 -> split end
test: next half-year period
```

支援：

- `--algo ppo` / `--algo sac`
- 多 seed，例如 `--seeds 42,43,44`
- 分期 metrics 與整體 OOS metrics

產出：

- `results_dir/metrics_{algo}_wf_seed{seed}.json`
- `results_dir/walk_forward_{algo}_seed{seed}.png`
- `experiment_report.md`

整理報告：

```bash
python experiment_report.py
```

排序原則：

1. OOS Sortino 高者優先。
2. Max Drawdown 低者優先。
3. Total Return 高者優先。

## 10. CMoney RPA 與每日流程

每日交易由 `daily_trade_runner.py` 編排：

```text
preopen macro guard
  -> auto_login.py
  -> pending buys unless guard is CRITICAL
  -> evaluate_portfolio.py
  -> cmoney_rpa.py --sell-only
```

RPA 特性：

- 自動登入與 cookie 更新。
- 自動抓取多個子帳戶 AID。
- 只處理現股庫存。
- T+1 pending buy 機制。
- `executed_signals.json` 防止重複下單。
- 預設 dry-run，需 `ENABLE_LIVE_TRADING=true` 才會送出實際委託。

Windows 工作排程建議：

- 程式或指令碼：`C:\Users\ggini\Desktop\cp\daily_trade.bat`
- 開始位置：`C:\Users\ggini\Desktop\cp`
- 時間依流程分工設定，盤前 pending buy 與收盤後評估不應混在同一個模糊時段。

## 11. 常用指令

測試：

```bash
python -m unittest tests.test_capital_flow_v2
python -m unittest discover -s tests
```

訓練：

```bash
python train_portfolio.py
python train_portfolio.py --overnight-feature-path capital_flow_analysis/data/overnight_gap_features_1d.csv
```

評估：

```bash
python evaluate_portfolio.py
python experiment_report.py
```

每日流程：

```bash
python daily_trade_runner.py
```

## 12. 文件索引

- `RL_Trading_DQN_PPO_Guide.md`：主系統入口指南。
- `docs_dir/trading_env_spec.md`：`TaiwanStockEnv` 規格。
- `macro_universe_separation.md`：RL macro 與 capital flow macro 分離原則。
- `experiment_report.md`：最新 walk-forward 實驗摘要。
- `capital_flow_analysis/README.md`：capital flow 模組使用說明。
- `capital_flow_analysis/docs/capital_flow_plan.md`：capital flow 設計與後續建議。
- `capital_flow_analysis/reports/*.md`：capital flow 特徵與模型評估報告。

## 13. 下一步

優先事項：

1. 將 preopen guard 落實為 `OK/WARN/CRITICAL`，並讓 WARN 可降低 pending buy 權重。
2. 補齊 PPO cash enabled 至至少 3 個 seeds，確認 Sortino、MDD 與 cash behavior 穩定。
3. 建立簡單且可解釋的 open gap baseline，例如只用 `tsm_adr_premium_chg`。
4. 對 gap fade 做 threshold calibration，不要固定使用 0.5。
5. 將 report 產出的 metrics JSON 化，方便 dashboard 或每日流程讀取。
