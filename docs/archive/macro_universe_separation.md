> Superseded by 2026-06-14 SL-first strategy.
> This document is retained as historical context only. It must not be treated as an active implementation queue unless explicitly updated after 2026-06-14.
# Macro Universe Separation

> 文件版本：v1.1
> 更新日期：2026-06-07
> **狀態**：✅ 已落地（見 `專案總覽.md` (`專案總覽.md`) §4.6）
> 目的：保持 RL baseline 可比較，同時允許 `capital_flow_analysis` 使用較大的全球資金流 universe。

## 1. 背景

本專案目前有兩條研究線：

- **RL Portfolio 主線**：以台股投資組合訓練、walk-forward、多 seed、PPO/SAC 對照與 RPA 訊號為主。
- **Capital Flow 輔助線**：位於 `capital_flow_analysis/`，觀察 ADR、SOX、VIX、FX、crypto、futures 等盤前資訊。

這兩條線的資料需求不同。若把所有 global macro ticker 都放進 RL baseline，會改變 observation space，使既有 PPO/SAC walk-forward 結果不再可比。因此 macro universe 必須分離。

## 2. 分離原則

### RL baseline 保持乾淨

RL 主線只使用與台股投資組合最直接相關、且過去實驗已採用的 macro：

```python
MACRO_TICKERS_RL = ["^TWII", "^IXIC", "USDTWD=X"]
```

這能保留 baseline 可比較性。

### Capital Flow 使用 extended universe

Capital flow 輔助線可使用更大的全球資金流 universe：

```python
MACRO_TICKERS_FLOW = [
    "^VIX",
    "^TNX",
    "BTC-USD",
    "ETH-USD",
    "NQ=F",
    "ES=F",
    "JPY=X",
    "DX-Y.NYB",
]
```

這些資料可用於：

- 盤前風控。
- open gap / gap fade 特徵工程。
- 宏觀觀察圖表。
- 未來新實驗組。

### 舊名稱保留相容

```python
MACRO_TICKERS = MACRO_TICKERS_RL
```

舊名稱保留只是為了相容，不應再被新程式主動 import。新程式應明確選擇 `MACRO_TICKERS_RL` 或 `MACRO_TICKERS_FLOW`。

## 3. 使用規則

| 模組 | 應使用 |
|:---|:---|
| `train_portfolio.py` | `MACRO_TICKERS_RL` |
| `walk_forward.py` | `MACRO_TICKERS_RL` |
| `evaluate_portfolio.py` | `MACRO_TICKERS_RL` |
| `data_loader.py` | 接收外部傳入的 `macro_tickers`，不強制固定 |
| `capital_flow_analysis/src/data_pipeline/global_macro_loader.py` | `MACRO_TICKERS_FLOW` |
| `capital_flow_analysis/src/data_pipeline/overnight_gap_features.py` | 自己定義 ADR/SOX/風險 ticker universe |

## 4. 未來整合方式

若要測試 extended macro 對 RL 是否有幫助，應建立明確實驗組，而不是悄悄修改 baseline。

建議命名：

- `macro_mode=rl`
- `macro_mode=extended`
- `flow_macro_enabled=True`
- `overnight_feature_path=capital_flow_analysis/data/overnight_gap_features_1d.csv`

評估時至少要輸出：

- observation shape。
- macro ticker list。
- overnight feature columns。
- walk-forward period。
- seed。
- Sortino、MDD、Total Return、Turnover、Cash behavior。

## 5. 不在此分離原則處理的事

- 不刪除 `capital_flow_analysis/` 的資料、圖表或報告。
- 不把 flow macro 自動接進 RL baseline。
- 不改 `data_loader.py` 的泛用參數設計。
- 不重跑長時間訓練。

## 6. 驗收條件

應符合：

- `MACRO_TICKERS == MACRO_TICKERS_RL`。
- RL 主線檔案不再 import 舊名稱 `MACRO_TICKERS`。
- Capital flow loader 使用 `MACRO_TICKERS_FLOW`。
- 新增 macro 實驗時，文件與 log 都能看出使用哪個 universe。
- 全測試通過。

## 7. 相關文件

- `RL_Trading_DQN_PPO_Guide.md`
- `docs_dir/trading_env_spec.md`
- `capital_flow_analysis/README.md`
- `capital_flow_analysis/docs/capital_flow_plan.md`
