# Milestone 3B: LightGBM/XGBoost 三分類選股模型結果

## IC-Filtered Features (32 features)

依照 Feature IC Dashboard (Milestone 3A) 篩選後，只保留 `|IC| > 0.02 @ 20d` 的黃金特徵進行訓練。


## Walk-Forward 回測績效

| Model    | Horizon   |   Seed | Total Return   | MDD   |   Sortino |   Sharpe |   Avg Turnover |
|:---------|:----------|-------:|:---------------|:------|----------:|---------:|---------------:|
| LIGHTGBM | 20d       |     42 | -25.4%         | 38.3% |     -0.67 |    -0.75 |          0.096 |


## 完整 Period 明細


### LIGHTGBM | Horizon=20d | Seed=42

| Period    | Return   | MDD   |   Sortino |   Sharpe |
|:----------|:---------|:------|----------:|---------:|
| 2022_BEAR | -18.3%   | 18.4% |     -1.77 |    -3.12 |
| 2024H2    | -14.1%   | 14.3% |     -1.78 |    -2.39 |
| 2025H1    | -5.2%    | 5.2%  |     -1.25 |    -2.51 |
| 2025H2    | -2.6%    | 13.3% |     -0.32 |    -0.25 |
| 2026H1    | 15.0%    | 13.6% |      2.05 |     1.66 |

