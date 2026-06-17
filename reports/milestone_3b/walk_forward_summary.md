# Milestone 3B: LightGBM/XGBoost 三分類選股模型結果

## IC-Filtered Features (32 features)

依照 Feature IC Dashboard (Milestone 3A) 篩選後，只保留 `|IC| > 0.02 @ 20d` 的黃金特徵進行訓練。


## Walk-Forward 回測績效

| Model    | Horizon   |   Seed | Total Return   | MDD   |   Sortino |   Sharpe |   Avg Turnover |
|:---------|:----------|-------:|:---------------|:------|----------:|---------:|---------------:|
| LIGHTGBM | 10d       |     42 | -30.5%         | 36.2% |     -0.76 |    -0.79 |          0.123 |


## 完整 Period 明細


### LIGHTGBM | Horizon=10d | Seed=42

| Period    | Return   | MDD   |   Sortino |   Sharpe |
|:----------|:---------|:------|----------:|---------:|
| 2022_BEAR | -13.1%   | 13.1% |     -1.25 |    -2.1  |
| 2024H2    | -13.6%   | 14.9% |     -2.63 |    -2.81 |
| 2025H1    | -2.5%    | 5.0%  |     -0.91 |    -0.85 |
| 2025H2    | -12.6%   | 14.0% |     -1.6  |    -1.14 |
| 2026H1    | 8.6%     | 15.2% |      1.01 |     0.9  |

