import pandas as pd
import numpy as np
from sl_pipeline.signal_generator import SignalGenerator, SignalGeneratorConfig
from data_pipeline.multi import fetch_multi_asset_data
from data_pipeline.utils import BASE_FEATURE_COLS
from data_pipeline.universe_builder import get_universe_builder

# Fetch universe for 2024H2
builder = get_universe_builder("dynamic")
tickers = builder.build_universe("2024-07-01", top_n=45)

train_data = fetch_multi_asset_data(
    tickers=tickers,
    start_date="2020-01-01",
    end_date="2024-06-30",
    macro_tickers=["^TWII", "^IXIC", "USDTWD=X"]
)

test_data = fetch_multi_asset_data(
    tickers=tickers,
    start_date="2024-04-01",
    end_date="2024-12-31",
    macro_tickers=["^TWII", "^IXIC", "USDTWD=X"]
)

gen = SignalGenerator(SignalGeneratorConfig(horizon=10))
scores, summary = gen.fit_period(
    train_data,
    test_data,
    train_end="2024-06-30",
    test_start="2024-07-01"
)

print("Top 10 Feature Importances:")
for k, v in summary.feature_importance_top10.items():
    print(f"{k}: {v}")

# Analyze test panel
test_panel = gen._prepare_features(
    gen._build_panel({**train_data, **test_data}, "test")
)
test_panel = test_panel[test_panel["date"] >= "2024-07-01"]
scored = gen.predict(test_panel)
test_panel["score"] = scored.values

# Check correlation of score with some features
features_to_check = ["Close_norm", "RSI_norm", "MACD_norm", "return_1d", "vol_20d", "beta_60d"]
for f in features_to_check:
    if f in test_panel.columns:
        corr = test_panel["score"].corr(test_panel[f])
        print(f"Corr(score, {f}): {corr:.3f}")
