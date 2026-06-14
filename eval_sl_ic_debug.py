import sys
sys.path.append('.')
from sl_pipeline.walk_forward_sl import build_period_plan, resolve_period
from sl_pipeline.signal_generator import SignalGenerator, SignalGeneratorConfig
from data_pipeline.universe_builder import get_universe_builder
from data_loader import fetch_multi_asset_data
from sl_pipeline.labels import build_labeled_panel, split_panel_by_date
import pandas as pd
from scipy.stats import spearmanr
import numpy as np

plan = build_period_plan()
builder = get_universe_builder('dynamic')
name = '2025H2' # Just test one period to be fast
period = resolve_period(name)
train_end = period['train_end']
test_end = period['test_end']
period_tickers = builder.build_universe(period['train_start'], top_n=45)

train_data = fetch_multi_asset_data(period_tickers, start_date=period['train_start'], end_date=train_end, macro_tickers=[])
test_fetch_start = str((pd.Timestamp(period['test_start']) - pd.Timedelta(days=90)).date())
test_data = fetch_multi_asset_data(period_tickers, start_date=test_fetch_start, end_date=test_end, macro_tickers=[])

config = SignalGeneratorConfig(horizon=10)
sg = SignalGenerator(config)
scores, summary = sg.fit_period(train_data, test_data, train_end=train_end, test_start=period['test_start'])

test_panel = build_labeled_panel(test_data, horizon=10, feature_cols=sg.feature_cols)
_, test_panel = split_panel_by_date(test_panel, train_end, period['test_start'])

preds = []
for i, row in test_panel.iterrows():
    ticker = row['ticker']
    date = row['date']
    if ticker in scores and date in scores[ticker].index:
        preds.append(scores[ticker].loc[date])
    else:
        preds.append(np.nan)
test_panel['pred'] = preds
test_panel = test_panel.dropna(subset=['pred'])

print("First few preds:")
print(test_panel[['date', 'ticker', 'target_10d_cross_demean', 'pred']].head(10))

print("Unique values in pred:", len(test_panel['pred'].unique()))

ic_vals = []
for date, df in test_panel.groupby('date'):
    if len(df) > 1:
        try:
            ic = float(spearmanr(df['pred'], df['target_10d_cross_demean'])[0])
            if not np.isnan(ic):
                ic_vals.append(ic)
        except:
            pass
avg_ic = np.mean(ic_vals) if ic_vals else np.nan
print(f"Period {name}: Avg Rank IC = {avg_ic:.4f}")
