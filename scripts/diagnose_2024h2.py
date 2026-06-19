"""Diagnose 2024H2 model failure.

Checks:
1. TWII market behavior during 2024H2 (Jul-Dec 2024)
2. Monthly IC breakdown (score vs actual 20d return)
3. Top-5 selection vs universe average
4. Average score distribution
"""
import sys
sys.path.append(".")
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

from data_loader import fetch_multi_asset_data
from data_pipeline.universe_builder import get_universe_builder
from sl_pipeline.labels import forward_log_return_t1

PRED_DIR = Path("reports/milestone_3b/lightgbm_20d_seed42/2024H2")
pred_df = pd.read_csv(PRED_DIR / "predictions_2024H2_h20.csv", parse_dates=["date"])
print(f"Predictions shape: {pred_df.shape}")
print(f"Date range: {pred_df['date'].min()} ~ {pred_df['date'].max()}")
print(f"Score stats: mean={pred_df['score'].mean():.4f}  std={pred_df['score'].std():.4f}")
print()

# Load TWII data to see what market was doing
import yfinance as yf
twii = yf.download("^TWII", start="2024-01-01", end="2025-01-01", progress=False, auto_adjust=True)
if isinstance(twii.columns, pd.MultiIndex):
    twii.columns = twii.columns.get_level_values(0)
twii_close = twii["Close"]
twii_rets = twii_close.pct_change()

print("=== TWII Monthly Performance 2024 ===")
monthly_twii = twii_rets.resample("ME").apply(lambda x: (1 + x).prod() - 1)
for dt, r in monthly_twii.items():
    if 2024 <= dt.year <= 2024:
        marker = " <-- CRASH" if r < -0.05 else (" <-- STRONG" if r > 0.05 else "")
        print(f"  {dt.strftime('%Y-%m')}: {r*100:+.1f}%{marker}")
print()

# Load actual returns for 2024H2
builder = get_universe_builder("dynamic")
tickers = builder.build_universe("2020-01-01", top_n=45)
data = fetch_multi_asset_data(
    tickers=tickers,
    start_date="2023-07-01",   # 6m before test to allow lookback
    end_date="2024-12-31",
    macro_tickers=[]
)

frames = []
for ticker, df in data.items():
    actual = forward_log_return_t1(df["log_return"], 20)
    tmp = pd.DataFrame({"date": df.index, "ticker": ticker, "actual_ret_20d": actual})
    frames.append(tmp)
actual_df = pd.concat(frames, ignore_index=True)

merged = pred_df[["date","ticker","score"]].merge(actual_df, on=["date","ticker"], how="inner").dropna()
merged = merged[merged["date"].dt.year == 2024]
print(f"Merged rows (2024 only): {len(merged)}")

# Monthly IC
merged["ym"] = merged["date"].dt.to_period("M")
monthly_ic = []
for ym, g in merged.groupby("ym"):
    if len(g) < 50:
        continue
    sc = g["score"].rank(pct=True)
    ar = g["actual_ret_20d"].rank(pct=True)
    ic = sc.corr(ar)
    monthly_ic.append({"month": str(ym), "ic": ic, "n": len(g)})

monthly_df = pd.DataFrame(monthly_ic)
print()
print("=== MONTHLY IC: 2024H2 OOS ===")
print(monthly_df.to_string(index=False))
print(f"\nMean IC: {monthly_df['ic'].mean():.4f}  Std: {monthly_df['ic'].std():.4f}  %positive: {100*(monthly_df['ic']>0).mean():.0f}%")

# Top-K vs universe in 2024H2
oos = merged[merged["date"] >= pd.Timestamp("2024-07-01")]
for k in [3, 5, 10]:
    top_k = oos.groupby("date").apply(lambda g: g.nlargest(k, "score")["actual_ret_20d"].mean()).mean()
    universe_avg = oos.groupby("date")["actual_ret_20d"].mean().mean()
    print(f"\nK={k}: Top-K 20d ret={top_k*100:.2f}%  Universe avg={universe_avg*100:.2f}%  Alpha={((top_k-universe_avg)*100):.2f}pp")

# Score distribution per month
print()
print("=== Score Distribution by Month (2024H2 OOS) ===")
oos["month"] = oos["date"].dt.to_period("M")
for month, g in oos.groupby("month"):
    print(f"  {month}: score mean={g['score'].mean():.3f} std={g['score'].std():.3f}  n_neg={100*(g['score']<0).mean():.0f}%")
