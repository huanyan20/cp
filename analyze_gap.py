import numpy as np
import pandas as pd

# Load data
df = pd.read_csv('capital_flow_analysis/data/overnight_gap_features_1d.csv', index_col=0)

# Clean up infinite/na
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['target_2330_open_gap', 'tsm_adr_premium'])

print("=== ADR Premium Predictive Power ===")
corr_open = df['target_2330_open_gap'].corr(df['tsm_adr_premium'])
corr_full = df['target_2330_full_day'].corr(df['tsm_adr_premium'])
print(f"TSM ADR Premium vs 2330 Open Gap: {corr_open:.4f}")
print(f"TSM ADR Premium vs 2330 Full Day: {corr_full:.4f}")

print("\n=== SOX / Nasdaq Stress Predictive Power ===")
corr_sox_open = df['target_2330_open_gap'].corr(df['sox_nasdaq_spread'])
print(f"SOX-Nasdaq Spread vs 2330 Open Gap: {corr_sox_open:.4f}")

print("\n=== High Conviction Signals (Precision) ===")
# if ADR premium > 1%, what is the chance open gap > 0?
high_premium = df[df['tsm_adr_premium'] > 0.01]
if len(high_premium) > 0:
    prob_up = (high_premium['target_2330_open_gap'] > 0).mean()
    print(f"When ADR Premium > 1% (N={len(high_premium)}), Probability of Open Gap > 0: {prob_up:.2%}")

# if ADR premium < -1%, what is the chance open gap < 0?
low_premium = df[df['tsm_adr_premium'] < -0.01]
if len(low_premium) > 0:
    prob_down = (low_premium['target_2330_open_gap'] < 0).mean()
    print(f"When ADR Premium < -1% (N={len(low_premium)}), Probability of Open Gap < 0: {prob_down:.2%}")

print("\n=== VIX Risk vs Intraday ===")
corr_vix_intra = df['target_2330_intraday'].corr(df['vix_ret'])
print(f"VIX Return vs 2330 Intraday Return: {corr_vix_intra:.4f}")
