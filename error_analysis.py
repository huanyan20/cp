from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

# Tech 30 tickers
TICKERS_TECH_EXPANDED = [
    "2330.TW", "2454.TW", "2303.TW", "2408.TW", "2379.TW",
    "3034.TW", "3443.TW", "3661.TW", "5269.TW", "3529.TWO",
    "8299.TWO", "5347.TWO", "6488.TWO", "5483.TWO", "6415.TW",
    "8016.TW", "3711.TW", "2317.TW", "2382.TW", "3231.TW",
    "2356.TW", "6669.TW", "2324.TW", "2357.TW", "2376.TW",
    "2308.TW", "6409.TW", "3017.TW", "3324.TWO", "3653.TW"
]

def analyze_momentum_breakdown(start_date, end_date, period_name):
    print(f"\n{'='*50}")
    print(f"Error Analysis for {period_name} ({start_date} ~ {end_date})")
    print(f"{'='*50}")

    # To calculate 20d momentum, we need 30 days of prior data
    start_dt = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=40)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=10) # for forward return
    
    # Download close prices
    data = yf.download(TICKERS_TECH_EXPANDED, start=start_dt.strftime("%Y-%m-%d"), end=end_dt.strftime("%Y-%m-%d"), progress=False)['Close']
    
    # Flatten MultiIndex if necessary
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
        
    all_data = []
    
    for ticker in data.columns:
        df = pd.DataFrame({'Close': data[ticker]})
        df = df.dropna()
        if len(df) < 30:
            continue
            
        # Calculate 20-day past return (momentum)
        df['mom_20d'] = df['Close'].pct_change(20)
        
        # Calculate 1-day forward return
        df['fwd_ret_1d'] = df['Close'].pct_change(1).shift(-1)
        
        # Calculate 5-day forward return
        df['fwd_ret_5d'] = df['Close'].pct_change(5).shift(-5)
        
        df_clean = df.loc[start_date:end_date].dropna(subset=['mom_20d', 'fwd_ret_1d', 'fwd_ret_5d'])
        df_clean['ticker'] = ticker
        
        all_data.append(df_clean[['ticker', 'mom_20d', 'fwd_ret_1d', 'fwd_ret_5d']])
        
    if not all_data:
        print("No enough data for analysis.")
        return
        
    combined_df = pd.concat(all_data)
    
    # Calculate cross-sectional correlation for each day (Rank IC)
    ic_1d_list = []
    ic_5d_list = []
    
    for _date, group in combined_df.groupby(combined_df.index):
        if len(group) < 10:
            continue
        ic_1d = group['mom_20d'].corr(group['fwd_ret_1d'], method='spearman')
        ic_5d = group['mom_20d'].corr(group['fwd_ret_5d'], method='spearman')
        ic_1d_list.append(ic_1d)
        ic_5d_list.append(ic_5d)
        
    avg_ic_1d = np.nanmean(ic_1d_list)
    avg_ic_5d = np.nanmean(ic_5d_list)
    
    win_rate_1d = np.nanmean(np.array(ic_1d_list) > 0)
    win_rate_5d = np.nanmean(np.array(ic_5d_list) > 0)
    
    print("Momentum Factor Analysis (Rank IC):")
    print(f"  - Avg 1-day Rank IC: {avg_ic_1d:.4f} (Win rate >0: {win_rate_1d:.2%})")
    print(f"  - Avg 5-day Rank IC: {avg_ic_5d:.4f} (Win rate >0: {win_rate_5d:.2%})")
    
    if avg_ic_1d < 0:
        print("  => [Warning] 1-day Momentum factor is INVERTED (mean reversion regime).")
    if avg_ic_5d < 0:
        print("  => [Warning] 5-day Momentum factor is INVERTED (mean reversion regime).")
        
    # Analyze market trend during this period (using 2330.TW as proxy)
    if '2330.TW' in data.columns:
        tsm_df = data['2330.TW'].loc[start_date:end_date].dropna()
        if len(tsm_df) > 0:
            period_ret = (tsm_df.iloc[-1] / tsm_df.iloc[0]) - 1
            max_drawdown = (tsm_df / tsm_df.cummax() - 1).min()
            print("\nMarket (2330.TW Proxy) Performance:")
            print(f"  - Period Return: {period_ret:.2%}")
            print(f"  - Max Drawdown: {max_drawdown:.2%}")

    print("\n")

if __name__ == "__main__":
    analyze_momentum_breakdown("2024-07-01", "2024-12-31", "2024H2")
    analyze_momentum_breakdown("2025-01-01", "2025-06-30", "2025H1")
    # For comparison, let's also analyze a good period
    analyze_momentum_breakdown("2026-01-01", "2026-06-06", "2026H1 (Good Period)")
