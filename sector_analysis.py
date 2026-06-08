import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stock_universe import TICKER_NAMES, get_ticker_sector


def analyze_sectors(trades_file="results_dir/trades_ppo_eval.csv"):
    if not os.path.exists(trades_file):
        print(f"Error: {trades_file} not found.")
        return

    df = pd.read_csv(trades_file)
    # date column might just be step integer or string, let's use the whole df
    if 'date' in df.columns:
        # Just in case we want to rename it
        pass
    else:
        print("Columns:", df.columns)
        return
    
    # Map sector and name (ensure column name is lowercase 'ticker')
    df['Sector'] = df['ticker'].apply(get_ticker_sector)
    df['Name'] = df['ticker'].map(TICKER_NAMES).fillna("Unknown")

    # Group by Action and Sector
    buys = df[df['trade_type'] == 'BUY'].copy()
    sells = df[df['trade_type'] == 'SELL'].copy()
    
    # 1. Trading Volume by Sector
    buys['Volume_TWD'] = buys['trade_amount_twd']
    sells['Volume_TWD'] = sells['trade_amount_twd']
    
    sector_buys = buys.groupby('Sector')['Volume_TWD'].sum().sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    sector_buys.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Total Buy Volume by Sector (2024H2 - 2025H1)')
    plt.ylabel('Buy Volume (TWD)')
    plt.xlabel('Sector')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('results_dir/sector_buy_volume.png')
    plt.close()
    print("[V] Saved: results_dir/sector_buy_volume.png")
    
    # 2. Sector Holding Count over time
    # We want to see if the model is concentrated in one sector
    # We can calculate net position size in TWD per sector over time
    # Instead of perfectly tracking PnL, we can just look at Net Buy Volume per step
    df['Net_Volume'] = np.where(df['trade_type'] == 'BUY', df['trade_amount_twd'], -df['trade_amount_twd'])
    daily_sector_net = df.groupby(['step', 'Sector'])['Net_Volume'].sum().reset_index()
    
    # Cumulative net position (approximation of exposure)
    daily_sector_net = daily_sector_net.pivot(index='step', columns='Sector', values='Net_Volume').fillna(0).cumsum()
    
    plt.figure(figsize=(12, 6))
    for col in daily_sector_net.columns:
        plt.plot(daily_sector_net.index, daily_sector_net[col], label=col, alpha=0.8)
    
    plt.axhline(0, color='black', linewidth=1, linestyle='--')
    plt.title('Approximated Cumulative Net Position by Sector (2024H2 - 2025H1)')
    plt.ylabel('Net Position (TWD)')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig('results_dir/sector_net_position.png')
    plt.close()
    print("[V] Saved: results_dir/sector_net_position.png")
    
    # 3. Analyze Losers: Which stocks caused the most turnover/losses?
    # Let's count the number of trades per stock
    trade_counts = df.groupby(['ticker', 'Name', 'Sector']).size().reset_index(name='Trade_Count')
    trade_counts = trade_counts.sort_values(by='Trade_Count', ascending=False).head(15)
    print("\n=== Top 15 Most Traded Stocks (Over-trading culprits) ===")
    print(trade_counts.to_string(index=False))

    # Output total trade percentage by sector
    total_trades = len(df)
    sector_trade_counts = df.groupby('Sector').size() / total_trades * 100
    sector_trade_counts = sector_trade_counts.sort_values(ascending=False)
    print("\n=== Percentage of Total Trades by Sector ===")
    for sector, pct in sector_trade_counts.items():
        print(f"{sector}: {pct:.1f}%")

if __name__ == "__main__":
    analyze_sectors()
