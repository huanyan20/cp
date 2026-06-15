import sys
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sl_pipeline.backtest import simulate_period
from sl_pipeline.rule_based_allocator import RuleBasedAllocator, RuleBasedAllocatorConfig
from data_loader import fetch_multi_asset_data

logging.basicConfig(level=logging.INFO)

def main():
    scores_path = r"C:\Users\ggini\AppData\Local\Temp\sl_2022_BEAR_20260615_231859\scores_2022_BEAR_h10.csv"
    if not Path(scores_path).exists():
        print("Scores not found!")
        return
        
    scores = pd.read_csv(scores_path, index_col='date', parse_dates=['date'])
    tickers = scores.columns.tolist()
    
    tickers = scores.columns.tolist()
    
    # Load combined_test
    print("Loading test data...")
    combined_test = fetch_multi_asset_data(tickers, "2021-01-01", "2022-12-31")

    # Run simulate_period
    print("Running simulate_period...")
    allocator = RuleBasedAllocator(RuleBasedAllocatorConfig())
    
    # Needs to match what simulate_period expects for target_weights
    backtest = simulate_period(
        combined_test,
        scores,
        allocator,
        tickers,
        test_start="2022-01-01",
        test_end="2022-12-31"
    )
    
    # Analysis
    dates = list(backtest["state_history"].keys())
    state_history = backtest["state_history"]
    cash_weights = backtest["cash_weights"]
    daily_returns = backtest["daily_returns"]
    
    records = []
    for i, dt in enumerate(dates):
        if i < len(daily_returns):
            ret = daily_returns[i]
            records.append({
                "date": dt,
                "pnl": ret,
                "cash": cash_weights[i],
                "state": state_history.get(dt, "OK")
            })
            
    df = pd.DataFrame(records)
    print("\n=== PnL Decomposition ===")
    
    # Group by state
    summary = df.groupby("state").agg(
        days=("date", "count"),
        avg_pnl=("pnl", "mean"),
        win_rate=("pnl", lambda x: (x > 0).mean()),
        avg_cash=("cash", "mean")
    )
    print("\n1. Performance by Macro State:")
    print(summary)
    
    # Group by month
    df['month'] = df['date'].dt.to_period('M')
    monthly = df.groupby("month").agg(
        pnl=("pnl", lambda x: (1 + x).prod() - 1),
        avg_cash=("cash", "mean"),
        critical_days=("state", lambda x: (x == "CRITICAL").sum())
    )
    print("\n2. Monthly Performance:")
    print(monthly)
    
    # We can also find top losers from cumulative returns
    # But positions are intraday... Let's just look at the monthly first.

if __name__ == "__main__":
    main()
