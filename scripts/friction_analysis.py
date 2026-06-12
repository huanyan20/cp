import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import os

import pandas as pd

from core.visualizations import plot_bar_chart, plot_histogram, plot_multi_line_chart, set_plot_style


def analyze_friction(csv_path="results_dir/trades_ppo_eval.csv", metrics_path="results_dir/metrics_ppo_eval.json"):
    print("=== Friction & Trade-Level Analysis ===")
    
    if not os.path.exists(csv_path):
        print(f"Cannot find {csv_path}. Make sure to run evaluate_portfolio.py first.")
        return

    # 1. Load metrics for turnover
    try:
        with open(metrics_path, encoding='utf-8') as f:
            metrics = json.load(f)
            print(f"Average Daily Turnover: {metrics.get('turnover', 0.0):.4f}")
    except Exception:
        pass

    # 2. Load trades
    df = pd.read_csv(csv_path)
    if df.empty:
        print("No trades found.")
        return
    
    print(f"Total Transactions: {len(df)}")
    
    # Cost analysis
    total_cost = df['cost'].sum()
    print(f"Total Friction Cost (Slippage + Fees + Tax): {total_cost:,.0f} TWD")
    
    # Trade behavior
    # Filter only SELL trades to see realized holding period
    sells = df[df['trade_type'] == 'SELL'].copy()
    
    if len(sells) > 0:
        avg_hold = sells['holding_period_days'].mean()
        print(f"Average Holding Period (Sells): {avg_hold:.1f} steps")
        print(f"Median Holding Period: {sells['holding_period_days'].median()} steps")
        
        # Plot histogram of holding periods
        set_plot_style()
        plot_histogram(
            data=sells['holding_period_days'],
            bins=50,
            title='Holding Period Distribution (SELLs)',
            xlabel='Holding Period (Steps)',
            ylabel='Frequency',
            color='skyblue',
            output_path='results_dir/holding_period_dist.png'
        )
        print("[V] Saved holding period distribution to: results_dir/holding_period_dist.png")
    
    # Plot trade count over time to identify over-trading periods
    trades_per_step = df.groupby('step').size()
    set_plot_style()
    plot_bar_chart(
        labels=trades_per_step.index,
        values=trades_per_step.values,
        title='Transactions per Step (Are we over-trading?)',
        xlabel='Step (Approximates Time)',
        ylabel='Number of Transactions',
        color='coral',
        output_path='results_dir/transactions_over_time.png'
    )
    print("[V] Saved transaction volume over time to: results_dir/transactions_over_time.png")

    # Plot cumulative friction
    df_step = df.groupby('step')['cost'].sum().reset_index()
    df_step['cum_cost'] = df_step['cost'].cumsum()
    
    set_plot_style()
    plot_multi_line_chart(
        df=df_step.set_index('step')[['cum_cost']],
        title='Cumulative Friction Cost over Time',
        ylabel='Cumulative Cost (TWD)',
        output_path='results_dir/cumulative_friction.png'
    )
    print("[V] Saved cumulative friction to: results_dir/cumulative_friction.png")

if __name__ == "__main__":
    analyze_friction()
