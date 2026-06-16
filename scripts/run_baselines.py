import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

sys.path.append('.')
from sl_pipeline.walk_forward_sl import build_period_plan, resolve_period
from data_pipeline.universe_builder import get_universe_builder
from data_loader import fetch_multi_asset_data
from metrics_utils import calculate_metrics

def evaluate_returns(daily_returns: pd.Series) -> dict:
    ret = daily_returns.dropna().values
    if len(ret) < 20:
        return {"CAGR(%)": 0.0, "MDD(%)": 0.0, "Sortino": 0.0, "Sharpe": 0.0}
        
    portfolio_history = np.cumprod(1 + ret)
    # prepend 1.0 to history
    portfolio_history = np.insert(portfolio_history, 0, 1.0)
    
    metrics = calculate_metrics(
        portfolio_history=portfolio_history.tolist(),
        positions_history=[],
        cash_history=[],
        daily_returns=ret.tolist()
    )
    
    # metrics_utils calculates total_return instead of cagr. We compute CAGR manually
    total_ret = metrics['total_return']
    years = len(ret) / 252.0
    cagr_val = ((1 + total_ret) ** (1 / years) - 1) if years > 0 else 0.0
    
    return {
        "CAGR(%)": float(cagr_val) * 100,
        "MDD(%)": float(metrics['max_drawdown']) * 100,
        "Sortino": float(metrics['sortino']),
        "Sharpe": float(metrics['sharpe'])
    }

def main():
    print("Starting Baseline Backtests (Milestone 1)...")
    
    plan = build_period_plan()
    builder = get_universe_builder('dynamic')
    
    results = []
    
    for planned in plan:
        name = planned['name']
        period = resolve_period(name)
        train_start = period['train_start']
        test_start = period['test_start']
        test_end = planned.get('effective_test_end', period['test_end'])
        
        print(f"Loading data for period: {name} (Test: {test_start} to {test_end})")
        period_tickers = builder.build_universe(train_start, top_n=45)
        
        data_dict = fetch_multi_asset_data(period_tickers, start_date=train_start, end_date=test_end, macro_tickers=['^TWII'])
        
        frames = []
        for ticker, df in data_dict.items():
            if df.empty or ticker == '^TWII':
                continue
            df = df.copy()
            df['ticker'] = ticker
            df['date'] = df.index
            
            if 'close' in df.columns:
                df['simple_ret'] = df['close'].pct_change()
                df['mom_20'] = df['close'].pct_change(20)
            elif 'log_return' in df.columns:
                df['simple_ret'] = np.exp(df['log_return']) - 1
                df['mom_20'] = df['simple_ret'].rolling(20).apply(lambda x: np.prod(1+x)-1)
            frames.append(df)
            
        if not frames:
            continue
            
        panel = pd.concat(frames, ignore_index=True)
        panel['date'] = pd.to_datetime(panel['date'])
        
        test_panel = panel[(panel['date'] >= pd.to_datetime(test_start)) & (panel['date'] <= pd.to_datetime(test_end))]
        
        # 1. Market Buy & Hold (^TWII)
        if '^TWII' in data_dict and not data_dict['^TWII'].empty:
            twii = data_dict['^TWII']
            twii = twii[(twii.index >= test_start) & (twii.index <= test_end)]
            if 'close' in twii.columns:
                twii_ret = twii['close'].pct_change()
            else:
                twii_ret = np.exp(twii['log_return']) - 1
            mkt_metrics = evaluate_returns(twii_ret)
        else:
            mkt_metrics = evaluate_returns(pd.Series())
            
        # 2. Equal Weight (Top 45 Dynamic Universe)
        ew_ret = test_panel.groupby('date')['simple_ret'].mean()
        ew_metrics = evaluate_returns(ew_ret)
        
        # 3. Momentum (Top 5)
        panel['mom_20_lag'] = panel.groupby('ticker')['mom_20'].shift(1)
        test_panel_mom = panel[(panel['date'] >= pd.to_datetime(test_start)) & (panel['date'] <= pd.to_datetime(test_end))]
        
        def top_k_ret(group, k=5):
            valid = group.dropna(subset=['mom_20_lag'])
            if len(valid) == 0:
                return 0.0
            top = valid.nlargest(k, 'mom_20_lag')
            return top['simple_ret'].mean()
            
        mom_ret = test_panel_mom.groupby('date').apply(top_k_ret)
        mom_metrics = evaluate_returns(mom_ret)
        
        for strat, metrics in [("Buy & Hold (^TWII)", mkt_metrics), 
                               ("Equal Weight (Univ 45)", ew_metrics), 
                               ("Momentum Top 5", mom_metrics)]:
            res = {"Period": name, "Strategy": strat}
            res.update(metrics)
            results.append(res)
            
    res_df = pd.DataFrame(results)
    summary = res_df.groupby('Strategy').mean(numeric_only=True).reset_index()
    
    print("\n--- Baseline Strategies Summary (Average Across Periods) ---")
    print(summary.to_string(index=False))
    
    Path("reports").mkdir(exist_ok=True)
    res_df.to_csv("reports/baseline_metrics_detailed.csv", index=False)
    summary.to_json("reports/baseline_metrics.json", orient='records', indent=2)
    print("\nDetailed results saved to reports/baseline_metrics_detailed.csv")
    print("Summary saved to reports/baseline_metrics.json")

if __name__ == "__main__":
    main()
