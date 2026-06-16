import sys
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

sys.path.append('.')
from sl_pipeline.walk_forward_sl import build_period_plan, resolve_period
from data_pipeline.universe_builder import get_universe_builder
from data_loader import fetch_multi_asset_data
from sl_pipeline.labels import forward_log_return_t1

def compute_rank_ic(panel_df: pd.DataFrame, feature_col: str, target_col: str) -> float:
    """Compute average daily Rank IC between a feature and a target."""
    ic_vals = []
    for date, df in panel_df.groupby('date'):
        # Filter valid rows
        valid_df = df.dropna(subset=[feature_col, target_col])
        if len(valid_df) > 10:  # Require at least 10 valid stocks
            try:
                ic = spearmanr(valid_df[feature_col], valid_df[target_col])[0]
                if not np.isnan(ic):
                    ic_vals.append(ic)
            except Exception:
                pass
    return np.mean(ic_vals) if ic_vals else np.nan

def main():
    print("Starting Capital Flow IC Analysis (Milestone 2)...")
    
    plan = build_period_plan()
    builder = get_universe_builder('dynamic')
    
    horizons = [5, 10, 20]
    
    results = []
    
    for planned in plan:
        name = planned['name']
        period = resolve_period(name)
        train_start = period['train_start']
        # For simplicity, we just use the entire train+test period for IC analysis
        # to maximize data points. We are just exploring feature alpha.
        test_end = planned.get('effective_test_end', period['test_end'])
        
        print(f"Loading data for period: {name} ({train_start} to {test_end})")
        period_tickers = builder.build_universe(train_start, top_n=45)
        
        data_dict = fetch_multi_asset_data(period_tickers, start_date=train_start, end_date=test_end, macro_tickers=[])
        
        # Build panel with forward returns
        frames = []
        for ticker, df in data_dict.items():
            if df.empty:
                continue
            df = df.copy()
            df['ticker'] = ticker
            df['date'] = df.index
            
            # Compute forward returns for various horizons
            for h in horizons:
                # Assuming df['log_return'] exists, forward_log_return_t1 expects it
                if 'log_return' in df.columns:
                    # Quick monkey-patch labels.py HORIZON_DAYS if needed, but we'll do it manually to avoid import errors
                    # log(P(t+h) / P(t+1))
                    arr = df['log_return'].values
                    n = len(arr)
                    out = np.full(n, np.nan, dtype=float)
                    for i in range(n):
                        start = i + 2
                        stop = i + h + 1
                        if stop <= n:
                            out[i] = float(np.nansum(arr[start:stop]))
                    df[f'target_{h}d'] = out
                
            frames.append(df)
            
        if not frames:
            continue
            
        panel = pd.concat(frames, ignore_index=True)
        
        # Determine capital flow related features
        # Assuming the features contain "capital_flow" or "flow"
        potential_features = [c for c in panel.columns if "flow" in c.lower() or "capital" in c.lower()]
        if not potential_features:
            # Fallback
            potential_features = ['capital_flow'] if 'capital_flow' in panel.columns else []
            
        if not potential_features:
            print(f"Warning: No capital flow features found in {name}.")
            continue
            
        for feature in potential_features:
            for h in horizons:
                target_col = f'target_{h}d'
                if target_col not in panel.columns:
                    continue
                ic = compute_rank_ic(panel, feature, target_col)
                
                results.append({
                    "Period": name,
                    "Feature": feature,
                    "Horizon": f"{h}d",
                    "Rank_IC": ic
                })
                
    if not results:
        print("No results computed.")
        return
        
    res_df = pd.DataFrame(results)
    
    # Average across periods
    summary = res_df.groupby(['Feature', 'Horizon'])['Rank_IC'].mean().reset_index()
    summary = summary.sort_values(by=['Horizon', 'Rank_IC'], ascending=[True, False])
    
    print("\n--- IC Analysis Summary ---")
    print(summary.to_string(index=False))
    
    Path("reports").mkdir(exist_ok=True)
    res_df.to_csv("reports/ic_report_detailed.csv", index=False)
    summary.to_markdown("reports/ic_report.md", index=False)
    print("\nDetailed results saved to reports/ic_report_detailed.csv")
    print("Summary saved to reports/ic_report.md")

if __name__ == "__main__":
    main()
