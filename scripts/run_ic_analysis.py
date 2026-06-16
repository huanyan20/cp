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
from sl_pipeline.labels import forward_log_return_t1, HORIZON_DAYS, _add_cross_sectional_ranks
from data_pipeline.utils import BASE_FEATURE_COLS

def compute_rank_ic(panel_df: pd.DataFrame, feature_col: str, target_col: str) -> float:
    """Compute average daily Rank IC between a feature and a target."""
    df = panel_df.dropna(subset=[feature_col, target_col]).copy()
    if len(df) < 10:
        return np.nan
        
    df['feat_rank'] = df.groupby('date')[feature_col].rank(pct=True)
    df['targ_rank'] = df.groupby('date')[target_col].rank(pct=True)
    
    # Pearson correlation on cross-sectionally ranked data = Rank IC (approximation of mean daily spearman)
    return df['feat_rank'].corr(df['targ_rank'], method='pearson')

def main():
    print("Starting Feature IC Dashboard Generation (Milestone 3A)...")
    
    plan = build_period_plan()
    builder = get_universe_builder('dynamic')
    
    horizons = HORIZON_DAYS # (5, 10, 20, 60)
    
    results = []
    
    for planned in plan:
        name = planned['name']
        period = resolve_period(name)
        train_start = period['train_start']
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
            if 'log_return' in df.columns:
                for h in horizons:
                    df[f'target_{h}d'] = forward_log_return_t1(df['log_return'], h)
                
            frames.append(df)
            
        if not frames:
            continue
            
        panel = pd.concat(frames, ignore_index=True)
        
        # Add cross-sectional ranks for all base features
        panel = _add_cross_sectional_ranks(panel, BASE_FEATURE_COLS)
        
        # All features to test (base + ranked)
        features_to_test = [col for col in panel.columns if col in BASE_FEATURE_COLS or col.startswith('rank_')]
        features_to_test = [f for f in features_to_test if f != "log_return" and f != "rank_log_return"]
            
        print(f"  -> Testing {len(features_to_test)} features...")
        for feature in features_to_test:
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
    
    # Pivot for dashboard
    pivot = summary.pivot(index="Feature", columns="Horizon", values="Rank_IC").reset_index()
    # Reorder columns logically
    horizon_cols = [f"{h}d" for h in horizons if f"{h}d" in pivot.columns]
    pivot = pivot[["Feature"] + horizon_cols]
    
    # Sort by absolute 20d IC if available
    sort_col = "20d" if "20d" in pivot.columns else horizon_cols[0]
    pivot = pivot.assign(abs_sort=pivot[sort_col].abs()).sort_values("abs_sort", ascending=False).drop(columns=["abs_sort"])
    
    print("\n=== Feature IC Dashboard (Milestone 3A) ===")
    print(pivot.to_markdown(index=False, floatfmt=".4f"))
    
    # Filter features: Keep those with |IC| > 0.02 at 20d
    threshold = 0.02
    if sort_col in pivot.columns:
        passed = pivot[pivot[sort_col].abs() > threshold]
        print(f"\n[!] Alpha Filtration: {len(passed)} / {len(pivot)} features passed the |IC| > {threshold} threshold at {sort_col}.")
        
    Path("reports").mkdir(exist_ok=True)
    res_df.to_csv("reports/ic_dashboard_detailed.csv", index=False)
    pivot.to_markdown("reports/feature_ic_dashboard.md", index=False)
    
    print("\nDetailed results saved to reports/ic_dashboard_detailed.csv")
    print("Summary saved to reports/feature_ic_dashboard.md")

if __name__ == "__main__":
    main()
