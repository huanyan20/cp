import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import shap
from stable_baselines3 import PPO

from data_loader import fetch_multi_asset_data
from stock_universe import MACRO_TICKERS_RL, TICKERS_TECH_EXPANDED

# Custom imports
from trading_env import TaiwanStockEnv


def collect_data_for_attribution(model_path, start_date="2024-07-01", end_date="2024-12-31"):
    print(f"=== Collecting PPO Actions & Features for {start_date} to {end_date} ===")
    
    tickers = TICKERS_TECH_EXPANDED
    macro_tickers = MACRO_TICKERS_RL
    overnight_path = "capital_flow_analysis/data/overnight_gap_features_1d.csv"
    
    # Fetch data slightly earlier to allow window_size
    fetch_start = "2024-01-01"
    end_date = "2025-06-30"
    df_dict = fetch_multi_asset_data(
        tickers, 
        start_date=fetch_start, 
        end_date=end_date, 
        macro_tickers=macro_tickers,
        overnight_feature_path=overnight_path
    )
    
    # 2. Create Env
    env = TaiwanStockEnv(
        df_dict=df_dict,
        window_size=20,
        initial_balance=1_000_000.0,
        topk=5,
        use_benchmark_reward=True,
        enable_cash_action=False,
        enable_margin_short=False,
    )
    
    model = PPO.load(model_path)
    
    # Get feature names
    feature_names = list(env.dfs[tickers[0]].columns)
    num_market_features = len(feature_names)
    window_size = env.window_size
    
    X_list = []
    Y_list = []
    
    obs, info = env.reset()
    done = False
    
    while not done:
        # PPO predicts action based on obs
        action, _ = model.predict(obs, deterministic=True)
        
        # We only extract the 'latest day' features for each stock to see local correlation
        # obs shape is (num_stocks, window_size * num_market_features + num_account_features)
        for i in range(env.num_stocks):
            # The market window is the first window_size * num_market_features elements
            market_window = obs[i, :window_size * num_market_features]
            # The latest day is the last 'num_market_features' elements of the window
            latest_day_features = market_window[-num_market_features:]
            
            X_list.append(latest_day_features)
            # action[i] is the raw logit/weight output for stock i
            Y_list.append(action[i])
            
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    X = np.array(X_list)
    Y = np.array(Y_list)
    
    return X, Y, feature_names

def run_shap_analysis():
    X, Y, feature_names = collect_data_for_attribution("ppo_portfolio_full_stock_seed42.zip")
    print(f"Collected {len(X)} samples with {len(feature_names)} features.")
    
    # Filter features to ignore mostly empty ones or standard open/high/low if we want
    # but let's keep them all for the surrogate model.
    
    print("Training surrogate LightGBM model to mimic PPO actions...")
    lgb_model = lgb.LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    lgb_model.fit(X, Y)
    
    r2_score = lgb_model.score(X, Y)
    print(f"Surrogate Model R2 Score: {r2_score:.4f}")
    if r2_score < 0.3:
        print("[!] Warning: Surrogate model R2 is low. PPO actions might depend heavily on cross-stock features or temporal sequence, not just latest day local features.")
    
    # 1. Plot LightGBM Feature Importance
    importance = lgb_model.feature_importances_
    idx = np.argsort(importance)[-20:] # top 20
    
    plt.figure(figsize=(10, 8))
    plt.barh(np.array(feature_names)[idx], importance[idx], color='lightgreen')
    plt.title('Top 20 Features Driving PPO Buy Actions (2024H2)\nSurrogate Model Feature Importance')
    plt.xlabel('Importance (Splits)')
    plt.tight_layout()
    plt.savefig('results_dir/feature_importance_lgbm.png')
    plt.close()
    print("[V] Saved: results_dir/feature_importance_lgbm.png")
    
    # 2. Run TreeSHAP
    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(lgb_model)
    # Use a subsample if X is too large for fast SHAP plotting
    X_sample = X[np.random.choice(X.shape[0], min(5000, X.shape[0]), replace=False)]
    shap_values = explainer.shap_values(X_sample)
    
    # SHAP summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.title('SHAP Summary: How Features Impact Buy Actions (2024H2)')
    plt.tight_layout()
    plt.savefig('results_dir/shap_summary.png')
    plt.close()
    print("[V] Saved: results_dir/shap_summary.png")

if __name__ == "__main__":
    run_shap_analysis()
