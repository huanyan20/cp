import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from settings import load_settings
import trading_env
from research_pipeline import build_eval_env, run_eval_loop, PERIODS
from core.model_trainer import ModelTrainer
from stable_baselines3 import PPO
from stock_universe import MACRO_TICKERS_RL

def main():
    settings = load_settings()
    model_path = "results_dir/wf_ppo_enabled_model_2024H2_seed42.zip"
    if not Path(model_path).exists():
        print(f"Model not found at {model_path}")
        return

    # Find the 2024H2 period
    period = next(p for p in PERIODS if p["name"] == "2024H2")
    test_start = period["test_start"]
    test_end = period["test_end"]

    from data_pipeline.universe_builder import get_universe_builder
    builder = get_universe_builder("dynamic")
    tickers = builder.build_universe(period["train_start"], top_n=45)

    print("=== Test 1: Churning Check (Sept/Oct 2024) ===")
    
    # 1. Evaluate with record_trades=True
    eval_env, _ = build_eval_env(
        tickers=tickers,
        test_start=test_start,
        test_end=test_end,
        window_size=20,
        macro_tickers=MACRO_TICKERS_RL,
        settings=settings,
        enable_cash_action=True,
        enable_margin_short=False,
    )
    # Enable trade recording manually on the env
    eval_env.record_trades = True
    
    trainer = ModelTrainer("ppo")
    model = trainer.load_model(model_path, eval_env)

    # Run eval loop
    obs, _ = eval_env.reset(seed=42)
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated

    total_return_base = eval_env.total_return
    print(f"Base Total Return (Slippage=0.001): {total_return_base:.2%}")
    
    trades = eval_env.trades_history
    print(f"Total trades recorded: {len(trades)}")
    
    # Calculate Annualized Turnover Rate
    total_traded_value = sum(t["trade_amount_twd"] for t in trades)
    # Total turnover is traded value divided by 2 (buy + sell = 1 complete turnover cycle)
    total_turnover = total_traded_value / 2.0
    
    # We use initial balance as a proxy for average AUM for simplicity
    turnover_ratio = total_turnover / eval_env.initial_balance
    
    # Annualize it
    num_days = eval_env._current_step - eval_env.window_size
    annualized_turnover = turnover_ratio * (252 / num_days)
    
    print(f"Total Traded Value: {total_traded_value:,.0f} TWD")
    print(f"Total Turnover Ratio: {turnover_ratio:.2%}")
    print(f"Annualized Turnover Rate: {annualized_turnover:.2%}")

    # Filter Sept/Oct trades
    target_trades = [t for t in trades if t["date"].startswith("2024-09") or t["date"].startswith("2024-10")]
    
    # Just show a sample of 5 consecutive days of trades around late Sept
    sample_dates = sorted(list(set(t["date"] for t in target_trades if t["date"] > "2024-09-20")))[:5]
    print(f"\nSample Trades for consecutive days: {sample_dates}")
    
    for d in sample_dates:
        day_trades = [t for t in target_trades if t["date"] == d]
        # Only show trades with > 1% of portfolio value to filter out tiny noise adjustments
        meaningful_trades = [t for t in day_trades if t["trade_amount_twd"] / eval_env.initial_balance > 0.01]
        print(f"--- {d} ---")
        for t in meaningful_trades[:5]: # Show max 5 per day
            print(f"  {t['ticker']}: {t['trade_type']} | prev={t['prev_weight']:.3f} -> target={t['target_weight']:.3f} | amount={t['trade_amount_twd']:,.0f}")
        if len(meaningful_trades) > 5:
            print(f"  ... and {len(meaningful_trades)-5} more meaningful trades.")
        if len(day_trades) > len(meaningful_trades):
            print(f"  ... plus {len(day_trades) - len(meaningful_trades)} micro-adjustments (<1% weight)")

    print("\n=== Test 2: Slippage Sensitivity ===")
    
    for test_slippage in [0.003, 0.005]:
        trading_env.SLIPPAGE_RATE = test_slippage
        
        eval_env, _ = build_eval_env(
            tickers=tickers,
            test_start=test_start,
            test_end=test_end,
            window_size=20,
            macro_tickers=MACRO_TICKERS_RL,
            settings=settings,
            enable_cash_action=True,
            enable_margin_short=False,
        )
        # Re-assign SLIPPAGE_RATE inside the environment just in case
        
        obs, _ = eval_env.reset(seed=42)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            
        print(f"Total Return with Slippage={test_slippage:.3f}: {eval_env.total_return:.2%}")

    # Reset
    trading_env.SLIPPAGE_RATE = 0.001

if __name__ == "__main__":
    main()
