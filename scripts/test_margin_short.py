import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from trading_env import TaiwanStockEnv


def test_margin_short():
    print("Testing margin short environment...")
    # Create dummy dataframe
    df1 = pd.DataFrame({
        "log_return": np.random.normal(0, 0.01, 100),
        "Close_norm": np.linspace(1, 2, 100)
    })
    df2 = pd.DataFrame({
        "log_return": np.random.normal(0, 0.01, 100),
        "Close_norm": np.linspace(1, 0.5, 100)
    })
    
    df_dict = {"STOCK_A": df1, "STOCK_B": df2}
    
    env = TaiwanStockEnv(
        df_dict=df_dict,
        window_size=20,
        initial_balance=1_000_000.0,
        topk=2,
        enable_margin_short=True,
        max_leverage=2.0
    )
    
    obs, info = env.reset()
    assert env.max_leverage == 2.0
    assert env.enable_margin_short
    
    # Test action transform with extreme values
    # Let's say action is [5.0, -5.0] (tanh = [0.9999, -0.9999])
    # Absolute sum = 2.0, which equals max_leverage.
    # No normalization needed.
    action = np.array([5.0, -5.0])
    weights = env._transform_action(action)
    print(f"Action: {action} -> Weights: {weights}")
    
    assert np.isclose(weights[0], 1.0, atol=1e-3)
    assert np.isclose(weights[1], -1.0, atol=1e-3)
    
    # Test normalization
    # Let's say action is [5.0, 5.0] -> tanh = [1.0, 1.0]. Sum abs = 2.0. No norm.
    # Wait, what if we have 3 stocks and output [5.0, 5.0, 5.0] -> sum = 3.0.
    # Max leverage is 2.0, so weights should be scaled to [0.66, 0.66, 0.66].
    env.num_stocks = 3
    env._topk = 3
    weights_3 = env._transform_action(np.array([5.0, 5.0, 5.0]))
    print(f"Action 3 ones -> Weights: {weights_3}")
    assert np.isclose(np.sum(weights_3), 2.0, atol=1e-3)
    
    # Test Step
    env.num_stocks = 2
    env._topk = 2
    action = np.array([5.0, -5.0]) # Long A, Short B
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"Portfolio Value after step 1: {info['portfolio_value']}")
    print(f"Positions: {info['positions']}")
    print(f"Cash Weight: {info['cash_weight']}")
    
    # Cash weight should be 1.0 - (1.0 + (-1.0)) = 1.0 ?
    # Wait, 1.0 - (1.0 - 1.0) = 1.0!
    # If we long 1.0 and short 1.0, we use $1M to buy A, and sell $1M of B (receive $1M).
    # Net cash = $1M (initial) - $1M (buy A) + $1M (sell B) = $1M!
    # So cash weight = 1.0. Correct!
    
    print("Test passed successfully!")

if __name__ == "__main__":
    test_margin_short()
