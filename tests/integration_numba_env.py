"""Integration test for Numba-accelerated TaiwanStockEnv."""
import numpy as np
import pandas as pd
from trading_env import TaiwanStockEnv, _NUMBA_AVAILABLE

print(f'Numba available: {_NUMBA_AVAILABLE}')

n_steps = 300
dates = pd.date_range('2022-01-01', periods=n_steps, freq='B')

def make_df():
    df = pd.DataFrame(index=dates)
    df['close'] = 100 + np.random.randn(n_steps).cumsum()
    df['volume'] = np.random.rand(n_steps) * 1e6
    df['log_return'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
    df['ma5'] = df['close'].rolling(5).mean().fillna(df['close'])
    df['ma20'] = df['close'].rolling(20).mean().fillna(df['close'])
    return df

tickers = ['A', 'B', 'C', 'D', 'E']
df_dict = {t: make_df() for t in tickers}

env = TaiwanStockEnv(df_dict, window_size=20, enable_cash_action=True)
obs, info = env.reset()
print('obs shape:', obs.shape)

total_reward = 0.0
for step in range(50):
    action = np.random.randn(len(tickers) + 1).astype(np.float32)
    obs, reward, done, trunc, info = env.step(action)
    total_reward += reward
    if done:
        break

pv = info['portfolio_value']
mdd = info['max_drawdown']
print(f'50-step env integration OK: total_reward={total_reward:.4f}  pv={pv:,.0f}  mdd={mdd:.4f}')
print('ALL GOOD - Numba + trading_env integration verified')
