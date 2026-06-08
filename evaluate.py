"""
evaluate_portfolio.py - 評估 Portfolio Manager 的績效與資金輪動
"""
import matplotlib.pyplot as plt
import numpy as np
from portfolio_env import PortfolioEnv
from stable_baselines3 import PPO

from data_loader import fetch_multi_asset_data

# 因為 PPO.load 需要知道自定義架構，直接 import 是為了確保類別可見

TICKERS = [
    "0052.TW", "00881.TW", "00891.TW", "00892.TW",
    "2454.TW", "2317.TW", "2382.TW", "3711.TW",
    "3131.TW", "3583.TW", "6187.TW", "3324.TW",
    "3017.TW", "3653.TW", "2421.TW", "8996.TW"
]
MACRO_TICKER = "^TWII"
TEST_START = "2024-11-01"
TEST_END = "2025-05-31"
WINDOW_SIZE = 20

def main():
    print(f"=== 下載驗證區間資料 ({TEST_START} ~ {TEST_END}) ===")
    enriched = fetch_multi_asset_data(
        tickers=TICKERS, start_date=TEST_START, end_date=TEST_END, 
        window_size=WINDOW_SIZE, macro_ticker=MACRO_TICKER
    )
    
    env = PortfolioEnv(df_dict=enriched, window_size=WINDOW_SIZE)
    
    print("\n=== 載入 PPO GNN 模型 ===")
    model = PPO.load("ppo_portfolio_gnn_model")
    
    obs, _ = env.reset()
    done = False
    
    # 紀錄歷史
    portfolio_history = []
    positions_history = []
    
    print("\n=== 開始回測 ===")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        portfolio_history.append(info["portfolio_value"])
        positions_history.append(info["positions"])
        
    print(f"\n[V] 回測完成！最終總資產: {portfolio_history[-1]:,.0f}")
    print(f"[V] 總報酬率: {((portfolio_history[-1]/1000000)-1)*100:.2f}%")
    
    # ── 繪圖 ─────────────────────────────────────────────────────────────
    plt.figure(figsize=(14, 10))
    
    # 1. 總資產淨值曲線
    plt.subplot(2, 1, 1)
    plt.plot(portfolio_history, label="Portfolio Manager (GNN)", color="blue", linewidth=2)
    plt.title("Portfolio Manager - Total Value Curve", fontsize=14, fontweight="bold")
    plt.ylabel("Portfolio Value (TWD)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    
    # 2. 資金輪動熱力圖 (板塊權重變化)
    plt.subplot(2, 1, 2)
    pos_matrix = np.array(positions_history)  # (Steps, N_stocks)
    # cmap=RdBu (Red=Short/Negative, Blue=Long/Positive)
    im = plt.imshow(pos_matrix.T, aspect='auto', cmap='RdBu', vmin=-1, vmax=1)
    plt.colorbar(im, label="Weight (Blue=Long, Red=Short)")
    plt.yticks(ticks=range(len(env.tickers)), labels=env.tickers)
    plt.title("Portfolio Capital Rotation & Allocation Heatmap", fontsize=14, fontweight="bold")
    plt.xlabel("Trading Days")
    plt.ylabel("Stocks")
    
    plt.tight_layout()
    plt.savefig("portfolio_evaluation.png", dpi=150)
    print("\n[V] 資金輪動熱力圖與資產曲線已儲存為 portfolio_evaluation.png")

if __name__ == "__main__":
    main()
