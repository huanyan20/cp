"""
train_portfolio.py - 投資組合經理人 (Portfolio Manager) 訓練腳本
整合 大盤特徵 (Macro Features) 與 GNN 特徵提取器。
"""
from portfolio_env import PortfolioEnv
from stable_baselines3 import PPO

from data_loader import fetch_multi_asset_data
from gnn_extractor import GnnFeatureExtractor

# ─────────────────────────────────────────────
# 訓練設定
# ─────────────────────────────────────────────
TICKERS = [
    "0052.TW", "00881.TW", "00891.TW", "00892.TW",
    "2454.TW", "2317.TW", "2382.TW", "3711.TW",
    "3131.TW", "3583.TW", "6187.TW", "3324.TW",
    "3017.TW", "3653.TW", "2421.TW", "8996.TW"
]
MACRO_TICKER = "^TWII"  # 大盤指標
TRAIN_START  = "2024-01-01"
TRAIN_END    = "2024-10-31"
WINDOW_SIZE  = 20
TIMESTEPS    = 50_000   # Portfolio 模型學習空間較大，可先跑 5 萬步測試

def main():
    # ── 1. 下載並對齊資料（含大盤特徵） ────────────────────────────
    print("=== 準備 Portfolio 訓練資料 ===")
    enriched = fetch_multi_asset_data(
        tickers=TICKERS,
        start_date=TRAIN_START,
        end_date=TRAIN_END,
        window_size=WINDOW_SIZE,
        macro_ticker=MACRO_TICKER
    )
    print(f"\n[V] 取得 {len(enriched)} 支股票的資料字典 (含大盤 {MACRO_TICKER} 特徵)。")

    # ── 2. 初始化 Portfolio 交易環境 ─────────────────────────────────
    print("\n=== 初始化 Portfolio 環境 ===")
    env = PortfolioEnv(df_dict=enriched, window_size=WINDOW_SIZE)

    # ── 3. 設定 PPO 與 GNN 特徵提取器 ─────────────────────────────────
    print(f"\n=== 開始訓練 Portfolio Manager (GNN 架構, {TIMESTEPS:,} 步) ===")
    
    # 告訴 PPO 使用我們自定義的 GnnFeatureExtractor
    policy_kwargs = dict(
        features_extractor_class=GnnFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[256, 256]  # Policy 和 Value Network 的後續層
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        policy_kwargs=policy_kwargs,
    )
    
    # 開始訓練
    model.learn(total_timesteps=TIMESTEPS, progress_bar=True)
    
    # 儲存模型
    model.save("ppo_portfolio_gnn_model")
    print("\n[V] 模型已儲存為 ppo_portfolio_gnn_model.zip")

if __name__ == "__main__":
    main()
