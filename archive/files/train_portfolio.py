"""
train_portfolio.py - 投資組合經理人 (Portfolio Manager) 訓練腳本 v2
整合 大盤特徵 (Macro Features) + 時空雙重注意力 GNN (Spatio-Temporal) + PPO

升級重點（相較 v1）：
  - GnnFeatureExtractor 現在接收 window_size，內部以 LSTM 先做時序編碼，
    再讓股票節點在 Graph Attention 層互相交換「含趨勢語意」的資訊。
  - features_extractor_kwargs 新增 window_size=WINDOW_SIZE 傳參。
  - 其餘訓練超參數維持 v1 設定不變（已調校過），避免引入新變數。
"""
import time
import winsound

from stable_baselines3 import PPO

from data_loader import fetch_multi_asset_data
from gnn_extractor import GnnFeatureExtractor
from trading_env import TaiwanStockEnv

# ─────────────────────────────────────────────
# 訓練設定
# ─────────────────────────────────────────────

# ── 初步測試：2 支股票 ─────────────────────────────────────────────
TICKERS_2 = ["3017.TW", "2317.TW"]   # 奇鋐 (散熱) + 鴻海 (組裝) 板塊差異大

# ── 正式訓練：16 支全板塊股票 ─────────────────────────────────────
TICKERS_FULL = [
    # 大盤與板塊 ETF
    "0052.TW", "00881.TW", "00891.TW", "00892.TW",
    # 大型權值與代工
    "2454.TW", "2317.TW", "2382.TW", "3711.TW",
    # 設備概念飆股
    "3131.TW", "3583.TW", "6187.TW",
    # 散熱概念飆股
    "3324.TW", "3017.TW", "3653.TW", "2421.TW", "8996.TW",
]

MACRO_TICKER = "^TWII"
TRAIN_START  = "2022-01-01"
TRAIN_END    = "2024-10-31"
WINDOW_SIZE  = 20
TIMESTEPS    = 1_000_000   # 正式版：100 萬步


def play_done_sound():
    """訓練完成後播放提示音（Windows 系統音效）"""
    for _ in range(3):
        winsound.Beep(1000, 300)
        time.sleep(0.2)
    winsound.Beep(1500, 500)


def main(tickers=TICKERS_2, timesteps=TIMESTEPS, model_name="ppo_portfolio_2stock"):
    print("=== Portfolio Manager 訓練腳本 v2 (Spatio-Temporal GNN) ===")
    print(f"股票清單：{tickers}")
    print(f"大盤指標：{MACRO_TICKER}")
    print(f"訓練步數：{timesteps:,}")

    # ── 1. 下載並對齊資料（含大盤特徵）────────────────────────────────
    print(f"\n=== 準備資料 ({TRAIN_START} ~ {TRAIN_END}) ===")
    enriched = fetch_multi_asset_data(
        tickers=tickers,
        start_date=TRAIN_START,
        end_date=TRAIN_END,
        window_size=WINDOW_SIZE,
        macro_ticker=MACRO_TICKER
    )
    print(f"\n[V] 取得 {len(enriched)} 支股票的資料字典 (含大盤 {MACRO_TICKER} 特徵)。")

    # ── 2. 初始化 Portfolio 環境 ──────────────────────────────────────
    env = TaiwanStockEnv(df_dict=enriched, window_size=WINDOW_SIZE)
    obs_shape = env.observation_space.shape
    print("\n[V] 環境初始化完成")
    print(f"    Observation Space : {obs_shape}")
    print(f"    Action Space      : {env.action_space.shape}")
    print(f"    num_stocks        : {obs_shape[0]}")
    print(f"    window_size       : {WINDOW_SIZE}")
    print(f"    features_per_step : {obs_shape[1] // WINDOW_SIZE}")

    # ── 3. 設定 PPO + 時空雙重注意力 GNN 特徵提取器 ──────────────────
    #
    #  升級說明：
    #   window_size 必須顯式傳入，讓 GnnFeatureExtractor 在內部將
    #   flatten 後的 obs 正確 reshape 為 (window, features_per_step)，
    #   再交給 LSTM 做時序編碼。
    #
    policy_kwargs = dict(
        features_extractor_class=GnnFeatureExtractor,
        features_extractor_kwargs=dict(
            features_dim=256,
            window_size=WINDOW_SIZE,   # ← v2 新增：時序編碼必要參數
            lstm_hidden=64,            # LSTM 隱藏層維度（輕量）
            embed_dim=64,              # Graph Attention 節點嵌入維度
            num_heads=4,               # Multi-Head Attention 頭數
            attn_dropout=0.1,          # Attention dropout（訓練時防過擬合）
        ),
        net_arch=[256, 256]
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,       # 保守學習步伐（v1 已調校）
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,           # 限制策略更新幅度（防策略崩潰）
        target_kl=0.01,           # KL > 0.01 就提前停止更新
        ent_coef=0.01,            # 維持探索性
        policy_kwargs=policy_kwargs,
    )

    # 顯示模型可訓練參數數量，確認 LSTM 已正確掛載
    total_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    print(f"\n[V] PPO 政策網路可訓練參數：{total_params:,}")

    # ── 4. 開始訓練 ───────────────────────────────────────────────────
    print(f"\n=== 開始訓練 PPO + Spatio-Temporal GNN ({timesteps:,} 步) ===")
    t_start = time.time()
    model.learn(total_timesteps=timesteps)
    elapsed = time.time() - t_start

    # ── 5. 儲存模型 ───────────────────────────────────────────────────
    model.save(model_name)
    print(f"\n[V] 模型已儲存為 {model_name}.zip")
    print(f"[V] 總訓練時間：{elapsed/60:.1f} 分鐘")

    # ── 6. 播放完成提示音 ─────────────────────────────────────────────
    play_done_sound()
    print("\n[DONE] Training complete!")


if __name__ == "__main__":
    main(
        tickers=TICKERS_FULL,
        timesteps=TIMESTEPS,
        model_name="ppo_portfolio_v3_st_gnn"   # v3 = Spatio-Temporal GNN
    )
