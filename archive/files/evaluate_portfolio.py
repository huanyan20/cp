"""
evaluate_portfolio.py - 評估 Portfolio Manager 的績效與資金輪動 v2
輸出：
  1. 組合總資產淨值曲線
  2. 各股資金權重配置熱力圖（板塊輪動視覺化）
  3. [NEW] AI 市場關聯性熱力圖（Attention Weights 視覺化）
     ─ 展示模型推論時每支股票「關注哪些其他股票」
     ─ 取測試期間所有時步的 attn_weights 平均，消除雜訊

升級重點：
  - 從 model.policy.features_extractor 擷取 GnnFeatureExtractor 實例，
    推論後呼叫 get_last_attn_weights() 收集每一步的注意力矩陣。
  - 繪製第三張子圖：平均 Attention Heatmap，提供可解釋性洞察。
"""
import time
import winsound

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from data_loader import fetch_multi_asset_data
from gnn_extractor import GnnFeatureExtractor  # 確保 PPO.load 能找到自定義架構
from trading_env import TaiwanStockEnv

TEST_START   = "2024-11-01"
TEST_END     = "2025-05-31"
WINDOW_SIZE  = 20
MACRO_TICKER = "^TWII"

# 對應 train_portfolio.py 的 TICKERS_2 或 TICKERS_FULL
TICKERS_2 = ["3017.TW", "2317.TW"]
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

TICKER_NAMES = {
    "0052.TW":  "Yuanta Tech",    "00881.TW": "Cathay 5G+",
    "00891.TW": "CTBC Semi",      "00892.TW": "Fubon Semi",
    "2454.TW":  "MediaTek",       "2317.TW":  "Foxconn",
    "2382.TW":  "Quanta",         "3711.TW":  "ASE Tech",
    "3324.TW":  "Shuang Hong",    "3017.TW":  "AVC",
    "3653.TW":  "Jentech",        "2421.TW":  "Sunon",
    "8996.TW":  "Kaori",
    "3131.TW":  "Adv. Semica",    "3583.TW":  "Scientech",
    "6187.TW":  "Wanrun",
}


# ════════════════════════════════════════════════════════════════════
# 主評估函式
# ════════════════════════════════════════════════════════════════════

def run_eval(model_path: str, tickers: list, output_file: str = "portfolio_evaluation.png"):
    print(f"=== 下載驗證資料 ({TEST_START} ~ {TEST_END}) ===")
    enriched = fetch_multi_asset_data(
        tickers=tickers, start_date=TEST_START, end_date=TEST_END,
        window_size=WINDOW_SIZE, macro_ticker=MACRO_TICKER
    )

    env = TaiwanStockEnv(df_dict=enriched, window_size=WINDOW_SIZE)

    print(f"\n=== 載入模型：{model_path} ===")
    model = PPO.load(model_path)

    # ── 取得 GnnFeatureExtractor 實例，用於擷取 attn_weights ──────────
    extractor: GnnFeatureExtractor = model.policy.features_extractor

    obs, _ = env.reset()
    done = False

    portfolio_history  = [env.initial_balance]
    positions_history  = []
    attn_weights_list  = []   # ← [NEW] 收集每步的 attention 矩陣

    print("\n=== 開始推論 ===")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        portfolio_history.append(info["portfolio_value"])
        positions_history.append(info["positions"].copy())

        # [NEW] 推論後立即擷取本步的 attn_weights (num_stocks × num_stocks)
        w = extractor.get_last_attn_weights()
        if w is not None:
            # 取 batch 維度的第一個（推論時 batch_size=1）
            attn_weights_list.append(w[0].cpu().numpy())

    # ── 計算績效指標 ─────────────────────────────────────────────────
    final_val  = portfolio_history[-1]
    total_ret  = (final_val / 1_000_000 - 1) * 100
    daily_rets = np.diff(portfolio_history) / portfolio_history[:-1]
    sharpe     = (daily_rets.mean() / (daily_rets.std() + 1e-9)) * np.sqrt(252)

    print(f"\n[V] 最終總資產：{final_val:,.0f}")
    print(f"[V] 總報酬率  ：{total_ret:+.2f}%")
    print(f"[V] 夏普比率  ：{sharpe:.2f}")
    print(f"[V] 最大回撤  ：{env._max_drawdown:.2%}")
    print(f"[V] 收集注意力矩陣：{len(attn_weights_list)} 步")

    # ── 計算平均注意力矩陣 ────────────────────────────────────────────
    has_attn = len(attn_weights_list) > 0
    if has_attn:
        avg_attn = np.mean(attn_weights_list, axis=0)   # (N, N)
        print(f"[V] 平均 Attention 矩陣計算完成 (shape: {avg_attn.shape})")

    # ════════════════════════════════════════════════════════════════
    # 繪圖：3 張子圖
    # ════════════════════════════════════════════════════════════════
    n_rows = 3 if has_attn else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 5 * n_rows))
    fig.suptitle(
        f"Portfolio Manager (Spatio-Temporal GNN) Evaluation\n{TEST_START} ~ {TEST_END}",
        fontsize=14, fontweight="bold"
    )
    stock_labels = [TICKER_NAMES.get(t, t) for t in env.tickers]

    # ── 圖 1：淨值曲線 ────────────────────────────────────────────────
    ax0 = axes[0]
    ax0.plot(portfolio_history, color="#1f77b4", linewidth=2,
             label=f"Portfolio ({total_ret:+.2f}%)  Sharpe={sharpe:.2f}")
    ax0.axhline(y=1_000_000, color="gray", linestyle="--", alpha=0.7, label="初始資金 100萬")
    ax0.fill_between(
        range(len(portfolio_history)), 1_000_000, portfolio_history,
        where=[v > 1_000_000 for v in portfolio_history],
        alpha=0.2, color="green"
    )
    ax0.fill_between(
        range(len(portfolio_history)), 1_000_000, portfolio_history,
        where=[v <= 1_000_000 for v in portfolio_history],
        alpha=0.2, color="red"
    )
    ax0.set_title("Total Portfolio Value", fontweight="bold")
    ax0.set_ylabel("Value (TWD)")
    ax0.legend()
    ax0.grid(True, linestyle="--", alpha=0.5)

    # ── 圖 2：資金輪動熱力圖 ──────────────────────────────────────────
    ax1 = axes[1]
    if positions_history:
        pos_matrix = np.array(positions_history).T   # (N_stocks, Steps)
        im1 = ax1.imshow(pos_matrix, aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
        plt.colorbar(im1, ax=ax1, label="Weight (Blue=Long / Red=Short)")
        ax1.set_yticks(range(len(env.tickers)))
        ax1.set_yticklabels(stock_labels, fontsize=9)
        ax1.set_title("Capital Rotation & Allocation Heatmap", fontweight="bold")
        ax1.set_xlabel("Trading Days")

    # ── 圖 3：[NEW] AI 市場關聯性熱力圖（平均 Attention Weights）────────
    if has_attn:
        ax2 = axes[2]

        # 自訂色彩：低值=米白，高值=深藍（強調高關注度）
        cmap_attn = matplotlib.colormaps.get_cmap("Blues")
        im2 = ax2.imshow(avg_attn, cmap=cmap_attn, vmin=0, vmax=avg_attn.max())
        plt.colorbar(im2, ax=ax2, label="Avg Attention Weight")

        # 座標軸標籤
        ax2.set_xticks(range(len(env.tickers)))
        ax2.set_yticks(range(len(env.tickers)))
        ax2.set_xticklabels(stock_labels, rotation=45, ha="right", fontsize=8)
        ax2.set_yticklabels(stock_labels, fontsize=8)

        # 在每格填入數值（僅當股票數 ≤ 12 時顯示，避免太擁擠）
        n = len(env.tickers)
        if n <= 12:
            for i in range(n):
                for j in range(n):
                    val = avg_attn[i, j]
                    text_color = "white" if val > avg_attn.max() * 0.6 else "black"
                    ax2.text(j, i, f"{val:.2f}", ha="center", va="center",
                             fontsize=7, color=text_color)

        ax2.set_title(
            "AI Market Correlation — Average Attention Weights\n"
            "（列 = 查詢節點 Query，欄 = 被關注節點 Key；數值越高代表 AI 越重視其影響）",
            fontweight="bold"
        )

        # 印出前 3 大關聯對（排除自身 i==j）
        print("\n[V] Attention 關聯最強 Top-5 股票對：")
        pairs = []
        for i in range(n):
            for j in range(n):
                if i != j:
                    pairs.append((avg_attn[i, j], stock_labels[i], stock_labels[j]))
        pairs.sort(reverse=True)
        for val, src, tgt in pairs[:5]:
            print(f"    {src:20s} → {tgt:20s}  attn={val:.4f}")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\n[V] 圖表已儲存：{output_file}")

    # 播放完成提示音
    for _ in range(3):
        winsound.Beep(1000, 300)
        time.sleep(0.2)
    winsound.Beep(1500, 500)


if __name__ == "__main__":
    run_eval(
        model_path="ppo_portfolio_v3_st_gnn",
        tickers=TICKERS_FULL,
        output_file="portfolio_evaluation_v3.png"
    )
