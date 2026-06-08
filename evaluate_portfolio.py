"""
evaluate_portfolio.py - 評估 Portfolio Manager 的績效與資金輪動
輸出：
  1. 組合總資產淨值曲線
  2. 各股資金權重配置熱力圖（板塊輪動視覺化）
"""

import argparse
import json
import os
import time
import winsound
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO, SAC

from data_loader import fetch_multi_asset_data
from metrics_utils import calculate_metrics
from settings import load_settings
from stock_universe import MACRO_TICKERS_RL, TICKER_NAMES, TICKERS_TECH_EXPANDED
from trading_env import TaiwanStockEnv

SETTINGS = load_settings()
TEST_START = SETTINGS.evaluation.test_start
TEST_END = datetime.now().strftime("%Y-%m-%d")
WINDOW_SIZE = SETTINGS.research.window_size


def _signal_aid(default: str = "2249294") -> str:
    return os.getenv("CMONEY_AID", default)


def run_eval(
    model_path: str,
    tickers: list,
    output_file: str = "portfolio_evaluation.png",
    overnight_feature_path: str | None = None,
    half_buys: bool = False,
):
    print(f"=== 下載驗證資料 ({TEST_START} ~ {TEST_END}) ===")
    enriched = fetch_multi_asset_data(
        tickers=tickers,
        start_date=TEST_START,
        end_date=TEST_END,
        window_size=WINDOW_SIZE,
        macro_tickers=MACRO_TICKERS_RL,
        overnight_feature_path=overnight_feature_path,
    )

    model_name_lower = model_path.lower()
    enable_cash_action = "cash" in model_name_lower or "sac" in model_name_lower
    enable_margin_short = "ls" in model_name_lower
    env = TaiwanStockEnv(
        df_dict=enriched,
        window_size=WINDOW_SIZE,
        enable_cash_action=enable_cash_action,
        enable_margin_short=enable_margin_short,
        max_leverage=2.0,
        record_trades=True,
    )

    print(f"\n=== 載入模型：{model_path} ===")
    if "sac" in model_name_lower:
        model = SAC.load(model_path)
    else:
        model = PPO.load(model_path)

    expected_shape = tuple(model.observation_space.shape)
    actual_shape = tuple(env.observation_space.shape)
    if actual_shape != expected_shape:
        if overnight_feature_path:
            print(
                "[!] Model observation shape mismatch: "
                f"model expects {expected_shape}, current data gives {actual_shape}. "
                "Retrying without overnight features."
            )
            enriched = fetch_multi_asset_data(
                tickers=tickers,
                start_date=TEST_START,
                end_date=TEST_END,
                window_size=WINDOW_SIZE,
                macro_tickers=MACRO_TICKERS_RL,
                overnight_feature_path=None,
            )
            env = TaiwanStockEnv(
                df_dict=enriched,
                window_size=WINDOW_SIZE,
                enable_cash_action=enable_cash_action,
                enable_margin_short=enable_margin_short,
                max_leverage=2.0,
                record_trades=True,
            )
            actual_shape = tuple(env.observation_space.shape)
        if actual_shape != expected_shape:
            raise ValueError(
                "Model observation shape mismatch after fallback: "
                f"model expects {expected_shape}, current data gives {actual_shape}."
            )

    obs, _ = env.reset()
    done = False
    portfolio_history = [env.initial_balance]
    positions_history = []
    cash_history = []
    turnover_history = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        portfolio_history.append(info["portfolio_value"])
        positions_history.append(info["positions"].copy())
        cash_history.append(info.get("cash_weight", 0.0))
        turnover_history.append(info.get("turnover", 0.0))

    final_val = portfolio_history[-1]
    
    val_array = np.array(portfolio_history)
    daily_returns = (val_array[1:] / val_array[:-1]) - 1.0
    
    metrics = calculate_metrics(
        portfolio_history,
        positions_history,
        cash_history,
        daily_returns,
        turnover_history,
        env.tickers
    )
    
    total_ret = metrics["total_return"] * 100
    print(f"\n[V] 最終總資產：{final_val:,.0f}")
    print(f"[V] 總報酬率：{total_ret:+.2f}%")
    print(f"[V] 最大回撤：{metrics['max_drawdown']:.2%}")
    print(f"[V] OOS Sortino: {metrics['sortino']:.2f}")
    print(f"[V] Avg Cash Weight: {metrics['avg_cash_weight']:.2%}")
    print(f"[V] Avg Long Exposure: {metrics['long_exposure']:.2f}")
    print(f"[V] Avg Short Exposure: {metrics['short_exposure']:.2f}")
    
    os.makedirs(str(SETTINGS.paths.results_dir), exist_ok=True)
    algo_name = "sac" if "sac" in model_name_lower else "ppo"
    
    out_metrics = metrics.copy()
    out_metrics["algo"] = algo_name
    out_metrics["train_test_period"] = f"{TEST_START} ~ {TEST_END}"
    out_metrics["seed"] = 42 # Default for now
    
    with open(SETTINGS.paths.results_dir / f"metrics_{algo_name}_eval.json", "w", encoding="utf-8") as f:
        json.dump(out_metrics, f, indent=4, ensure_ascii=False)
        
    trades_history = info.get("trades_history", [])
    if trades_history:
        import pandas as pd
        trades_path = SETTINGS.paths.results_dir / f"trades_{algo_name}_eval.csv"
        pd.DataFrame(trades_history).to_csv(trades_path, index=False)
        print(f"[V] Trade logs saved to: {trades_path}")

    # ── 產生最新訊號 (使用完整資料到最後一天) ──────────────────────────────────
    env._current_step = env.max_steps
    final_obs = env._get_observation()
    final_action, _ = model.predict(final_obs, deterministic=True)
    
    next_day_target_weights = env._transform_action(final_action)
    cash_weight = float(env._cash_weight)

    if half_buys:
        next_day_target_weights = next_day_target_weights * 0.5
        cash_weight = 1.0 - np.sum(next_day_target_weights)
        print(f"\n[!] WARN Guard Active: Target weights halved. New cash weight: {cash_weight:.2%}")

    signal_id = f"tech30-{algo_name}-{datetime.now().strftime('%Y%m%d')}"
    weights_dict = {
        t: float(w) for t, w in zip(tickers, next_day_target_weights, strict=True) if w > 0.01
    }
    signal_data = {
        "signal_id": signal_id,
        "created_at": datetime.now().isoformat(),
        "aid": _signal_aid(),
        "target_weights": weights_dict,
        "cash_weight": cash_weight,
    }
    with open(SETTINGS.paths.signal_path, "w", encoding="utf-8") as f:
        json.dump(signal_data, f, indent=4, ensure_ascii=False)
    print(
        f"\n[V] 明日 PPO Target Weights (Top-{env._topk}) 已儲存為 {SETTINGS.paths.signal_path}: {weights_dict}"
    )

    # ── 繪圖 ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(
        f"Portfolio Manager (GNN) Evaluation\n{TEST_START} ~ {TEST_END}", fontsize=14
    )

    # 1. 淨值曲線
    axes[0].plot(
        portfolio_history,
        color="#1f77b4",
        linewidth=2,
        label=f"Portfolio ({total_ret:+.2f}%)",
    )
    axes[0].axhline(
        y=1_000_000, color="gray", linestyle="--", alpha=0.7, label="Initial capital 1M"
    )
    axes[0].fill_between(
        range(len(portfolio_history)),
        1_000_000,
        portfolio_history,
        where=[v > 1_000_000 for v in portfolio_history],
        alpha=0.2,
        color="green",
    )
    axes[0].fill_between(
        range(len(portfolio_history)),
        1_000_000,
        portfolio_history,
        where=[v <= 1_000_000 for v in portfolio_history],
        alpha=0.2,
        color="red",
    )
    axes[0].set_title("Total Portfolio Value", fontweight="bold")
    axes[0].set_ylabel("Value (TWD)")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # 2. 資金輪動熱力圖
    if positions_history:
        pos_matrix = np.array(positions_history).T  # (N_stocks, Steps)
        if cash_history:
            pos_matrix = np.vstack([pos_matrix, np.array(cash_history)])
        stock_labels = [TICKER_NAMES.get(t, t) for t in env.tickers] + ["CASH"]
        im = axes[1].imshow(pos_matrix, aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
        plt.colorbar(im, ax=axes[1], label="Weight (Blue=Long / Red=Short)")
        axes[1].set_yticks(range(len(env.tickers) + 1))
        axes[1].set_yticklabels(stock_labels, fontsize=9)
        axes[1].set_title("Capital Rotation & Allocation Heatmap", fontweight="bold")
        axes[1].set_xlabel("Trading Days")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\n[V] 圖表已儲存：{output_file}")

    # 播放完成提示音
    for _ in range(3):
        winsound.Beep(1000, 300)
        time.sleep(0.2)
    winsound.Beep(1500, 500)

def run_momentum_eval(
    tickers: list,
    lookback=20,
    topk=1,
    power=1.0,
    use_ma_filter=False,
    output_file="portfolio_evaluation_momentum.png",
):
    print(f"=== 下載驗證資料 (Momentum, {TEST_START} ~ {TEST_END}) ===")
    enriched = fetch_multi_asset_data(
        tickers=tickers,
        start_date=TEST_START,
        end_date=TEST_END,
        window_size=lookback,
        macro_tickers=MACRO_TICKERS_RL,
    )

    env = TaiwanStockEnv(df_dict=enriched, window_size=lookback)

    obs, _ = env.reset()
    done = False
    portfolio_history = [env.initial_balance]
    positions_history = []

    target_weights = np.zeros(len(tickers))

    while not done:
        # Get historical closes for the last `lookback` days for momentum calculation
        action = np.zeros(len(tickers))

        # Simplified momentum calculation: uses current day's log_return sum over last `lookback` days
        # using the environment's current observation block
        # For simplicity in this env, we rely on the enriched dataframe directly for the exact dates,
        # but since env step controls the date, we can use env._current_step.

        current_step = env._current_step
        if current_step >= lookback:
            scores = []
            for _i, ticker in enumerate(env.tickers):
                df = env.dfs[ticker]
                # Log returns for the last `lookback` days
                log_rets = (
                    df["log_return"].iloc[current_step - lookback : current_step].sum()
                )

                # Risk control: drop if below 20MA
                if use_ma_filter:
                    current_close = df["Close_norm"].iloc[current_step - 1]
                    ma20 = (
                        df["Close_norm"]
                        .iloc[current_step - lookback : current_step]
                        .mean()
                    )
                    if current_close < ma20:
                        log_rets = -9999.0  # Penalize to drop allocation

                scores.append(log_rets)

            scores = np.array(scores)
            # Find topk
            topk_indices = np.argsort(scores)[-topk:]
            for idx in topk_indices:
                if scores[idx] > -990:  # Only allocate if not penalized
                    action[idx] = 1.0

            if np.sum(action) > 0:
                action = action / np.sum(action)

        target_weights = action
        # bypass_action_transform=True: 一對一把 Momentum 計算的正規化權重直接送給環境，
        # 跳過 env.step() 內的 Softmax+TopK 管道（該管道僅適用於 RL logits）。
        obs, _, terminated, truncated, info = env.step(
            action, bypass_action_transform=True
        )
        done = terminated or truncated
        portfolio_history.append(info["portfolio_value"])
        positions_history.append(info["positions"].copy())

    final_val = portfolio_history[-1]
    total_ret = (final_val / 1_000_000 - 1) * 100
    print(f"\n[V] 最終總資產：{final_val:,.0f}")
    print(f"[V] 總報酬率：{total_ret:+.2f}%")
    print(f"[V] 最大回撤：{env._max_drawdown:.2%}")

    # Output JSON signal
    signal_id = f"tech30-mom-{datetime.now().strftime('%Y%m%d')}"
    weights_dict = {
        t: float(w) for t, w in zip(tickers, target_weights, strict=True) if w > 0
    }
    signal_data = {
        "signal_id": signal_id,
        "created_at": datetime.now().isoformat(),
        "aid": _signal_aid(),  # Prefer the configured live account aid.
        "target_weights": weights_dict,
    }
    with open(SETTINGS.paths.signal_path, "w", encoding="utf-8") as f:
        json.dump(signal_data, f, indent=4, ensure_ascii=False)
    print(f"[V] Signal JSON 已儲存為 {SETTINGS.paths.signal_path}: {weights_dict}")

    # 繪圖
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(
        f"Momentum {lookback}D Top-{topk} Evaluation\n{TEST_START} ~ {TEST_END}",
        fontsize=14,
    )

    axes[0].plot(
        portfolio_history,
        color="#1f77b4",
        linewidth=2,
        label=f"Portfolio ({total_ret:+.2f}%)",
    )
    axes[0].axhline(
        y=1_000_000, color="gray", linestyle="--", alpha=0.7, label="Initial capital 1M"
    )
    axes[0].set_title("Total Portfolio Value", fontweight="bold")
    axes[0].set_ylabel("Value (TWD)")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    if positions_history:
        pos_matrix = np.array(positions_history).T
        stock_labels = [TICKER_NAMES.get(t, t) for t in env.tickers]
        im = axes[1].imshow(pos_matrix, aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
        plt.colorbar(im, ax=axes[1], label="Weight (Blue=Long / Red=Short)")
        axes[1].set_yticks(range(len(env.tickers)))
        axes[1].set_yticklabels(stock_labels, fontsize=9)
        axes[1].set_title("Capital Rotation & Allocation Heatmap", fontweight="bold")
        axes[1].set_xlabel("Trading Days")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\n[V] 圖表已儲存：{output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate portfolio manager model.")
    parser.add_argument(
        "--model-path",
        default=str(SETTINGS.paths.models_dir / SETTINGS.evaluation.model_name),
    )
    parser.add_argument("--output-file", default=SETTINGS.evaluation.output_file)
    parser.add_argument(
        "--overnight-feature-path",
        default=SETTINGS.research.overnight_feature_path,
        help="Optional overnight_gap_features_1d.csv path for RL observation features.",
    )
    parser.add_argument(
        "--half-buys",
        action="store_true",
        help="Halve the target weights and increase cash weight for risk control.",
    )
    parser.add_argument("--test-start", default=SETTINGS.evaluation.test_start)
    parser.add_argument(
        "--test-end",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Evaluation end date. Defaults to today.",
    )
    args = parser.parse_args()

    TEST_START = args.test_start
    TEST_END = args.test_end

    # 執行模型評估
    run_eval(
        model_path=args.model_path,
        tickers=TICKERS_TECH_EXPANDED,
        output_file=args.output_file,
        overnight_feature_path=args.overnight_feature_path,
        half_buys=args.half_buys,
    )
