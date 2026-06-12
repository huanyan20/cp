"""
scripts/prepare_dataset.py - 靜態特徵預編譯腳本

這個腳本會預先下載股票與大盤的歷史資料，並計算所有的技術指標與跨股特徵，
最後將原本會在 TradingEnv 裡動態組合的 Pandas DataFrames 直接轉換為連續的
Numpy Array (.npz 格式)，讓 RL 環境的初始化時間從數十秒縮減至毫秒級別，
並節省跨進程 (VecEnv) 的記憶體佔用。
"""

import argparse
import os
from pathlib import Path
import numpy as np

# 加入專案根目錄到 sys.path，以便讀取設定與模組
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from data_loader import fetch_multi_asset_data
from stock_universe import TICKERS_TECH_EXPANDED, MACRO_TICKERS_RL
from settings import load_settings

def prepare_and_save_dataset(
    tickers: list[str],
    macro_tickers: list[str],
    start_date: str,
    end_date: str,
    window_size: int,
    output_path: str,
    overnight_feature_path: str | None = None,
):
    print(f"=== 開始靜態特徵預編譯 ({start_date} ~ {end_date}) ===")
    # 1. 抓取並計算所有特徵
    df_dict = fetch_multi_asset_data(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        window_size=window_size,
        macro_tickers=macro_tickers,
        overnight_feature_path=overnight_feature_path,
    )

    if not df_dict:
        print("[!] 錯誤：沒有取回任何資料。")
        return

    # 取得實際有資料的 ticker 列表
    actual_tickers = list(df_dict.keys())
    
    # 2. 轉換為 Numpy Arrays (與 TradingEnv 的邏輯對齊)
    print("\n=== 將 DataFrame 轉換為 Numpy Tensor ===")
    market_data = np.stack(
        [df_dict[t].to_numpy(dtype=np.float32) for t in actual_tickers], axis=1
    )
    
    log_returns = np.stack(
        [df_dict[t]["log_return"].to_numpy(dtype=np.float64) for t in actual_tickers],
        axis=1,
    )
    
    # 取出共同的 index 作為 dates
    first_ticker = actual_tickers[0]
    dates = df_dict[first_ticker].index.strftime("%Y-%m-%d").to_numpy()
    
    num_steps, num_stocks, num_features = market_data.shape
    print(f"[V] Market Data 陣列大小: {market_data.shape}")
    print(f"[V] 佔用記憶體: {market_data.nbytes / 1024 / 1024:.2f} MB")

    # 3. 儲存為 .npz 格式
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(
        output_path,
        market_data=market_data,
        log_returns=log_returns,
        tickers=actual_tickers,
        dates=dates,
        window_size=window_size,
        num_features=num_features,
    )
    
    print(f"\n[V] 預編譯特徵庫已成功儲存至: {output_path}")

if __name__ == "__main__":
    settings = load_settings()
    
    parser = argparse.ArgumentParser(description="靜態特徵預編譯工具 (RL Env)")
    parser.add_argument("--start", type=str, default=settings.research.train_start)
    parser.add_argument("--end", type=str, default="2024-06-30")
    parser.add_argument("--window-size", type=int, default=settings.research.window_size)
    parser.add_argument("--out", type=str, default=".cache/npz_data/dataset.npz")
    parser.add_argument("--overnight", type=str, default=settings.research.overnight_feature_path)
    
    args = parser.parse_args()
    
    prepare_and_save_dataset(
        tickers=TICKERS_TECH_EXPANDED,
        macro_tickers=MACRO_TICKERS_RL,
        start_date=args.start,
        end_date=args.end,
        window_size=args.window_size,
        output_path=args.out,
        overnight_feature_path=args.overnight if args.overnight else None,
    )
