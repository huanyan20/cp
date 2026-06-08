from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from FinMind.data import DataLoader
except ImportError:
    DataLoader = None


def fetch_fii_futures_oi(start_date: str) -> pd.DataFrame:
    if DataLoader is None:
        print("[WARN] FinMind not installed, skipping FII futures OI.")
        return pd.DataFrame()
    dl = DataLoader()
    try:
        df = dl.taiwan_futures_institutional_investors(data_id="TX", start_date=start_date)
        if df.empty:
            return pd.DataFrame()
            
        df_fii = df[df["institutional_investors"].str.contains("外資", na=False)].copy()
        df_fii["fii_tx_net_oi"] = df_fii["long_open_interest_balance_volume"] - df_fii["short_open_interest_balance_volume"]
        
        df_fii["date"] = pd.to_datetime(df_fii["date"])
        df_fii = df_fii.set_index("date")[["fii_tx_net_oi"]]
        return df_fii.sort_index()
    except Exception as e:
        print(f"[ERROR] fetching FII Futures OI: {e}")
        return pd.DataFrame()


def fetch_retail_long_short_ratio(start_date: str) -> pd.DataFrame:
    if DataLoader is None:
        return pd.DataFrame()
    dl = DataLoader()
    try:
        df_mtx_inst = dl.taiwan_futures_institutional_investors(data_id="MTX", start_date=start_date)
        
        # We need total open interest for MTX.
        df_mtx_total = dl.taiwan_futures_daily_trade(data_id="MTX", start_date=start_date)
        
        if df_mtx_inst.empty or df_mtx_total.empty:
            return pd.DataFrame()
            
        df_mtx_inst["net_oi"] = df_mtx_inst["long_open_interest_balance_volume"] - df_mtx_inst["short_open_interest_balance_volume"]
        inst_net_oi = df_mtx_inst.groupby("date")["net_oi"].sum()
        
        # Filter for regular session and calculate total OI
        if "trading_session" in df_mtx_total.columns:
            df_mtx_total = df_mtx_total[df_mtx_total["trading_session"] == "一般"]
            
        total_oi = df_mtx_total.groupby("date")["open_interest"].max() # use max across contracts for the day, or sum? usually sum of all contracts
        # Wait, open_interest is usually per contract. We should sum them.
        total_oi = df_mtx_total.groupby("date")["open_interest"].sum()
        
        df = pd.DataFrame({"inst_net_oi": inst_net_oi, "total_oi": total_oi}).dropna()
        df["retail_net_oi"] = -df["inst_net_oi"]
        df["retail_long_short_ratio"] = df["retail_net_oi"] / df["total_oi"]
        
        df.index = pd.to_datetime(df.index)
        return df[["retail_long_short_ratio"]].sort_index()
    except Exception as e:
        print(f"[ERROR] fetching Retail MTX Ratio: {e}")
        return pd.DataFrame()


def build_taiwan_chip_features(start_date: str) -> pd.DataFrame:
    print("Fetching Taiwan FII Futures OI...")
    df_fii = fetch_fii_futures_oi(start_date)
    print("Fetching Taiwan Retail MTX Long/Short Ratio...")
    df_retail = fetch_retail_long_short_ratio(start_date)
    
    if df_fii.empty and df_retail.empty:
        return pd.DataFrame()
        
    df = df_fii.join(df_retail, how="outer").ffill()
    
    # Calculate Deltas and Z-scores
    if "fii_tx_net_oi" in df.columns:
        df["fii_tx_net_oi_delta"] = df["fii_tx_net_oi"].diff(3)
        mean = df["fii_tx_net_oi_delta"].rolling(60, min_periods=10).mean()
        std = df["fii_tx_net_oi_delta"].rolling(60, min_periods=10).std().replace(0.0, np.nan)
        df["fii_tx_net_oi_delta_z"] = (df["fii_tx_net_oi_delta"] - mean) / std
        
    if "retail_long_short_ratio" in df.columns:
        df["retail_ratio_delta"] = df["retail_long_short_ratio"].diff(3)
        mean = df["retail_long_short_ratio"].rolling(60, min_periods=10).mean()
        std = df["retail_long_short_ratio"].rolling(60, min_periods=10).std().replace(0.0, np.nan)
        df["retail_ratio_z"] = (df["retail_long_short_ratio"] - mean) / std
        
    return df
