from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from .utils import BASE_FEATURE_COLS

# keep yfinance cache config
YFINANCE_CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "yfinance"
YFINANCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
yf.set_tz_cache_location(str(YFINANCE_CACHE_DIR))


def fetch_and_process_data(
    ticker: str = "2330.TW",
    start_date: str = "2018-01-01",
    end_date: str = "2023-12-31",
    window_size: int = 20,
) -> pd.DataFrame:
    cache_dir = YFINANCE_CACHE_DIR / "raw_parquet"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{ticker}_{start_date}_{end_date}.parquet"

    if cache_path.exists():
        print(f"> 從 Parquet 快取讀取 {ticker} ({start_date} ~ {end_date})...")
        raw = pd.read_parquet(cache_path)
    else:
        print(f"> 正在下載 {ticker} 歷史數據 ({start_date} ~ {end_date})...")
        raw = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
        
        if raw.empty:
            raise ValueError(f"無法下載 {ticker} 的資料，請確認股票代號與網路連線。")
            
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
            
        raw.to_parquet(cache_path)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)

    # RSI
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss
    df["RSI_14"] = (100 - (100 / (1 + rs))).fillna(50.0)

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD_12_26_9"] = ema12 - ema26
    df["MACDs_12_26_9"] = df["MACD_12_26_9"].ewm(span=9, adjust=False).mean()
    df["MACDh_12_26_9"] = df["MACD_12_26_9"] - df["MACDs_12_26_9"]

    # Bollinger Bands
    df["BBM_20_2.0_2.0"] = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()
    df["BBU_20_2.0_2.0"] = df["BBM_20_2.0_2.0"] + 2 * std
    df["BBL_20_2.0_2.0"] = df["BBM_20_2.0_2.0"] - 2 * std
    band_diff = (df["BBU_20_2.0_2.0"] - df["BBL_20_2.0_2.0"]).replace(0, 1e-8)
    df["BBB_20_2.0_2.0"] = band_diff / df["BBM_20_2.0_2.0"] * 100
    df["BBP_20_2.0_2.0"] = (df["Close"] - df["BBL_20_2.0_2.0"]) / band_diff

    # ATR
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - df["Close"].shift()).abs()
    tr3 = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATRr_14"] = tr.rolling(14).mean()

    # ADX
    up_move = df["High"] - df["High"].shift()
    down_move = df["Low"].shift() - df["Low"]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    tr14 = tr.rolling(14).sum().replace(0, 1e-8)
    df["DMP_14"] = 100 * pd.Series(plus_dm, index=df.index).rolling(14).sum() / tr14
    df["DMN_14"] = 100 * pd.Series(minus_dm, index=df.index).rolling(14).sum() / tr14
    dx = (
        100
        * (df["DMP_14"] - df["DMN_14"]).abs()
        / (df["DMP_14"] + df["DMN_14"]).replace(0, 1e-8)
    )
    df["ADX_14"] = dx.rolling(14).mean()

    # STOCH
    lowest_low = df["Low"].rolling(14).min()
    highest_high = df["High"].rolling(14).max()
    stoch_k = 100 * (df["Close"] - lowest_low) / (highest_high - lowest_low)
    df["STOCHk_14_3_3"] = stoch_k.rolling(3).mean()
    df["STOCHd_14_3_3"] = df["STOCHk_14_3_3"].rolling(3).mean()

    # OBV
    df["OBV"] = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()

    # MFI
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    raw_money_flow = typical_price * df["Volume"]
    pos_flow = np.where(typical_price > typical_price.shift(), raw_money_flow, 0)
    neg_flow = np.where(typical_price < typical_price.shift(), raw_money_flow, 0)
    pos_flow_sum = pd.Series(pos_flow, index=df.index).rolling(14).sum()
    neg_flow_sum = pd.Series(neg_flow, index=df.index).rolling(14).sum()
    mfi_ratio = pos_flow_sum / neg_flow_sum.replace(0, 1e-8)
    df["MFI_14"] = (100 - (100 / (1 + mfi_ratio))).fillna(50.0)

    df.dropna(inplace=True)

    # 正規化
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["open_return"] = np.log(df["Open"] / df["Close"].shift(1))

    bbm = df["BBM_20_2.0_2.0"]
    for col in ["Open", "High", "Low", "Close"]:
        df[f"{col}_norm"] = (df[col] - bbm) / bbm.replace(0, np.nan)

    vol_ma = df["Volume"].rolling(20).mean()
    df["Volume_norm"] = np.log1p(df["Volume"] / vol_ma.replace(0, 1.0))

    df["RSI_norm"] = df["RSI_14"] / 100.0
    df["MACD_norm"] = df["MACD_12_26_9"] / df["Close"]
    df["MACDs_norm"] = df["MACDs_12_26_9"] / df["Close"]
    df["MACDh_norm"] = df["MACDh_12_26_9"] / df["Close"]

    df["BB_bandwidth"] = df["BBB_20_2.0_2.0"] / 100.0
    df["BB_pct_b"] = df["BBP_20_2.0_2.0"]
    df["BBU_norm"] = (df["BBU_20_2.0_2.0"] - bbm) / bbm.replace(0, np.nan)
    df["BBL_norm"] = (df["BBL_20_2.0_2.0"] - bbm) / bbm.replace(0, np.nan)

    df["ADX_norm"] = df["ADX_14"] / 100.0
    df["DMP_norm"] = df["DMP_14"] / 100.0
    df["DMN_norm"] = df["DMN_14"] / 100.0
    df["ATR_norm"] = df["ATRr_14"] / df["Close"]

    if "STOCHk_14_3_3" in df.columns:
        df["STOCHk_norm"] = df["STOCHk_14_3_3"] / 100.0
        df["STOCHd_norm"] = df["STOCHd_14_3_3"] / 100.0
    else:
        df["STOCHk_norm"] = 0.5
        df["STOCHd_norm"] = 0.5

    obv_ma = df["OBV"].rolling(20).mean()
    obv_std = df["OBV"].rolling(20).std()
    df["OBV_norm"] = (df["OBV"] - obv_ma) / obv_std.replace(0, 1.0)

    if "MFI_14" in df.columns:
        df["MFI_norm"] = df["MFI_14"].fillna(50.0) / 100.0
    else:
        df["MFI_norm"] = 0.5

    # === Milestone 3A: Feature Zoo Generation ===
    # Group 1: Momentum Family
    df["ret_5d"] = df["log_return"].rolling(5).sum()
    df["ret_10d"] = df["log_return"].rolling(10).sum()
    df["ret_20d"] = df["log_return"].rolling(20).sum()
    df["ret_60d"] = df["log_return"].rolling(60).sum()
    
    df["price_ma20_ratio"] = df["Close"] / df["Close"].rolling(20).mean()
    df["price_ma60_ratio"] = df["Close"] / df["Close"].rolling(60).mean()
    df["price_ma120_ratio"] = df["Close"] / df["Close"].rolling(120).mean()
    
    # Group 2: Volatility Family
    tr_series = df["High"] - df["Low"]
    tr2_series = (df["High"] - df["Close"].shift()).abs()
    tr3_series = (df["Low"] - df["Close"].shift()).abs()
    true_range = pd.concat([tr_series, tr2_series, tr3_series], axis=1).max(axis=1)
    
    df["atr_20"] = true_range.rolling(20).mean() / df["Close"]
    df["atr_60"] = true_range.rolling(60).mean() / df["Close"]
    
    df["rolling_std_20"] = df["log_return"].rolling(20).std()
    df["rolling_std_60"] = df["log_return"].rolling(60).std()
    
    # Group 3: Liquidity Family
    vol_mean_20 = df["Volume"].rolling(20).mean()
    vol_std_20 = df["Volume"].rolling(20).std().replace(0, 1.0)
    df["volume_zscore_20"] = (df["Volume"] - vol_mean_20) / vol_std_20
    
    vol_mean_60 = df["Volume"].rolling(60).mean()
    vol_std_60 = df["Volume"].rolling(60).std().replace(0, 1.0)
    df["volume_zscore_60"] = (df["Volume"] - vol_mean_60) / vol_std_60
    
    df["dollar_volume_log"] = np.log1p(df["Close"] * df["Volume"])
    df["volume_ma60_ratio"] = df["Volume"] / vol_mean_60.replace(0, np.nan)

    # Group 4: Market Regime Features (Milestone 3B fix)
    # These are stock-level features derived from own price history,
    # but serve as proxies for market regime state.
    # The 200-day MA ratio is the classic bull/bear regime indicator.
    ma200 = df["Close"].rolling(200).mean()
    df["price_ma200_ratio"] = (df["Close"] / ma200.replace(0, np.nan)).clip(0.5, 2.0)

    # 60-day trend slope (linearly detrended return rate, annualized)
    # Positive = uptrend, Negative = downtrend
    df["trend_slope_60d"] = df["log_return"].rolling(60).mean() * 252

    # Regime flag: is price > 120-day MA? (1=bull, 0=bear)
    ma120 = df["Close"].rolling(120).mean()
    df["above_ma120"] = (df["Close"] > ma120).astype(float)

    df.dropna(inplace=True)

    result = df[BASE_FEATURE_COLS].copy()

    print(
        f"[V] {ticker} 資料處理完成！有效交易日：{len(result)} 筆，特徵數：{len(BASE_FEATURE_COLS)}"
    )
    if len(result) < window_size * 2:
        print(f"[!] 警告：資料筆數 ({len(result)}) 偏少，建議延長日期範圍。")

    return result
