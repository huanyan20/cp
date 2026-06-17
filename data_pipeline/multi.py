import pandas as pd
import numpy as np

from stock_universe import SECTOR_GROUPS, get_ticker_sector

from .core import fetch_and_process_data
from .overnight import load_overnight_features
from .utils import BASE_FEATURE_COLS, CROSS_ASSET_COLS, build_feature_schema


# Columns added from ^TWII market index (Option A: Market Regime)
TWII_REGIME_COLS = [
    "twii_ma60_ratio",    # TWII Close / 60d MA
    "twii_ma120_ratio",   # TWII Close / 120d MA
    "twii_ma200_ratio",   # TWII Close / 200d MA (classic bull/bear)
    "twii_trend_60d",     # TWII 60d log-return trend (annualised)
    "twii_above_ma120",   # Binary: TWII > 120d MA (1=bull, 0=bear)
]


def _compute_twii_regime(twii_raw_df: pd.DataFrame) -> pd.DataFrame:
    """Compute market-level regime features from raw TWII OHLCV DataFrame.

    All signals are point-in-time safe (no future leakage).
    The raw_df index is the trading-date index as returned by fetch_and_process_data.
    """
    # fetch_and_process_data returns the BASE_FEATURE_COLS slice of TWII.
    # We need the original Close & log_return – use the already computed rolling features.
    # Since fetch_and_process_data already computes price_ma60_ratio etc. for TWII,
    # we can reuse them directly.
    regime = pd.DataFrame(index=twii_raw_df.index)

    if "price_ma60_ratio" in twii_raw_df.columns:
        regime["twii_ma60_ratio"] = twii_raw_df["price_ma60_ratio"].clip(0.5, 2.0)
    else:
        regime["twii_ma60_ratio"] = 1.0

    if "price_ma120_ratio" in twii_raw_df.columns:
        regime["twii_ma120_ratio"] = twii_raw_df["price_ma120_ratio"].clip(0.5, 2.0)
        regime["twii_above_ma120"] = (twii_raw_df["price_ma120_ratio"] > 1.0).astype(float)
    else:
        regime["twii_ma120_ratio"] = 1.0
        regime["twii_above_ma120"] = 0.5

    if "price_ma200_ratio" in twii_raw_df.columns:
        regime["twii_ma200_ratio"] = twii_raw_df["price_ma200_ratio"].clip(0.5, 2.0)
    else:
        regime["twii_ma200_ratio"] = 1.0

    if "trend_slope_60d" in twii_raw_df.columns:
        regime["twii_trend_60d"] = twii_raw_df["trend_slope_60d"]
    elif "log_return" in twii_raw_df.columns:
        regime["twii_trend_60d"] = twii_raw_df["log_return"].rolling(60).mean() * 252
    else:
        regime["twii_trend_60d"] = 0.0

    return regime.ffill().bfill()


def fetch_multi_asset_data(
    tickers: list | None = None,
    start_date: str = "2023-06-01",
    end_date: str = "2024-06-30",
    window_size: int = 20,
    macro_tickers: list | None = None,
    overnight_feature_path: str | None = None,
    overnight_feature_cols: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    if tickers is None:
        tickers = ["2330.TW", "00919.TW", "00929.TW"]
    if macro_tickers is None:
        macro_tickers = ["^TWII", "^IXIC", "USDTWD=X"]

    print(f"\n=== 下載並對齊多股票資料 ({start_date} ~ {end_date}) ===")

    raw_dfs = {}
    for ticker in tickers:
        try:
            raw_dfs[ticker] = fetch_and_process_data(
                ticker, start_date=start_date, end_date=end_date, window_size=window_size
            )
        except Exception as e:
            print(f"[!] {ticker} 下載失敗，跳過：{e}")

    if len(raw_dfs) < 2:
        raise RuntimeError("至少需要 2 支股票才能計算跨股特徵！")

    macro_dfs = {}
    twii_raw_df = None  # keep raw TWII for regime feature computation
    if macro_tickers:
        for m_tick in macro_tickers:
            try:
                m_df = fetch_and_process_data(
                    m_tick, start_date=start_date, end_date=end_date, window_size=window_size
                )
                if m_tick == "^TWII":
                    twii_raw_df = m_df  # save before prefixing
                m_df = m_df.add_prefix(f"macro_{m_tick}_")
                
                # [FIX] Strict Point-in-Time alignment for foreign indices (prevent future leakage)
                if m_tick != "^TWII":
                    m_df = m_df.shift(1)
                    
                macro_dfs[m_tick] = m_df
            except Exception as e:
                print(f"[!] 大盤 {m_tick} 下載失敗，跳過：{e}")

    available = list(raw_dfs.keys())

    lr_dict = {t: raw_dfs[t]["log_return"] for t in available}
    lr_aligned = pd.DataFrame(lr_dict).dropna()

    overnight_features = None
    if overnight_feature_path:
        try:
            overnight_features = load_overnight_features(
                overnight_feature_path, feature_cols=overnight_feature_cols
            ).reindex(lr_aligned.index).ffill().fillna(0.0)
            print(f"[V] overnight features loaded: {overnight_features.shape[1]} columns")
        except Exception as e:
            print(f"[!] overnight features 載入失敗，略過：{e}")
            overnight_features = None

    # Compute TWII market regime features (Option A) once for all stocks
    twii_regime_df = None
    if twii_raw_df is not None:
        try:
            twii_regime_df = _compute_twii_regime(twii_raw_df)
            print(f"[V] TWII Regime features computed: {list(twii_regime_df.columns)}")
        except Exception as e:
            print(f"[!] TWII Regime feature computation failed: {e}")
            twii_regime_df = None

    print(f"\n[V] 多股對齊後有效交易日：{len(lr_aligned)} 筆")
    if len(lr_aligned) < window_size * 2:
        print("[!] 警告：對齊後資料筆數偏少，建議擴展日期範圍。")

    enriched = {}
    for ticker in available:
        df = raw_dfs[ticker].reindex(lr_aligned.index).copy()

        if macro_dfs:
            for m_tick, m_df in macro_dfs.items():
                macro_aligned = m_df.reindex(lr_aligned.index).ffill().bfill()
                valid_dates = lr_aligned.index
                weekend_gaps = pd.Series(0.0, index=valid_dates)
                m_ret_col = f"macro_{m_tick}_log_return"

                if m_ret_col in m_df.columns:
                    m_returns = m_df[m_ret_col]
                    m_log_ret_df = pd.DataFrame({m_ret_col: m_returns})
                    m_log_ret_df["tw_date"] = pd.NaT

                    intersection = m_log_ret_df.index.intersection(valid_dates)
                    m_log_ret_df.loc[intersection, "tw_date"] = intersection
                    m_log_ret_df["tw_date"] = m_log_ret_df["tw_date"].bfill()

                    m_log_ret_df["is_tw_trading_day"] = m_log_ret_df.index.isin(valid_dates)
                    gap_only = m_log_ret_df[~m_log_ret_df["is_tw_trading_day"]]

                    gap_sums = gap_only.groupby("tw_date")[m_ret_col].sum()
                    weekend_gaps.update(gap_sums)

                macro_aligned[f"macro_{m_tick}_weekend_gap"] = weekend_gaps.values
                df = pd.concat([df, macro_aligned], axis=1)

        # Attach TWII market regime features (clean column names, point-in-time safe)
        if twii_regime_df is not None:
            regime_aligned = twii_regime_df.reindex(lr_aligned.index).ffill().bfill()
            df = pd.concat([df, regime_aligned], axis=1)

        if overnight_features is not None:
            df = pd.concat([df, overnight_features.reindex(df.index).fillna(0.0)], axis=1)

        df.dropna(inplace=True)

        all_cols = BASE_FEATURE_COLS + CROSS_ASSET_COLS
        if macro_dfs:
            for m_tick, m_df in macro_dfs.items():
                all_cols += list(m_df.columns)
                all_cols += [f"macro_{m_tick}_weekend_gap"]
        # Add TWII regime columns to schema if computed
        if twii_regime_df is not None:
            all_cols += TWII_REGIME_COLS
        if overnight_features is not None:
            all_cols += list(overnight_features.columns)

        schema = build_feature_schema(
            macro_features=all_cols[len(BASE_FEATURE_COLS + CROSS_ASSET_COLS) :],
            overnight_features=(),
        )
        schema.validate(df)
        enriched[ticker] = df[list(schema.columns)].copy()

        print(
            f"  [{ticker}] 跨股/大盤特徵附加完成 → "
            f"{len(enriched[ticker])} 筆 × {schema.observation_dim} 特徵"
        )

    return enriched

