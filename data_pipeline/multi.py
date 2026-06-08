import pandas as pd

from stock_universe import SECTOR_GROUPS, get_ticker_sector

from .core import fetch_and_process_data
from .overnight import load_overnight_features
from .utils import BASE_FEATURE_COLS, CROSS_ASSET_COLS, build_feature_schema


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
    if macro_tickers:
        for m_tick in macro_tickers:
            try:
                m_df = fetch_and_process_data(
                    m_tick, start_date=start_date, end_date=end_date, window_size=window_size
                )
                m_df = m_df.add_prefix(f"macro_{m_tick}_")
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

    print(f"\n[V] 多股對齊後有效交易日：{len(lr_aligned)} 筆")
    if len(lr_aligned) < window_size * 2:
        print("[!] 警告：對齊後資料筆數偏少，建議擴展日期範圍。")

    cumret_5d = {t: lr_aligned[t].rolling(5).sum() for t in available}
    avg_5d = pd.DataFrame(cumret_5d).mean(axis=1)

    sector_vol_dict: dict[str, pd.Series] = {}
    for sector_name, sector_tickers in SECTOR_GROUPS.items():
        sector_avail = [t for t in sector_tickers if t in available]
        if sector_avail:
            vols = pd.DataFrame(
                {t: raw_dfs[t]["Volume_norm"].reindex(lr_aligned.index) for t in sector_avail}
            )
            sector_vol_dict[sector_name] = vols.sum(axis=1)

    enriched = {}
    for ticker in available:
        peers = [t for t in available if t != ticker]
        df = raw_dfs[ticker].reindex(lr_aligned.index).copy()

        df["peer1_logret"] = lr_aligned[peers[0]].values
        df["peer2_logret"] = lr_aligned[peers[1]].values if len(peers) > 1 else 0.0

        self_lr = lr_aligned[ticker]
        df["corr_peer1_20d"] = self_lr.rolling(20).corr(lr_aligned[peers[0]]).values
        df["corr_peer2_20d"] = (
            self_lr.rolling(20).corr(lr_aligned[peers[1]]).values if len(peers) > 1 else 0.0
        )

        df["rel_strength"] = (cumret_5d[ticker] - avg_5d).values

        sector_name = get_ticker_sector(ticker)
        own_vol = raw_dfs[ticker]["Volume_norm"].reindex(lr_aligned.index)
        if sector_name in sector_vol_dict:
            sec_total = sector_vol_dict[sector_name]
            sec_share = own_vol / sec_total.replace(0, 1e-8)
            share_mean = sec_share.rolling(20, min_periods=5).mean()
            share_std = sec_share.rolling(20, min_periods=5).std().replace(0, 1.0)
            df["sector_flow"] = (
                (((sec_share - share_mean) / share_std).clip(-3.0, 3.0) / 3.0)
                .reindex(df.index)
                .fillna(0.0)
            )
        else:
            df["sector_flow"] = 0.0

        advancers = pd.DataFrame({t: raw_dfs[t]["Close_norm"].reindex(lr_aligned.index) > 0 for t in available}).sum(axis=1)
        df["market_breadth"] = (advancers / len(available)).values

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

        if overnight_features is not None:
            df = pd.concat([df, overnight_features.reindex(df.index).fillna(0.0)], axis=1)

        df.dropna(inplace=True)

        all_cols = BASE_FEATURE_COLS + CROSS_ASSET_COLS
        if macro_dfs:
            for m_tick, m_df in macro_dfs.items():
                all_cols += list(m_df.columns)
                all_cols += [f"macro_{m_tick}_weekend_gap"]
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
