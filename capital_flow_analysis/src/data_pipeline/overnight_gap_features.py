from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from capital_flow_analysis.src.data_pipeline.market_calendar import (  # noqa: E402,F401
    TAIPEI_TZ,
    map_available_to_tw_trade_date,
    source_age_hours,
    us_close_available_at_taipei,
)
from capital_flow_analysis.src.data_pipeline.market_data_provider import (  # noqa: E402
    CompositeProvider,
    FetchResult,
)

TW_TICKERS = ["2330.TW", "2303.TW", "3711.TW"]
ADR_TICKERS = ["TSM", "UMC", "ASX"]
MACRO_TICKERS = ["^SOX", "^IXIC", "^VIX", "JPY=X", "DX-Y.NYB", "USDTWD=X", "BTC-USD", "IWM", "QQQ", "HG=F", "GC=F", "EWT"]
ALL_TICKERS = TW_TICKERS + ADR_TICKERS + MACRO_TICKERS

TAIWAN_ADR_PAIRS = {
    "TSM": "2330.TW",
    "UMC": "2303.TW",
    "ASX": "3711.TW",
}

ADR_SHARE_RATIO = {
    "TSM": [{"start": "1997-01-01", "ordinary_shares": 5}],
    "UMC": [{"start": "2000-09-19", "ordinary_shares": 5}],
    "ASX": [{"start": "2018-04-30", "ordinary_shares": 2}],
}

US_SYMBOL_LABELS = {
    "TSM": "TSM",
    "UMC": "UMC",
    "ASX": "ASX",
    "^SOX": "SOX",
    "^IXIC": "IXIC",
    "^VIX": "VIX",
    "JPY=X": "USD_JPY",
    "DX-Y.NYB": "DXY",
    "BTC-USD": "BTC",
    "IWM": "IWM",
    "QQQ": "QQQ",
    "HG=F": "COPPER",
    "GC=F": "GOLD",
    "EWT": "EWT",
}

PRIMARY_OUTPUT_COLUMNS = [
    "tw_trade_date",
    "target_2330_open_gap",
    "target_2330_intraday",
    "target_2330_full_day",
    "target_2303_open_gap",
    "target_3711_open_gap",
    "target_gap_fade",
    "tsm_adr_premium",
    "tsm_adr_premium_chg",
    "tsm_adr_ret",
    "sox_ret",
    "ixic_ret",
    "sox_nasdaq_spread",
    "sox_spread_z",
    "semi_hardware_stress_flag",
    "semi_adr_positive_count",
    "semi_adr_weighted_ret",
    "tsm_only_strength_flag",
    "vix_ret",
    "vix_ret_z",
    "vix_level_z",
    "vix_panic_combo",
    "jpy_strength",
    "dxy_ret",
    "carry_unwind_flag",
    "btc_weekend_gap",
    "copper_ret",
    "gold_ret",
    "gold_copper_ratio",
    "gold_copper_ratio_chg",
    "iwm_ret",
    "qqq_ret",
    "iwm_qqq_spread",
    "ewt_ret",
    "source_age_hours",
    "fx_data_source_risk",
    "corporate_action_flag",
    "gap_followthrough_risk",
    "data_provider",
    "data_stale_flag",
    "fii_tx_net_oi",
    "fii_tx_net_oi_delta",
    "fii_tx_net_oi_delta_z",
    "retail_long_short_ratio",
    "retail_ratio_delta",
    "retail_ratio_z",
]


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(-1)
    idx = pd.DatetimeIndex(pd.to_datetime(out.index))
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    out.index = idx.normalize()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def extract_symbol_frame(df_raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()
    if isinstance(df_raw.columns, pd.MultiIndex):
        if symbol not in df_raw.columns.get_level_values(0):
            return pd.DataFrame()
        return normalize_ohlcv(df_raw[symbol])
    return normalize_ohlcv(df_raw)


def safe_log_return(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    ret = np.log(values / values.shift(1))
    return ret.replace([np.inf, -np.inf], np.nan)


def rolling_zscore(series: pd.Series, window: int = 252, min_periods: int | None = None) -> pd.Series:
    if min_periods is None:
        min_periods = max(5, window // 2)
    values = series.astype(float)
    mean = values.rolling(window, min_periods=min_periods).mean()
    std = values.rolling(window, min_periods=min_periods).std().replace(0.0, np.nan)
    return (values - mean) / std


def get_adr_ratio(symbol: str, trade_date: pd.Timestamp) -> float:
    rules = ADR_SHARE_RATIO[symbol]
    date_value = pd.Timestamp(trade_date).normalize()
    applicable = [
        rule
        for rule in rules
        if pd.Timestamp(rule["start"]).normalize() <= date_value
    ]
    if not applicable:
        raise ValueError(f"No ADR ratio configured for {symbol} on {trade_date}")
    return float(applicable[-1]["ordinary_shares"])


def align_us_asset_to_tw_dates(
    asset_df: pd.DataFrame,
    tw_trade_dates: pd.DatetimeIndex,
    label: str,
) -> pd.DataFrame:
    df = normalize_ohlcv(asset_df)
    dates = pd.DatetimeIndex(pd.to_datetime(tw_trade_dates)).normalize()
    if df.empty or "Close" not in df.columns:
        return pd.DataFrame(index=dates)

    work = pd.DataFrame(index=df.index)
    work["close"] = df["Close"].astype(float)
    work["volume"] = df["Volume"].astype(float) if "Volume" in df.columns else np.nan
    work["log_return"] = safe_log_return(work["close"])
    work["available_at_taipei"] = [us_close_available_at_taipei(d) for d in work.index]
    work["tw_trade_date"] = [
        map_available_to_tw_trade_date(ts, dates)
        for ts in work["available_at_taipei"]
    ]
    work = work.dropna(subset=["tw_trade_date"]).sort_values("available_at_taipei")
    if work.empty:
        return pd.DataFrame(index=dates)

    grouped = work.groupby("tw_trade_date", sort=True)
    aligned = pd.DataFrame(index=pd.DatetimeIndex(grouped.size().index).normalize())
    aligned[f"{label}_close"] = grouped["close"].last()
    aligned[f"{label}_ret"] = grouped["log_return"].sum(min_count=1)
    aligned[f"{label}_volume"] = grouped["volume"].sum(min_count=1)
    aligned[f"{label}_available_at_taipei"] = grouped["available_at_taipei"].last()
    
    aligned = aligned.reindex(dates)
    aligned[f"{label}_close"] = aligned[f"{label}_close"].ffill()
    aligned[f"{label}_ret"] = aligned[f"{label}_ret"].fillna(0.0)
    aligned[f"{label}_volume"] = aligned[f"{label}_volume"].fillna(0.0)
    aligned[f"{label}_available_at_taipei"] = aligned[f"{label}_available_at_taipei"].ffill()
    
    aligned[f"{label}_source_age_hours"] = [
        source_age_hours(trade_date, available_at) if pd.notna(available_at) else np.nan
        for trade_date, available_at in aligned[f"{label}_available_at_taipei"].items()
    ]
    return aligned


def align_daily_fx_to_tw_dates(fx_df: pd.DataFrame, tw_trade_dates: pd.DatetimeIndex) -> pd.Series:
    df = normalize_ohlcv(fx_df)
    dates = pd.DatetimeIndex(pd.to_datetime(tw_trade_dates)).normalize()
    if df.empty or "Close" not in df.columns:
        return pd.Series(np.nan, index=dates, name="fx_usdtwd_available")

    close = df["Close"].astype(float).sort_index()
    values = []
    for trade_date in dates:
        eligible = close.loc[close.index < trade_date]
        values.append(float(eligible.iloc[-1]) if not eligible.empty else np.nan)
    return pd.Series(values, index=dates, name="fx_usdtwd_available").ffill()


def build_corporate_action_flag(
    frames: dict[str, pd.DataFrame],
    tw_trade_dates: pd.DatetimeIndex,
) -> pd.Series:
    dates = pd.DatetimeIndex(pd.to_datetime(tw_trade_dates)).normalize()
    flag = pd.Series(False, index=dates)
    for ticker in TW_TICKERS + ADR_TICKERS:
        df = normalize_ohlcv(frames.get(ticker, pd.DataFrame()))
        if df.empty:
            continue
        action_dates = pd.Series(False, index=dates)
        for col in ["Dividends", "Stock Splits", "Capital Gains"]:
            if col in df.columns:
                action_dates = action_dates | (
                    df[col].reindex(dates).fillna(0.0).astype(float) != 0.0
                )
        flag = flag | action_dates
    return flag.rename("corporate_action_flag")


def liquidity_weighted_mean(row: pd.Series, ret_cols: list[str], weight_cols: list[str]) -> float:
    returns = []
    weights = []
    for ret_col, weight_col in zip(ret_cols, weight_cols, strict=True):
        value = row.get(ret_col)
        weight = row.get(weight_col)
        if pd.notna(value) and pd.notna(weight) and weight > 0:
            returns.append(float(value))
            weights.append(float(weight))
    if not weights or sum(weights) <= 0:
        return np.nan
    return float(np.average(returns, weights=weights))


def load_all_data(start_date: str = "2020-01-01", end_date: str | None = None) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    provider = CompositeProvider()
    result = provider.fetch(ALL_TICKERS, start=start_date, end=end_date, interval="1d")
    if result.data.empty:
        raise RuntimeError(f"No market data available. Attempts: {result.metadata.get('attempts', [])}")
    if result.warnings:
        print(f"[WARN] provider={result.provider} warnings={result.warnings}")
    return result.data, TW_TICKERS, ADR_TICKERS, MACRO_TICKERS


def build_overnight_gap_features(
    frames: dict[str, pd.DataFrame],
    provider_name: str = "in_memory",
    provider_warnings: list[str] | None = None,
    drop_incomplete: bool = True,
) -> pd.DataFrame:
    if "2330.TW" not in frames:
        raise KeyError("2330.TW is required to build target labels")

    tw_2330 = normalize_ohlcv(frames["2330.TW"])
    if tw_2330.empty or not {"Open", "Close"}.issubset(tw_2330.columns):
        raise ValueError("2330.TW must contain Open and Close columns")

    tw_dates = pd.DatetimeIndex(tw_2330.index).normalize()
    result = pd.DataFrame(index=tw_dates)
    result.index.name = "tw_trade_date"

    for ticker in TW_TICKERS:
        frame = normalize_ohlcv(frames.get(ticker, pd.DataFrame()))
        if frame.empty or not {"Open", "Close"}.issubset(frame.columns):
            continue
        code = ticker.replace(".TW", "")
        prev_close = frame["Close"].astype(float).shift(1).reindex(result.index)
        result[f"{ticker}_Close_prev"] = prev_close
        result[f"target_{code}_open_gap"] = np.log(
            frame["Open"].astype(float).reindex(result.index) / prev_close
        )
        result[f"target_{code}_intraday"] = np.log(
            frame["Close"].astype(float).reindex(result.index)
            / frame["Open"].astype(float).reindex(result.index)
        )
        result[f"target_{code}_full_day"] = np.log(
            frame["Close"].astype(float).reindex(result.index) / prev_close
        )

    for symbol, label in US_SYMBOL_LABELS.items():
        aligned = align_us_asset_to_tw_dates(frames.get(symbol, pd.DataFrame()), tw_dates, label)
        result = result.join(aligned)

    result["fx_usdtwd_available"] = align_daily_fx_to_tw_dates(
        frames.get("USDTWD=X", pd.DataFrame()),
        tw_dates,
    )
    result["fx_data_source_risk"] = 1
    result["corporate_action_flag"] = build_corporate_action_flag(frames, tw_dates)

    for adr_symbol, tw_symbol in TAIWAN_ADR_PAIRS.items():
        prefix = adr_symbol.lower()
        tw_frame = normalize_ohlcv(frames.get(tw_symbol, pd.DataFrame()))
        if tw_frame.empty or "Close" not in tw_frame.columns:
            result[f"{prefix}_adr_premium"] = np.nan
            result[f"{prefix}_adr_premium_chg"] = np.nan
            result[f"{prefix}_adr_ret"] = result.get(f"{adr_symbol}_ret", np.nan)
            continue
        ratio = pd.Series(
            [get_adr_ratio(adr_symbol, trade_date) for trade_date in result.index],
            index=result.index,
        )
        prev_local_close = tw_frame["Close"].astype(float).shift(1).reindex(result.index)
        result[f"{prefix}_adr_twd"] = (
            result[f"{adr_symbol}_close"] * result["fx_usdtwd_available"] / ratio
        )
        result[f"{prefix}_adr_premium"] = np.log(
            result[f"{prefix}_adr_twd"] / prev_local_close
        )
        result[f"{prefix}_adr_premium_chg"] = result[f"{prefix}_adr_premium"].diff()
        result[f"{prefix}_adr_ret"] = result[f"{adr_symbol}_ret"]

    result["sox_ret"] = result["SOX_ret"]
    result["ixic_ret"] = result["IXIC_ret"]
    result["sox_nasdaq_spread"] = result["sox_ret"] - result["ixic_ret"]
    result["sox_spread_z"] = rolling_zscore(result["sox_nasdaq_spread"])
    result["semi_hardware_stress_flag"] = (
        (result["sox_spread_z"] <= -1.5) & (result["sox_ret"] < 0)
    )

    adr_ret_cols = ["tsm_adr_ret", "umc_adr_ret", "asx_adr_ret"]
    adr_weight_cols = []
    valid_cols = []
    for adr_symbol in ADR_TICKERS:
        volume = result.get(f"{adr_symbol}_volume", pd.Series(np.nan, index=result.index)).astype(float)
        result[f"adr_volume_z_{adr_symbol}"] = rolling_zscore(np.log1p(volume.clip(lower=0)))
        rolling_median = volume.rolling(60, min_periods=5).median()
        valid = (volume > rolling_median * 0.3).fillna(volume > 0)
        valid_col = f"adr_liquidity_valid_{adr_symbol}"
        weight_col = f"adr_liquidity_weight_{adr_symbol}"
        result[valid_col] = valid
        result[weight_col] = np.where(valid, np.log1p(volume.clip(lower=0)), 0.0)
        valid_cols.append(valid_col)
        adr_weight_cols.append(weight_col)

    positive_count = pd.Series(0, index=result.index, dtype=int)
    for ret_col, valid_col in zip(adr_ret_cols, valid_cols, strict=True):
        positive_count += ((result[ret_col] > 0) & result[valid_col]).astype(int)
    result["semi_adr_positive_count"] = positive_count
    result["semi_adr_weighted_ret"] = result.apply(
        liquidity_weighted_mean,
        axis=1,
        ret_cols=adr_ret_cols,
        weight_cols=adr_weight_cols,
    )
    result["tsm_only_strength_flag"] = (
        (result["tsm_adr_ret"] > 0)
        & (result["umc_adr_ret"] <= 0)
        & (result["asx_adr_ret"] <= 0)
    )

    # De-fragment the DataFrame to prevent PerformanceWarnings from further insertions
    result = result.copy()

    result["vix_ret"] = result["VIX_ret"]
    result["vix_ret_z"] = rolling_zscore(result["vix_ret"])
    result["vix_level_z"] = rolling_zscore(np.log(result["VIX_close"]))
    result["vix_panic_combo"] = (result["vix_ret_z"] >= 1.5) & (result["vix_level_z"] >= 1.0)

    result["jpy_strength"] = -result["USD_JPY_ret"]
    result["jpy_strength_z"] = rolling_zscore(result["jpy_strength"])
    result["dxy_ret"] = result["DXY_ret"]
    result["carry_unwind_flag"] = (result["jpy_strength_z"] >= 1.5) & (result["sox_ret"] < 0)

    # Weekend/holiday compression: any BTC sessions mapped to the same TW date are already summed.
    result["btc_weekend_gap"] = result.get("BTC_ret", pd.Series(0.0, index=result.index)).fillna(0.0)

    result["copper_ret"] = result.get("COPPER_ret", pd.Series(np.nan, index=result.index))
    result["gold_ret"] = result.get("GOLD_ret", pd.Series(np.nan, index=result.index))
    if "GOLD_close" in result.columns and "COPPER_close" in result.columns:
        result["gold_copper_ratio"] = np.log(result["GOLD_close"] / result["COPPER_close"].replace(0, np.nan))
        result["gold_copper_ratio_chg"] = result["gold_copper_ratio"].diff()
    else:
        result["gold_copper_ratio"] = np.nan
        result["gold_copper_ratio_chg"] = np.nan

    result["iwm_ret"] = result.get("IWM_ret", pd.Series(np.nan, index=result.index))
    result["qqq_ret"] = result.get("QQQ_ret", pd.Series(np.nan, index=result.index))
    result["iwm_qqq_spread"] = result["iwm_ret"] - result["qqq_ret"]

    result["ewt_ret"] = result.get("EWT_ret", pd.Series(np.nan, index=result.index))

    age_cols = [col for col in result.columns if col.endswith("_source_age_hours")]
    result["source_age_hours"] = result[age_cols].max(axis=1) if age_cols else np.nan
    result["data_provider"] = provider_name
    result["data_stale_flag"] = result["source_age_hours"].fillna(999) > 36
    if provider_warnings:
        result["data_provider_warnings"] = "; ".join(provider_warnings)
    else:
        result["data_provider_warnings"] = ""

    result["target_gap_fade"] = (
        (result["target_2330_open_gap"] > 0.01)
        & (result["target_2330_intraday"] < -0.005)
    )
    result["gap_followthrough_risk"] = (
        (result["vix_panic_combo"] | result["carry_unwind_flag"])
        & (result["tsm_adr_premium"] > 0)
    )

    if drop_incomplete:
        result = result.dropna(
            subset=[
                "target_2330_open_gap",
                "target_2330_intraday",
                "target_2330_full_day",
                "tsm_adr_premium",
            ]
        )

    out = result.reset_index()
    ordered = [col for col in PRIMARY_OUTPUT_COLUMNS if col in out.columns]
    remaining = [col for col in out.columns if col not in ordered]
    return out[ordered + remaining]


def build_overnight_features(start_date: str = "2020-01-01", end_date: str | None = None) -> pd.DataFrame:
    provider = CompositeProvider()
    fetch_result: FetchResult = provider.fetch(
        ALL_TICKERS,
        start=start_date,
        end=end_date,
        interval="1d",
    )
    if fetch_result.data.empty:
        raise RuntimeError(f"No market data available: {fetch_result.metadata.get('attempts', [])}")

    frames = {symbol: extract_symbol_frame(fetch_result.data, symbol) for symbol in ALL_TICKERS}
    features = build_overnight_gap_features(
        frames,
        provider_name=fetch_result.provider,
        provider_warnings=fetch_result.warnings,
    )

    try:
        from capital_flow_analysis.src.data_pipeline.taiwan_chip_features import (
            build_taiwan_chip_features,
        )
        chip_features = build_taiwan_chip_features(start_date)
        if not chip_features.empty:
            chip_features.index.name = "tw_trade_date"
            features = features.merge(chip_features, on="tw_trade_date", how="left")
    except Exception as e:
        print(f"Warning: Failed to merge Taiwan chip features: {e}")

    output_dir = Path(__file__).resolve().parents[2] / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "overnight_gap_features_1d.csv"
    features.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
    return features


def write_feature_report(features: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    target_cols = {
        "target_2330_open_gap",
        "target_2330_intraday",
        "target_2330_full_day",
        "target_gap_fade",
    }
    numeric_cols = features.select_dtypes(include=[np.number]).columns
    candidate_features = [col for col in numeric_cols if col not in target_cols]
    corr_cols = ["target_2330_open_gap"] + candidate_features
    corr = features[corr_cols].corr(numeric_only=True)
    top_open_gap = (
        corr["target_2330_open_gap"]
        .drop(labels=list(target_cols), errors="ignore")
        .dropna()
        .sort_values(key=lambda s: s.abs(), ascending=False)
        .head(12)
    )
    missing = features.isna().mean().sort_values(ascending=False).head(12)
    large_mask = features["target_2330_open_gap"].abs() >= 0.01
    fade_rate = float(features["target_gap_fade"].mean()) if "target_gap_fade" in features else np.nan
    start_date = features["tw_trade_date"].min() if not features.empty else "N/A"
    end_date = features["tw_trade_date"].max() if not features.empty else "N/A"

    lines = [
        "# Overnight ADR/SOX Feature Report",
        "",
        "> Data file: `capital_flow_analysis/data/overnight_gap_features_1d.csv`",
        "> Purpose: Describe overnight ADR/SOX/macro feature quality and open-gap relationships.",
        "> Note: This report is descriptive; use walk-forward validation before RL integration.",
        "",
        "## 1. Dataset Summary",
        "",
        f"- Rows: {len(features)}",
        f"- Date range: {start_date} ~ {end_date}",
        f"- Corporate-action flagged rows: {int(features['corporate_action_flag'].sum()) if 'corporate_action_flag' in features else 0}",
        f"- Stale-data flagged rows: {int(features['data_stale_flag'].sum()) if 'data_stale_flag' in features else 0}",
        f"- Gap fade positive rate: {fade_rate:.2%}" if pd.notna(fade_rate) else "- Gap fade positive rate: N/A",
        f"- abs(open gap) >= 1.0% rows: {int(large_mask.sum())}",
        "",
        "## 2. Interpretation",
        "",
        "- High ADR premium correlation is useful as a research signal, but it is not a trading rule by itself.",
        "- Target-side columns such as other Taiwan open-gap labels should not be used as pre-open features.",
        "- Missing rolling or diff features should be filled using train-set or historical-only logic.",
        "",
        "## Top Open-Gap Correlations",
        "",
    ]
    if top_open_gap.empty:
        lines.append("- Not enough data for correlation analysis.")
    else:
        for col, value in top_open_gap.items():
            lines.append(f"- `{col}`: {value:.4f}")
    lines.extend(["", "## Highest Missing Rates", ""])
    for col, value in missing.items():
        lines.append(f"- `{col}`: {value:.2%}")
    lines.extend(
        [
            "",
            "## 5. Modeling Notes",
            "",
            "- `source_age_hours` captures stale US data after holidays or closures.",
            "- Rows with `corporate_action_flag=True` should be excluded from primary evaluation.",
            "- Rows with `data_stale_flag=True` should not be used for model promotion decisions.",
            "- Prefer small, stable ADR/SOX feature sets before expanding into full macro features.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_data_quality_json(features: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if features.empty:
        data = {"status": "empty_dataframe"}
    else:
        latest_date = features["tw_trade_date"].max()
        latest_row = features[features["tw_trade_date"] == latest_date].iloc[0]
        missing_cols = [str(col) for col in features.columns if pd.isna(latest_row[col])]
        stale_flag = bool(latest_row.get("data_stale_flag", False))
        warnings = str(latest_row.get("data_provider_warnings", ""))
        
        data = {
            "date": latest_date.strftime("%Y-%m-%d"),
            "data_stale_flag": stale_flag,
            "missing_columns": missing_cols,
            "data_provider_warnings": warnings
        }
        
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ADR/SOX overnight features.")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--report", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = build_overnight_features(args.start, args.end)
    if args.report:
        report_path = Path(__file__).resolve().parents[2] / "reports" / "overnight_gap_feature_report.md"
        json_path = Path(__file__).resolve().parents[2] / "reports" / "overnight_data_quality.json"
        write_feature_report(features, report_path)
        write_data_quality_json(features, json_path)
        print(f"Reports saved to {report_path} and {json_path}")
    print(features.tail())


if __name__ == "__main__":
    main()
