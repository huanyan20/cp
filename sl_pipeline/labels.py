"""Cross-sectional demeaned forward-return labels for pooled LightGBM training."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_pipeline.utils import build_feature_schema

HORIZON_DAYS = (5, 10)


def label_column_name(horizon: int) -> str:
    if horizon not in HORIZON_DAYS:
        raise ValueError(f"Unsupported horizon {horizon}; expected one of {HORIZON_DAYS}")
    return f"target_{horizon}d_cross_demean"


def default_feature_columns(macro_feature_cols: list[str] | None = None) -> list[str]:
    """Feature columns aligned with base RL observation (no overnight, no ticker id)."""
    schema = build_feature_schema(macro_features=tuple(macro_feature_cols or ()))
    return list(schema.columns)


def forward_log_return_t1(log_return: pd.Series, end_day: int) -> pd.Series:
    """log(P(t+end_day) / P(t+1)) at decision date t (T+1 execution alignment).

    Uses per-day ``log_return[d] = log(P(d) / P(d-1))``, summing from t+2 through t+end_day.
    """
    if end_day not in HORIZON_DAYS:
        raise ValueError(f"Unsupported end_day {end_day}; expected one of {HORIZON_DAYS}")

    arr = pd.to_numeric(log_return, errors="coerce").astype(float).values
    n = len(arr)
    out = np.full(n, np.nan, dtype=float)
    for i in range(n):
        start = i + 2
        stop = i + end_day + 1
        if stop <= n:
            out[i] = float(np.nansum(arr[start:stop]))
    return pd.Series(out, index=log_return.index, name=f"raw_{end_day}d_return_t1")


def build_cross_demean_frame(
    enriched: dict[str, pd.DataFrame],
    horizon: int,
) -> pd.DataFrame:
    """Wide frame of cross-demeaned forward returns (index=date, columns=tickers)."""
    raw = {
        ticker: forward_log_return_t1(df["log_return"], horizon)
        for ticker, df in enriched.items()
    }
    raw_df = pd.DataFrame(raw)
    median = raw_df.median(axis=1, skipna=True)
    return raw_df.sub(median, axis=0)


def build_labeled_panel(
    enriched: dict[str, pd.DataFrame],
    horizon: int = 5,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Long panel: one row per (date, ticker) with features + cross-demean label."""
    feature_cols = feature_cols or default_feature_columns()
    label_col = label_column_name(horizon)
    cross_demean = build_cross_demean_frame(enriched, horizon)

    frames: list[pd.DataFrame] = []
    for ticker, df in enriched.items():
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{ticker} missing feature columns: {missing}")
        part = df[feature_cols].copy()
        part["ticker"] = ticker
        part["date"] = df.index
        part[label_col] = cross_demean[ticker].reindex(df.index).values
        frames.append(part.reset_index(drop=True))

    panel = pd.concat(frames, ignore_index=True)
    return panel.dropna(subset=[label_col]).sort_values(["date", "ticker"]).reset_index(drop=True)


def build_feature_panel(
    enriched: dict[str, pd.DataFrame],
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Inference panel without labels (OOS scoring)."""
    feature_cols = feature_cols or default_feature_columns()
    frames: list[pd.DataFrame] = []
    for ticker, df in enriched.items():
        part = df[feature_cols].copy()
        part["ticker"] = ticker
        part["date"] = df.index
        frames.append(part.reset_index(drop=True))
    return pd.concat(frames, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)


def split_panel_by_date(
    panel: pd.DataFrame,
    train_end: str,
    test_start: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-ordered train/test split (no shuffle)."""
    dates = pd.to_datetime(panel["date"])
    train_end_ts = pd.Timestamp(train_end)
    test_start_ts = pd.Timestamp(test_start)
    train = panel.loc[dates <= train_end_ts].copy()
    test = panel.loc[dates >= test_start_ts].copy()
    return train, test
