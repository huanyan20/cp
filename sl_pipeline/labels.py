"""Cross-sectional demeaned forward-return labels for pooled LightGBM training."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_pipeline.utils import BASE_FEATURE_COLS

HORIZON_DAYS = (5, 10, 20, 60)


def label_column_name(horizon: int) -> str:
    if horizon not in HORIZON_DAYS:
        raise ValueError(f"Unsupported horizon {horizon}; expected one of {HORIZON_DAYS}")
    return f"target_{horizon}d_class"


def default_feature_columns() -> list[str]:
    """Feature columns aligned with Milestone 3A."""
    return list(BASE_FEATURE_COLS)


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


def build_classification_label_frame(
    enriched: dict[str, pd.DataFrame],
    horizon: int,
) -> pd.DataFrame:
    """Wide frame of 3-class forward returns."""
    raw = {
        ticker: forward_log_return_t1(df["log_return"], horizon)
        for ticker, df in enriched.items()
    }
    raw_df = pd.DataFrame(raw)
    rank = raw_df.rank(axis=1, pct=True)
    
    # 3-class classification
    # 2: Top 20%
    # 1: Middle 60%
    # 0: Bottom 20%
    classes = pd.DataFrame(np.nan, index=raw_df.index, columns=raw_df.columns)
    classes[rank >= 0.80] = 2.0
    classes[(rank < 0.80) & (rank > 0.20)] = 1.0
    classes[rank <= 0.20] = 0.0
    
    return classes


def _add_cross_sectional_ranks(panel: pd.DataFrame, base_cols: list[str]) -> pd.DataFrame:
    """Add rank columns for all base features."""
    for col in base_cols:
        if col in panel.columns and col != "log_return":
            rank_col_name = f"rank_{col}"
            panel[rank_col_name] = panel.groupby("date")[col].rank(pct=True)
    return panel


def build_labeled_panel(
    enriched: dict[str, pd.DataFrame],
    horizon: int = 20,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Long panel: one row per (date, ticker) with features + classification label."""
    feature_cols = feature_cols or default_feature_columns()
    label_col = label_column_name(horizon)
    label_frame = build_classification_label_frame(enriched, horizon)

    frames: list[pd.DataFrame] = []
    for ticker, df in enriched.items():
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{ticker} missing feature columns: {missing}")
        part = df[feature_cols].copy()
        part["ticker"] = ticker
        part["date"] = df.index
        part[label_col] = label_frame[ticker].reindex(df.index).values
        frames.append(part.reset_index(drop=True))

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.dropna(subset=[label_col]).sort_values(["date", "ticker"]).reset_index(drop=True)
    panel = _add_cross_sectional_ranks(panel, feature_cols)
    return panel


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
        
    panel = pd.concat(frames, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)
    panel = _add_cross_sectional_ranks(panel, feature_cols)
    return panel


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
