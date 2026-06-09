"""SL observation features for RLAllocator spike (S5): score + rank + rule weight."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sl_pipeline.allocator import MarketContext, PortfolioState
from sl_pipeline.backtest import build_vols_as_of
from sl_pipeline.rule_based_allocator import RuleBasedAllocator

SL_FEATURE_VERSION = "v1"
SL_FEATURES_PER_STOCK = 3  # score_z, rank_norm, rule_weight


@dataclass(frozen=True)
class SLFeatureConfig:
    score_clip: float = 3.0


def cross_sectional_zscore(scores: dict[str, float]) -> dict[str, float]:
    """Z-score alpha scores across the universe at one decision date."""
    values = np.array(list(scores.values()), dtype=float)
    if len(values) == 0:
        return {}
    mean = float(np.nanmean(values))
    std = float(np.nanstd(values))
    if std < 1e-8:
        std = 1.0
    return {ticker: (float(scores[ticker]) - mean) / std for ticker in scores}


def cross_sectional_rank_norm(scores: dict[str, float]) -> dict[str, float]:
    """Percentile rank in [0, 1] (1 = highest score)."""
    if not scores:
        return {}
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    n = len(ranked)
    if n == 1:
        return {ranked[0][0]: 1.0}
    return {
        ticker: 1.0 - (idx / (n - 1))
        for idx, (ticker, _) in enumerate(ranked)
    }


def sl_features_at_date(
    scores: dict[str, float],
    rule_weights: dict[str, float],
    *,
    config: SLFeatureConfig | None = None,
) -> dict[str, np.ndarray]:
    """Per-ticker SL feature vector for one date."""
    config = config or SLFeatureConfig()
    zscores = cross_sectional_zscore(scores)
    ranks = cross_sectional_rank_norm(scores)
    out: dict[str, np.ndarray] = {}
    for ticker in scores:
        score_z = float(np.clip(zscores.get(ticker, 0.0), -config.score_clip, config.score_clip))
        rank_norm = float(ranks.get(ticker, 0.0))
        rule_w = float(rule_weights.get(ticker, 0.0))
        out[ticker] = np.array([score_z, rank_norm, rule_w], dtype=np.float32)
    return out


def build_rule_weight_history(
    enriched: dict[str, pd.DataFrame],
    scores: dict[str, pd.Series],
    allocator: RuleBasedAllocator,
    tickers: list[str],
    *,
    vol_window: int = 20,
) -> dict[str, dict[pd.Timestamp, float]]:
    """Replay RuleBasedAllocator to get daily baseline weights (for SL obs)."""
    dates = sorted(
        set.intersection(
            *[
                set(scores[t].dropna().index)
                for t in tickers
                if t in scores and not scores[t].dropna().empty
            ]
        )
    )
    positions: dict[str, float] = {}
    cash_weight = 1.0
    portfolio_value = 1.0
    peak_value = 1.0
    history: dict[str, dict[pd.Timestamp, float]] = {t: {} for t in tickers}

    for signal_date in dates:
        score_row = {
            t: float(scores[t].loc[signal_date])
            for t in tickers
            if t in scores
            and signal_date in scores[t].index
            and np.isfinite(scores[t].loc[signal_date])
        }
        if not score_row:
            continue
        vols = build_vols_as_of(
            enriched,
            tickers,
            signal_date,
            vol_window=vol_window,
            min_vol_obs=5,
            vol_floor=0.05,
        )
        rolling_mdd = (peak_value - portfolio_value) / max(peak_value, 1e-12)
        state = PortfolioState(
            positions=dict(positions),
            cash_weight=cash_weight,
            portfolio_value=portfolio_value,
            peak_value=peak_value,
            rolling_mdd=float(rolling_mdd),
        )
        target = allocator.allocate(score_row, vols, state, MarketContext())
        for ticker in tickers:
            history[ticker][signal_date] = float(target.target_weights.get(ticker, 0.0))
        positions = dict(target.target_weights)
        cash_weight = float(target.cash_weight)

    return history


def build_sl_feature_arrays(
    enriched: dict[str, pd.DataFrame],
    scores: dict[str, pd.Series],
    tickers: list[str],
    *,
    allocator: RuleBasedAllocator | None = None,
    config: SLFeatureConfig | None = None,
) -> dict[str, np.ndarray]:
    """Build per-ticker (n_steps, 3) arrays aligned with env DataFrame rows."""
    config = config or SLFeatureConfig()
    allocator = allocator or RuleBasedAllocator()
    rule_hist = build_rule_weight_history(enriched, scores, allocator, tickers)

    arrays: dict[str, np.ndarray] = {}
    for ticker in tickers:
        if ticker not in enriched:
            continue
        df = enriched[ticker]
        n = len(df)
        arr = np.zeros((n, SL_FEATURES_PER_STOCK), dtype=np.float32)
        score_series = scores.get(ticker)
        if score_series is None:
            arrays[ticker] = arr
            continue

        for step, date in enumerate(df.index):
            if date not in score_series.index or not np.isfinite(score_series.loc[date]):
                continue
            score_row = {
                t: float(scores[t].loc[date])
                for t in tickers
                if t in scores and date in scores[t].index and np.isfinite(scores[t].loc[date])
            }
            if ticker not in score_row:
                continue
            rule_w = {t: rule_hist.get(t, {}).get(date, 0.0) for t in tickers}
            feats = sl_features_at_date(score_row, rule_w, config=config)
            if ticker in feats:
                arr[step] = feats[ticker]
        arrays[ticker] = arr
    return arrays
