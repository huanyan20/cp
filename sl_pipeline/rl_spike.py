"""S5 smoke: validate SL features + RLAllocator + env observation wiring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sl_pipeline.rl_allocator import RLAllocator, RLAllocatorConfig
from sl_pipeline.sl_features import (
    SL_FEATURE_VERSION,
    SL_FEATURES_PER_STOCK,
    build_sl_feature_arrays,
)
from trading_env import TaiwanStockEnv


def _synthetic_enriched(tickers: list[str], days: int = 60) -> dict[str, pd.DataFrame]:
    from data_pipeline import BASE_FEATURE_COLS, CROSS_ASSET_COLS

    index = pd.date_range("2024-01-01", periods=days, freq="B")
    out: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        data: dict[str, np.ndarray] = {}
        for idx, col in enumerate(BASE_FEATURE_COLS):
            if col == "log_return":
                data[col] = np.full(days, 0.01)
            elif col in ("Volume_norm", "Close_norm"):
                data[col] = np.linspace(0.5, 1.5, days)
            else:
                data[col] = np.full(days, idx / 100.0)
        for idx, col in enumerate(CROSS_ASSET_COLS):
            data[col] = np.full(days, (idx + 1) / 50.0)
        out[ticker] = pd.DataFrame(data, index=index)
    return out


def validate_spike(
    enriched: dict,
    scores: dict,
    tickers: list[str],
    *,
    window_size: int = 20,
    enable_cash_action: bool = True,
) -> dict:
    """Return validation payload without training RL."""
    sl_arrays = build_sl_feature_arrays(enriched, scores, tickers)
    base_env = TaiwanStockEnv(
        df_dict=enriched,
        window_size=window_size,
        enable_cash_action=enable_cash_action,
        enable_sl_features=False,
    )
    sl_env = TaiwanStockEnv(
        df_dict=enriched,
        window_size=window_size,
        enable_cash_action=enable_cash_action,
        enable_sl_features=True,
        sl_features_by_ticker=sl_arrays,
    )

    obs_base, _ = base_env.reset(seed=42)
    obs_sl, _ = sl_env.reset(seed=42)

    allocator = RLAllocator(RLAllocatorConfig(top_k=5))
    action_dim = sl_env.action_space.shape[0]
    action = sl_env.action_space.sample()
    step = window_size
    scores_t = {t: float(scores[t].iloc[step]) for t in tickers if t in scores}
    vols_t = {t: 0.2 for t in tickers}
    state = sl_env.portfolio_state_snapshot()
    target = allocator.allocate_from_action(
        action,
        scores_t,
        vols_t,
        state,
        tickers,
        enable_cash_action=enable_cash_action,
    )

    base_dim = base_env._obs_dim_per_stock
    sl_dim = sl_env._obs_dim_per_stock
    delta = sl_dim - base_dim
    return {
        "sl_feature_version": SL_FEATURE_VERSION,
        "sl_features_per_stock": SL_FEATURES_PER_STOCK,
        "obs_base_dim_per_stock": base_dim,
        "obs_sl_dim_per_stock": sl_dim,
        "obs_dim_delta": delta,
        "obs_base_shape": list(obs_base.shape),
        "obs_sl_shape": list(obs_sl.shape),
        "action_dim": int(action_dim),
        "target_stock_count": len(target.target_weights),
        "target_cash_weight": float(target.cash_weight),
        "tickers": len(tickers),
        "spike_ok": (
            delta == SL_FEATURES_PER_STOCK
            and obs_sl.shape[0] == obs_base.shape[0]
            and obs_sl.shape[1] == obs_base.shape[1] + SL_FEATURES_PER_STOCK
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S5 RL+SL spike validation")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    tickers = ["2330.TW", "2317.TW"]
    enriched = _synthetic_enriched(tickers, days=60)
    scores = {
        t: pd.Series(np.linspace(0.1, 0.5, 60), index=enriched[t].index, name=t)
        for t in tickers
    }
    result = validate_spike(enriched, scores, tickers)
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0 if result.get("spike_ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
