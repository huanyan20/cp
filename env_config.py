"""Environment configuration snapshot and fingerprinting for experiment versioning (O1)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import trading_env

# Bump manually when reward / regime logic changes intentionally (e.g. r4, r5).
ENV_CONFIG_VERSION = "r4"


def build_env_config_snapshot(
    settings: Any | None = None,
    *,
    sl_features: str | None = None,
) -> dict[str, Any]:
    """Capture env + research knobs that invalidate cross-run metric comparison."""
    topk = 5
    softmax_temp = 0.5
    max_leverage = 2.0
    if settings is not None:
        topk = settings.research.default_topk
        softmax_temp = settings.research.default_softmax_temp
        max_leverage = settings.risk_limits.max_leverage

    config = {
        "version": ENV_CONFIG_VERSION,
        "lambda_cost": trading_env.LAMBDA_COST,
        "lambda_turnover": trading_env.LAMBDA_TURNOVER,
        "lambda_cash": trading_env.LAMBDA_CASH,
        "lambda_drawdown": trading_env.LAMBDA_DRAWDOWN,
        "lambda_cash_defensive": trading_env.LAMBDA_CASH_DEFENSIVE,
        "reward_ref_dd": trading_env.REWARD_REF_DD,
        "regime_dd_threshold": trading_env.REGIME_DD_THRESHOLD,
        "regime_penalty_coef": trading_env.REGIME_PENALTY_COEF,
        "topk": topk,
        "softmax_temp": softmax_temp,
        "max_leverage": max_leverage,
        "use_benchmark_reward": True,
    }
    if sl_features:
        config["sl_features"] = sl_features
    config["hash"] = compute_env_config_hash(config)
    return config


def compute_env_config_hash(config: dict[str, Any]) -> str:
    """Stable 8-char fingerprint from env knobs (excludes human version label)."""
    payload = {k: v for k, v in config.items() if k not in ("hash", "version")}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]


def get_current_env_config_hash(settings: Any | None = None) -> str:
    return build_env_config_snapshot(settings)["hash"]
