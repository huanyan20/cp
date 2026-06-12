from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import fetch_multi_asset_data
from env_config import build_env_config_snapshot
from metrics_utils import calculate_metrics
from trading_env import TaiwanStockEnv

PERIODS = [
    {
        "name": "2024H2",
        "train_start": "2020-01-01",
        "train_end": "2024-06-30",
        "test_start": "2024-07-01",
        "test_end": "2024-12-31",
    },
    {
        "name": "2025H1",
        "train_start": "2020-01-01",
        "train_end": "2024-12-31",
        "test_start": "2025-01-01",
        "test_end": "2025-06-30",
    },
    {
        "name": "2025H2",
        "train_start": "2020-01-01",
        "train_end": "2025-06-30",
        "test_start": "2025-07-01",
        "test_end": "2025-12-31",
    },
    {
        "name": "2026H1",
        "train_start": "2020-01-01",
        "train_end": "2025-12-31",
        "test_start": "2026-01-01",
        "test_end": "2026-06-30",
    },
]


def clamp_periods(periods=None, today=None):
    """Clamp future walk-forward periods to the present day."""
    periods = periods or PERIODS
    today_ts = pd.Timestamp(today or datetime.now().date())
    clamped = []
    for period in periods:
        p = dict(period)
        test_start = pd.Timestamp(p["test_start"])
        test_end = pd.Timestamp(p["test_end"])
        train_end = pd.Timestamp(p["train_end"])

        if test_start > today_ts:
            p["skip_reason"] = f"test_start {test_start.date()} is after today {today_ts.date()}"
            clamped.append(p)
            continue

        effective_test_end = min(test_end, today_ts)
        effective_train_end = min(train_end, today_ts)
        p["effective_test_end"] = effective_test_end.strftime("%Y-%m-%d")
        p["effective_train_end"] = effective_train_end.strftime("%Y-%m-%d")
        p["was_clamped"] = effective_test_end < test_end or effective_train_end < train_end
        clamped.append(p)
    return clamped


def build_period_plan(periods=None, today=None):
    """Return the current period plan used by walk-forward experiments."""
    return clamp_periods(periods=periods, today=today)


def build_artifact_paths(
    algo: str,
    cash_mode: str,
    seed: int,
    feature_suffix: str = "",
    results_dir: str = "results_dir",
    period_name: str = "2024H2",
) -> dict[str, str]:
    """Centralize the artifact file names for walk-forward runs."""
    metrics_path = f"{results_dir}/metrics_{algo}_{cash_mode}{feature_suffix}_wf_seed{seed}.json"
    model_path = f"{results_dir}/wf_{algo}_{cash_mode}{feature_suffix}_model_{period_name}_seed{seed}"
    chart_path = f"{results_dir}/walk_forward_{algo}_{cash_mode}{feature_suffix}_seed{seed}.png"
    return {
        "metrics": metrics_path,
        "model": model_path,
        "chart": chart_path,
    }


def feature_suffix_from_path(overnight_feature_path: str | None) -> str:
    return "_with_features" if overnight_feature_path else ""


def should_skip_artifact(metrics_path: str, overwrite: bool = False) -> bool:
    return (not overwrite) and Path(metrics_path).exists()


def build_pending_walk_forward_tasks(
    algos: list[str],
    cash_modes: list[bool],
    seeds: list[int],
    results_dir: str,
    overnight_feature_path: str | None = None,
    overwrite: bool = False,
) -> list[tuple[str, bool, int]]:
    pending = []
    feature_suffix = feature_suffix_from_path(overnight_feature_path)
    for algo in algos:
        for enable_cash_action in cash_modes:
            cash_mode = "enabled" if enable_cash_action else "disabled"
            for seed in seeds:
                metrics_path = build_artifact_paths(
                    algo, cash_mode, seed, feature_suffix, results_dir
                )["metrics"]
                if should_skip_artifact(metrics_path, overwrite=overwrite):
                    continue
                pending.append((algo, enable_cash_action, seed))
    return pending


def build_seed_metrics(
    algo: str,
    seed: int,
    cash_mode: str,
    enable_cash_action: bool,
    enable_margin_short: bool,
    timesteps: int,
    settings=None,
) -> dict:
    env_config = build_env_config_snapshot(settings)
    return {
        "algo": algo,
        "seed": seed,
        "cash_mode": cash_mode,
        "enable_cash_action": enable_cash_action,
        "enable_margin_short": enable_margin_short,
        "train_test_period": "Walk-Forward",
        "timesteps": timesteps,
        "env_config_version": env_config["version"],
        "env_config_hash": env_config["hash"],
        "env_config": env_config,
        "overall": {},
        "periods": {},
        "skipped_periods": {},
    }


def write_metrics_json(metrics: dict, path: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=4, ensure_ascii=False), encoding="utf-8")


# ============================================================================
# Train/Eval Orchestration Functions
# ============================================================================


def build_train_env(
    tickers: list[str],
    train_start: str,
    train_end: str,
    window_size: int,
    macro_tickers: list[str],
    settings,
    enable_cash_action: bool = True,
    enable_margin_short: bool = False,
    overnight_feature_path: str | None = None,
    enable_sl_features: bool = False,
    sl_scores: dict | None = None,
) -> tuple[TaiwanStockEnv, dict]:
    """
    Build training environment from period data.
    
    Args:
        tickers: Stock universe
        train_start: Training period start date
        train_end: Training period end date
        window_size: Technical indicator window size
        macro_tickers: Macro index tickers
        settings: AppSettings object
        enable_cash_action: Enable cash action in env
        enable_margin_short: Enable margin shorting
        overnight_feature_path: Optional overnight features CSV path
        enable_sl_features: S5 spike — append SL score/rank/rule_weight to observation
        sl_scores: Per-ticker score Series aligned with train_data (required if enable_sl_features)
    
    Returns:
        Tuple of (TaiwanStockEnv, dict with training data)
    """
    train_data = fetch_multi_asset_data(
        tickers=tickers,
        start_date=train_start,
        end_date=train_end,
        window_size=window_size,
        macro_tickers=macro_tickers,
        overnight_feature_path=overnight_feature_path,
    )
    
    sl_feature_arrays = None
    if enable_sl_features:
        if not sl_scores:
            raise ValueError("enable_sl_features requires sl_scores.")
        from sl_pipeline.sl_features import build_sl_feature_arrays

        sl_feature_arrays = build_sl_feature_arrays(train_data, sl_scores, tickers)

    train_env = TaiwanStockEnv(
        df_dict=train_data,
        window_size=window_size,
        topk=settings.research.default_topk,
        softmax_temp=settings.research.default_softmax_temp,
        use_benchmark_reward=True,
        enable_cash_action=enable_cash_action,
        enable_margin_short=enable_margin_short,
        max_leverage=settings.risk_limits.max_leverage,
        enable_sl_features=enable_sl_features,
        sl_features_by_ticker=sl_feature_arrays,
    )

    return train_env, train_data


def train_and_save_model(
    algo: str,
    train_env: TaiwanStockEnv,
    timesteps: int,
    model_path: str,
    temporal_extractor: bool = False,
) -> None:
    """
    Train model and save to disk.
    
    Args:
        algo: Algorithm name ('ppo' or 'sac')
        train_env: Training environment
        timesteps: Total training timesteps
        model_path: Path to save trained model
        temporal_extractor: Use GRU temporal feature extractor
    """
    from core.model_trainer import ModelTrainer
    from settings import SETTINGS
    
    trainer = ModelTrainer(algo, device=SETTINGS.research.torch_device)
    model, callback = trainer.build_model(train_env, timesteps, temporal_extractor=temporal_extractor)
    
    if callback is not None:
        model.learn(total_timesteps=timesteps, progress_bar=True, callback=callback)
    else:
        model.learn(total_timesteps=timesteps, progress_bar=True)
    model.save(model_path)


def build_eval_env(
    tickers: list[str],
    test_start: str,
    test_end: str,
    window_size: int,
    macro_tickers: list[str],
    settings,
    enable_cash_action: bool = True,
    enable_margin_short: bool = False,
    overnight_feature_path: str | None = None,
    enable_sl_features: bool = False,
    sl_scores: dict | None = None,
) -> tuple[TaiwanStockEnv, dict]:
    """
    Build evaluation environment from period data.
    
    Handles data fetching with one year lookback and slicing to test_start.
    
    Args:
        tickers: Stock universe
        test_start: Test period start date
        test_end: Test period end date (clamped to today)
        window_size: Technical indicator window size
        macro_tickers: Macro index tickers
        settings: AppSettings object
        enable_cash_action: Enable cash action in env
        enable_margin_short: Enable margin shorting
        overnight_feature_path: Optional overnight features CSV path
    
    Returns:
        Tuple of (TaiwanStockEnv, dict with test data)
    """
    # Fetch data from one year before test_start for warmup
    test_fetch_start = str(int(test_start[:4]) - 1) + test_start[4:]
    test_data_raw = fetch_multi_asset_data(
        tickers=tickers,
        start_date=test_fetch_start,
        end_date=test_end,
        window_size=window_size,
        macro_tickers=macro_tickers,
        overnight_feature_path=overnight_feature_path,
    )
    
    # Slice to test_start (keep window_size lookback)
    test_data = {}
    for ticker, df in test_data_raw.items():
        mask = df.index >= pd.to_datetime(test_start)
        if not mask.any():
            test_data[ticker] = df
        else:
            start_idx = np.argmax(mask)
            slice_start = max(0, start_idx - window_size)
            test_data[ticker] = df.iloc[slice_start:]
    
    sl_feature_arrays = None
    if enable_sl_features:
        if not sl_scores:
            raise ValueError("enable_sl_features requires sl_scores.")
        from sl_pipeline.sl_features import build_sl_feature_arrays

        sl_feature_arrays = build_sl_feature_arrays(test_data, sl_scores, tickers)

    test_env = TaiwanStockEnv(
        df_dict=test_data,
        window_size=window_size,
        topk=settings.research.default_topk,
        softmax_temp=settings.research.default_softmax_temp,
        use_benchmark_reward=True,
        enable_cash_action=enable_cash_action,
        enable_margin_short=enable_margin_short,
        max_leverage=settings.risk_limits.max_leverage,
        enable_sl_features=enable_sl_features,
        sl_features_by_ticker=sl_feature_arrays,
    )
    
    return test_env, test_data


def run_eval_loop(
    model,
    test_env: TaiwanStockEnv,
    seed: int,
) -> dict:
    """
    Execute evaluation loop and collect period metrics.
    
    Args:
        model: Trained RL model
        test_env: Evaluation environment
        seed: Random seed for environment reset
    
    Returns:
        Dict with:
            - daily_returns: List of daily returns
            - positions: List of position arrays
            - cash_weights: List of cash weight values
            - turnover: List of turnover values
            - portfolio_hist: List of portfolio values
    """
    period_returns = []
    period_positions = []
    period_cash = []
    period_turnover = []
    
    obs, _ = test_env.reset(seed=seed)
    done = False
    portfolio_val = test_env.initial_balance
    period_portfolio_hist = [portfolio_val]
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated
        
        new_portfolio_val = info["portfolio_value"]
        daily_ret = (new_portfolio_val / portfolio_val) - 1.0
        portfolio_val = new_portfolio_val
        
        position = info["positions"].copy()
        cash_weight = info.get("cash_weight", 0.0)
        turnover = info.get("turnover", 0.0)
        
        period_returns.append(daily_ret)
        period_positions.append(position)
        period_cash.append(cash_weight)
        period_turnover.append(turnover)
        period_portfolio_hist.append(portfolio_val)
    
    return {
        "daily_returns": period_returns,
        "positions": period_positions,
        "cash_weights": period_cash,
        "turnover": period_turnover,
        "portfolio_hist": period_portfolio_hist,
    }


def persist_period_metrics(
    algo: str,
    cash_mode: str,
    seed: int,
    feature_suffix: str,
    tickers: list[str],
    test_start: str,
    test_end: str,
    eval_results: dict,
    period_name: str,
    results_dir: str,
) -> dict:
    """
    Calculate and store metrics for a single period.
    
    Args:
        algo: Algorithm name
        cash_mode: 'enabled' or 'disabled'
        seed: Random seed
        feature_suffix: Feature suffix from overnight_feature_path
        tickers: Stock universe
        test_start: Test period start date
        test_end: Test period end date
        eval_results: Dict from run_eval_loop()
        period_name: Period name (e.g., '2024H2')
        results_dir: Results directory
    
    Returns:
        Dict with period metrics
    """
    p_metrics = calculate_metrics(
        eval_results["portfolio_hist"],
        eval_results["positions"],
        eval_results["cash_weights"],
        eval_results["daily_returns"],
        eval_results["turnover"],
        tickers,
    )
    p_metrics["test_start"] = test_start
    p_metrics["test_end"] = test_end
    
    return p_metrics
