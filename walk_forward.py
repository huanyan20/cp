import argparse
import concurrent.futures
import gc
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.utils import set_random_seed

from metrics_utils import calculate_metrics
from research_pipeline import (
    build_artifact_paths,
    build_eval_env,
    build_pending_walk_forward_tasks,
    build_period_plan,
    build_seed_metrics,
    build_train_env,
    clamp_periods,
    feature_suffix_from_path,
    persist_period_metrics,
    run_eval_loop,
    train_and_save_model,
    write_metrics_json,
)
from settings import (
    TIER_PRESETS,
    load_settings,
    resolve_tier,
    resolve_torch_device,
)
from stock_universe import MACRO_TICKERS_RL, TICKER_NAMES, TICKERS_TECH_EXPANDED

SETTINGS = load_settings()

DEFAULT_TIMESTEPS = SETTINGS.research.walk_forward_timesteps
DEFAULT_SEEDS = [int(s.strip()) for s in SETTINGS.research.default_seeds.split(",") if s.strip()]
WINDOW_SIZE = SETTINGS.research.window_size
RESULTS_DIR = str(SETTINGS.paths.results_dir)

__all__ = [
    "cash_modes_from_arg",
    "clamp_periods",
    "parse_seeds",
    "run_candidate_set",
    "run_research_matrix",
    "run_walk_forward",
]

# O3 — curated default candidate set. Not a cartesian matrix: only the two best
# historical performers (SAC enabled #1, PPO disabled close #2) so a routine
# research iteration trains 2 models x seeds instead of the full 4-combo matrix.
CANDIDATE_PAIRS: list[tuple[str, bool]] = [
    ("sac", True),
    ("ppo", False),
]


def parse_seeds(seed_text: str) -> list[int]:
    return [int(s.strip()) for s in seed_text.split(",") if s.strip()]


def cash_modes_from_arg(cash_mode: str) -> list[bool]:
    if cash_mode == "enabled":
        return [True]
    if cash_mode == "disabled":
        return [False]
    if cash_mode == "both":
        return [True, False]
    raise ValueError(f"Unsupported cash mode: {cash_mode}")


def cash_mode_name(enable_cash_action: bool) -> str:
    return "enabled" if enable_cash_action else "disabled"


def saved_model_exists(model_path: str) -> bool:
    """Stable-Baselines saves .zip files even when model_path has no suffix."""
    path = Path(model_path)
    return path.exists() or path.with_suffix(".zip").exists()


def run_research_matrix(
    timesteps: int = DEFAULT_TIMESTEPS,
    algos: list[str] | None = None,
    cash_modes: list[bool] | None = None,
    seeds: list[int] | None = None,
    enable_margin_short: bool = False,
    max_workers: int = 1,
    overwrite: bool = False,
    overnight_feature_path: str | None = None,
    temporal_extractor: bool = False,
):
    algos = algos or ["ppo", "sac"]
    cash_modes = cash_modes or [True, False]
    seeds = seeds or DEFAULT_SEEDS

    pending_tasks = build_pending_walk_forward_tasks(
        algos=algos,
        cash_modes=cash_modes,
        seeds=seeds,
        results_dir=RESULTS_DIR,
        overnight_feature_path=overnight_feature_path,
        overwrite=overwrite,
    )

    if not pending_tasks:
        print("All experiments have already been completed. Exiting.")
        return

    if max_workers == 1:
        for algo, enable_cash_action, seed in pending_tasks:
            _run_single_walk_forward(
                timesteps,
                algo,
                seed,
                enable_cash_action,
                enable_margin_short,
                overnight_feature_path,
                temporal_extractor,
                overwrite,
            )
    else:
        print(f"Starting multiprocessing with {max_workers} workers for {len(pending_tasks)} tasks...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for algo, enable_cash_action, seed in pending_tasks:
                futures.append(
                    executor.submit(
                        _run_single_walk_forward,
                        timesteps,
                        algo,
                        seed,
                        enable_cash_action,
                        enable_margin_short,
                        overnight_feature_path,
                        temporal_extractor,
                        overwrite,
                    )
                )
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"[Error] Task failed: {e}")


def run_candidate_set(
    timesteps: int = DEFAULT_TIMESTEPS,
    seeds: list[int] | None = None,
    enable_margin_short: bool = False,
    max_workers: int = 1,
    overwrite: bool = False,
    overnight_feature_path: str | None = None,
    temporal_extractor: bool = False,
    candidate_pairs: list[tuple[str, bool]] | None = None,
):
    """O3: train only the curated candidate set (SAC enabled + PPO disabled).

    Explicit (algo, enable_cash_action) pairs rather than a cartesian product,
    so we avoid retraining the 4-combo matrix on every iteration.
    """
    seeds = seeds or DEFAULT_SEEDS
    candidate_pairs = candidate_pairs or CANDIDATE_PAIRS

    pending: list[tuple[str, bool, int]] = []
    for algo, enable_cash_action in candidate_pairs:
        pending.extend(
            build_pending_walk_forward_tasks(
                algos=[algo],
                cash_modes=[enable_cash_action],
                seeds=seeds,
                results_dir=RESULTS_DIR,
                overnight_feature_path=overnight_feature_path,
                overwrite=overwrite,
            )
        )

    if not pending:
        print("All candidate-set experiments have already been completed. Exiting.")
        return

    print(
        f"Candidate set: {[(a, 'enabled' if c else 'disabled') for a, c in candidate_pairs]} "
        f"x seeds {seeds} -> {len(pending)} task(s)."
    )

    if max_workers == 1:
        for algo, enable_cash_action, seed in pending:
            _run_single_walk_forward(
                timesteps,
                algo,
                seed,
                enable_cash_action,
                enable_margin_short,
                overnight_feature_path,
                temporal_extractor,
                overwrite,
            )
    else:
        print(f"Starting multiprocessing with {max_workers} workers for {len(pending)} tasks...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for algo, enable_cash_action, seed in pending:
                futures.append(
                    executor.submit(
                        _run_single_walk_forward,
                        timesteps,
                        algo,
                        seed,
                        enable_cash_action,
                        enable_margin_short,
                        overnight_feature_path,
                        temporal_extractor,
                        overwrite,
                    )
                )
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"[Error] Task failed: {e}")


def run_walk_forward(
    timesteps: int = DEFAULT_TIMESTEPS,
    algo: str = "ppo",
    seeds: list[int] | None = None,
    enable_cash_action: bool = True,
    enable_margin_short: bool = False,
    max_workers: int = 1,
    overwrite: bool = False,
    overnight_feature_path: str | None = None,
    temporal_extractor: bool = False,
):
    seeds = seeds or DEFAULT_SEEDS
    pending_tasks = build_pending_walk_forward_tasks(
        algos=[algo],
        cash_modes=[enable_cash_action],
        seeds=seeds,
        results_dir=RESULTS_DIR,
        overnight_feature_path=overnight_feature_path,
        overwrite=overwrite,
    )
    pending_seeds = [seed for _, _, seed in pending_tasks]

    if not pending_seeds:
        print("All seeds have already been completed. Exiting.")
        return

    if max_workers == 1:
        for seed in pending_seeds:
            _run_single_walk_forward(
                timesteps,
                algo,
                seed,
                enable_cash_action,
                enable_margin_short,
                overnight_feature_path,
                temporal_extractor,
                overwrite,
            )
    else:
        print(f"Starting multiprocessing with {max_workers} workers for {len(pending_seeds)} tasks...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for seed in pending_seeds:
                futures.append(
                    executor.submit(
                        _run_single_walk_forward,
                        timesteps,
                        algo,
                        seed,
                        enable_cash_action,
                        enable_margin_short,
                        overnight_feature_path,
                        temporal_extractor,
                        overwrite,
                    )
                )
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"[Error] Task failed: {e}")


def _run_single_walk_forward(
    timesteps: int,
    algo: str,
    seed: int,
    enable_cash_action: bool,
    enable_margin_short: bool,
    overnight_feature_path: str | None = None,
    temporal_extractor: bool = False,
    overwrite: bool = False,
):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tickers = TICKERS_TECH_EXPANDED
    cash_mode = cash_mode_name(enable_cash_action)
    periods = build_period_plan()

    print("\n=======================================================")
    print(
        f"=== Walk-Forward (algo={algo}, cash={cash_mode}, seed={seed}, "
        f"timesteps={timesteps}) ==="
    )
    print("=======================================================")

    set_random_seed(seed)
    all_daily_returns = []
    all_positions = []
    all_cash_weights = []
    all_turnover = []
    period_start_indices = []

    seed_metrics = build_seed_metrics(
        algo=algo,
        seed=seed,
        cash_mode=cash_mode,
        enable_cash_action=enable_cash_action,
        enable_margin_short=enable_margin_short,
        timesteps=timesteps,
        settings=SETTINGS,
    )

    for i, period in enumerate(periods):
        name = period["name"]
        if "skip_reason" in period:
            print(f"[{i + 1}/{len(periods)}] Skip {name}: {period['skip_reason']}")
            seed_metrics["skipped_periods"][name] = period["skip_reason"]
            continue

        train_end = period["effective_train_end"]
        test_end = period["effective_test_end"]
        test_start = period["test_start"]

        print(f"\n[{i + 1}/{len(periods)}] Period {name}")
        print(f"  Train: {period['train_start']} ~ {train_end}")
        print(f"  Test:  {test_start} ~ {test_end}")
        if period.get("was_clamped"):
            print("  Note: period end was clamped to available calendar date.")

        feature_suffix = feature_suffix_from_path(overnight_feature_path)
        model_path = build_artifact_paths(
            algo,
            cash_mode,
            seed,
            feature_suffix,
            RESULTS_DIR,
            period_name=name,
        )["model"]

        train_env = None
        if saved_model_exists(model_path) and not overwrite:
            print(f"  Resume: found existing model, skip training: {model_path}.zip")
        else:
            # Build training environment only when this period still needs training.
            train_env, _ = build_train_env(
                tickers=tickers,
                train_start=period["train_start"],
                train_end=train_end,
                window_size=WINDOW_SIZE,
                macro_tickers=MACRO_TICKERS_RL,
                settings=SETTINGS,
                enable_cash_action=enable_cash_action,
                enable_margin_short=enable_margin_short,
                overnight_feature_path=overnight_feature_path,
            )

            train_and_save_model(
                algo=algo,
                train_env=train_env,
                timesteps=timesteps,
                model_path=model_path,
                temporal_extractor=temporal_extractor,
            )

        # Build evaluation environment
        test_env, _ = build_eval_env(
            tickers=tickers,
            test_start=test_start,
            test_end=test_end,
            window_size=WINDOW_SIZE,
            macro_tickers=MACRO_TICKERS_RL,
            settings=SETTINGS,
            enable_cash_action=enable_cash_action,
            enable_margin_short=enable_margin_short,
            overnight_feature_path=overnight_feature_path,
        )

        # Load trained model
        from stable_baselines3 import PPO, SAC
        
        model_class = PPO if algo == "ppo" else SAC
        device = resolve_torch_device(SETTINGS.research.torch_device)
        model = model_class.load(model_path, env=test_env, device=device)

        # Run evaluation loop
        eval_results = run_eval_loop(
            model=model,
            test_env=test_env,
            seed=seed,
        )

        # Persist period metrics
        period_start_indices.append((name, len(all_daily_returns)))
        p_metrics = persist_period_metrics(
            algo=algo,
            cash_mode=cash_mode,
            seed=seed,
            feature_suffix=feature_suffix,
            tickers=tickers,
            test_start=test_start,
            test_end=test_end,
            eval_results=eval_results,
            period_name=name,
            results_dir=RESULTS_DIR,
        )
        p_metrics["was_clamped"] = bool(period.get("was_clamped"))
        seed_metrics["periods"][name] = p_metrics

        # Collect data for overall metrics
        all_daily_returns.extend(eval_results["daily_returns"])
        all_positions.extend(eval_results["positions"])
        all_cash_weights.extend(eval_results["cash_weights"])
        all_turnover.extend(eval_results["turnover"])

        print(
            f"  Return={p_metrics['total_return'] * 100:.2f}% | "
            f"MDD={p_metrics['max_drawdown'] * 100:.2f}% | "
            f"Sortino={p_metrics['sortino']:.2f} | "
            f"Cash={p_metrics['avg_cash_weight'] * 100:.2f}%"
        )

        del model, test_env
        if train_env is not None:
            del train_env
        gc.collect()

    if not all_daily_returns:
        print(f"Seed {seed} produced no OOS data.")
        return

    cum_returns = 1_000_000 * np.cumprod(1.0 + np.array(all_daily_returns))
    portfolio_history = [1_000_000] + list(cum_returns)
    overall_metrics = calculate_metrics(
        portfolio_history,
        all_positions,
        all_cash_weights,
        all_daily_returns,
        all_turnover,
        tickers,
    )
    seed_metrics["overall"] = overall_metrics

    feature_suffix = feature_suffix_from_path(overnight_feature_path)
    json_path = build_artifact_paths(algo, cash_mode, seed, feature_suffix, RESULTS_DIR)["metrics"]
    write_metrics_json(seed_metrics, json_path)

    print(f"\n[OK] metrics: {json_path}")
    print(
        f"Overall Return={overall_metrics['total_return'] * 100:+.2f}% | "
        f"MDD={overall_metrics['max_drawdown'] * 100:.2f}% | "
        f"Sortino={overall_metrics['sortino']:.2f} | "
        f"Avg Cash={overall_metrics['avg_cash_weight'] * 100:.2f}%"
    )

    plot_walk_forward(
        algo,
        cash_mode,
        seed,
        portfolio_history,
        all_positions,
        all_cash_weights,
        period_start_indices,
        tickers,
        overall_metrics,
        overnight_feature_path,
    )


def plot_walk_forward(
    algo,
    cash_mode,
    seed,
    portfolio_history,
    all_positions,
    all_cash_weights,
    period_start_indices,
    tickers,
    overall_metrics,
    overnight_feature_path=None,
):
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(
        f"Walk-Forward Validation ({algo.upper()}, cash={cash_mode}, seed={seed})",
        fontsize=16,
    )

    axes[0].plot(
        portfolio_history,
        color="#1f77b4",
        linewidth=2,
        label=f"Portfolio ({overall_metrics['total_return'] * 100:+.2f}%)",
    )
    axes[0].axhline(y=1_000_000, color="gray", linestyle="--", alpha=0.7)
    for name, idx in period_start_indices:
        axes[0].axvline(x=idx, color="red", linestyle=":", alpha=0.8)
        axes[0].text(idx + 1, 1_000_000 * 1.05, name, color="red", fontsize=10)
    axes[0].set_title("Walk-Forward Out-of-Sample Portfolio Value", fontweight="bold")
    axes[0].set_ylabel("Value (TWD)")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    if all_positions:
        pos_matrix = np.array(all_positions).T
        if all_cash_weights:
            pos_matrix = np.vstack([pos_matrix, np.array(all_cash_weights)])
        stock_labels = [TICKER_NAMES.get(t, t) for t in tickers] + ["CASH"]
        im = axes[1].imshow(pos_matrix, aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
        plt.colorbar(im, ax=axes[1], label="Weight")
        axes[1].set_yticks(range(len(stock_labels)))
        axes[1].set_yticklabels(stock_labels, fontsize=9)
        axes[1].set_title("OOS Allocation Heatmap", fontweight="bold")
        axes[1].set_xlabel("Trading Days")
        for _, idx in period_start_indices:
            axes[1].axvline(x=idx, color="black", linestyle=":", alpha=0.8)

    plt.tight_layout()
    feature_suffix = feature_suffix_from_path(overnight_feature_path)
    output_file = build_artifact_paths(algo, cash_mode, seed, feature_suffix, RESULTS_DIR)["chart"]
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] chart: {output_file}")


def parse_args():
    parser = argparse.ArgumentParser(description="Walk-forward research validation")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--algo", choices=["ppo", "sac", "both"], default=SETTINGS.research.default_algo)
    parser.add_argument(
        "--cash-mode",
        choices=["enabled", "disabled", "both"],
        default=SETTINGS.research.walk_forward_cash_mode,
    )
    parser.add_argument("--seeds", type=str, default=SETTINGS.research.default_seeds)
    parser.add_argument(
        "--tier",
        choices=sorted(TIER_PRESETS),
        default=SETTINGS.research.research_tier or None,
        help=(
            "O2 layered training tier. Overrides --timesteps and --seeds. "
            "All tiers train 300K timesteps; they differ by seed count: "
            "smoke=1 seed, candidate=2 seeds, promotion=3 seeds."
        ),
    )
    parser.add_argument(
        "--candidates",
        action="store_true",
        help="O3: train only the curated candidate set (SAC enabled + PPO disabled).",
    )
    parser.add_argument(
        "--matrix",
        action="store_true",
        help=(
            "Run the full PPO/SAC x cash/no-cash matrix (opt-in). "
            "Cost: 4 combos x seeds x 4 periods x timesteps (~18M steps at 300K x 3 seeds). "
            "Prefer --candidates for routine iterations."
        ),
    )
    parser.add_argument("--enable-margin-short", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing completed models")
    parser.add_argument(
        "--overnight-feature-path",
        default=None,
        help=(
            "Opt-in overnight_gap_features_1d.csv path for RL observation features. "
            "Default None keeps base features (with_features is a separate risk-overlay line, "
            "not the main ranking; see R5/O6)."
        ),
    )
    parser.add_argument(
        "--temporal-extractor",
        action="store_true",
        help="Use GRU-over-window TemporalGnnFeatureExtractor.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    timesteps = args.timesteps
    if args.tier:
        timesteps, seeds = resolve_tier(args.tier, seeds)
        print(
            f"[Tier] {args.tier}: timesteps={timesteps:,}, seeds={seeds} "
            "(overrides --timesteps/--seeds)"
        )
    if args.candidates:
        run_candidate_set(
            timesteps=timesteps,
            seeds=seeds,
            enable_margin_short=args.enable_margin_short,
            max_workers=args.workers,
            overwrite=args.overwrite,
            overnight_feature_path=args.overnight_feature_path,
            temporal_extractor=args.temporal_extractor,
        )
    else:
        algos = ["ppo", "sac"] if args.algo == "both" else [args.algo]
        cash_modes = cash_modes_from_arg(args.cash_mode)
        if args.matrix:
            algos = ["ppo", "sac"]
            cash_modes = [True, False]

        run_research_matrix(
            timesteps=timesteps,
            algos=algos,
            cash_modes=cash_modes,
            seeds=seeds,
            enable_margin_short=args.enable_margin_short,
            max_workers=args.workers,
            overwrite=args.overwrite,
            overnight_feature_path=args.overnight_feature_path,
            temporal_extractor=args.temporal_extractor,
        )
