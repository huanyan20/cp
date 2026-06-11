#!/usr/bin/env python3
"""P10 PPO training efficiency ablation (A0–A3).

Runs single-period train + OOS eval for each stage and writes
``results_dir/ppo_efficiency_ablation.json``.

Example:
    ..\\cp\\env\\Scripts\\python.exe scripts\\ppo_efficiency_ablation.py --period 2025H1 --seed 42
    ..\\cp\\env\\Scripts\\python.exe scripts\\ppo_efficiency_ablation.py --smoke --stages A0,A1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stable_baselines3.common.utils import set_random_seed

import research_pipeline
from data_loader import fetch_multi_asset_data
from metrics_utils import calculate_metrics
from research_pipeline import build_eval_env, run_eval_loop
from settings import load_settings
from stock_universe import MACRO_TICKERS_RL, TICKERS_TECH_EXPANDED
from trading_env import TaiwanStockEnv
from train_portfolio import PpoEfficiencyConfig, train_ppo_with_config

SETTINGS = load_settings()
RESULTS_DIR = SETTINGS.paths.results_dir
ROOT = Path(__file__).resolve().parents[1]


def _main_repo_root() -> Path:
    """P10 worktree → sibling cp/; running from cp/ → self."""
    sibling = ROOT.parent / "cp"
    if (sibling / ".research").is_dir():
        return sibling
    return ROOT


def _write_p10_handoff(payload: dict, ablation_path: Path) -> Path:
    handoff_path = _main_repo_root() / ".research" / "handoffs" / "P10.json"
    handoff_path.parent.mkdir(parents=True, exist_ok=True)
    a0 = next((s for s in payload["stages"] if s["stage"] == "A0"), None)
    a3 = next((s for s in payload["stages"] if s["stage"] == "A3"), None)
    a0_mdd = (a0 or {}).get("oos") or {}
    a3_mdd = (a3 or {}).get("oos") or {}
    mdd_delta = None
    if a0_mdd.get("max_drawdown") is not None and a3_mdd.get("max_drawdown") is not None:
        mdd_delta = abs(a3_mdd["max_drawdown"] - a0_mdd["max_drawdown"])
    summary = (
        f"P10 ablation {payload['period']} seed{payload['seed']}: "
        f"A3 fps_vs_a0={(a3 or {}).get('fps_vs_a0', 'n/a')}"
    )
    if mdd_delta is not None:
        summary += f", |ΔMDD|={mdd_delta * 100:.1f}pp"
    handoff = {
        "task_id": "P10",
        "phase": "handoff",
        "agent": "cursor",
        "tool": "cursor-ide",
        "branch": "feat/p10-ppo-vecenv",
        "worktree": "../cp-p10-ppo",
        "files_touched": [
            "train_portfolio.py",
            "settings.py",
            "scripts/ppo_efficiency_ablation.py",
            "tests/test_ppo_vecenv.py",
        ],
        "pytest": "pass",
        "commands_run": [
            "..\\cp\\env\\Scripts\\python.exe -m pytest tests/test_ppo_vecenv.py -q",
            f"..\\cp\\env\\Scripts\\python.exe scripts\\ppo_efficiency_ablation.py "
            f"--period {payload['period']} --seed {payload['seed']}",
        ],
        "summary": summary,
        "ablation_report": str(ablation_path.resolve()),
        "acceptance": {
            "a1_fps_vs_a0_min": 1.5,
            "a3_wall_vs_a0_max": 0.5,
            "mdd_delta_pp_max": 5.0,
            "a3_fps_vs_a0": (a3 or {}).get("fps_vs_a0"),
            "mdd_delta_pp": round(mdd_delta * 100, 2) if mdd_delta is not None else None,
        },
        "ready_for_cross_review": True,
    }
    handoff_path.write_text(json.dumps(handoff, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[OK] handoff {handoff_path}")
    return handoff_path

# Parallel env count: 4 needs ~4x dataset RAM; default 2 for 16GB machines (GTX 1060).
_DEFAULT_PPO_N_ENVS = int(os.environ.get("PPO_N_ENVS", "2"))

ABLATION_MATRIX: list[tuple[str, PpoEfficiencyConfig]] = [
    ("A0", PpoEfficiencyConfig(n_envs=1, vecenv="dummy", n_steps=256, n_epochs=10)),
    (
        "A1",
        PpoEfficiencyConfig(
            n_envs=_DEFAULT_PPO_N_ENVS, vecenv="subproc", n_steps=256, n_epochs=10
        ),
    ),
    (
        "A2",
        PpoEfficiencyConfig(
            n_envs=_DEFAULT_PPO_N_ENVS, vecenv="subproc", n_steps=256, n_epochs=5
        ),
    ),
    (
        "A3",
        PpoEfficiencyConfig(
            n_envs=_DEFAULT_PPO_N_ENVS, vecenv="subproc", n_steps=512, n_epochs=5
        ),
    ),
]


def _resolve_period(name: str) -> dict:
    period = next((p for p in research_pipeline.PERIODS if p["name"] == name), None)
    if period is None:
        choices = [p["name"] for p in research_pipeline.PERIODS]
        raise SystemExit(f"Unknown period {name!r}; choices: {choices}")
    clamped = research_pipeline.clamp_periods([period])
    if clamped[0].get("skip_reason"):
        raise SystemExit(clamped[0]["skip_reason"])
    return clamped[0]


def _apply_n_envs(cfg: PpoEfficiencyConfig, n_envs: int | None) -> PpoEfficiencyConfig:
    if n_envs is None:
        return cfg
    return PpoEfficiencyConfig(
        n_envs=n_envs,
        vecenv=cfg.vecenv,
        n_steps=cfg.n_steps,
        n_epochs=cfg.n_epochs,
        batch_size=cfg.batch_size,
    )


def _new_train_env(train_data: dict) -> TaiwanStockEnv:
    return TaiwanStockEnv(
        df_dict=train_data,
        window_size=SETTINGS.research.window_size,
        topk=SETTINGS.research.default_topk,
        softmax_temp=SETTINGS.research.default_softmax_temp,
        use_benchmark_reward=True,
        enable_cash_action=False,
        enable_margin_short=False,
        max_leverage=SETTINGS.risk_limits.max_leverage,
    )


def _preload_train_data(period: dict) -> dict:
    train_end = period["effective_train_end"]
    print(f"[preload] train data {period['train_start']} ~ {train_end}")
    return fetch_multi_asset_data(
        tickers=TICKERS_TECH_EXPANDED,
        start_date=period["train_start"],
        end_date=train_end,
        window_size=SETTINGS.research.window_size,
        macro_tickers=MACRO_TICKERS_RL,
        overnight_feature_path=None,
    )


def _make_env_factory(train_data: dict, seed: int):
    def _factory():
        set_random_seed(seed)
        return _new_train_env(train_data)

    return _factory


def _smoke_cfg(stage_id: str, cfg: PpoEfficiencyConfig) -> PpoEfficiencyConfig:
    """Smoke: no SubprocVecEnv (avoids 4x RAM + yfinance storm on 16GB)."""
    if stage_id == "A0":
        return cfg
    return PpoEfficiencyConfig(
        n_envs=min(2, cfg.n_envs),
        vecenv="dummy",
        n_steps=cfg.n_steps,
        n_epochs=cfg.n_epochs,
        batch_size=cfg.batch_size,
    )


def run_stage(
    stage_id: str,
    cfg: PpoEfficiencyConfig,
    period: dict,
    seed: int,
    timesteps: int,
    model_dir: Path,
    train_data: dict,
    skip_eval: bool = False,
) -> dict:
    print(f"\n{'=' * 60}\n=== P10 {stage_id} ===\n{'=' * 60}")
    env_factory = _make_env_factory(train_data, seed)
    model_path = model_dir / f"ppo_eff_{stage_id}_seed{seed}.zip"

    try:
        model, elapsed_s = train_ppo_with_config(
            env_factory,
            cfg,
            timesteps=timesteps,
        )
    except Exception as exc:
        if cfg.n_envs > 1 and cfg.vecenv == "subproc":
            fallback = max(1, cfg.n_envs // 2)
            print(f"[{stage_id}] failed with n_envs={cfg.n_envs}, retry n_envs={fallback}: {exc}")
            cfg = _apply_n_envs(cfg, fallback)
            model, elapsed_s = train_ppo_with_config(env_factory, cfg, timesteps=timesteps)
        else:
            raise

    model.save(str(model_path.with_suffix("")))
    fps = timesteps / elapsed_s if elapsed_s > 0 else 0.0

    metrics: dict = {}
    if skip_eval:
        print(f"[{stage_id}] skip_eval=True (smoke train-only)")
    else:
        test_env, _ = build_eval_env(
            tickers=TICKERS_TECH_EXPANDED,
            test_start=period["test_start"],
            test_end=period["effective_test_end"],
            window_size=SETTINGS.research.window_size,
            macro_tickers=MACRO_TICKERS_RL,
            settings=SETTINGS,
            enable_cash_action=False,
            enable_margin_short=False,
            overnight_feature_path=None,
        )
        eval_results = run_eval_loop(model=model, test_env=test_env, seed=seed)
        metrics = calculate_metrics(
            eval_results["portfolio_hist"],
            eval_results["positions"],
            eval_results["cash_weights"],
            eval_results["daily_returns"],
            turnover_history=eval_results["turnover"],
        )

    row = {
        "stage": stage_id,
        "config": {
            "n_envs": cfg.n_envs,
            "vecenv": cfg.vecenv,
            "n_steps": cfg.n_steps,
            "n_epochs": cfg.n_epochs,
            "batch_size": cfg.batch_size,
        },
        "timesteps": timesteps,
        "elapsed_s": round(elapsed_s, 2),
        "fps": round(fps, 2),
        "model_path": str(model_path.with_suffix(".zip")),
        "oos": None if skip_eval else {
            "total_return": metrics.get("total_return"),
            "max_drawdown": metrics.get("max_drawdown"),
            "sortino": metrics.get("sortino"),
            "avg_cash_weight": metrics.get("avg_cash_weight"),
            "turnover": metrics.get("avg_turnover"),
        },
    }
    if skip_eval:
        print(f"[{stage_id}] fps={row['fps']:.1f} elapsed={elapsed_s:.1f}s (train-only)")
    else:
        print(
            f"[{stage_id}] fps={row['fps']:.1f} elapsed={elapsed_s / 60:.1f}m "
            f"Return={metrics.get('total_return', 0) * 100:+.2f}% "
            f"MDD={metrics.get('max_drawdown', 0) * 100:.2f}% "
            f"Sortino={metrics.get('sortino', 0):.2f}"
        )
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period", default="2025H1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--smoke", action="store_true", help="Use 1_000 timesteps per stage")
    parser.add_argument(
        "--stages",
        default="A0,A1,A2,A3",
        help="Comma-separated stage ids (default: all)",
    )
    parser.add_argument("--n-envs", type=int, default=None, help="Override n_envs for A1–A3")
    parser.add_argument(
        "--output",
        default=None,
        help="JSON output path (default: results_dir/ppo_efficiency_ablation.json)",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Directory for stage checkpoints (default: temp dir)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timesteps = 1_000 if args.smoke else args.timesteps
    stage_ids = {s.strip() for s in args.stages.split(",") if s.strip()}
    stages = [(sid, cfg) for sid, cfg in ABLATION_MATRIX if sid in stage_ids]
    if not stages:
        raise SystemExit(f"No stages matched {args.stages!r}")

    period = _resolve_period(args.period)
    period_name = period["name"]
    train_data = _preload_train_data(period)
    skip_eval = args.smoke
    output_path = Path(args.output or RESULTS_DIR / "ppo_efficiency_ablation.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.model_dir:
        model_dir = Path(args.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        temp_ctx = None
    else:
        temp_ctx = tempfile.TemporaryDirectory(prefix=f"ppo_eff_{period_name}_")
        model_dir = Path(temp_ctx.name)

    results: list[dict] = []
    try:
        for stage_id, cfg in stages:
            cfg = _apply_n_envs(cfg, args.n_envs)
            if args.smoke:
                cfg = _smoke_cfg(stage_id, cfg)
            results.append(
                run_stage(
                    stage_id,
                    cfg,
                    period,
                    args.seed,
                    timesteps,
                    model_dir,
                    train_data,
                    skip_eval=skip_eval,
                )
            )
    finally:
        if temp_ctx is not None and args.smoke:
            temp_ctx.cleanup()

    payload = {
        "generated_at": datetime.now(UTC).astimezone().isoformat(timespec="seconds"),
        "period": period_name,
        "seed": args.seed,
        "timesteps_per_stage": timesteps,
        "smoke": args.smoke,
        "stages": results,
    }
    if results:
        a0_fps = next((r["fps"] for r in results if r["stage"] == "A0"), None)
        for row in results:
            if a0_fps and a0_fps > 0:
                row["fps_vs_a0"] = round(row["fps"] / a0_fps, 3)

    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n[OK] wrote {output_path}")
    if not args.smoke and len(results) == len(stages):
        _write_p10_handoff(payload, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
