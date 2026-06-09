"""Single-period walk-forward validation into an ISOLATED results dir.

Train one model on ONE walk-forward period at a chosen timestep budget without
touching the canonical ``results_dir/``. Useful for fast worst-case-direction
checks (e.g. the 2025H1 bear period at the candidate tier, 150K).

Run from repo root:
    python scripts/validate_period.py --period 2025H1 --timesteps 300000
    python scripts/validate_period.py --period 2025H1 --timesteps 300000 --algo ppo --cash disabled
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import tempfile

import research_pipeline
import walk_forward
from env_config import ENV_CONFIG_VERSION, get_current_env_config_hash


def main():
    parser = argparse.ArgumentParser(description="Isolated single-period validation")
    parser.add_argument("--period", default="2025H1", help="Period name from research_pipeline.PERIODS")
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--algo", choices=["sac", "ppo"], default="sac")
    parser.add_argument("--cash", choices=["enabled", "disabled"], default="enabled")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    period = next((p for p in research_pipeline.PERIODS if p["name"] == args.period), None)
    if period is None:
        choices = [p["name"] for p in research_pipeline.PERIODS]
        raise SystemExit(f"Unknown period {args.period}; choices: {choices}")

    tmp = tempfile.mkdtemp(prefix=f"valid_{args.period}_")
    walk_forward.RESULTS_DIR = tmp  # redirect ALL artifacts to a throwaway dir
    # Restrict the walk-forward plan to the single requested period.
    walk_forward.build_period_plan = lambda *a, **k: research_pipeline.clamp_periods([period])

    enable_cash = args.cash == "enabled"
    print(
        f"[validate] env={ENV_CONFIG_VERSION}/{get_current_env_config_hash()} "
        f"{args.algo}/{args.cash} seed{args.seed} period={args.period} "
        f"timesteps={args.timesteps}"
    )
    print(f"[validate] isolated results_dir={tmp}")

    walk_forward.run_walk_forward(
        timesteps=args.timesteps,
        algo=args.algo,
        seeds=[args.seed],
        enable_cash_action=enable_cash,
        overwrite=True,
        overnight_feature_path=None,
    )

    print("\n=== VALIDATION SUMMARY ===")
    for path in sorted(Path(tmp).glob("metrics_*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        for pname, pm in d.get("periods", {}).items():
            print(
                f"{d.get('algo')}/{d.get('cash_mode')} seed{d.get('seed')} {pname}: "
                f"MDD={pm.get('max_drawdown', 0) * 100:.2f}% "
                f"Return={pm.get('total_return', 0) * 100:+.2f}% "
                f"Sortino={pm.get('sortino', 0):.2f} "
                f"AvgCash={pm.get('avg_cash_weight', 0) * 100:.2f}% "
                f"env={d.get('env_config_version')}/{d.get('env_config_hash')}"
            )
    print(f"\n[validate] artifacts in {tmp} (safe to delete).")


if __name__ == "__main__":
    main()
