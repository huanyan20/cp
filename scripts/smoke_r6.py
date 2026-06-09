"""R6 smoke validation (O2 smoke tier) into an ISOLATED results dir.

Purpose: sanity-check the R4 reward (lambda_drawdown 0.8, regime/defensive-cash)
end-to-end without touching the canonical ``results_dir/``. Artifact filenames do
not encode the tier, so a smoke run reuses the promotion filenames; redirecting to
a throwaway directory keeps the real metrics intact.

Run from repo root:
    python scripts/smoke_r6.py            # SAC enabled (default, fastest signal)
    python scripts/smoke_r6.py --both     # SAC enabled + PPO disabled

Smoke models are undertrained (30K). Read MDD *direction*, not absolute quality.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import tempfile

import walk_forward
from env_config import ENV_CONFIG_VERSION, get_current_env_config_hash
from settings import resolve_tier


def main():
    parser = argparse.ArgumentParser(description="Isolated R6 smoke validation")
    parser.add_argument(
        "--both",
        action="store_true",
        help="Run SAC enabled + PPO disabled (default: SAC enabled only).",
    )
    args = parser.parse_args()

    candidate_pairs = [("sac", True)]
    if args.both:
        candidate_pairs = [("sac", True), ("ppo", False)]

    tmp = tempfile.mkdtemp(prefix="smoke_r6_")
    walk_forward.RESULTS_DIR = tmp  # redirect ALL artifacts to a throwaway dir
    timesteps, seeds = resolve_tier("smoke", [42])

    print(
        f"[smoke] env={ENV_CONFIG_VERSION}/{get_current_env_config_hash()} "
        f"timesteps={timesteps} seeds={seeds} "
        f"pairs={[(a, 'enabled' if c else 'disabled') for a, c in candidate_pairs]}"
    )
    print(f"[smoke] isolated results_dir={tmp}")

    walk_forward.run_candidate_set(
        timesteps=timesteps,
        seeds=seeds,
        overwrite=True,
        overnight_feature_path=None,
        candidate_pairs=candidate_pairs,
    )

    print("\n=== SMOKE SUMMARY (undertrained 30K — check MDD direction only) ===")
    for path in sorted(Path(tmp).glob("metrics_*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        o = d.get("overall", {})
        print(
            f"{d.get('algo')}/{d.get('cash_mode')} seed{d.get('seed')}: "
            f"MDD={o.get('max_drawdown', 0) * 100:.2f}% "
            f"Sortino={o.get('sortino', 0):.2f} "
            f"Return={o.get('total_return', 0) * 100:+.2f}% "
            f"AvgCash={o.get('avg_cash_weight', 0) * 100:.2f}% "
            f"env={d.get('env_config_version')}/{d.get('env_config_hash')}"
        )
        for pname, pm in d.get("periods", {}).items():
            print(
                f"    {pname}: MDD={pm.get('max_drawdown', 0) * 100:.2f}% "
                f"Return={pm.get('total_return', 0) * 100:+.2f}%"
            )
    print(f"\n[smoke] artifacts in {tmp} (safe to delete).")


if __name__ == "__main__":
    main()
