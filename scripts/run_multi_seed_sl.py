"""Multi-seed confirmation script for SL models."""

import argparse
import json
import logging
from pathlib import Path
import sys
from datetime import UTC, datetime

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sl_pipeline.walk_forward_sl import run_walk_forward_sl
from sl_pipeline.candidate import CURRENT_CANDIDATE_ID
from sl_pipeline.gate import run_sl_promotion_gate, save_sl_gate_result
from settings import load_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("MultiSeed")

def main():
    parser = argparse.ArgumentParser(description="Run SL walk-forward across multiple seeds to confirm stability.")
    parser.add_argument("--seeds", type=str, default="42,43,44", help="Comma-separated list of seeds")
    parser.add_argument("--horizon", type=int, default=10, help="Prediction horizon (e.g. 10 for h10)")
    parser.add_argument("--allocator", type=str, default="rule", help="Allocator name")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save manifests")
    parser.add_argument(
        "--allow-blocked-exit-zero",
        action="store_true",
        help="Return exit code 0 even when the multiseed gate is blocked.",
    )
    args = parser.parse_args()
    settings = load_settings()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        logger.error("No valid seeds provided.")
        sys.exit(1)

    logger.info(f"Starting multi-seed confirmation for SL horizon {args.horizon} over seeds: {seeds}")

    results = {}
    all_passed = True
    failed_seeds = []
    stress_by_seed = {}

    out_dir = Path(args.output_dir) if args.output_dir else None

    for seed in seeds:
        logger.info(f"\n{'='*40}\nEvaluating Seed: {seed}\n{'='*40}")
        try:
            res = run_walk_forward_sl(
                horizon=args.horizon,
                allocator_name=args.allocator,
                seed=seed,
                output_dir=out_dir,
                run_gate=False
            )

            results[seed] = {
                "candidate_id": res.get("candidate_id"),
                "metrics_path": res.get("metrics_path"),
                "stress_summary_path": res.get("stress_summary_path"),
                "overall_sortino": res.get("overall", {}).get("sortino", 0.0),
                "overall_mdd": res.get("overall", {}).get("max_drawdown", 0.0),
                "overall_return": res.get("overall", {}).get("total_return", 0.0),
            }
            stress_by_seed[seed] = res.get("stress_summary", {})

            logger.info(
                "Seed %s done: return=%.2f%% sortino=%.2f mdd=%.2f%%",
                seed,
                results[seed]["overall_return"] * 100,
                results[seed]["overall_sortino"],
                results[seed]["overall_mdd"] * 100,
            )

        except Exception as e:
            logger.error(f"Failed to evaluate seed {seed}: {e}")
            all_passed = False
            failed_seeds.append(seed)
            results[seed] = {"error": str(e), "core_gate_approved": False, "full_gate_approved": False}

    aggregate_tests = {}
    for seed, stress in stress_by_seed.items():
        for name, data in stress.get("tests", {}).items():
            current = aggregate_tests.setdefault(
                name,
                {
                    "total_return": data.get("total_return", 0.0),
                    "max_drawdown": data.get("max_drawdown", 0.0),
                    "worst_return_seed": seed,
                    "worst_mdd_seed": seed,
                },
            )
            if data.get("total_return", 0.0) < current["total_return"]:
                current["total_return"] = data.get("total_return", 0.0)
                current["worst_return_seed"] = seed
            if data.get("max_drawdown", 0.0) > current["max_drawdown"]:
                current["max_drawdown"] = data.get("max_drawdown", 0.0)
                current["worst_mdd_seed"] = seed
    aggregate_stress = {
        "candidate_id": CURRENT_CANDIDATE_ID,
        "horizon": args.horizon,
        "allocator": args.allocator,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "source_files": [
            item["stress_summary_path"]
            for item in results.values()
            if isinstance(item, dict) and item.get("stress_summary_path")
        ],
        "tests": aggregate_tests,
    }
    stress_path = settings.paths.stress_summary_path
    stress_path.write_text(json.dumps(aggregate_stress, indent=2), encoding="utf-8")

    gate_result, raw_summary, _ = run_sl_promotion_gate(
        results_dir=settings.paths.results_dir,
        target_horizon=args.horizon,
        target_candidate_id=CURRENT_CANDIDATE_ID,
    )
    gate_path = save_sl_gate_result(
        gate_result,
        raw_summary,
        results_dir=settings.paths.results_dir,
        horizon=args.horizon,
        allocator=args.allocator,
        seed=None,
    )
    all_passed = all_passed and gate_result.core_gate_approved

    logger.info(f"\n{'='*40}\nMulti-Seed Confirmation Summary\n{'='*40}")
    print(json.dumps({
        "candidate_id": CURRENT_CANDIDATE_ID,
        "gate_result_path": str(gate_path),
        "promotion_gate": {
            "core_gate_approved": gate_result.core_gate_approved,
            "full_gate_approved": gate_result.full_gate_approved,
            "risk_level": gate_result.risk_level,
            "summary": gate_result.summary,
        },
        "seeds": results,
    }, indent=2))

    if all_passed:
        logger.info("SUCCESS: All seeds passed the promotion gate. Model behavior is stable.")
        sys.exit(0)
    else:
        logger.error(
            "FAILURE: Multiseed gate is blocked. Failed seeds/errors: %s. Do NOT promote this model to live.",
            failed_seeds,
        )
        sys.exit(0 if args.allow_blocked_exit_zero else 1)

if __name__ == "__main__":
    main()
