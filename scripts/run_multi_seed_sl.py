"""Multi-seed confirmation script for SL models."""

import argparse
import json
import logging
from pathlib import Path
import sys

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sl_pipeline.walk_forward_sl import run_walk_forward_sl

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("MultiSeed")

def main():
    parser = argparse.ArgumentParser(description="Run SL walk-forward across multiple seeds to confirm stability.")
    parser.add_argument("--seeds", type=str, default="42,43,44", help="Comma-separated list of seeds")
    parser.add_argument("--horizon", type=int, default=10, help="Prediction horizon (e.g. 10 for h10)")
    parser.add_argument("--allocator", type=str, default="rule", help="Allocator name")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save manifests")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        logger.error("No valid seeds provided.")
        sys.exit(1)

    logger.info(f"Starting multi-seed confirmation for SL horizon {args.horizon} over seeds: {seeds}")

    results = {}
    all_passed = True
    failed_seeds = []

    out_dir = Path(args.output_dir) if args.output_dir else None

    for seed in seeds:
        logger.info(f"\n{'='*40}\nEvaluating Seed: {seed}\n{'='*40}")
        try:
            # We enforce run_gate=True to get the promotion check
            res = run_walk_forward_sl(
                horizon=args.horizon,
                allocator_name=args.allocator,
                seed=seed,
                output_dir=out_dir,
                run_gate=True
            )
            
            gate = res.get("promotion_gate", {})
            core_gate_approved = gate.get("core_gate_approved", False)
            full_gate_approved = gate.get("full_gate_approved", False)
            summary = gate.get("summary", "No summary")

            results[seed] = {
                "core_gate_approved": core_gate_approved,
                "full_gate_approved": full_gate_approved,
                "summary": summary,
                "overall_sortino": res.get("overall", {}).get("sortino", 0.0),
                "overall_mdd": res.get("overall", {}).get("max_drawdown", 0.0)
            }

            logger.info(f"Seed {seed} Gate Result: {'APPROVED' if core_gate_approved else 'BLOCKED'}")
            logger.info(f"Summary: {summary}")

            if not core_gate_approved:
                all_passed = False
                failed_seeds.append(seed)

        except Exception as e:
            logger.error(f"Failed to evaluate seed {seed}: {e}")
            all_passed = False
            failed_seeds.append(seed)
            results[seed] = {"error": str(e), "core_gate_approved": False, "full_gate_approved": False}

    logger.info(f"\n{'='*40}\nMulti-Seed Confirmation Summary\n{'='*40}")
    print(json.dumps(results, indent=2))

    if all_passed:
        logger.info("SUCCESS: All seeds passed the promotion gate. Model behavior is stable.")
        sys.exit(0)
    else:
        logger.error(f"FAILURE: The following seeds failed the gate: {failed_seeds}. Do NOT promote this model to live.")
        sys.exit(1)

if __name__ == "__main__":
    main()
