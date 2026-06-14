import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import tempfile
import walk_forward
import research_pipeline
from settings import resolve_tier

# Mock the period plan to only return 2025H1
def custom_period_plan():
    # Calling the original to get the standard definitions
    # Actually wait, research_pipeline.build_period_plan returns a list of dicts.
    all_periods = research_pipeline.build_period_plan(None, None)
    return [p for p in all_periods if p["name"] == "2025H1"]

walk_forward.build_period_plan = custom_period_plan

tmp = tempfile.mkdtemp(prefix="smoke_2025h1_")
walk_forward.RESULTS_DIR = tmp
timesteps = 50000
seeds = [42]
candidate_pairs = [("sac", True)]

print(f"Running 2025H1 smoke test in {tmp}...")
walk_forward.run_candidate_set(
    timesteps=timesteps,
    seeds=seeds,
    overwrite=True,
    overnight_feature_path=None,
    candidate_pairs=candidate_pairs,
)

print("\n=== 2025H1 SMOKE SUMMARY ===")
for path in sorted(Path(tmp).glob("metrics_*.json")):
    d = json.loads(path.read_text(encoding="utf-8"))
    o = d.get("overall", {})
    print(
        f"{d.get('algo')}/{d.get('cash_mode')} seed{d.get('seed')}: "
        f"MDD={o.get('max_drawdown', 0) * 100:.2f}% "
        f"Sortino={o.get('sortino', 0):.2f} "
        f"Return={o.get('total_return', 0) * 100:+.2f}% "
        f"AvgCash={o.get('avg_cash_weight', 0) * 100:.2f}% "
        f"Turnover={o.get('turnover', 0) * 100:.2f}%"
    )
