import sys
sys.path.append(".")
from research_pipeline import PERIODS, build_period_plan

plan = build_period_plan()
for p in plan:
    period = next((x for x in PERIODS if x["name"] == p["name"]), {})
    train_start = period.get("train_start", "?")
    test_start = period.get("test_start", "?")
    eff_train_end = p.get("effective_train_end", "?")
    eff_test_end = p.get("effective_test_end", "?")
    skip = p.get("skip_reason", "")
    print(f"{p['name']:12s}  train={train_start}~{eff_train_end}  test={test_start}~{eff_test_end}  skip={skip}")
