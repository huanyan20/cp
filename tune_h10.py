import json
import subprocess
import itertools

def evaluate(vol_target, yellow_mdd, red_mdd, weight_band):
    code = f"""
from sl_pipeline.rule_based_allocator import RuleBasedAllocator, RuleBasedAllocatorConfig
from sl_pipeline.walk_forward_sl import run_walk_forward_sl

def build_allocator(name):
    from settings import load_settings
    settings = load_settings()
    return RuleBasedAllocator(
        RuleBasedAllocatorConfig(
            top_k=settings.research.default_topk,
            max_single_weight=settings.risk_limits.max_single_weight,
            enable_vol_target=True,
            target_vol_annual={vol_target},
            yellow_mdd={yellow_mdd},
            red_mdd={red_mdd},
            weight_band={weight_band}
        )
    )

import sl_pipeline.walk_forward_sl
sl_pipeline.walk_forward_sl.build_allocator = build_allocator

if __name__ == "__main__":
    result = sl_pipeline.walk_forward_sl.run_walk_forward_sl(
        horizon=10,
        allocator_name="rule",
        seed=42,
        run_gate=False
    )
    overall = result["overall"]
    print(json.dumps({{"return": overall["total_return"], "mdd": overall["max_drawdown"], "turnover": overall["turnover"]}}))
"""
    with open("temp_run.py", "w") as f:
        f.write(code)
    
    out = subprocess.check_output(["python", "temp_run.py"], env={"PYTHONPATH": "."})
    res = json.loads(out.decode('utf-8').strip().split('\n')[-1])
    return res

print("Testing configurations...")
configs = [
    (0.15, 0.08, 0.12, 0.05),
    (0.18, 0.10, 0.15, 0.05),
    (0.12, 0.05, 0.10, 0.08),
    (0.20, 0.15, 0.20, 0.05),
    (0.10, 0.08, 0.12, 0.10)
]

for c in configs:
    try:
        res = evaluate(*c)
        print(f"Config {c}: MDD: {res['mdd']*100:.2f}%, Turnover: {res['turnover']*100:.2f}%, Return: {res['return']*100:.2f}%")
    except Exception as e:
        print(f"Config {c} failed: {e}")
