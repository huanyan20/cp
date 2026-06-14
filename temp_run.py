
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
            target_vol_annual=0.1,
            yellow_mdd=0.08,
            red_mdd=0.12,
            weight_band=0.1
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
    print(json.dumps({"return": overall["total_return"], "mdd": overall["max_drawdown"], "turnover": overall["turnover"]}))
