import json
from pathlib import Path

# Load metrics to see performance
p = Path(r"c:\Users\ggini\Desktop\cp\results_dir\metrics_sl_rule_h10_seed42.json")
if p.exists():
    with open(p, "r") as f:
        data = json.load(f)
    print("overall return:", data["overall"]["total_return"])
    for p_name, p_data in data["periods"].items():
        print(f"{p_name}: return={p_data['total_return']:.4f}, mdd={p_data['max_drawdown']:.4f}, cash={p_data['avg_cash_weight']:.4f}")
