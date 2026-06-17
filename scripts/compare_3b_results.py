import json
from pathlib import Path

paths = {
    'TWII+h20': 'reports/milestone_3b/lightgbm_20d_seed42/run_summary.json',
    'TWII+h5':  'reports/milestone_3b/lightgbm_5d_seed42/run_summary.json',
    'TWII+h10': 'reports/milestone_3b/lightgbm_10d_seed42/run_summary.json',
}

period_names = ['2022_BEAR', '2024H2', '2025H1', '2025H2', '2026H1']
header = f"{'Period':<13} | {'TWII+h20':>10} | {'TWII+h5':>10} | {'TWII+h10':>10}"
print(header)
print('-' * len(header))

for name in period_names:
    cols = [name]
    for label, p in paths.items():
        d = json.loads(Path(p).read_text('utf-8'))
        m = d['periods'].get(name, {})
        ret = m.get('total_return', float('nan'))
        cols.append(f'{ret*100:+.1f}%')
    print(f'{cols[0]:<13} | {cols[1]:>10} | {cols[2]:>10} | {cols[3]:>10}')

print('-' * len(header))

# Overall
overall_cols = ['OVERALL Ret']
mdd_cols = ['OVERALL MDD']
cash_cols = ['AvgCash']
for label, p in paths.items():
    d = json.loads(Path(p).read_text('utf-8'))
    m = d['overall']
    overall_cols.append(f'{m["total_return"]*100:+.1f}%')
    mdd_cols.append(f'{m["max_drawdown"]*100:.1f}%')
    cash_cols.append(f'{m["avg_cash_weight"]*100:.0f}%')

print(f'{overall_cols[0]:<13} | {overall_cols[1]:>10} | {overall_cols[2]:>10} | {overall_cols[3]:>10}')
print(f'{mdd_cols[0]:<13} | {mdd_cols[1]:>10} | {mdd_cols[2]:>10} | {mdd_cols[3]:>10}')
print(f'{cash_cols[0]:<13} | {cash_cols[1]:>10} | {cash_cols[2]:>10} | {cash_cols[3]:>10}')
