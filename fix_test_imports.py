import os
import glob

mapping = {
    'import cmoney_rpa': 'import rpa_pipeline.cmoney_rpa as cmoney_rpa',
    'import daily_trade_runner': 'import rpa_pipeline.daily_trade_runner as daily_trade_runner',
    'import trade_guard': 'import rpa_pipeline.trade_guard as trade_guard',
    'import experiment_report': 'import scripts.experiment_report as experiment_report',
    'import p5_analysis': 'import scripts.p5_analysis as p5_analysis',
    'import evaluate_portfolio': 'import scripts.evaluate_portfolio as evaluate_portfolio',
    'import optuna_tune': 'import scripts.optuna_tune as optuna_tune',
    'from experiment_report': 'from scripts.experiment_report',
}

for f in glob.glob('tests/*.py'):
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()
    
    modified = False
    for k, v in mapping.items():
        if k in content:
            content = content.replace(k, v)
            modified = True
            
    if modified:
        with open(f, 'w', encoding='utf-8') as file:
            file.write(content)
        print(f"Updated {f}")
