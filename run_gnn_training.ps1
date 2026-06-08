$ErrorActionPreference = "Stop"

Write-Host "==================================================="
Write-Host "Starting PPO Training with Temporal GNN Extractor"
Write-Host "==================================================="

# Run a shorter training run to demonstrate the integration
.\env\Scripts\python.exe train_portfolio.py --algo ppo --timesteps 10000 --temporal-extractor --seed 42

Write-Host "Training finished successfully!"
