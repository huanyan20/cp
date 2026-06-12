param(
    [int]$Timesteps = 300000,
    [string]$Seeds = "42,43,44",
    [int]$Workers = 4
)

Write-Host "Running research matrix: PPO/SAC x cash enabled/disabled x seeds $Seeds with $Workers workers"
.\env\Scripts\python.exe walk_forward.py --matrix --timesteps $Timesteps --seeds $Seeds --workers $Workers
if ($LASTEXITCODE -ne 0) {
    Write-Host "Research matrix failed"
    exit $LASTEXITCODE
}

Write-Host "Generating experiment report..."
$env:PYTHONPATH = (Get-Location).Path
.\env\Scripts\python.exe scripts\experiment_report.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Experiment report failed"
    exit $LASTEXITCODE
}

Write-Host "Done."
