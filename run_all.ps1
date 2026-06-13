param(
    [int]$Timesteps = 200000,
    [string]$Seeds = "42,43,44",
    [int]$Workers = 1
)

Write-Host "Running SAC validation: cash enabled x seeds $Seeds with $Workers workers"
.\env\Scripts\python.exe walk_forward.py --algo sac --cash-mode enabled --timesteps $Timesteps --seeds $Seeds --workers $Workers
if ($LASTEXITCODE -ne 0) {
    Write-Host "SAC validation failed"
    exit $LASTEXITCODE
}

Write-Host "Running SL baseline..."
foreach ($seed in $Seeds.Split(",")) {
    .\env\Scripts\python.exe -m sl_pipeline.walk_forward_sl --allocator rule --seed $seed
    if ($LASTEXITCODE -ne 0) {
        Write-Host "SL baseline failed for seed $seed"
        exit $LASTEXITCODE
    }
}

Write-Host "Generating experiment report..."
$env:PYTHONPATH = (Get-Location).Path
.\env\Scripts\python.exe scripts\experiment_report.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Experiment report failed"
    exit $LASTEXITCODE
}

Write-Host "Done."
