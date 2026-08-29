$ErrorActionPreference = "Continue"

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$logDir = "logs/lcycle-history/overnight-$stamp"
New-Item -ItemType Directory -Force $logDir | Out-Null

$manifest = Join-Path $logDir "manifest.tsv"
"run`tmethod`tseed`tstatus`truntime_minutes" | Set-Content $manifest

$experiments = @()

foreach ($seed in 0..4) {
    $experiments += @{
        Seed = $seed
        Method = "gated"
        RunMethod = "gated"
    }
}

foreach ($seed in 0..4) {
    $experiments += @{
        Seed = $seed
        Method = "parameter_matched"
        RunMethod = "parammatched"
    }
}

foreach ($exp in $experiments) {

    $seed = $exp.Seed
    $method = $exp.Method
    $runMethod = $exp.RunMethod

    $run = "proposal-h3l6-l2-$runMethod-40ep-seed$seed"

    $metricsDir = "results/lcycle-history/$run"
    $checkpointDir = "checkpoints/lcycle-history/$run"
    $finalMetric = Join-Path $metricsDir "metrics_step_10000.json"
    $log = Join-Path $logDir "$run.log"

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "RUN: $run"
    Write-Host "============================================================"

    if (Test-Path $finalMetric) {

        Write-Host "SKIP: final metric already exists"

        "$run`t$method`t$seed`tSKIPPED_EXISTING`t0" |
            Add-Content $manifest

        continue
    }

    $args = @(
        "pretrain.py",
        "--config-name", "cfg_baseline_v2",

        "arch.halt_max_steps=4",
        "arch.H_cycles=3",
        "arch.L_cycles=6",
        "arch.L_layers=2",

        "+arch.history_enabled=false",
        "+arch.history_aggregator=none",

        "+arch.lcycle_history_enabled=true",
        "+arch.lcycle_history_method=$method",
        "+arch.lcycle_history_rank=16",
        "+arch.lcycle_history_heads=4",
        "+arch.lcycle_history_gate_init=-2.0",
        "+arch.lcycle_history_pre_norm=true",

        "epochs=40",
        "eval_interval=5",
        "seed=$seed",

        "run_name=$run",
        "checkpoint_path=$checkpointDir",
        "metrics_dir=$metricsDir"
    )

    $start = Get-Date

    & $env:TRM_PYTHON @args 2>&1 |
        Tee-Object -FilePath $log

    $exitCode = $LASTEXITCODE

    $elapsed = (Get-Date) - $start
    $minutes = [math]::Round($elapsed.TotalMinutes, 3)

    if (($exitCode -eq 0) -and (Test-Path $finalMetric)) {
        $status = "PASS"
    }
    else {
        $status = "FAIL"
    }

    "$run`t$method`t$seed`t$status`t$minutes" |
        Add-Content $manifest

    Write-Host ""
    Write-Host "FINISHED: $run"
    Write-Host "STATUS: $status"
    Write-Host "RUNTIME: $minutes minutes"
}

Write-Host ""
Write-Host "============================================================"
Write-Host "OVERNIGHT CAMPAIGN COMPLETE"
Write-Host "============================================================"
Write-Host "Manifest: $manifest"
