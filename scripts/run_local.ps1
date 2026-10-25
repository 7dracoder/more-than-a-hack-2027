param(
    [switch]$Download,
    [switch]$OpenTest,
    [switch]$GapRobust,
    [string]$OutputDirectory = 'data/local-v07/reproduction'
)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $taskRoot
try {
    $taskTemp = Join-Path $taskRoot 'data\test-tmp'
    New-Item -ItemType Directory -Force -Path $taskTemp | Out-Null
    $env:TEMP = $taskTemp
    $env:TMP = $taskTemp
    $env:OMP_NUM_THREADS = '2'
    $env:MKL_NUM_THREADS = '2'
    $env:OPENBLAS_NUM_THREADS = '2'
    $env:PYTHONIOENCODING = 'utf-8'
    $env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
    $taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
    if (!(Test-Path -LiteralPath $taskPython)) { throw 'Set up the project CUDA environment first; see LOCAL_V07.md.' }
    function Invoke-LocalPython {
        & $taskPython @args
        if ($LASTEXITCODE -ne 0) { throw "Local Python step failed with exit code $LASTEXITCODE" }
    }
    if ($Download) { Invoke-LocalPython scripts/fetch_local_data.py --external --competition }
    Invoke-LocalPython scripts/prepare_external.py
    Invoke-LocalPython -m unittest discover -s tests -v
    $taskFirst = Join-Path $OutputDirectory 'tcn'
    $taskSecond = Join-Path $OutputDirectory 'bigru'
    $taskEnsemble = Join-Path $OutputDirectory 'ensemble'
    Invoke-LocalPython scripts/run_local.py --train --resume --out $taskFirst
    if ($GapRobust) {
        Invoke-LocalPython scripts/prepare_external_gaps.py --matches J03WMX J03WN1 J03WOH J03WOY J03WPY
        $taskGaps = Join-Path $OutputDirectory 'gaps'
        New-Item -ItemType Directory -Force -Path $taskGaps | Out-Null
        foreach ($taskFile in @('baseline.joblib','baseline_training.json','baseline_validation.json')) {
            $taskDestination = Join-Path $taskGaps $taskFile
            if (!(Test-Path -LiteralPath $taskDestination)) {
                Copy-Item -LiteralPath (Join-Path $taskFirst $taskFile) -Destination $taskDestination
            }
        }
        Invoke-LocalPython scripts/run_local.py --train --resume --config configs/local_v07_gaps.json --out $taskGaps
        Invoke-LocalPython scripts/evaluate_gap_stress.py --run $taskGaps
        Invoke-LocalPython scripts/run_local.py --infer --config configs/local_v07_gaps.json --out $taskGaps
        Invoke-LocalPython evaluation/validate_submission.py (Join-Path $taskGaps 'candidate\CoreOrigin.csv')
        if ($OpenTest) { Invoke-LocalPython scripts/run_local.py --test --config configs/local_v07_gaps.json --out $taskGaps }
        return
    }
    New-Item -ItemType Directory -Force -Path $taskSecond | Out-Null
    foreach ($taskFile in @('baseline.joblib','baseline_training.json','baseline_validation.json')) {
        $taskDestination = Join-Path $taskSecond $taskFile
        if (!(Test-Path -LiteralPath $taskDestination)) {
            Copy-Item -LiteralPath (Join-Path $taskFirst $taskFile) -Destination $taskDestination
        }
    }
    Invoke-LocalPython scripts/run_local.py --train --resume --config configs/local_v07_bigru.json --out $taskSecond
    if (!(Test-Path -LiteralPath (Join-Path $taskEnsemble 'selected.pt'))) {
        Invoke-LocalPython scripts/select_local_ensemble.py --first $taskFirst --second $taskSecond --out $taskEnsemble
    }
    $taskConfiguration = Join-Path $taskEnsemble 'config.json'
    Invoke-LocalPython scripts/run_local.py --infer --config $taskConfiguration --out $taskEnsemble
    Invoke-LocalPython evaluation/validate_submission.py (Join-Path $taskEnsemble 'candidate\CoreOrigin.csv')
    if ($OpenTest) { Invoke-LocalPython scripts/run_local.py --test --config $taskConfiguration --out $taskEnsemble }
}
finally { Pop-Location }
