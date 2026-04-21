$workspace = Split-Path -Parent $MyInvocation.MyCommand.Path

$pythonCandidates = @()

if ($env:MEDICALAI_PYTHON) {
    $pythonCandidates += $env:MEDICALAI_PYTHON
}

$pythonCandidates += @(
    (Join-Path $workspace ".venv\Scripts\python.exe"),
    (Join-Path $env:USERPROFILE "anaconda3\envs\torch_env\python.exe"),
    (Join-Path $env:USERPROFILE "anaconda3\envs\pytorchenv\python.exe"),
    "python"
)

$pythonPath = $null
foreach ($candidate in ($pythonCandidates | Select-Object -Unique)) {
    if ($candidate -eq "python") {
        $pythonPath = $candidate
        break
    }

    if (Test-Path $candidate) {
        $pythonPath = $candidate
        break
    }
}

if (-not $pythonPath) {
    throw "No usable Python interpreter was found. Set MEDICALAI_PYTHON or create a local .venv."
}

Write-Host "Starting Automated Medical Image Analysis for Disease Diagnosis Using Deep Learning using $pythonPath"
& $pythonPath (Join-Path $workspace "app.py")
