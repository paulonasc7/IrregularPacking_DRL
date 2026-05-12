param(
    [string]$VenvDir = ".venv"
)

$ErrorActionPreference = "Stop"

function Test-Command {
    param([string]$Name)
    return $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

$pythonExe = $null
$pythonArgs = @()

if (Test-Command "py") {
    $probe311 = & py -3.11 -c "import sys; print(sys.version)" 2>$null
    if ($LASTEXITCODE -eq 0) {
        $pythonExe = "py"
        $pythonArgs = @("-3.11")
    }
}

if ($null -eq $pythonExe) {
    if (Test-Command "python") {
        $pythonExe = "python"
    } else {
        throw "Neither 'py' nor 'python' was found on PATH. Install Python 3.10-3.12 and rerun."
    }
}

$pyVersionText = & $pythonExe @pythonArgs -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
$pyVersionParts = $pyVersionText.Split(".") | ForEach-Object { [int]$_ }

if ($pyVersionParts[0] -eq 3 -and $pyVersionParts[1] -ge 13) {
    Write-Warning "Detected Python $pyVersionText. PyTorch wheels may be unavailable for this version."
    Write-Warning "Recommended: Python 3.10-3.12."
}

if (-not (Test-Path $VenvDir)) {
    & $pythonExe @pythonArgs -m venv $VenvDir
}

$venvPython = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "Virtual environment python was not found at $venvPython"
}

& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r requirements.txt

Write-Host ""
Write-Host "Setup complete."
Write-Host "Activate with:"
Write-Host "  .\$VenvDir\Scripts\Activate.ps1"
Write-Host "Train with:"
Write-Host "  python scripts/train_hrl_packing.py"
