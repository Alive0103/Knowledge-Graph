param(
    [string]$PythonExe = "D:\software\anaconda\python.exe",
    [string]$WorkDir = "",
    [string]$ScriptPath = "",
    [string]$Mode = ""
)

$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not $WorkDir) {
    $WorkDir = Join-Path $repoRoot "work_wyy"
}
if (-not $ScriptPath) {
    $ScriptPath = Join-Path $WorkDir "search_vllm.py"
}

if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

if (-not (Test-Path $ScriptPath)) {
    throw "search_vllm.py not found: $ScriptPath"
}

if (-not $env:KG_ZHIPU_API_KEY -and -not $env:ZHIPUAI_API_KEY) {
    throw "Missing KG_ZHIPU_API_KEY or ZHIPUAI_API_KEY."
}

Set-Location $WorkDir

$arguments = @($ScriptPath)
if ($Mode) {
    $arguments += $Mode
}

& $PythonExe @arguments
