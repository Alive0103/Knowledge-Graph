param(
    [string]$PythonExe = "D:\software\anaconda\python.exe",
    [string]$WorkDir = "",
    [string]$LauncherPath = "",
    [string]$StateFile = ""
)

$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not $WorkDir) {
    $WorkDir = Join-Path $repoRoot "work_wyy"
}
if (-not $LauncherPath) {
    $LauncherPath = Join-Path $PSScriptRoot "start_work_wyy_eval.ps1"
}
if (-not $StateFile) {
    $StateFile = Join-Path $PSScriptRoot "output\work_wyy_remaining_recovery_state.json"
}
$logFile = Join-Path $PSScriptRoot "output\work_wyy_remaining_recovery.log"

if (-not (Test-Path $LauncherPath)) {
    throw "Launcher script not found: $LauncherPath"
}
if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}
if (-not $env:KG_ZHIPU_API_KEY -and -not $env:ZHIPUAI_API_KEY) {
    throw "Missing KG_ZHIPU_API_KEY or ZHIPUAI_API_KEY."
}

$workWyyStateFile = $env:KG_EVAL_STATE_FILE
$stateDir = Split-Path -Parent $StateFile
if ($stateDir -and -not (Test-Path $stateDir)) {
    New-Item -ItemType Directory -Path $stateDir -Force | Out-Null
}

$modes = @(
    @{ name = "llm_only"; flag = "--llm-only" },
    @{ name = "vector_with_llm_always"; flag = "--vector-llm-always" },
    @{ name = "vector_with_llm"; flag = "--vector-llm" }
)

$state = [ordered]@{
    started_at = (Get-Date).ToString("o")
    updated_at = (Get-Date).ToString("o")
    status = "running"
    current_mode = $null
    completed_modes = @()
    skipped_modes = @()
    failed_mode = $null
    launcher_path = $LauncherPath
    state_file = $StateFile
    work_wyy_state_file = $workWyyStateFile
    log_file = $logFile
}

function Write-State {
    param($CurrentState)
    $CurrentState.updated_at = (Get-Date).ToString("o")
    $json = $CurrentState | ConvertTo-Json -Depth 6
    Set-Content -LiteralPath $StateFile -Value $json -Encoding utf8
}

function Write-Log {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date).ToString("o"), $Message
    Add-Content -LiteralPath $logFile -Value $line -Encoding utf8
}

function Get-ExistingModeState {
    param([string]$ModeName)

    if (-not $workWyyStateFile -or -not (Test-Path $workWyyStateFile)) {
        return $null
    }

    try {
        $payload = Get-Content -LiteralPath $workWyyStateFile -Raw -Encoding utf8 | ConvertFrom-Json
    }
    catch {
        return $null
    }

    if (-not $payload.modes) {
        return $null
    }

    return $payload.modes.$ModeName
}

Write-State -CurrentState $state
Write-Log "wrapper started"

foreach ($mode in $modes) {
    $existingModeState = Get-ExistingModeState -ModeName $mode.name
    if (
        $existingModeState -and
        $existingModeState.status -eq "completed" -and
        $existingModeState.report_file -and
        (Test-Path $existingModeState.report_file)
    ) {
        $completed = @($state.completed_modes)
        if ($completed -notcontains $mode.name) {
            $completed += $mode.name
        }
        $state.completed_modes = $completed

        $skipped = @($state.skipped_modes)
        if ($skipped -notcontains $mode.name) {
            $skipped += $mode.name
        }
        $state.skipped_modes = $skipped
        Write-State -CurrentState $state
        Write-Log "skip completed mode: $($mode.name)"
        continue
    }

    $state.current_mode = $mode.name
    $state.status = "running"
    Write-State -CurrentState $state
    Write-Log "start mode: $($mode.name)"

    & $LauncherPath -PythonExe $PythonExe -WorkDir $WorkDir -Mode $mode.flag
    Write-Log "mode exited: $($mode.name), exit_code=$LASTEXITCODE"
    if ($LASTEXITCODE -ne 0) {
        $state.status = "failed"
        $state.failed_mode = $mode.name
        Write-State -CurrentState $state
        Write-Log "wrapper failed on mode: $($mode.name)"
        exit $LASTEXITCODE
    }

    $completed = @($state.completed_modes)
    if ($completed -notcontains $mode.name) {
        $completed += $mode.name
    }
    $state.completed_modes = $completed
    Write-State -CurrentState $state
    Write-Log "mode completed: $($mode.name)"
}

$state.status = "completed"
$state.current_mode = $null
$state.failed_mode = $null
$state.finished_at = (Get-Date).ToString("o")
Write-State -CurrentState $state
Write-Log "wrapper completed"
