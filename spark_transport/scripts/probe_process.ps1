#requires -Version 7.0
# Bound local command execution, including output collection, without a shell.
function Invoke-ProbeProcess {
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [string[]]$ArgumentList = @(),
        [double]$TimeoutSeconds = 30
    )
    if ([double]::IsNaN($TimeoutSeconds) -or [double]::IsInfinity($TimeoutSeconds) `
        -or $TimeoutSeconds -le 0 -or $TimeoutSeconds -gt 86400) {
        throw "Command timeout must be finite and between 0 and 86400 seconds"
    }
    $info = [System.Diagnostics.ProcessStartInfo]::new()
    $info.FileName = $FilePath
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    foreach ($argument in $ArgumentList) { $info.ArgumentList.Add($argument) }
    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo = $info
    $clock = [System.Diagnostics.Stopwatch]::StartNew()
    $budget = [int][Math]::Ceiling($TimeoutSeconds * 1000)
    $timedOut = $false
    try {
        [void]$process.Start()
        $stdout = $process.StandardOutput.ReadToEndAsync()
        $stderr = $process.StandardError.ReadToEndAsync()
        $remaining = [Math]::Max(0, $budget - [int]$clock.ElapsedMilliseconds)
        if (-not $process.WaitForExit($remaining)) {
            $timedOut = $true
            try { $process.Kill($true) } catch { if (-not $process.HasExited) { throw } }
            # Process-tree termination has its own bounded cleanup allowance.
            if (-not $process.WaitForExit(5000)) {
                throw "Timed-out command did not exit within the cleanup allowance"
            }
        }
        foreach ($reader in @($stdout, $stderr)) {
            $remaining = [Math]::Max(0, $budget - [int]$clock.ElapsedMilliseconds)
            if (-not $reader.Wait($remaining)) { $timedOut = $true }
        }
        [pscustomobject]@{
            ExitCode = $(if ($timedOut) { 124 } else { $process.ExitCode })
            TimedOut = $timedOut
            StandardOutput = $(if ($stdout.IsCompletedSuccessfully) { $stdout.Result } else { "" })
            StandardError = $(if ($stderr.IsCompletedSuccessfully) { $stderr.Result } else { "" })
        }
    }
    finally {
        $process.Dispose()
        $clock.Stop()
    }
}

# Runners supply RemoteTimeoutSeconds and an optional command executor.
# An injected executor returns the same fields as Invoke-ProbeProcess.
function Invoke-ProbeRemote {
    param([string]$FilePath, [string[]]$ArgumentList, [double]$TimeoutSeconds,
          [scriptblock]$Executor)
    if ($Executor) {
        $result = & $Executor $FilePath $ArgumentList $TimeoutSeconds
    } else {
        $result = Invoke-ProbeProcess -FilePath $FilePath -ArgumentList $ArgumentList `
            -TimeoutSeconds $TimeoutSeconds
    }
    if ($result.StandardError) {
        Write-Error -Message $result.StandardError.TrimEnd() -ErrorAction Continue
    }
    if ($result.TimedOut) {
        Write-Error -Message "$FilePath exceeded the ${TimeoutSeconds}s operation deadline" -ErrorAction Continue
    }
    if ($result.StandardOutput) {
        $lines = @($result.StandardOutput -split "\r?\n")
        if ($lines[-1] -eq "") { $lines = @($lines | Select-Object -SkipLast 1) }
        $lines | Write-Output
    }
    $global:LASTEXITCODE = [int]$result.ExitCode
}

function Invoke-ProbeSsh {
    Invoke-ProbeRemote -FilePath ssh -ArgumentList @($args) `
        -TimeoutSeconds $RemoteTimeoutSeconds -Executor $RemoteExecutor
}

function Invoke-ProbeScp {
    Invoke-ProbeRemote -FilePath scp -ArgumentList @($args) `
        -TimeoutSeconds $RemoteTimeoutSeconds -Executor $RemoteExecutor
}
