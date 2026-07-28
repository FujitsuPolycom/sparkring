param(
    [ValidateRange(43, 1000000)]
    [int]$Cycles = 100,

    [ValidateRange(1024, 65534)]
    [int]$ControlPort0 = 9462,

    [ValidateRange(1024, 65534)]
    [int]$ControlPort1 = 9463,

    [ValidatePattern("^[0-9,-]+$")]
    [string]$CpuSet = "10,11,12,13,14",

    [ValidateRange(0, 4095)]
    [int]$SubmitCpu = 10,

    [ValidateRange(0, 4095)]
    [int]$ProgressCpu = 14,

    [ValidateRange(15, 3600)]
    [int]$WatchdogSeconds = 120,

    [string]$ProbeBinary =
        "/tmp/spark_tp4_indexer_graph_probe-20260727",
    [string]$Image = "<your-vllm-image>",
    [string[]]$Targets = ($env:SPARKRING_TARGETS -split ",").Trim(),
    [string[]]$RankHosts = ($env:SPARKRING_RANK_HOSTS -split ",").Trim(),

    [switch]$DestructiveMismatchQ,
    [switch]$ConfirmDestructiveMismatchQ,
    [switch]$KeepContainers
)

$ErrorActionPreference = "Stop"

if (@($Targets | Where-Object { $_ }).Count -ne 4) {
    throw ("SPARKRING_TARGETS (or -Targets) must be a comma-separated " +
        "list of 4 SSH targets (user@host) in rank order, e.g. " +
        "'user@spark0,user@spark1,user@spark2,user@spark3'")
}
if (@($RankHosts | Where-Object { $_ }).Count -ne 4) {
    throw ("SPARKRING_RANK_HOSTS (or -RankHosts) must be a " +
        "comma-separated list of 4 rank host IPs in rank order, e.g. " +
        "'192.0.2.1,192.0.2.2,192.0.2.3,192.0.2.4'")
}
if ($Image -eq "<your-vllm-image>") {
    throw "set -Image to your vLLM container image tag"
}

$ringCapacity = 64L
$requiredRingWraps = 2L
$qPattern = "1,23,40"
$expectedCapturedQMask = 549760008193L

if ($ControlPort0 -eq $ControlPort1) {
    throw "ControlPort0 and ControlPort1 must differ"
}
if ($SubmitCpu -eq $ProgressCpu) {
    throw "SubmitCpu and ProgressCpu must differ"
}
if ($DestructiveMismatchQ -ne $ConfirmDestructiveMismatchQ) {
    throw "destructive mismatch-Q mode requires both -DestructiveMismatchQ and -ConfirmDestructiveMismatchQ"
}

function Expand-CpuSet {
    param(
        [Parameter(Mandatory)]
        [string]$Value
    )

    $expanded = @()
    foreach ($part in $Value.Split(",")) {
        if ($part -match "^([0-9]+)-([0-9]+)$") {
            $first = [int]$Matches[1]
            $last = [int]$Matches[2]
            if ($last -lt $first) {
                throw "CpuSet range is reversed: $part"
            }
            $expanded += $first..$last
        }
        elseif ($part -match "^[0-9]+$") {
            $expanded += [int]$part
        }
        else {
            throw "CpuSet contains an invalid component: $part"
        }
    }
    return @($expanded | Sort-Object -Unique)
}

$expandedCpuSet = @(Expand-CpuSet -Value $CpuSet)
if ($SubmitCpu -notin $expandedCpuSet) {
    throw "CpuSet must include SubmitCpu $SubmitCpu"
}
if ($ProgressCpu -notin $expandedCpuSet) {
    throw "CpuSet must include ProgressCpu $ProgressCpu"
}

$expectedLaunches = [long]$Cycles * 3L
$expectedRingWraps = [Math]::Floor($expectedLaunches / $ringCapacity)
$expectedValidatedBytes = [long]$Cycles * 4194304L
if (-not $DestructiveMismatchQ -and
    $expectedRingWraps -lt $requiredRingWraps) {
    throw "normal probe must execute through at least two command-ring wraps"
}

$nodes = @(
    [pscustomobject]@{
        Rank = 0
        Target = $Targets[0]
        Peer0 = $RankHosts[1]
        Peer1 = $RankHosts[3]
    },
    [pscustomobject]@{
        Rank = 1
        Target = $Targets[1]
        Peer0 = $RankHosts[0]
        Peer1 = $RankHosts[2]
    },
    [pscustomobject]@{
        Rank = 2
        Target = $Targets[2]
        Peer0 = $RankHosts[3]
        Peer1 = $RankHosts[1]
    },
    [pscustomobject]@{
        Rank = 3
        Target = $Targets[3]
        Peer0 = $RankHosts[2]
        Peer1 = $RankHosts[0]
    }
)

function Get-ProbeContainerName {
    param(
        [Parameter(Mandatory)]
        [int]$Rank
    )

    if ($DestructiveMismatchQ) {
        return "spark-tp4-indexer-graph-mismatch-r$Rank"
    }
    return "spark-tp4-indexer-graph-r$Rank"
}

function Invoke-NodeSsh {
    param(
        [Parameter(Mandatory)]
        [pscustomobject]$Node,

        [Parameter(Mandatory)]
        [string]$Command
    )

    & ssh -o BatchMode=yes -o ConnectTimeout=8 $Node.Target $Command
    return $LASTEXITCODE
}

function Get-ContainerState {
    param(
        [Parameter(Mandatory)]
        [pscustomobject]$Node
    )

    $name = Get-ProbeContainerName -Rank $Node.Rank
    $state = (& ssh -o BatchMode=yes -o ConnectTimeout=8 $Node.Target `
        "docker inspect $name --format '{{.State.Status}}:{{.State.ExitCode}}'" 2>$null)
    if ($LASTEXITCODE -ne 0) {
        return "missing"
    }
    return $state.Trim()
}

$hashes = @()
foreach ($node in $nodes) {
    $runningModel = (& ssh -o BatchMode=yes -o ConnectTimeout=8 `
        $node.Target `
        "docker ps --filter name=^/glm52-trace$ --format '{{.Names}}'")
    if ($LASTEXITCODE -ne 0) {
        throw "failed to inspect running containers on rank $($node.Rank)"
    }
    if (($runningModel -join "`n").Trim() -eq "glm52-trace") {
        throw "rank $($node.Rank) still runs glm52-trace; the model-free indexer graph probe requires a model-down window"
    }

    $hash = (& ssh -o BatchMode=yes -o ConnectTimeout=8 `
        $node.Target "test -x '$ProbeBinary' && sha256sum '$ProbeBinary'")
    if ($LASTEXITCODE -ne 0) {
        throw "rank $($node.Rank) is missing the staged indexer graph probe"
    }
    $hashes += ,(($hash -join "`n").Trim())
}
if (@($hashes | Sort-Object -Unique).Count -ne 1) {
    throw "indexer graph probe SHA-256 values differ across ranks"
}
Write-Output "preflight=pass model_down=true identical_sha256=true destructive_mismatch_q=$($DestructiveMismatchQ.ToString().ToLowerInvariant())"
Write-Output $hashes[0]

$failed = $false
$timedOut = $false
$destructiveTimedOut = $false
$destructiveArgs = @()
if ($DestructiveMismatchQ) {
    $destructiveArgs = @(
        "--destructive-mismatch-q"
        "--i-understand-mismatch-may-abort"
    )
}

try {
    foreach ($node in $nodes) {
        $name = Get-ProbeContainerName -Rank $node.Rank
        $command = @(
            "docker rm -f $name >/dev/null 2>&1 || true;"
            "docker run -d --name $name"
            "--privileged --gpus all --network host --ipc host"
            "--cpuset-cpus=$CpuSet"
            "--ulimit memlock=-1"
            "-v ${ProbeBinary}:/probe:ro"
            $Image
            "timeout --signal=TERM --kill-after=5s ${WatchdogSeconds}s"
            "env -u SPARK_TRANSPORT_TRACE"
            "taskset -c $SubmitCpu /probe"
            "--rank $($node.Rank)"
            "--peer0 $($node.Peer0)"
            "--peer1 $($node.Peer1)"
            "--device0 rocep1s0f0 --device1 rocep1s0f1"
            "--gid0 3 --gid1 3"
            "--control-port0 $ControlPort0"
            "--control-port1 $ControlPort1"
            "--submit-cpu $SubmitCpu"
            "--progress-cpu $ProgressCpu"
            "--cycles $Cycles"
            $destructiveArgs
            ">/dev/null"
        ) -join " "

        $exitCode = Invoke-NodeSsh -Node $node -Command $command
        if ($exitCode -ne 0) {
            throw "failed to launch indexer graph probe rank $($node.Rank)"
        }
    }

    $deadline = [DateTime]::UtcNow.AddSeconds($WatchdogSeconds + 15)
    do {
        $states = @($nodes | ForEach-Object {
            Get-ContainerState -Node $_
        })
        $running = @($states | Where-Object {
            $_ -like "running:*"
        }).Count
        if ($running -eq 0) {
            break
        }
        Start-Sleep -Milliseconds 250
    } while ([DateTime]::UtcNow -lt $deadline)

    if ($running -ne 0) {
        $timedOut = $true
        $failed = $true
    }

    foreach ($node in $nodes) {
        $name = Get-ProbeContainerName -Rank $node.Rank
        $state = Get-ContainerState -Node $node
        $log = (& ssh -o BatchMode=yes -o ConnectTimeout=8 `
            $node.Target "docker logs $name 2>&1")
        Write-Output "rank=$($node.Rank) state=$state"

        if ($DestructiveMismatchQ) {
            $result = @($log | Where-Object {
                $_ -like "TP4_INDEXER_GRAPH_MISMATCH*"
            })
            $result | Write-Output
            $gate = $log -join " "
            $expectedLocalQ = if (($node.Rank % 2) -eq 0) { 1 } else { 23 }
            if ($state -eq "exited:124" -or $state -eq "exited:137") {
                $destructiveTimedOut = $true
                $failed = $true
            }
            if ($result.Count -ne 1 `
                -or $state -ne "exited:134" `
                -or $gate -notmatch "TP4_INDEXER_GRAPH_MISMATCH rank=$($node.Rank) armed=true destructive=true confirmation=true local_q=$expectedLocalQ expected_outcome=bounded_transport_abort" `
                -or $gate -notmatch "FATAL asynchronous TP4 indexer graph failed:" `
                -or $gate -match "TP4_INDEXER_GRAPH_MISMATCH_UNEXPECTED" `
                -or $gate -match "passed=true(?:\s|$)") {
                $failed = $true
                Write-Output "rank=$($node.Rank) destructive_failure_log:"
                $log | Select-Object -Last 60 | Write-Output
            }
            continue
        }

        $result = @($log | Where-Object {
            $_ -like "TP4_INDEXER_GRAPH rank=*"
        })
        $result | Write-Output
        $gate = $result -join " "
        if ($result.Count -ne 1 `
            -or $state -ne "exited:0" `
            -or $gate -notmatch "rank=$($node.Rank)(?:\s|$)" `
            -or $gate -notmatch "mode=normal(?:\s|$)" `
            -or $gate -notmatch "publisher=device(?:\s|$)" `
            -or $gate -notmatch "q_pattern=$qPattern(?:\s|$)" `
            -or $gate -notmatch "cycles=$Cycles(?:\s|$)" `
            -or $gate -notmatch "ring_capacity=$ringCapacity(?:\s|$)" `
            -or $gate -notmatch "required_ring_wraps=$requiredRingWraps(?:\s|$)" `
            -or $gate -notmatch "ring_wraps=$expectedRingWraps(?:\s|$)" `
            -or $gate -notmatch "graph_launches=$expectedLaunches(?:\s|$)" `
            -or $gate -notmatch "captured_nodes=3(?:\s|$)" `
            -or $gate -notmatch "captured_q_mask=$expectedCapturedQMask(?:\s|$)" `
            -or $gate -notmatch "census_q1=1(?:\s|$)" `
            -or $gate -notmatch "census_q23=1(?:\s|$)" `
            -or $gate -notmatch "census_q40=1(?:\s|$)" `
            -or $gate -notmatch "capture_configured=true(?:\s|$)" `
            -or $gate -notmatch "polling_enabled=true(?:\s|$)" `
            -or $gate -notmatch "host_native_atomics_supported=true(?:\s|$)" `
            -or $gate -notmatch "submit_affinity_verified=true(?:\s|$)" `
            -or $gate -notmatch "progress_affinity_verified=true(?:\s|$)" `
            -or $gate -notmatch "submit_cpu=$SubmitCpu(?:\s|$)" `
            -or $gate -notmatch "progress_cpu=$ProgressCpu(?:\s|$)" `
            -or $gate -notmatch "published=$expectedLaunches(?:\s|$)" `
            -or $gate -notmatch "consumed=$expectedLaunches(?:\s|$)" `
            -or $gate -notmatch "completed=$expectedLaunches(?:\s|$)" `
            -or $gate -notmatch "overflow=0(?:\s|$)" `
            -or $gate -notmatch "validated_bytes=$expectedValidatedBytes(?:\s|$)" `
            -or $gate -notmatch "mismatched_int32=0(?:\s|$)" `
            -or $gate -notmatch "byte_exact=true(?:\s|$)" `
            -or $gate -notmatch "monotonic_sequences=true(?:\s|$)" `
            -or $gate -notmatch "passed=true(?:\s|$)") {
            $failed = $true
            Write-Output "rank=$($node.Rank) failure_log:"
            $log | Select-Object -Last 60 | Write-Output
        }
    }
}
finally {
    if (-not $KeepContainers) {
        foreach ($node in $nodes) {
            $name = Get-ProbeContainerName -Rank $node.Rank
            Invoke-NodeSsh -Node $node `
                -Command "docker rm -f $name >/dev/null 2>&1 || true" |
                Out-Null
        }
    }
}

if ($timedOut) {
    throw "TP4 indexer graph probe exceeded the outer watchdog"
}
if ($destructiveTimedOut) {
    throw "destructive mismatch-Q transport hung until the watchdog instead of aborting within its bounded protocol timeout"
}
if ($failed) {
    throw "one or more TP4 indexer graph ranks failed"
}

if ($DestructiveMismatchQ) {
    Write-Output "gate=pass ranks=4 destructive_mismatch_q=true bounded_transport_abort=true"
}
else {
    Write-Output "gate=pass ranks=4 q_pattern=$qPattern graph_launches=$expectedLaunches ring_wraps=$expectedRingWraps byte_exact=true"
}
