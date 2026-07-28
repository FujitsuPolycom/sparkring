param(
    [ValidateRange(1, 4096)]
    [int]$PrefixQueries = 66,

    [ValidateSet("decode-buckets", "q40")]
    [string]$PrefixPattern = "decode-buckets",

    [ValidateSet("shared", "switch-at-67", "split-capture")]
    [string]$StreamMode = "shared",

    [ValidateSet("none", "eager-stream", "device")]
    [string]$Pre67Sync = "none",

    [ValidateRange(0, 600000)]
    [int]$Sequence67DelayMs = 0,

    [ValidateRange(1, 10000)]
    [int]$GraphReplays = 1,

    [ValidateRange(1, 4096)]
    [int]$MaxInflight = 64,

    [ValidateRange(5, 3600)]
    [int]$EagerReadyTimeoutSeconds = 15,

    [ValidateRange(5, 3600)]
    [int]$ProtocolTimeoutSeconds = 15,

    [ValidateRange(1024, 65532)]
    [int]$EagerPort0 = 9910,

    [ValidateRange(1024, 65533)]
    [int]$EagerPort1 = 9911,

    [ValidateRange(1024, 65534)]
    [int]$GraphPort0 = 9912,

    [ValidateRange(1024, 65535)]
    [int]$GraphPort1 = 9913,

    [ValidatePattern("^[0-9,-]+$")]
    [string]$CpuSet = "10,13",

    [ValidateRange(0, 4095)]
    [int]$SubmitCpu = 10,

    [ValidateRange(0, 4095)]
    [int]$ProgressCpu = 13,

    [ValidateSet("adaptive-yield", "dedicated-spin")]
    [string]$PollPolicy = "adaptive-yield",

    [ValidateRange(20, 3600)]
    [int]$WatchdogSeconds = 60,

    [string]$ProbeBinary =
        "/tmp/spark_tp4_dcp_sequence67_probe",
    [string]$Library =
        "/tmp/libspark_transport_capi.so",
    # Replace <your-vllm-image> with your locally built vLLM container tag.
    [string]$Image = "<your-vllm-image>",
    # Rank SSH targets default from SPARKRING_TARGETS, a comma-separated
    # user@host list in rank order 0-3; -RankNTarget overrides per rank.
    [string]$Rank0Target = ($env:SPARKRING_TARGETS -split ",")[0],
    [string]$Rank0ProxyJump = "",
    [string]$Rank1Target = ($env:SPARKRING_TARGETS -split ",")[1],
    [string]$Rank2Target = ($env:SPARKRING_TARGETS -split ",")[2],
    [string]$Rank3Target = ($env:SPARKRING_TARGETS -split ",")[3],
    [string]$Rank3ProxyJump = "",
    [switch]$PhaseTrace,
    [switch]$KeepContainers
)

$ErrorActionPreference = "Stop"

$ports = @($EagerPort0, $EagerPort1, $GraphPort0, $GraphPort1)
if (@($ports | Sort-Object -Unique).Count -ne 4) {
    throw "all eager and graph control ports must be distinct"
}
if ($SubmitCpu -eq $ProgressCpu) {
    throw "SubmitCpu and ProgressCpu must differ"
}
if ($WatchdogSeconds -le $EagerReadyTimeoutSeconds) {
    throw "WatchdogSeconds must exceed EagerReadyTimeoutSeconds"
}
if (@($Rank0Target, $Rank1Target, $Rank2Target, $Rank3Target) |
        Where-Object { [string]::IsNullOrWhiteSpace($_) }) {
    throw ("missing rank SSH targets: set SPARKRING_TARGETS to a " +
        "comma-separated user@host list for ranks 0-3, or pass " +
        "-Rank0Target..-Rank3Target explicitly")
}

# Peer0/Peer1 are RFC 5737 placeholder addresses for the point-to-point
# fabric links between ring neighbours; replace them with your cluster's
# per-link fabric IPs.
$nodes = @(
    [pscustomobject]@{
        Rank = 0
        Target = $Rank0Target
        ProxyJump = $Rank0ProxyJump
        Peer0 = "192.0.2.11"
        Peer1 = "192.0.2.41"
    },
    [pscustomobject]@{
        Rank = 1
        Target = $Rank1Target
        ProxyJump = ""
        Peer0 = "192.0.2.10"
        Peer1 = "192.0.2.21"
    },
    [pscustomobject]@{
        Rank = 2
        Target = $Rank2Target
        ProxyJump = ""
        Peer0 = "192.0.2.30"
        Peer1 = "192.0.2.20"
    },
    [pscustomobject]@{
        Rank = 3
        Target = $Rank3Target
        ProxyJump = $Rank3ProxyJump
        Peer0 = "192.0.2.31"
        Peer1 = "192.0.2.40"
    }
)

function Get-NodeSshArguments {
    param(
        [Parameter(Mandatory)]
        [pscustomobject]$Node
    )

    $arguments = @(
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=8",
        "-o", "StrictHostKeyChecking=accept-new"
    )
    if (-not [string]::IsNullOrWhiteSpace($Node.ProxyJump)) {
        $arguments += @("-J", $Node.ProxyJump)
    }
    $arguments += $Node.Target
    return $arguments
}

function Invoke-NodeSsh {
    param(
        [Parameter(Mandatory)]
        [pscustomobject]$Node,

        [Parameter(Mandatory)]
        [string]$Command
    )

    $arguments = @(Get-NodeSshArguments -Node $Node)
    & ssh @arguments $Command
    return $LASTEXITCODE
}

function Get-ContainerState {
    param(
        [Parameter(Mandatory)]
        [pscustomobject]$Node
    )

    $name = "spark-tp4-dcp-sequence67-r$($Node.Rank)"
    $arguments = @(Get-NodeSshArguments -Node $Node)
    $state = (& ssh @arguments `
        "docker inspect $name --format '{{.State.Status}}:{{.State.ExitCode}}'" `
        2>$null)
    if ($LASTEXITCODE -ne 0) {
        return "missing"
    }
    return $state.Trim()
}

$artifactHashes = @()
foreach ($node in $nodes) {
    $sshArguments = @(Get-NodeSshArguments -Node $node)
    $runningModel = (& ssh @sshArguments `
        "docker ps --filter name=^/glm52-trace$ --format '{{.Names}}'")
    if ($LASTEXITCODE -ne 0) {
        throw "failed to inspect running containers on rank $($node.Rank)"
    }
    if (($runningModel -join "`n").Trim() -eq "glm52-trace") {
        throw (
            "rank $($node.Rank) still runs glm52-trace; the sequence-67 " +
            "probe requires a model-down window"
        )
    }

    $hash = (& ssh @sshArguments `
        "test -x '$ProbeBinary' && test -f '$Library' && sha256sum '$ProbeBinary' '$Library'")
    if ($LASTEXITCODE -ne 0) {
        throw "rank $($node.Rank) is missing a staged sequence-67 artifact"
    }
    $artifactHashes += ,(($hash -join "`n").Trim())
}
if (@($artifactHashes | Sort-Object -Unique).Count -ne 1) {
    throw "sequence-67 artifact SHA-256 values differ across ranks"
}
Write-Output "preflight=pass model_down=true identical_sha256=true"
Write-Output $artifactHashes[0]

$failed = $false
$timedOut = $false
$knownReadinessTimeout = $false
$traceEnvironment = if ($PhaseTrace) {
    "SPARK_TRANSPORT_TRACE=1"
}
else {
    "-u SPARK_TRANSPORT_TRACE"
}

try {
    foreach ($node in $nodes) {
        $name = "spark-tp4-dcp-sequence67-r$($node.Rank)"
        $command = @(
            "docker rm -f $name >/dev/null 2>&1 || true;"
            "docker run -d --name $name"
            "--privileged --gpus all --network host --ipc host"
            "--cpuset-cpus=$CpuSet"
            "--ulimit memlock=-1"
            "-v ${ProbeBinary}:/probe:ro"
            "-v ${Library}:/opt/spark/lib/libspark_transport_capi.so:ro"
            "-e LD_LIBRARY_PATH=/opt/spark/lib"
            $Image
            "timeout --signal=TERM --kill-after=5s ${WatchdogSeconds}s"
            "env $traceEnvironment"
            "SPARK_TP4_MAX_INFLIGHT=$MaxInflight"
            "SPARK_TP4_DCP_EAGER_READY_TIMEOUT_SECONDS=$EagerReadyTimeoutSeconds"
            "SPARK_TP4_EAGER_PROTOCOL_TIMEOUT_SECONDS=$ProtocolTimeoutSeconds"
            "SPARK_TP4_DCP_GRAPH_POLL_POLICY=$PollPolicy"
            "taskset -c $SubmitCpu /probe"
            "--rank $($node.Rank)"
            "--peer0 $($node.Peer0)"
            "--peer1 $($node.Peer1)"
            "--device0 rocep1s0f0 --device1 rocep1s0f1"
            "--gid0 3 --gid1 3"
            "--eager-port0 $EagerPort0"
            "--eager-port1 $EagerPort1"
            "--graph-port0 $GraphPort0"
            "--graph-port1 $GraphPort1"
            "--submit-cpu $SubmitCpu"
            "--progress-cpu $ProgressCpu"
            "--prefix-queries $PrefixQueries"
            "--prefix-pattern $PrefixPattern"
            "--stream-mode $StreamMode"
            "--pre67-sync $Pre67Sync"
            "--sequence67-delay-ms $Sequence67DelayMs"
            "--graph-replays $GraphReplays"
            ">/dev/null"
        ) -join " "

        $exitCode = Invoke-NodeSsh -Node $node -Command $command
        if ($exitCode -ne 0) {
            throw "failed to launch sequence-67 rank $($node.Rank)"
        }
    }

    $deadline = [DateTime]::UtcNow.AddSeconds($WatchdogSeconds + 15)
    do {
        $states = @($nodes | ForEach-Object {
            Get-ContainerState -Node $_
        })
        $running = @($states | Where-Object { $_ -like "running:*" }).Count
        if ($running -eq 0) {
            break
        }
        Start-Sleep -Milliseconds 250
    } while ([DateTime]::UtcNow -lt $deadline)

    if ($running -ne 0) {
        $timedOut = $true
        $failed = $true
    }

    $transitionSequence = [long]$PrefixQueries + 1L
    $capturedNodes = 34L
    $expectedGraphSequence = $capturedNodes * [long]$GraphReplays
    foreach ($node in $nodes) {
        $name = "spark-tp4-dcp-sequence67-r$($node.Rank)"
        $state = Get-ContainerState -Node $node
        $sshArguments = @(Get-NodeSshArguments -Node $node)
        $log = (& ssh @sshArguments "docker logs $name 2>&1")
        $result = @($log | Where-Object {
            $_ -like "TP4_DCP_SEQUENCE67*"
        })
        $phaseLines = @($log | Where-Object {
            $_ -like "DCP_SEQUENCE67_PHASE*"
        })
        Write-Output "rank=$($node.Rank) state=$state"
        $phaseLines | Write-Output
        $result | Write-Output

        $joinedLog = $log -join "`n"
        if (
            $joinedLog -match (
                "timed out waiting for GPU DCP eager input readiness " +
                "sequence $transitionSequence"
            )
        ) {
            $knownReadinessTimeout = $true
            Write-Output (
                "rank=$($node.Rank) " +
                "failure_class=eager_input_readiness_timeout " +
                "sequence=$transitionSequence"
            )
        }

        $gate = $result -join " "
        if ($state -ne "exited:0" `
            -or $gate -notmatch "prefix_queries=$PrefixQueries(?:\s|$)" `
            -or $gate -notmatch (
                "transition_sequence=$transitionSequence(?:\s|$)"
            ) `
            -or $gate -notmatch "transition_q=40(?:\s|$)" `
            -or $gate -notmatch "prefix_pattern=$PrefixPattern(?:\s|$)" `
            -or $gate -notmatch "stream_mode=$StreamMode(?:\s|$)" `
            -or $gate -notmatch "pre67_sync=$Pre67Sync(?:\s|$)" `
            -or $gate -notmatch "piecewise_graphs=18(?:\s|$)" `
            -or $gate -notmatch "full_graphs=16(?:\s|$)" `
            -or $gate -notmatch (
                "graph_expected_sequence=$expectedGraphSequence(?:\s|$)"
            ) `
            -or $gate -notmatch (
                "graph_published=$expectedGraphSequence(?:\s|$)"
            ) `
            -or $gate -notmatch (
                "graph_consumed=$expectedGraphSequence(?:\s|$)"
            ) `
            -or $gate -notmatch (
                "graph_completed=$expectedGraphSequence(?:\s|$)"
            ) `
            -or $gate -notmatch "graph_overflow=0(?:\s|$)" `
            -or $gate -notmatch "eager_mismatches=0(?:\s|$)" `
            -or $gate -notmatch "graph_mismatches=0(?:\s|$)" `
            -or $gate -notmatch "passed=true(?:\s|$)") {
            $failed = $true
            Write-Output "rank=$($node.Rank) failure_log:"
            $log | Select-Object -Last 120 | Write-Output
        }

        foreach ($requiredPhase in @(
            "eager_prefix_submitted",
            "piecewise_captured",
            "sequence67_submitted",
            "full_captured",
            "graphs_replayed",
            "sequence67_completed"
        )) {
            if (
                -not @($phaseLines | Where-Object {
                    $_ -match "phase=$requiredPhase(?:\s|$)"
                })
            ) {
                $failed = $true
                Write-Output (
                    "rank=$($node.Rank) " +
                    "missing_phase=$requiredPhase"
                )
            }
        }
    }
}
finally {
    if (-not $KeepContainers) {
        foreach ($node in $nodes) {
            $name = "spark-tp4-dcp-sequence67-r$($node.Rank)"
            Invoke-NodeSsh -Node $node `
                -Command "docker rm -f $name >/dev/null 2>&1 || true" |
                Out-Null
        }
    }
}

if ($timedOut) {
    throw "TP4 DCP sequence-67 probe exceeded the watchdog"
}
if ($failed) {
    $classification = if ($knownReadinessTimeout) {
        "eager_input_readiness_timeout"
    }
    else {
        "other"
    }
    throw (
        "one or more TP4 DCP sequence-67 ranks failed; " +
        "failure_class=$classification"
    )
}

Write-Output (
    "gate=pass ranks=4 prefix_queries=$PrefixQueries " +
    "transition_sequence=$transitionSequence transition_q=40 " +
    "stream_mode=$StreamMode pre67_sync=$Pre67Sync " +
    "graph_expected_sequence=$expectedGraphSequence"
)
