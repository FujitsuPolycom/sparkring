param(
    [ValidateRange(1, 1000000)]
    [int]$Iterations = 32,

    [ValidateSet(4, 5)]
    [int]$MtpTokens = 4,

    [ValidateRange(1024, 65534)]
    [int]$ControlPort0 = 10210,

    [ValidateRange(1024, 65534)]
    [int]$ControlPort1 = 10211,

    [ValidatePattern("^[0-9,-]+$")]
    [string]$CpuSet = "10,11,12",

    [ValidateRange(0, 4095)]
    [int]$SubmitCpu = 10,

    [ValidateRange(0, 4095)]
    [int]$TpProgressCpu = 11,

    [ValidateRange(0, 4095)]
    [int]$VocabProgressCpu = 12,

    [ValidateRange(10, 3600)]
    [int]$WatchdogSeconds = 120,

    [string]$Library =
        "/tmp/libspark_transport_capi-vocab-graph-q6-20260726.so",

    [ValidatePattern("^[0-9a-f]{64}$")]
    [string]$ExpectedLibrarySha256 =
        "8657306b1807c4111687f5f128725ff6726be2c8c8c4bebc67b520a730c0acca",

    [string]$Image = "<your-vllm-image>",
    [string[]]$Targets = ($env:SPARKRING_TARGETS -split ",").Trim(),
    [string[]]$RankHosts = ($env:SPARKRING_RANK_HOSTS -split ",").Trim(),

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

if ($ControlPort0 -eq $ControlPort1) {
    throw "ControlPort0 and ControlPort1 must differ"
}
if (@($SubmitCpu, $TpProgressCpu, $VocabProgressCpu |
        Sort-Object -Unique).Count -ne 3) {
    throw "submit, TP progress, and vocabulary progress CPUs must differ"
}

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$probeSource = Join-Path $repoRoot `
    "spark_transport\integrations\vllm\probe_vocab_graph_stream_switch.py"
$adapterSource = Join-Path $repoRoot `
    "spark_transport\integrations\vllm\spark_tp4_vocab_allgather_backend.py"
$queryContractSource = Join-Path $repoRoot `
    "spark_transport\integrations\vllm\spark_tp4_query_contract.py"
if (-not (Test-Path -LiteralPath $probeSource -PathType Leaf)) {
    throw "missing stream-switch probe source: $probeSource"
}
if (-not (Test-Path -LiteralPath $adapterSource -PathType Leaf)) {
    throw "missing vocabulary adapter source: $adapterSource"
}
if (-not (Test-Path -LiteralPath $queryContractSource -PathType Leaf)) {
    throw "missing query-width contract source: $queryContractSource"
}

$probeHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $probeSource).
    Hash.ToLowerInvariant()
$adapterHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $adapterSource).
    Hash.ToLowerInvariant()
$queryContractHash = (
    Get-FileHash -Algorithm SHA256 -LiteralPath $queryContractSource
).Hash.ToLowerInvariant()
$stageId = (
    "$($probeHash.Substring(0, 12))-" +
    "$($adapterHash.Substring(0, 12))-" +
    "$($queryContractHash.Substring(0, 12))"
)
$remoteStage = "/tmp/spark-vocab-stream-switch-$stageId"

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

    $name = "spark-vocab-stream-switch-r$($Node.Rank)"
    $state = (& ssh -o BatchMode=yes -o ConnectTimeout=8 $Node.Target `
        "docker inspect $name --format '{{.State.Status}}:{{.State.ExitCode}}'" `
        2>$null)
    if ($LASTEXITCODE -ne 0) {
        return "missing"
    }
    return $state.Trim()
}

$artifactHashes = @()
foreach ($node in $nodes) {
    $runningModel = (& ssh -o BatchMode=yes -o ConnectTimeout=8 `
        $node.Target `
        "docker ps --filter name=^/glm52-trace$ --format '{{.Names}}'")
    if ($LASTEXITCODE -ne 0) {
        throw "failed to inspect running containers on rank $($node.Rank)"
    }
    if (($runningModel -join "`n").Trim() -eq "glm52-trace") {
        throw "rank $($node.Rank) still runs glm52-trace; the stream-switch probe requires the model-down memory window"
    }

    $mkdirExit = Invoke-NodeSsh -Node $node `
        -Command "mkdir -p '$remoteStage'"
    if ($mkdirExit -ne 0) {
        throw "failed to create stage directory on rank $($node.Rank)"
    }
    & scp -q -o BatchMode=yes -o ConnectTimeout=8 `
        $probeSource "$($node.Target):$remoteStage/probe.py"
    if ($LASTEXITCODE -ne 0) {
        throw "failed to stage probe on rank $($node.Rank)"
    }
    & scp -q -o BatchMode=yes -o ConnectTimeout=8 `
        $adapterSource "$($node.Target):$remoteStage/adapter.py"
    if ($LASTEXITCODE -ne 0) {
        throw "failed to stage adapter on rank $($node.Rank)"
    }
    & scp -q -o BatchMode=yes -o ConnectTimeout=8 `
        $queryContractSource `
        "$($node.Target):$remoteStage/spark_tp4_query_contract.py"
    if ($LASTEXITCODE -ne 0) {
        throw "failed to stage query-width contract on rank $($node.Rank)"
    }
    $hashCommand = (
        "test -f '$Library' && sha256sum '$remoteStage/probe.py' " +
        "'$remoteStage/adapter.py' " +
        "'$remoteStage/spark_tp4_query_contract.py' '$Library'"
    )
    $hash = (& ssh -o BatchMode=yes -o ConnectTimeout=8 `
        $node.Target $hashCommand)
    if ($LASTEXITCODE -ne 0) {
        throw "rank $($node.Rank) is missing a staged probe artifact"
    }
    $hashLines = @($hash)
    if ($hashLines.Count -ne 4 `
        -or $hashLines[0] -notmatch "^$probeHash\s" `
        -or $hashLines[1] -notmatch "^$adapterHash\s" `
        -or $hashLines[2] -notmatch "^$queryContractHash\s" `
        -or $hashLines[3] -notmatch "^$ExpectedLibrarySha256\s") {
        throw "rank $($node.Rank) staged source hash mismatch"
    }
    $artifactHashes += ,(($hashLines -join "`n").Trim())
}

if (@($artifactHashes | Sort-Object -Unique).Count -ne 1) {
    throw "stream-switch probe artifact SHA-256 values differ across ranks"
}
Write-Output "preflight=pass model_down=true identical_sha256=true"
Write-Output $artifactHashes[0]

$failed = $false
$timedOut = $false
try {
    foreach ($node in $nodes) {
        $name = "spark-vocab-stream-switch-r$($node.Rank)"
        $command = @(
            "docker rm -f $name >/dev/null 2>&1 || true;"
            "docker run -d --name $name"
            "--privileged --gpus all --network host --ipc host"
            "--cpuset-cpus=$CpuSet"
            "--ulimit memlock=-1"
            "-v ${remoteStage}/probe.py:/probe/probe.py:ro"
            "-v ${remoteStage}/adapter.py:/probe/spark_tp4_vocab_allgather_backend.py:ro"
            "-v ${remoteStage}/spark_tp4_query_contract.py:/probe/spark_tp4_query_contract.py:ro"
            "-v ${Library}:/opt/spark/lib/libspark_transport_capi.so:ro"
            "-e PYTHONPATH=/probe"
            "-e SPARK_TP4_LIBRARY=/opt/spark/lib/libspark_transport_capi.so"
            "-e VLLM_SPARK_TP4_VOCAB_MODE=custom"
            "-e VLLM_SPARK_TP4_GRAPH_Q1=1"
            "-e SPARK_TP4_PEER0=$($node.Peer0)"
            "-e SPARK_TP4_PEER1=$($node.Peer1)"
            "-e SPARK_TP4_DEVICE0=rocep1s0f0"
            "-e SPARK_TP4_DEVICE1=rocep1s0f1"
            "-e SPARK_TP4_GID0=3"
            "-e SPARK_TP4_GID1=3"
            "-e SPARK_TP4_GRAPH_VOCAB_CONTROL_PORT0=$ControlPort0"
            "-e SPARK_TP4_GRAPH_VOCAB_CONTROL_PORT1=$ControlPort1"
            "-e SPARK_TP4_GRAPH_VOCAB_PROGRESS_CPU=$VocabProgressCpu"
            $Image
            "timeout --signal=TERM --kill-after=5s ${WatchdogSeconds}s"
            "taskset -c $SubmitCpu python3 /probe/probe.py"
            "--rank $($node.Rank)"
            "--iterations $Iterations"
            "--mtp-tokens $MtpTokens"
            "--submit-cpu $SubmitCpu"
            "--tp-progress-cpu $TpProgressCpu"
            ">/dev/null"
        ) -join " "

        $exitCode = Invoke-NodeSsh -Node $node -Command $command
        if ($exitCode -ne 0) {
            throw "failed to launch stream-switch rank $($node.Rank)"
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

    $expectedNodes = [long]$MtpTokens + 1L
    $expectedSequence = $expectedNodes * [long]$Iterations
    $expectedWarmups = 0L
    $expectedPattern = (
        @($expectedNodes) + (@(1L) * $MtpTokens)
    ) -join ","
    foreach ($node in $nodes) {
        $name = "spark-vocab-stream-switch-r$($node.Rank)"
        $state = Get-ContainerState -Node $node
        $log = (& ssh -o BatchMode=yes -o ConnectTimeout=8 `
            $node.Target "docker logs $name 2>&1")
        $result = @($log | Where-Object {
            $_ -like "TP4_VOCAB_STREAM_SWITCH*"
        })
        Write-Output "rank=$($node.Rank) state=$state"
        $result | Write-Output

        $gate = $result -join " "
        if ($state -ne "exited:0" `
            -or $gate -notmatch "pattern=$expectedPattern(?:\s|$)" `
            -or $gate -notmatch "stock_warmups=$expectedWarmups(?:\s|$)" `
            -or $gate -notmatch "captured_nodes=$expectedNodes(?:\s|$)" `
            -or $gate -notmatch "published=$expectedSequence(?:\s|$)" `
            -or $gate -notmatch "consumed=$expectedSequence(?:\s|$)" `
            -or $gate -notmatch "completed=$expectedSequence(?:\s|$)" `
            -or $gate -notmatch "overflow=0(?:\s|$)" `
            -or $gate -notmatch "warmup_mismatches=0(?:\s|$)" `
            -or $gate -notmatch "post_capture_mismatches=0(?:\s|$)" `
            -or $gate -notmatch "mismatches=0(?:\s|$)" `
            -or $gate -notmatch "eager_native_created=true(?:\s|$)" `
            -or $gate -notmatch "eager_session_reused=true(?:\s|$)" `
            -or $gate -notmatch "passed=true(?:\s|$)") {
            $failed = $true
            Write-Output "rank=$($node.Rank) failure_log:"
            $log | Select-Object -Last 80 | Write-Output
        }
    }
}
finally {
    if (-not $KeepContainers) {
        foreach ($node in $nodes) {
            $name = "spark-vocab-stream-switch-r$($node.Rank)"
            Invoke-NodeSsh -Node $node `
                -Command "docker rm -f $name >/dev/null 2>&1 || true" |
                Out-Null
        }
    }
}

if ($timedOut) {
    throw "TP4 vocabulary stream-switch probe exceeded the watchdog"
}
if ($failed) {
    throw "one or more TP4 vocabulary stream-switch ranks failed"
}

Write-Output (
    "gate=pass ranks=4 pattern=$expectedPattern " +
    "expected_sequence=$expectedSequence stream_switch=A-to-B-to-C"
)
