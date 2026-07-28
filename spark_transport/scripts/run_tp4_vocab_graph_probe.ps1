param(
    [ValidateRange(0, 10000000)]
    [int]$Warmup = 2,

    [ValidateRange(1, 10000000)]
    [int]$Iterations = 100,

    [ValidateRange(1024, 65534)]
    [int]$ControlPort0 = 10110,

    [ValidateRange(1024, 65534)]
    [int]$ControlPort1 = 10111,

    [ValidatePattern("^[0-9,-]+$")]
    [string]$CpuSet = "10,12",

    [ValidateRange(0, 4095)]
    [int]$SubmitCpu = 10,

    [ValidateRange(0, 4095)]
    [int]$ProgressCpu = 12,

    [ValidateSet(4, 5)]
    [int]$MtpTokens = 4,

    [ValidateRange(10, 3600)]
    [int]$WatchdogSeconds = 120,

    [string]$ProbeBinary =
        "/tmp/spark_tp4_vocab_graph_q6_probe-20260726",
    [string]$Library =
        "/tmp/libspark_transport_capi-vocab-graph-q6-20260726.so",
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
if ($SubmitCpu -eq $ProgressCpu) {
    throw "SubmitCpu and ProgressCpu must differ"
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

    $name = "spark-tp4-vocab-graph-r$($Node.Rank)"
    $state = (& ssh -o BatchMode=yes -o ConnectTimeout=8 $Node.Target `
        "docker inspect $name --format '{{.State.Status}}:{{.State.ExitCode}}'" 2>$null)
    if ($LASTEXITCODE -ne 0) {
        return "missing"
    }
    return $state.Trim()
}

$artifactHashes = @()
foreach ($node in $nodes) {
    $runningModel = (& ssh -o BatchMode=yes -o ConnectTimeout=8 $node.Target `
        "docker ps --filter name=^/glm52-trace$ --format '{{.Names}}'")
    if ($LASTEXITCODE -ne 0) {
        throw "failed to inspect running containers on rank $($node.Rank)"
    }
    if (($runningModel -join "`n").Trim() -eq "glm52-trace") {
        throw "rank $($node.Rank) still runs glm52-trace; the vocabulary graph probe requires the model-down memory window"
    }

    $hash = (& ssh -o BatchMode=yes -o ConnectTimeout=8 $node.Target `
        "test -x '$ProbeBinary' && test -f '$Library' && sha256sum '$ProbeBinary' '$Library'")
    if ($LASTEXITCODE -ne 0) {
        throw "rank $($node.Rank) is missing a staged vocabulary graph artifact"
    }
    $artifactHashes += ,(($hash -join "`n").Trim())
}
if (@($artifactHashes | Sort-Object -Unique).Count -ne 1) {
    throw "vocabulary graph artifact SHA-256 values differ across ranks"
}
Write-Output "preflight=pass model_down=true identical_sha256=true"
Write-Output $artifactHashes[0]

$failed = $false
$timedOut = $false
try {
    foreach ($node in $nodes) {
        $name = "spark-tp4-vocab-graph-r$($node.Rank)"
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
            "--mtp-tokens $MtpTokens"
            "--warmup $Warmup"
            "--iterations $Iterations"
            ">/dev/null"
        ) -join " "

        $exitCode = Invoke-NodeSsh -Node $node -Command $command
        if ($exitCode -ne 0) {
            throw "failed to launch vocabulary graph rank $($node.Rank)"
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

    $expectedNodes = [long]$MtpTokens + 1L
    $expected = ([long]$Warmup + [long]$Iterations) * $expectedNodes
    $targetQ = [long]$MtpTokens + 1L
    $expectedPattern = (
        @($targetQ) + (@(1L) * $MtpTokens)
    ) -join ","
    foreach ($node in $nodes) {
        $name = "spark-tp4-vocab-graph-r$($node.Rank)"
        $state = Get-ContainerState -Node $node
        $log = (& ssh -o BatchMode=yes -o ConnectTimeout=8 $node.Target `
            "docker logs $name 2>&1")
        $result = @($log | Where-Object { $_ -like "TP4_VOCAB_GRAPH*" })
        Write-Output "rank=$($node.Rank) state=$state"
        $result | Write-Output

        $gate = $result -join " "
        if ($state -ne "exited:0" `
            -or $gate -notmatch "mtp_tokens=$MtpTokens(?:\s|$)" `
            -or $gate -notmatch "pattern=$expectedPattern(?:\s|$)" `
            -or $gate -notmatch "captured_nodes=$expectedNodes(?:\s|$)" `
            -or $gate -notmatch "published=$expected(?:\s|$)" `
            -or $gate -notmatch "consumed=$expected(?:\s|$)" `
            -or $gate -notmatch "completed=$expected(?:\s|$)" `
            -or $gate -notmatch "overflow=0(?:\s|$)" `
            -or $gate -notmatch "submit_cpu=$SubmitCpu(?:\s|$)" `
            -or $gate -notmatch "progress_cpu=$ProgressCpu(?:\s|$)" `
            -or $gate -notmatch "mismatches=0(?:\s|$)" `
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
            $name = "spark-tp4-vocab-graph-r$($node.Rank)"
            Invoke-NodeSsh -Node $node `
                -Command "docker rm -f $name >/dev/null 2>&1 || true" | Out-Null
        }
    }
}

if ($timedOut) {
    throw "TP4 vocabulary graph probe exceeded the watchdog"
}
if ($failed) {
    throw "one or more TP4 vocabulary graph ranks failed"
}

Write-Output "gate=pass ranks=4 mtp_tokens=$MtpTokens pattern=$expectedPattern expected_sequence=$expected"
