param(
    [ValidateRange(0, 10000000)]
    [int]$Warmup = 1000,

    [ValidateRange(1, 10000000)]
    [int]$Iterations = 10000,

    [ValidateRange(2, 1073741824)]
    [int]$Bytes = 12288,

    [ValidateRange(1024, 65534)]
    [int]$ControlPort0 = 9480,

    [ValidateRange(1024, 65534)]
    [int]$ControlPort1 = 9481,

    [ValidateRange(10, 3600)]
    [int]$WatchdogSeconds = 90,

    [ValidateRange(0, 300000)]
    [int]$QueuedDelayMs = 0,

    [ValidateRange(0, 3)]
    [int]$QueuedDelayRank = 1,

    [string]$Binary = "/tmp/spark_tp4_tensor_probe",
    [string]$Image = "<your-vllm-image>",
    [string[]]$Targets = ($env:SPARKRING_TARGETS -split ",").Trim(),
    [string[]]$RankHosts = ($env:SPARKRING_RANK_HOSTS -split ",").Trim(),
    [switch]$AlternateStreams,
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

if (($Bytes % 2) -ne 0) {
    throw "Bytes must be a multiple of two for BF16 elements"
}
if ($ControlPort0 -eq $ControlPort1) {
    throw "ControlPort0 and ControlPort1 must differ"
}
if (($QueuedDelayMs -gt 0) -and
    ((-not $AlternateStreams) -or ($Warmup -lt 1) -or ($Iterations -lt 2))) {
    throw "QueuedDelayMs requires AlternateStreams, Warmup >= 1, and Iterations >= 2"
}

$alternateStreamsArgument = if ($AlternateStreams) {
    "--alternate-streams"
} else {
    ""
}
$queuedDelayArgument = if ($QueuedDelayMs -gt 0) {
    "--queued-delay-ms $QueuedDelayMs --queued-delay-rank $QueuedDelayRank"
} else {
    ""
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

    $name = "spark-tp4-tensor-r$($Node.Rank)"
    $state = (& ssh -o BatchMode=yes -o ConnectTimeout=8 $Node.Target `
        "docker inspect $name --format '{{.State.Status}}:{{.State.ExitCode}}'" 2>$null)
    if ($LASTEXITCODE -ne 0) {
        return "missing"
    }
    return $state.Trim()
}

$failed = $false
$timedOut = $false

try {
    foreach ($node in $nodes) {
        $name = "spark-tp4-tensor-r$($node.Rank)"
        $command = @(
            "docker rm -f $name >/dev/null 2>&1 || true;"
            "docker run -d --name $name"
            "--privileged --gpus all --network host --ipc host"
            "--ulimit memlock=-1"
            "-v ${Binary}:/probe:ro"
            $Image
            "timeout --signal=TERM --kill-after=5s ${WatchdogSeconds}s"
            "taskset -c 10 /probe"
            "--rank $($node.Rank)"
            "--peer0 $($node.Peer0)"
            "--peer1 $($node.Peer1)"
            "--device0 rocep1s0f0 --device1 rocep1s0f1"
            "--gid0 3 --gid1 3"
            "--control-port0 $ControlPort0"
            "--control-port1 $ControlPort1"
            "--bytes $Bytes"
            "--warmup $Warmup"
            "--iterations $Iterations"
            $queuedDelayArgument
            $alternateStreamsArgument
            ">/dev/null"
        ) -join " "

        $exitCode = Invoke-NodeSsh -Node $node -Command $command
        if ($exitCode -ne 0) {
            throw "failed to launch rank $($node.Rank)"
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
        Start-Sleep -Milliseconds 500
    } while ([DateTime]::UtcNow -lt $deadline)

    if ($running -ne 0) {
        $timedOut = $true
        $failed = $true
    }

    for ($index = 0; $index -lt $nodes.Count; ++$index) {
        $node = $nodes[$index]
        $name = "spark-tp4-tensor-r$($node.Rank)"
        $state = Get-ContainerState -Node $node
        Write-Output "rank=$($node.Rank) state=$state"
        & ssh -o BatchMode=yes -o ConnectTimeout=8 $node.Target `
            "docker logs $name 2>&1 | grep '^TP4_TENSOR' || true"
        if ($state -ne "exited:0") {
            $failed = $true
            Write-Output "rank=$($node.Rank) failure_log:"
            & ssh -o BatchMode=yes -o ConnectTimeout=8 $node.Target `
                "docker logs --tail 40 $name 2>&1"
        }
    }
}
finally {
    if (-not $KeepContainers) {
        foreach ($node in $nodes) {
            $name = "spark-tp4-tensor-r$($node.Rank)"
            Invoke-NodeSsh -Node $node `
                -Command "docker rm -f $name >/dev/null 2>&1 || true" | Out-Null
        }
    }
}

if ($timedOut) {
    throw "TP4 tensor probe exceeded the $WatchdogSeconds-second watchdog"
}
if ($failed) {
    throw "one or more TP4 tensor ranks failed"
}
