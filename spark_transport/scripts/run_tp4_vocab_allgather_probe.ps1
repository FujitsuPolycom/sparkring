param(
    [ValidateRange(0, 10000)]
    [int]$Warmup = 4,

    [ValidateRange(1, 100000)]
    [int]$Iterations = 100,

    [ValidateRange(1, 100000)]
    [int]$ProductionRounds = 1000,

    [ValidateRange(10, 600)]
    [int]$WatchdogSeconds = 120,

    [ValidateRange(1024, 65534)]
    [int]$ControlPort0 = 9990,

    [ValidateRange(1024, 65534)]
    [int]$ControlPort1 = 9991,

    [ValidateRange(0, 300000)]
    [int]$QueuedDelayMs = 0,

    [ValidateRange(0, 3)]
    [int]$QueuedDelayRank = 1,

    [string]$Binary = "/tmp/spark_tp4_vocab_allgather_probe",
    [string]$Library = "/tmp/libspark_transport_capi.so",
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

if ($ControlPort0 -eq $ControlPort1) {
    throw "ControlPort0 and ControlPort1 must differ"
}
if (($QueuedDelayMs -gt 0) -and
    ($QueuedDelayMs -lt 5500)) {
    throw "QueuedDelayMs must be zero or at least 5500"
}
if (($QueuedDelayMs -gt 0) -and
    ((-not $AlternateStreams) -or ($Warmup -lt 1) -or
     ($ProductionRounds -lt 1))) {
    throw "QueuedDelayMs requires AlternateStreams, Warmup >= 1, and ProductionRounds >= 1"
}
if (($QueuedDelayMs -gt 0) -and
    (($QueuedDelayMs + 10000) -ge ($WatchdogSeconds * 1000))) {
    throw "WatchdogSeconds must exceed QueuedDelayMs by more than 10 seconds"
}

$alternateStreamsArgument = if ($AlternateStreams) {
    "--alternate-streams"
} else {
    ""
}
$productionPatternArgument = if ($AlternateStreams) {
    "--production-rounds $ProductionRounds"
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

    $output = & ssh -o BatchMode=yes -o ConnectTimeout=10 `
        $Node.Target $Command
    $exitCode = $LASTEXITCODE
    if ($output) {
        Write-Verbose ($output -join [Environment]::NewLine)
    }
    return $exitCode
}

function Get-ContainerState {
    param(
        [Parameter(Mandatory)]
        [pscustomobject]$Node
    )

    $name = "spark-tp4-vocab-r$($Node.Rank)"
    $state = (& ssh -o BatchMode=yes -o ConnectTimeout=10 $Node.Target `
        "docker inspect $name --format '{{.State.Status}}:{{.State.ExitCode}}'" `
        2>$null)
    if ($LASTEXITCODE -ne 0) {
        return "missing"
    }
    return $state.Trim()
}

$failed = $false
$timedOut = $false
try {
    foreach ($node in $nodes) {
        $name = "spark-tp4-vocab-r$($node.Rank)"
        $command = @(
            "chmod 0755 $Binary $Library;"
            "docker rm -f $name >/dev/null 2>&1 || true;"
            "docker run -d --name $name"
            "--privileged --gpus all --network host --ipc host"
            "--ulimit memlock=-1"
            "-e LD_LIBRARY_PATH=/opt/spark/lib"
            "-v ${Binary}:/opt/spark/bin/tp4_vocab_probe:ro"
            "-v ${Library}:/opt/spark/lib/libspark_transport_capi.so:ro"
            $Image
            "timeout --signal=TERM --kill-after=5s ${WatchdogSeconds}s"
            "taskset -c 8-15 /opt/spark/bin/tp4_vocab_probe"
            "--rank $($node.Rank)"
            "--peer0 $($node.Peer0)"
            "--peer1 $($node.Peer1)"
            "--device0 rocep1s0f0 --device1 rocep1s0f1"
            "--gid0 3 --gid1 3"
            "--control-port0 $ControlPort0"
            "--control-port1 $ControlPort1"
            "--warmup $Warmup --iterations $Iterations"
            $alternateStreamsArgument
            $productionPatternArgument
            $queuedDelayArgument
        ) -join " "
        if ((Invoke-NodeSsh -Node $node -Command $command) -ne 0) {
            throw "failed to launch vocabulary probe rank $($node.Rank)"
        }
    }

    $deadline = [DateTime]::UtcNow.AddSeconds($WatchdogSeconds + 15)
    do {
        $states = @($nodes | ForEach-Object {
            Get-ContainerState -Node $_
        })
        $running = @(
            $states | Where-Object { $_ -like "running:*" }
        ).Count
        if ($running -eq 0) {
            break
        }
        Start-Sleep -Milliseconds 250
    } while ([DateTime]::UtcNow -lt $deadline)

    if ($running -ne 0) {
        $timedOut = $true
        $failed = $true
    }

    $expectedMeasuredSubmissions = 4L * [long]$ProductionRounds
    $expectedDelaySequence = 4L * [long]$Warmup + 1L
    foreach ($node in $nodes) {
        $name = "spark-tp4-vocab-r$($node.Rank)"
        $state = Get-ContainerState -Node $node
        $log = (& ssh -o BatchMode=yes -o ConnectTimeout=10 `
            $node.Target "docker logs $name 2>&1")
        $result = @($log | Where-Object {
            $_ -like "TP4_VOCAB_ALLGATHER*"
        })
        Write-Output "rank=$($node.Rank) state=$state"
        $result | Write-Output

        $validEvidence = $true
        if ($AlternateStreams) {
            $gate = $result -join " "
            $measuredGate = (
                "measured_submissions=$expectedMeasuredSubmissions" +
                "(?:\s|$)"
            )
            $validEvidence = (
                $result.Count -eq 1 `
                -and $gate -match "pattern=5,1,1,1(?:\s|$)" `
                -and $gate -match "production_rounds=$ProductionRounds(?:\s|$)" `
                -and $gate -match $measuredGate `
                -and $gate -match "alternate_streams=true(?:\s|$)" `
                -and $gate -match "mismatches=0(?:\s|$)"
            )
        }
        if ($QueuedDelayMs -gt 0) {
            $expectedApplied = if (
                $node.Rank -eq $QueuedDelayRank
            ) {
                "true"
            } else {
                "false"
            }
            $validEvidence = (
                $validEvidence `
                -and $gate -match "queued_delay_ms=$QueuedDelayMs(?:\s|$)" `
                -and $gate -match "queued_delay_rank=$QueuedDelayRank(?:\s|$)" `
                -and $gate -match "queued_delay_applied=$expectedApplied(?:\s|$)" `
                -and $gate -match "queued_delay_sequence=$expectedDelaySequence(?:\s|$)"
            )
        }
        if ($state -ne "exited:0" -or -not $validEvidence) {
            $failed = $true
            $log | Select-Object -Last 80 | Write-Output
        }
    }
}
finally {
    if (-not $KeepContainers) {
        foreach ($node in $nodes) {
            $name = "spark-tp4-vocab-r$($node.Rank)"
            Invoke-NodeSsh -Node $node `
                -Command "docker rm -f $name >/dev/null 2>&1 || true" |
                Out-Null
        }
    }
}

if ($timedOut) {
    throw "TP4 vocabulary probe exceeded the watchdog"
}
if ($failed) {
    throw "one or more TP4 vocabulary probe ranks failed"
}
