param(
    [int[]]$InputBytes = @(16384, 77440, 753664),

    [ValidateRange(0, 10000)]
    [int]$Warmup = 4,

    [ValidateRange(1, 100000)]
    [int]$Iterations = 100,

    [ValidateRange(10, 600)]
    [int]$WatchdogSeconds = 120,

    [ValidateRange(1024, 65400)]
    [int]$BasePort = 9490,

    [ValidateRange(0, 300000)]
    [int]$QueuedDelayMs = 0,

    [ValidateRange(0, 3)]
    [int]$QueuedDelayRank = 1,

    [string]$Binary = "/tmp/spark_tp4_allgather_probe",
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

    & ssh -o BatchMode=yes -o ConnectTimeout=10 $Node.Target $Command
    return $LASTEXITCODE
}

function Get-ContainerState {
    param(
        [Parameter(Mandatory)]
        [pscustomobject]$Node,

        [Parameter(Mandatory)]
        [string]$Name
    )

    $state = (& ssh -o BatchMode=yes -o ConnectTimeout=10 $Node.Target `
        "docker inspect $Name --format '{{.State.Status}}:{{.State.ExitCode}}'" `
        2>$null)
    if ($LASTEXITCODE -ne 0) {
        return "missing"
    }
    return $state.Trim()
}

foreach ($bytes in $InputBytes) {
    if ($bytes -le 0 -or $bytes % 16 -ne 0) {
        throw "input size must be a positive multiple of 16: $bytes"
    }
}
if (($QueuedDelayMs -gt 0) -and
    ($QueuedDelayMs -lt 5500)) {
    throw "QueuedDelayMs must be zero or at least 5500"
}
if (($QueuedDelayMs -gt 0) -and
    (($Warmup -lt 1) -or ($Iterations -lt 2))) {
    throw "QueuedDelayMs requires Warmup >= 1 and Iterations >= 2"
}
if (($QueuedDelayMs -gt 0) -and
    (($QueuedDelayMs + 10000) -ge ($WatchdogSeconds * 1000))) {
    throw "WatchdogSeconds must exceed QueuedDelayMs by more than 10 seconds"
}

$queuedDelayArgument = if ($QueuedDelayMs -gt 0) {
    "--queued-delay-ms $QueuedDelayMs --queued-delay-rank $QueuedDelayRank"
} else {
    ""
}

for ($shapeIndex = 0; $shapeIndex -lt $InputBytes.Count; ++$shapeIndex) {
    $bytes = $InputBytes[$shapeIndex]
    $port0 = $BasePort + $shapeIndex * 10
    $port1 = $port0 + 1
    $failed = $false
    try {
        foreach ($node in $nodes) {
            $name = "spark-tp4-allgather-b${bytes}-r$($node.Rank)"
            $command = @(
                "chmod 0755 $Binary;"
                "docker rm -f $name >/dev/null 2>&1 || true;"
                "docker run -d --name $name"
                "--privileged --gpus all --network host --ipc host"
                "--ulimit memlock=-1"
                "-v ${Binary}:/opt/spark/bin/tp4_allgather_probe:ro"
                $Image
                "timeout --signal=TERM --kill-after=5s ${WatchdogSeconds}s"
                "taskset -c 8-15 /opt/spark/bin/tp4_allgather_probe"
                "--rank $($node.Rank)"
                "--peer0 $($node.Peer0)"
                "--peer1 $($node.Peer1)"
                "--device0 rocep1s0f0 --device1 rocep1s0f1"
                "--gid0 3 --gid1 3"
                "--control-port0 $port0 --control-port1 $port1"
                "--bytes $bytes --warmup $Warmup"
                "--iterations $Iterations"
                $queuedDelayArgument
                ">/dev/null"
            ) -join " "
            if ((Invoke-NodeSsh -Node $node -Command $command) -ne 0) {
                throw "failed to launch rank $($node.Rank), bytes=$bytes"
            }
        }

        $deadline = [DateTime]::UtcNow.AddSeconds($WatchdogSeconds + 15)
        do {
            $running = 0
            foreach ($node in $nodes) {
                $name = "spark-tp4-allgather-b${bytes}-r$($node.Rank)"
                if ((Get-ContainerState -Node $node -Name $name) `
                    -like "running:*") {
                    ++$running
                }
            }
            if ($running -eq 0) {
                break
            }
            Start-Sleep -Milliseconds 250
        } while ([DateTime]::UtcNow -lt $deadline)

        if ($running -ne 0) {
            throw "all-gather shape $bytes exceeded watchdog"
        }
        foreach ($node in $nodes) {
            $name = "spark-tp4-allgather-b${bytes}-r$($node.Rank)"
            $state = Get-ContainerState -Node $node -Name $name
            Write-Output "bytes=$bytes rank=$($node.Rank) state=$state"
            $log = (& ssh -o BatchMode=yes -o ConnectTimeout=10 `
                $node.Target "docker logs $name 2>&1")
            $result = @($log | Where-Object {
                $_ -like "TP4_ALLGATHER*"
            })
            $result | Write-Output

            $validEvidence = $true
            if ($QueuedDelayMs -gt 0) {
                $expectedApplied = if (
                    $node.Rank -eq $QueuedDelayRank
                ) {
                    "true"
                } else {
                    "false"
                }
                $gate = $result -join " "
                $validEvidence = (
                    $result.Count -eq 1 `
                    -and $gate -match "queued_delay_ms=$QueuedDelayMs(?:\s|$)" `
                    -and $gate -match "queued_delay_rank=$QueuedDelayRank(?:\s|$)" `
                    -and $gate -match "queued_delay_applied=$expectedApplied(?:\s|$)" `
                    -and $gate -match "queued_delay_sequence=$($Warmup + 1)(?:\s|$)" `
                    -and $gate -match "mismatches=0(?:\s|$)"
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
                $name = "spark-tp4-allgather-b${bytes}-r$($node.Rank)"
                Invoke-NodeSsh -Node $node `
                    -Command "docker rm -f $name >/dev/null 2>&1 || true" |
                    Out-Null
            }
        }
    }
    if ($failed) {
        throw "one or more all-gather ranks failed for bytes=$bytes"
    }
}
