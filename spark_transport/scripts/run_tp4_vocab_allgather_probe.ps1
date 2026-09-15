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
    [ValidateSet("documented-cycle")]
    [string]$DevicePreset,
    [string[]]$Device0 = @(),
    [string[]]$Device1 = @(),
    [ValidateRange(1, 3600)]
    [int]$RemoteTimeoutSeconds = 60,
    [scriptblock]$RemoteExecutor,
    [switch]$KeepContainers
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot/posix_shell_argument.ps1"
. "$PSScriptRoot/probe_process.ps1"
$runIdentity = [Guid]::NewGuid().ToString("N")
$ownedNodes = @()

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
# Index only the non-empty entries so an embedded empty element cannot shift
# the rank-to-target mapping after the count check.
$Targets = @($Targets | Where-Object { $_ })
$RankHosts = @($RankHosts | Where-Object { $_ })

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

. "$PSScriptRoot/tp4_device_mapping.ps1"
$deviceMapping = @(Resolve-Tp4DeviceMapping -Preset $DevicePreset -Device0 $Device0 -Device1 $Device1)

$nodes = @(
    [pscustomobject]@{
        Rank = 0
        Device0 = $deviceMapping[0].Device0
        Device1 = $deviceMapping[0].Device1
        Target = $Targets[0]
        Peer0 = $RankHosts[1]
        Peer1 = $RankHosts[3]
    },
    [pscustomobject]@{
        Rank = 1
        Device0 = $deviceMapping[1].Device0
        Device1 = $deviceMapping[1].Device1
        Target = $Targets[1]
        Peer0 = $RankHosts[0]
        Peer1 = $RankHosts[2]
    },
    [pscustomobject]@{
        Rank = 2
        Device0 = $deviceMapping[2].Device0
        Device1 = $deviceMapping[2].Device1
        Target = $Targets[2]
        Peer0 = $RankHosts[3]
        Peer1 = $RankHosts[1]
    },
    [pscustomobject]@{
        Rank = 3
        Device0 = $deviceMapping[3].Device0
        Device1 = $deviceMapping[3].Device1
        Target = $Targets[3]
        Peer0 = $RankHosts[2]
        Peer1 = $RankHosts[0]
    }
)

foreach ($node in $nodes) {
    Write-Output "rank=$($node.Rank) peer0=$($node.Peer0) device0=$($node.Device0) peer1=$($node.Peer1) device1=$($node.Device1) gid0=3 gid1=3"
}

function Invoke-NodeSsh {
    param(
        [Parameter(Mandatory)]
        [pscustomobject]$Node,

        [Parameter(Mandatory)]
        [string]$Command
    )

    $output = Invoke-ProbeSsh -o BatchMode=yes -o ConnectTimeout=10 `
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

    $name = "spark-tp4-vocab-$runIdentity-r$($Node.Rank)"
    $state = (Invoke-ProbeSsh -o BatchMode=yes -o ConnectTimeout=10 $Node.Target `
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
        $name = "spark-tp4-vocab-$runIdentity-r$($node.Rank)"
        $command = @(
            # Missing artifacts must fail here instead of becoming empty
            # directories at the bind-mount paths.
            "test -f $(ConvertTo-PosixShellArgument $Binary) -a -f $(ConvertTo-PosixShellArgument $Library) || exit 97;"
            "chmod 0755 $(ConvertTo-PosixShellArgument $Binary) $(ConvertTo-PosixShellArgument $Library) || exit 97;"
            "docker run -d --name $name"
            "--privileged --gpus all --network host --ipc host"
            "--ulimit memlock=-1"
            "-e LD_LIBRARY_PATH=/opt/spark/lib"
            "-v $(ConvertTo-PosixShellArgument "${Binary}:/opt/spark/bin/tp4_vocab_probe:ro")"
            "-v $(ConvertTo-PosixShellArgument "${Library}:/opt/spark/lib/libspark_transport_capi.so:ro")"
            (ConvertTo-PosixShellArgument $Image)
            "timeout --signal=TERM --kill-after=5s ${WatchdogSeconds}s"
            "taskset -c 8-15 /opt/spark/bin/tp4_vocab_probe"
            "--rank $($node.Rank)"
            "--peer0 $(ConvertTo-PosixShellArgument $node.Peer0)"
            "--peer1 $(ConvertTo-PosixShellArgument $node.Peer1)"
            "--device0 $($node.Device0) --device1 $($node.Device1)"
            "--gid0 3 --gid1 3"
            "--control-port0 $ControlPort0"
            "--control-port1 $ControlPort1"
            "--warmup $Warmup --iterations $Iterations"
            $alternateStreamsArgument
            $productionPatternArgument
            $queuedDelayArgument
        ) -join " "
        # The invocation-specific name also identifies a launch whose SSH reply is lost.
        $ownedNodes += $node
        $launchExit = Invoke-NodeSsh -Node $node -Command $command
        if ($launchExit -eq 97) {
            throw "rank $($node.Rank) is missing the probe binary or library"
        }
        if ($launchExit -ne 0) {
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
        $name = "spark-tp4-vocab-$runIdentity-r$($node.Rank)"
        $state = Get-ContainerState -Node $node
        $log = (Invoke-ProbeSsh -o BatchMode=yes -o ConnectTimeout=10 `
            $node.Target "docker logs $name 2>&1")
        $result = @($log | Where-Object {
            $_ -like "TP4_VOCAB_ALLGATHER*"
        })
        Write-Output "rank=$($node.Rank) state=$state"
        $result | Write-Output

        $validEvidence = $true
        if ($AlternateStreams) {
            $gate = if ($result.Count -eq 1) { $result[0] } else { "" }
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
        foreach ($node in $ownedNodes) {
            $name = "spark-tp4-vocab-$runIdentity-r$($node.Rank)"
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
