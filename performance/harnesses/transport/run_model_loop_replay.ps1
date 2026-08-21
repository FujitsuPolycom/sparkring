param(
    [ValidateSet("Both", "Socket", "Spark")]
    [string]$Mode = "Both",

    [ValidateRange(0, 10000)]
    [int]$WarmupTokens = 2,

    [ValidateRange(1, 10000)]
    [int]$Tokens = 16,

    [ValidateRange(0.0, 1000000.0)]
    [double]$ComputeUs = 0.0,

    [ValidateRange(10, 3600)]
    [int]$WatchdogSeconds = 180,

    [ValidateRange(1024, 65532)]
    [int]$BasePort = 9600,

    [string]$Binary = "/tmp/spark_model_loop_replay",
    [string]$Library = "/tmp/libspark_transport_capi.so",
    [string]$Image = "<your-vllm-image>",
    [string[]]$Targets = ($env:SPARKRING_TARGETS -split ",").Trim(),
    [string[]]$RankHosts = ($env:SPARKRING_RANK_HOSTS -split ",").Trim(),
    [switch]$TraceTransport,
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
$master = $RankHosts[0]
$computeText = $ComputeUs.ToString(
    "0.###",
    [Globalization.CultureInfo]::InvariantCulture
)

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
        [pscustomobject]$Node,

        [Parameter(Mandatory)]
        [string]$ContainerName
    )

    $state = (& ssh -o BatchMode=yes -o ConnectTimeout=8 $Node.Target `
        "docker inspect $ContainerName --format '{{.State.Status}}:{{.State.ExitCode}}'" `
        2>$null)
    if ($LASTEXITCODE -ne 0) {
        return "missing"
    }
    return $state.Trim()
}

function Invoke-ReplayMode {
    param(
        [Parameter(Mandatory)]
        [ValidateSet("socket", "spark")]
        [string]$ReplayMode,

        [Parameter(Mandatory)]
        [int]$PortOffset
    )

    $rendezvousPort = $BasePort + $PortOffset
    $controlPort0 = $rendezvousPort + 1
    $controlPort1 = $rendezvousPort + 2
    if ($controlPort1 -gt 65535) {
        throw "derived control port exceeds 65535"
    }

    $failed = $false
    $timedOut = $false
    try {
        foreach ($node in $nodes) {
            $name = "spark-model-replay-$ReplayMode-r$($node.Rank)"
            $environment = if ($ReplayMode -eq "socket") {
                @(
                    "-e NCCL_NET=Socket"
                    "-e NCCL_IB_DISABLE=1"
                    "-e NCCL_SOCKET_IFNAME=enp1s0f0np0"
                    "-e NCCL_DEBUG=WARN"
                ) -join " "
            }
            else {
                if ($TraceTransport) {
                    "-e SPARK_TRANSPORT_TRACE=1"
                }
                else {
                    ""
                }
            }
            $command = @(
                "chmod 0755 ${Binary};"
                "test -r ${Library};"
                "docker rm -f $name >/dev/null 2>&1 || true;"
                "docker run -d --name $name"
                "--privileged --gpus all --network host --ipc host"
                "--ulimit memlock=-1"
                $environment
                "-e LD_LIBRARY_PATH=/opt/spark/lib:/opt/venv/lib/python3.12/site-packages/nvidia/nccl/lib"
                "-v ${Binary}:/opt/spark/bin/model_loop_replay:ro"
                "-v ${Library}:/opt/spark/lib/libspark_transport_capi.so:ro"
                $Image
                "timeout --signal=TERM --kill-after=5s ${WatchdogSeconds}s"
                "taskset -c 8-15 /opt/spark/bin/model_loop_replay"
                "--mode $ReplayMode"
                "--rank $($node.Rank)"
                "--master $master"
                "--peer0 $($node.Peer0)"
                "--peer1 $($node.Peer1)"
                "--device0 rocep1s0f0 --device1 rocep1s0f1"
                "--gid0 3 --gid1 3"
                "--rendezvous-port $rendezvousPort"
                "--control-port0 $controlPort0"
                "--control-port1 $controlPort1"
                "--warmup-tokens $WarmupTokens"
                "--tokens $Tokens"
                "--compute-us $computeText >/dev/null"
            ) -join " "

            $exitCode = Invoke-NodeSsh -Node $node -Command $command
            if ($exitCode -ne 0) {
                throw "failed to launch $ReplayMode rank $($node.Rank)"
            }
        }

        $deadline = [DateTime]::UtcNow.AddSeconds($WatchdogSeconds + 15)
        do {
            $running = 0
            foreach ($node in $nodes) {
                $name = "spark-model-replay-$ReplayMode-r$($node.Rank)"
                if ((Get-ContainerState -Node $node -ContainerName $name) `
                    -like "running:*") {
                    ++$running
                }
            }
            if ($running -eq 0) {
                break
            }
            Start-Sleep -Milliseconds 500
        } while ([DateTime]::UtcNow -lt $deadline)

        if ($running -ne 0) {
            $timedOut = $true
            $failed = $true
        }

        foreach ($node in $nodes) {
            $name = "spark-model-replay-$ReplayMode-r$($node.Rank)"
            $state = Get-ContainerState -Node $node -ContainerName $name
            Write-Output "mode=$ReplayMode rank=$($node.Rank) state=$state"
            & ssh -o BatchMode=yes -o ConnectTimeout=8 $node.Target `
                "docker logs $name 2>&1 | grep '^MODEL_REPLAY' || true"
            if ($state -ne "exited:0") {
                $failed = $true
                Write-Output "mode=$ReplayMode rank=$($node.Rank) failure_log:"
                & ssh -o BatchMode=yes -o ConnectTimeout=8 $node.Target `
                    "docker logs --tail 60 $name 2>&1"
            }
        }
    }
    finally {
        if (-not $KeepContainers) {
            foreach ($node in $nodes) {
                $name = "spark-model-replay-$ReplayMode-r$($node.Rank)"
                Invoke-NodeSsh -Node $node `
                    -Command "docker rm -f $name >/dev/null 2>&1 || true" |
                    Out-Null
            }
        }
    }

    if ($timedOut) {
        throw "$ReplayMode replay exceeded the watchdog"
    }
    if ($failed) {
        throw "one or more $ReplayMode replay ranks failed"
    }
}

$modes = @(switch ($Mode) {
    "Socket" { @("socket") }
    "Spark" { @("spark") }
    default { @("socket", "spark") }
})

for ($index = 0; $index -lt $modes.Count; ++$index) {
    Invoke-ReplayMode -ReplayMode $modes[$index] -PortOffset ($index * 10)
}
