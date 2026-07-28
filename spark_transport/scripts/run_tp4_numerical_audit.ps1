param(
    [ValidateRange(1, 1000000)]
    [int]$Iterations = 1000,

    [ValidateRange(1024, 65534)]
    [int]$MasterPort = 29600,

    [ValidateRange(10, 3600)]
    [int]$WatchdogSeconds = 180,

    [string]$Source = "/tmp/spark-vllm-tp4-v2",
    [string]$Library = "/tmp/libspark_transport_capi-v2.so",
    [string]$Image = "<your-vllm-image>",
    [string[]]$Targets = ($env:SPARKRING_TARGETS -split ",").Trim(),
    [string[]]$RankHosts = ($env:SPARKRING_RANK_HOSTS -split ",").Trim(),
    [string]$ManagementNic = "wlP9s9",
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
    [pscustomobject]@{ Rank = 0; Target = $Targets[0] },
    [pscustomobject]@{ Rank = 1; Target = $Targets[1] },
    [pscustomobject]@{ Rank = 2; Target = $Targets[2] },
    [pscustomobject]@{ Rank = 3; Target = $Targets[3] }
)
$headIp = $RankHosts[0]

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

    $name = "spark-tp4-numerical-r$($Node.Rank)"
    $state = (& ssh -o BatchMode=yes -o ConnectTimeout=8 $Node.Target `
        "docker inspect $name --format '{{.State.Status}}:{{.State.ExitCode}}'" 2>$null)
    if ($LASTEXITCODE -ne 0) {
        return "missing"
    }
    return $state.Trim()
}

foreach ($node in $nodes) {
    $runningGlm = (& ssh -o BatchMode=yes -o ConnectTimeout=8 $node.Target `
        "docker inspect glm52-trace --format '{{.State.Running}}' 2>/dev/null || echo false")
    if ($runningGlm.Trim() -eq "true") {
        throw "glm52-trace is still running on rank $($node.Rank); stop the model explicitly before this audit"
    }
}

$failed = $false
$timedOut = $false

try {
    foreach ($node in $nodes) {
        $name = "spark-tp4-numerical-r$($node.Rank)"
        $command = @(
            "test -f $Source/tp4_numerical_audit.py"
            "&& test -f $Library"
            "&& docker rm -f $name >/dev/null 2>&1 || true;"
            "docker run -d --name $name"
            "--network host --ipc host --gpus all"
            "--cap-add IPC_LOCK --ulimit memlock=-1:-1"
            "--ulimit nofile=1048576:1048576"
            "--device /dev/infiniband:/dev/infiniband"
            "-v ${Source}:/opt/spark-vllm:ro"
            "-v ${Library}:/opt/spark-transport/libspark_transport_capi.so:ro"
            "-e PYTHONPATH=/opt/spark-vllm"
            "-e SPARK_TP4_LIBRARY=/opt/spark-transport/libspark_transport_capi.so"
            "-e RANK=$($node.Rank) -e WORLD_SIZE=4"
            "-e MASTER_ADDR=$headIp -e MASTER_PORT=$MasterPort"
            "-e ITERATIONS=$Iterations"
            "-e NCCL_NET=Socket -e NCCL_IB_DISABLE=1"
            "-e NCCL_SOCKET_IFNAME=$ManagementNic"
            "-e GLOO_SOCKET_IFNAME=$ManagementNic"
            "-e NCCL_CUMEM_ENABLE=0 -e NCCL_PROTO=Simple"
            $Image
            "timeout --signal=TERM --kill-after=5s ${WatchdogSeconds}s"
            "python3 /opt/spark-vllm/tp4_numerical_audit.py >/dev/null"
        ) -join " "

        $exitCode = Invoke-NodeSsh -Node $node -Command $command
        if ($exitCode -ne 0) {
            throw "failed to launch numerical-audit rank $($node.Rank)"
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
        Start-Sleep -Milliseconds 500
    } while ([DateTime]::UtcNow -lt $deadline)

    if ($running -ne 0) {
        $timedOut = $true
        $failed = $true
    }

    foreach ($node in $nodes) {
        $name = "spark-tp4-numerical-r$($node.Rank)"
        $state = Get-ContainerState -Node $node
        Write-Output "rank=$($node.Rank) state=$state"
        & ssh -o BatchMode=yes -o ConnectTimeout=8 $node.Target `
            "docker logs $name 2>&1 | grep '^TP4_NUMERICAL' || true"
        if ($state -ne "exited:0") {
            $failed = $true
            Write-Output "rank=$($node.Rank) failure_log:"
            & ssh -o BatchMode=yes -o ConnectTimeout=8 $node.Target `
                "docker logs --tail 60 $name 2>&1"
        }
    }
}
finally {
    if (-not $KeepContainers) {
        foreach ($node in $nodes) {
            $name = "spark-tp4-numerical-r$($node.Rank)"
            Invoke-NodeSsh -Node $node `
                -Command "docker rm -f $name >/dev/null 2>&1 || true" | Out-Null
        }
    }
}

if ($timedOut) {
    throw "TP4 numerical audit exceeded the $WatchdogSeconds-second watchdog"
}
if ($failed) {
    throw "one or more TP4 numerical-audit ranks failed"
}
