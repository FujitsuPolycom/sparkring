param(
    [int]$Warmup = 1000,
    [int]$Iterations = 10000,
    [int]$Bytes = 12288,
    [int]$ControlPort0 = 9460,
    [int]$ControlPort1 = 9461,
    [string]$Binary = "/tmp/spark_tp4_probe",
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

foreach ($node in $nodes) {
    $name = "spark-tp4-r$($node.Rank)"
    $command = @(
        "docker rm -f $name >/dev/null 2>&1 || true;"
        "docker run -d --name $name"
        "--privileged --gpus all --network host --ipc host"
        "--ulimit memlock=-1"
        "-v ${Binary}:/probe:ro"
        $Image
        "timeout 60s taskset -c 10 /probe"
        "--rank $($node.Rank)"
        "--peer0 $($node.Peer0)"
        "--peer1 $($node.Peer1)"
        "--device0 rocep1s0f0 --device1 rocep1s0f1"
        "--gid0 3 --gid1 3"
        "--control-port0 $ControlPort0"
        "--control-port1 $ControlPort1"
        "--bytes $Bytes"
        "--warmup $Warmup"
        "--iterations $Iterations >/dev/null"
    ) -join " "

    & ssh -o BatchMode=yes -o ConnectTimeout=8 $node.Target $command
    if ($LASTEXITCODE -ne 0) {
        throw "failed to launch rank $($node.Rank)"
    }
}

Start-Sleep -Seconds 5
$failed = $false
foreach ($node in $nodes) {
    $name = "spark-tp4-r$($node.Rank)"
    $state = (& ssh -o BatchMode=yes -o ConnectTimeout=8 $node.Target `
        "docker inspect $name --format '{{.State.Status}}:{{.State.ExitCode}}'").Trim()
    Write-Output "rank=$($node.Rank) state=$state"
    & ssh -o BatchMode=yes -o ConnectTimeout=8 $node.Target `
        "docker logs $name 2>&1 | grep -E '^(TP4_BF16|PHASE)'"
    if ($state -ne "exited:0") {
        $failed = $true
    }
}

if (-not $KeepContainers) {
    foreach ($node in $nodes) {
        $name = "spark-tp4-r$($node.Rank)"
        & ssh -o BatchMode=yes -o ConnectTimeout=8 $node.Target `
            "docker rm $name >/dev/null"
    }
}

if ($failed) {
    throw "one or more TP4 ranks failed"
}
