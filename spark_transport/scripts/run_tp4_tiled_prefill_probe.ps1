param(
    [Parameter(Mandatory)]
    [ValidateSet(
        "q40_isolated",
        "q40_steady",
        "q512_isolated",
        "q512_steady",
        "q1024_isolated",
        "q1024_steady",
        "q4096_isolated",
        "q4096_steady",
        "glm4096_q1024_steady",
        "glm4096_q2048_steady",
        "glm4096_q4096_steady",
        "glm4096_q8192_steady",
        "q513_boundary_correctness",
        "q1025_boundary_correctness",
        "q4096_backpressure_edge0",
        "q4096_backpressure_edge1",
        "q40_poison_unexpected_generation",
        "q512_poison_credit_regression"
    )]
    [string]$ArmId,

    [ValidateRange(1024, 65534)]
    [int]$ControlPort0 = 12500,

    [ValidateRange(1024, 65534)]
    [int]$ControlPort1 = 12501,

    [ValidateRange(10, 3600)]
    [int]$WatchdogSeconds = 300,

    [ValidatePattern("^[0-9,-]+$")]
    [string]$CpuSet = "10,11,12",

    [ValidateRange(0, 4095)]
    [int]$SubmitCpu = 10,

    [ValidateRange(0, 4095)]
    [int]$Edge0ProgressCpu = 11,

    [ValidateRange(0, 4095)]
    [int]$Edge1ProgressCpu = 12,

    [string]$ProbeBinary = "/tmp/spark_tp4_tiled_prefill_probe",
    [string]$Image = "<your-vllm-image>",
    [string]$Python = "python",
    [string[]]$Targets = ($env:SPARKRING_TARGETS -split ",").Trim(),
    [string[]]$RankHosts = ($env:SPARKRING_RANK_HOSTS -split ",").Trim(),
    [switch]$Execute,
    [switch]$KeepContainers
)

$ErrorActionPreference = "Stop"

# Status: research-only. Without -Execute this script only prints the exact
# arm and never validates remote configuration, invokes SSH, or starts CUDA.
$repositoryRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$qualificationModule =
    "spark_transport.experiments.tiled_prefill.qualification"
$receiptPrefix = "TP4_TILED_PREFILL_RECEIPT "
$expectedRegisteredBytesPerEdge = 8389632L

function Invoke-QualificationPython {
    param(
        [Parameter(Mandatory)]
        [string[]]$Arguments
    )

    Push-Location $repositoryRoot
    try {
        $output = & $Python -m $qualificationModule @Arguments
        $exitCode = $LASTEXITCODE
    }
    finally {
        Pop-Location
    }
    if ($exitCode -ne 0) {
        throw "tiled-prefill qualification CLI failed with exit $exitCode"
    }
    return @($output)
}

$planJson = (Invoke-QualificationPython -Arguments @("plan", "--compact")) `
    -join "`n"
$plan = $planJson | ConvertFrom-Json
$matchingArms = @($plan.arms | Where-Object { $_.arm_id -eq $ArmId })
if ($matchingArms.Count -ne 1) {
    throw "qualification plan does not contain exactly one arm named $ArmId"
}
$arm = $matchingArms[0]

if ([long]$plan.geometry.registered_tile_storage_bytes_per_edge -ne `
    $expectedRegisteredBytesPerEdge) {
    throw ("qualification geometry drift: expected " +
        "$expectedRegisteredBytesPerEdge registered bytes per edge")
}
if ($plan.runnable -ne $true -or $plan.status -ne "research-only") {
    throw "qualification plan does not expose the research-only native probe"
}
if ($ControlPort0 -eq $ControlPort1) {
    throw "ControlPort0 and ControlPort1 must differ"
}
if (@($SubmitCpu, $Edge0ProgressCpu, $Edge1ProgressCpu |
        Sort-Object -Unique).Count -ne 3) {
    throw "SubmitCpu, Edge0ProgressCpu, and Edge1ProgressCpu must differ"
}

$configuredTargets = @($Targets | Where-Object { $_ })
$configuredRankHosts = @($RankHosts | Where-Object { $_ })
$nodes = @()
if ($configuredTargets.Count -ne 0 -or $configuredRankHosts.Count -ne 0) {
    if ($configuredTargets.Count -ne 4) {
        throw ("SPARKRING_TARGETS (or -Targets) must contain exactly four " +
            "SSH targets in rank order")
    }
    if ($configuredRankHosts.Count -ne 4) {
        throw ("SPARKRING_RANK_HOSTS (or -RankHosts) must contain exactly " +
            "four rank host addresses in rank order")
    }
    if (@($configuredRankHosts | Sort-Object -Unique).Count -ne 4) {
        throw "rank host addresses must be unique"
    }

    $nodes = @(
        [pscustomobject]@{
            Rank = 0
            Target = $configuredTargets[0]
            Local = $configuredRankHosts[0]
            Peer0 = $configuredRankHosts[1]
            Peer1 = $configuredRankHosts[3]
            Device0 = "rocep1s0f0"
            Device1 = "rocep1s0f1"
        },
        [pscustomobject]@{
            Rank = 1
            Target = $configuredTargets[1]
            Local = $configuredRankHosts[1]
            Peer0 = $configuredRankHosts[0]
            Peer1 = $configuredRankHosts[2]
            Device0 = "rocep1s0f1"
            Device1 = "rocep1s0f0"
        },
        [pscustomobject]@{
            Rank = 2
            Target = $configuredTargets[2]
            Local = $configuredRankHosts[2]
            Peer0 = $configuredRankHosts[3]
            Peer1 = $configuredRankHosts[1]
            Device0 = "rocep1s0f0"
            Device1 = "rocep1s0f1"
        },
        [pscustomobject]@{
            Rank = 3
            Target = $configuredTargets[3]
            Local = $configuredRankHosts[3]
            Peer0 = $configuredRankHosts[2]
            Peer1 = $configuredRankHosts[0]
            Device0 = "rocep1s0f1"
            Device1 = "rocep1s0f0"
        }
    )

    foreach ($node in $nodes) {
        if ($node.Peer0 -eq $node.Local -or $node.Peer1 -eq $node.Local) {
            throw "rank $($node.Rank) topology contains a self peer"
        }
        $targetMatch = [regex]::Match(
            $node.Target, "(?:^|@)(\d{1,3}(?:\.\d{1,3}){3})$")
        if ($targetMatch.Success -and
                $targetMatch.Groups[1].Value -ne $node.Local) {
            throw ("rank $($node.Rank) SSH target address does not match " +
                "its rank-order host address")
        }
        $edge0Role = if ($node.Rank -gt ($node.Rank -bxor 1)) {
            "server"
        } else {
            "client"
        }
        $edge1Role = if ($node.Rank -gt ($node.Rank -bxor 3)) {
            "server"
        } else {
            "client"
        }
        Write-Output ("tiled_prefill_endpoint rank=$($node.Rank) " +
            "target=$($node.Target) local=$($node.Local) " +
            "edge0_peer=$($node.Peer0) edge0_role=$edge0Role " +
            "edge0_port=$ControlPort0 edge0_device=$($node.Device0) " +
            "edge1_peer=$($node.Peer1) edge1_role=$edge1Role " +
            "edge1_port=$ControlPort1 edge1_device=$($node.Device1)")
    }
}

$action = if ($Execute) { "execute" } else { "plan" }
Write-Output ("tiled_prefill_arm=$ArmId action=$action " +
    "q=$($arm.query_rows) timing_mode=$($arm.timing_mode) " +
    "warmup=$($arm.warmup_operations) " +
    "measured=$($arm.measured_operations) " +
    "tiles=$($arm.logical_tile_count) " +
    "tail_active_bytes=$($arm.tail_active_bytes) " +
    "credit_delay_edge=$($arm.credit_delay_edge) " +
    "credit_delay_us=$($arm.credit_delay_us) " +
    "poison=$($arm.poison) expected_exit=$($arm.expected_exit_code) " +
    "registered_tile_storage_bytes_per_edge=" +
    "$expectedRegisteredBytesPerEdge")

if (-not $Execute) {
    Write-Output ("tiled_prefill_probe=plan ranks=4 remote_contact=false " +
        "serving_integration=false receipt_schema=$($plan.receipt_schema) " +
        "executor_stream_policy=single_stream_correctness_only " +
        "protocol_nodes_status=standalone-native-executor-bound " +
        "performance_claim_allowed=false")
    return
}

if ($nodes.Count -ne 4) {
    throw ("execution requires SPARKRING_TARGETS and " +
        "SPARKRING_RANK_HOSTS with four entries in rank order")
}
if ($Image -eq "<your-vllm-image>") {
    throw "set -Image before using -Execute"
}

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

    $name = "spark-tp4-tiled-$ArmId-r$($Node.Rank)"
    $state = (& ssh -o BatchMode=yes -o ConnectTimeout=8 $Node.Target `
        "docker inspect $name --format '{{.State.Status}}:{{.State.ExitCode}}'" `
        2>$null)
    if ($LASTEXITCODE -ne 0) {
        return "missing"
    }
    return $state.Trim()
}

$hashes = @()
foreach ($node in $nodes) {
    $hash = (& ssh -o BatchMode=yes -o ConnectTimeout=8 $node.Target `
        "test -x '$ProbeBinary' && sha256sum '$ProbeBinary'")
    if ($LASTEXITCODE -ne 0) {
        throw "rank $($node.Rank) is missing the tiled-prefill probe"
    }
    $digest = (($hash -join "`n").Trim() -split "\s+")[0]
    if ($digest -notmatch "^[0-9a-fA-F]{64}$") {
        throw "rank $($node.Rank) returned an invalid probe SHA-256"
    }
    $hashes += $digest.ToLowerInvariant()
}
if (@($hashes | Sort-Object -Unique).Count -ne 1) {
    throw "tiled-prefill probe SHA-256 values differ across ranks"
}
Write-Output "preflight=pass ranks=4 identical_sha256=true sha256=$($hashes[0])"

$probeArguments = @($arm.probe_arguments | ForEach-Object { [string]$_ })
$probeArgumentString = $probeArguments -join " "
$failed = $false
$timedOut = $false
$receiptLines = @()
$expectedState = "exited:$($arm.expected_exit_code)"

try {
    foreach ($node in $nodes) {
        $name = "spark-tp4-tiled-$ArmId-r$($node.Rank)"
        $command = @(
            "docker rm -f $name >/dev/null 2>&1 || true;"
            "docker run -d --name $name"
            "--privileged --gpus all --network host --ipc host"
            "--cpuset-cpus=$CpuSet"
            "--ulimit memlock=-1"
            "-v ${ProbeBinary}:/probe:ro"
            "--entrypoint /usr/bin/env"
            $Image
            "timeout --signal=TERM --kill-after=5s ${WatchdogSeconds}s"
            "env -u SPARK_TRANSPORT_TRACE"
            "taskset -c $SubmitCpu /probe"
            "--rank $($node.Rank) --world-size 4"
            "--peer0 $($node.Peer0) --peer1 $($node.Peer1)"
            "--device0 $($node.Device0) --device1 $($node.Device1)"
            "--gid0 3 --gid1 3"
            "--control-port0 $ControlPort0"
            "--control-port1 $ControlPort1"
            "--submit-cpu $SubmitCpu"
            "--edge0-progress-cpu $Edge0ProgressCpu"
            "--edge1-progress-cpu $Edge1ProgressCpu"
            $probeArgumentString
            ">/dev/null"
        ) -join " "
        $exitCode = Invoke-NodeSsh -Node $node -Command $command
        if ($exitCode -ne 0) {
            throw "failed to launch tiled-prefill probe rank $($node.Rank)"
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

    foreach ($node in $nodes) {
        $name = "spark-tp4-tiled-$ArmId-r$($node.Rank)"
        $state = Get-ContainerState -Node $node
        $log = (& ssh -o BatchMode=yes -o ConnectTimeout=8 $node.Target `
            "docker logs $name 2>&1")
        $rankReceipts = @($log | Where-Object {
            $_ -like "${receiptPrefix}*"
        })
        Write-Output "rank=$($node.Rank) state=$state receipts=$($rankReceipts.Count)"
        $rankReceipts | Write-Output
        if ($state -ne $expectedState -or $rankReceipts.Count -ne 1) {
            $failed = $true
            Write-Output "rank=$($node.Rank) failure_log:"
            $log | Select-Object -Last 40 | Write-Output
        }
        else {
            $receiptLines += $rankReceipts[0]
        }
    }

    if (-not $failed) {
        $receiptFile = Join-Path ([IO.Path]::GetTempPath()) `
            ("sparkring-tiled-prefill-" + [Guid]::NewGuid() + ".jsonl")
        try {
            $utf8WithoutBom = New-Object System.Text.UTF8Encoding($false)
            [IO.File]::WriteAllLines(
                $receiptFile, [string[]]$receiptLines, $utf8WithoutBom)
            $gate = Invoke-QualificationPython -Arguments @(
                "validate",
                "--arm-id",
                $ArmId,
                "--receipts",
                $receiptFile
            )
            $gate | Write-Output
        }
        finally {
            Remove-Item -LiteralPath $receiptFile -Force `
                -ErrorAction SilentlyContinue
        }
    }
}
finally {
    if (-not $KeepContainers) {
        foreach ($node in $nodes) {
            $name = "spark-tp4-tiled-$ArmId-r$($node.Rank)"
            Invoke-NodeSsh -Node $node -Command `
                "docker rm -f $name >/dev/null 2>&1 || true" | Out-Null
        }
    }
}

if ($timedOut) {
    throw "tiled-prefill probe exceeded the watchdog"
}
if ($failed) {
    throw "one or more tiled-prefill probe ranks failed"
}

Write-Output ("tiled_prefill_probe=pass arm=$ArmId ranks=4 " +
    "expected_exit=$($arm.expected_exit_code) " +
    "registered_tile_storage_bytes_per_edge=" +
    "$expectedRegisteredBytesPerEdge")
