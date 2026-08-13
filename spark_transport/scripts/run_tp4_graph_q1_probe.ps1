param(
    [ValidateRange(0, 10000000)]
    [int]$Warmup = 10,

    [ValidateRange(1, 10000000)]
    [int]$Iterations = 100,

    [ValidateRange(1, 4096)]
    [int]$OperationsPerGraph = 1,

    [ValidateSet("serial_ack", "two_slot_deferred_ack")]
    [string]$AllreduceProtocol = "serial_ack",

    [ValidateSet("fused", "split_64k", "tiered_64k")]
    [string]$GraphKernel = "fused",

    [ValidateSet("sequential", "dual_port_striped")]
    [string]$WireSchedule = "sequential",

    [ValidateSet("burst", "isolated")]
    [string]$TimingMode = "burst",

    [switch]$MultiGraphValidation,

    [switch]$MixedQValidation,

    [ValidateRange(1, 512)]
    [int]$FixedQ = 1,

    [ValidateRange(6, 512)]
    [int]$MaximumQ = 6,

    [ValidateRange(1, 16)]
    [int]$GraphAOperations = 3,

    [ValidateRange(128, 128)]
    [int]$GraphBOperations = 128,

    [ValidateRange(1024, 65534)]
    [int]$ControlPort0 = 9960,

    [ValidateRange(1024, 65534)]
    [int]$ControlPort1 = 9961,

    [ValidateRange(0.001, 1000000)]
    [double]$MaxGraphSubmitUs = 25,

    [ValidateRange(0.001, 1000000)]
    [double]$MaxDeviceUs = 75,

    [switch]$DisablePerformanceGates,

    [ValidateRange(10, 3600)]
    [int]$WatchdogSeconds = 90,

    [ValidatePattern("^[0-9,-]+$")]
    [string]$CpuSet = "10,11",

    [ValidateRange(0, 4095)]
    [int]$SubmitCpu = 10,

    [ValidateRange(0, 4095)]
    [int]$ProgressCpu = 11,

    [string]$ProbeBinary =
        "/tmp/spark_tp4_graph_q1_probe",
    [string]$Image = "<your-vllm-image>",
    [string[]]$Targets = ($env:SPARKRING_TARGETS -split ",").Trim(),
    [string[]]$RankHosts = ($env:SPARKRING_RANK_HOSTS -split ",").Trim(),
    [switch]$ValidateOnly,
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
if ($MultiGraphValidation -and -not $DisablePerformanceGates) {
    throw "MultiGraphValidation requires DisablePerformanceGates because it synchronizes and verifies every replay"
}
if ($MixedQValidation -and -not $MultiGraphValidation) {
    throw "MixedQValidation requires MultiGraphValidation"
}
if ($TimingMode -eq "isolated" -and `
    ($MultiGraphValidation -or $MixedQValidation -or `
        $OperationsPerGraph -ne 1)) {
    throw "TimingMode=isolated requires one single-graph collective"
}
if ($FixedQ -ne 1 -and `
    ($MultiGraphValidation -or $MixedQValidation -or `
        $OperationsPerGraph -ne 1)) {
    throw "FixedQ above one requires one single-graph collective"
}
if ($GraphKernel -eq "split_64k" -and `
    ($MultiGraphValidation -or $MixedQValidation -or `
        $OperationsPerGraph -ne 1)) {
    throw ("GraphKernel=split_64k requires fixed Q1 through Q512 " +
        "and one single-graph collective")
}
if ($WireSchedule -eq "dual_port_striped" -and `
    (($FixedQ -ne 40 -and $FixedQ -ne 512) -or `
        $AllreduceProtocol -ne "two_slot_deferred_ack" -or `
        $GraphKernel -ne "fused" -or `
        $MultiGraphValidation -or $MixedQValidation -or `
        $OperationsPerGraph -ne 1)) {
    throw ("WireSchedule=dual_port_striped requires FixedQ 40 or 512, " +
        "AllreduceProtocol=two_slot_deferred_ack, GraphKernel=fused, " +
        "and one single-graph collective")
}
if ($WireSchedule -eq "dual_port_striped") {
    $selectedTransportKernelPath = "dual_port_striped_dag"
}
else {
    $selectedTransportKernelPath = "sequential_$GraphKernel"
}

if ($ValidateOnly) {
    Write-Output ("validation=pass graph_kernel=$GraphKernel " +
        "wire_schedule=$WireSchedule protocol=$AllreduceProtocol " +
        "transport_kernel_path=$selectedTransportKernelPath " +
        "timing_mode=$TimingMode " +
        "fixed_q=$FixedQ maximum_q=$MaximumQ " +
        "multi_graph=$($MultiGraphValidation.IsPresent.ToString().ToLowerInvariant()) " +
        "mixed_q=$($MixedQValidation.IsPresent.ToString().ToLowerInvariant())")
    return
}

$nodes = @(
    [pscustomobject]@{
        Rank = 0
        Target = $Targets[0]
        Peer0 = $RankHosts[1]
        Peer1 = $RankHosts[3]
        Device0 = "rocep1s0f0"
        Device1 = "rocep1s0f1"
    },
    [pscustomobject]@{
        Rank = 1
        Target = $Targets[1]
        Peer0 = $RankHosts[0]
        Peer1 = $RankHosts[2]
        Device0 = "rocep1s0f1"
        Device1 = "rocep1s0f0"
    },
    [pscustomobject]@{
        Rank = 2
        Target = $Targets[2]
        Peer0 = $RankHosts[3]
        Peer1 = $RankHosts[1]
        Device0 = "rocep1s0f0"
        Device1 = "rocep1s0f1"
    },
    [pscustomobject]@{
        Rank = 3
        Target = $Targets[3]
        Peer0 = $RankHosts[2]
        Peer1 = $RankHosts[0]
        Device0 = "rocep1s0f1"
        Device1 = "rocep1s0f0"
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

    $name = "spark-tp4-graph-q1-r$($Node.Rank)"
    $state = (& ssh -o BatchMode=yes -o ConnectTimeout=8 $Node.Target `
        "docker inspect $name --format '{{.State.Status}}:{{.State.ExitCode}}'" 2>$null)
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
        throw "rank $($node.Rank) is missing the staged graph probe"
    }
    $hashes += ,(($hash -join "`n").Trim())
}
if (@($hashes | Sort-Object -Unique).Count -ne 1) {
    throw "graph probe SHA-256 values differ across ranks"
}
Write-Output "preflight=pass identical_sha256=true"
Write-Output $hashes[0]

$failed = $false
$timedOut = $false
$performanceArgs = @()
if (-not $DisablePerformanceGates) {
    $performanceArgs = @(
        "--max-graph-submit-us $MaxGraphSubmitUs"
        "--max-device-us $MaxDeviceUs"
    )
}
$graphValidationArgs = @()
if ($MultiGraphValidation) {
    $graphValidationArgs = @(
        "--multi-graph-validation"
        "--graph-a-operations $GraphAOperations"
        "--graph-b-operations $GraphBOperations"
    )
}
if ($MixedQValidation) {
    $graphValidationArgs += "--mixed-q-validation"
}

try {
    foreach ($node in $nodes) {
        $name = "spark-tp4-graph-q1-r$($node.Rank)"
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
            "--rank $($node.Rank)"
            "--peer0 $($node.Peer0)"
            "--peer1 $($node.Peer1)"
            "--device0 $($node.Device0) --device1 $($node.Device1)"
            "--gid0 3 --gid1 3"
            "--control-port0 $ControlPort0"
            "--control-port1 $ControlPort1"
            "--warmup $Warmup"
            "--iterations $Iterations"
            "--operations-per-graph $OperationsPerGraph"
            "--allreduce-protocol $AllreduceProtocol"
            "--graph-kernel $GraphKernel"
            "--wire-schedule $WireSchedule"
            "--timing-mode $TimingMode"
            "--fixed-q $FixedQ"
            "--maximum-q $MaximumQ"
            "--graph-submit-cpu $SubmitCpu"
            "--graph-progress-cpu $ProgressCpu"
            $graphValidationArgs
            $performanceArgs
            ">/dev/null"
        ) -join " "

        $exitCode = Invoke-NodeSsh -Node $node -Command $command
        if ($exitCode -ne 0) {
            throw "failed to launch graph probe rank $($node.Rank)"
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

    if ($MultiGraphValidation) {
        $expected = ([long]$Warmup + [long]$Iterations) * `
            ([long]$GraphAOperations + [long]$GraphBOperations)
        $expectedCapturedNodes = [long]$GraphAOperations + `
            [long]$GraphBOperations
        $expectedLaunches = ([long]$Warmup + [long]$Iterations) * 2
        $expectedMode = "multi"
        $expectedGraphAOperations = [long]$GraphAOperations
        $expectedGraphBOperations = [long]$GraphBOperations
    }
    else {
        $expected = ([long]$Warmup + [long]$Iterations) * `
            [long]$OperationsPerGraph
        $expectedCapturedNodes = [long]$OperationsPerGraph
        $expectedLaunches = [long]$Warmup + [long]$Iterations
        $expectedMode = "single"
        $expectedGraphAOperations = [long]$OperationsPerGraph
        $expectedGraphBOperations = 0
    }
    $expectedQHistogram = @(
        for ($q = 1; $q -le 512; $q++) {
            0L
        }
    )
    if ($MixedQValidation) {
        $graphAPattern = @(1, 4, 6)
        for ($operation = 0; $operation -lt $GraphAOperations; $operation++) {
            $q = $graphAPattern[$operation % $graphAPattern.Count]
            $expectedQHistogram[$q - 1]++
        }
        $graphBPattern = @(
            for ($q = 1; $q -le [Math]::Min($MaximumQ, 40); $q++) {
                $q
            }
            foreach ($q in @(48, 72, 144, 512)) {
                if ($q -le $MaximumQ) {
                    $q
                }
            }
        )
        if ($graphBPattern[-1] -ne $MaximumQ) {
            $graphBPattern += $MaximumQ
        }
        for ($operation = 0; $operation -lt $GraphBOperations; $operation++) {
            $q = $graphBPattern[$operation % $graphBPattern.Count]
            $expectedQHistogram[$q - 1]++
        }
        $expectedSessionCapacityBytes = [long]$MaximumQ * 6144L * 2L
        $expectedMixedQ = "true"
    }
    else {
        $expectedQHistogram[$FixedQ - 1] = $expectedCapturedNodes
        $expectedSessionCapacityBytes = [long]$FixedQ * 6144L * 2L
        $expectedMixedQ = "false"
    }
    $expectedActiveBytesPerGraphCycle = 0L
    for ($q = 1; $q -le 512; $q++) {
        $expectedActiveBytesPerGraphCycle += `
            $expectedQHistogram[$q - 1] * [long]$q * 6144L * 2L
    }
    $expectedValidatedActiveBytesTotal = `
        $expectedActiveBytesPerGraphCycle * `
        ([long]$Warmup + [long]$Iterations)
    $expectedQ48 = $expectedQHistogram[47]
    $expectedQ40 = $expectedQHistogram[39]
    $expectedQ72 = $expectedQHistogram[71]
    $expectedQ144 = $expectedQHistogram[143]
    $expectedQ512 = $expectedQHistogram[511]
    if ($GraphKernel -eq "fused") {
        $expectedKernelFusedNodes = $expectedCapturedNodes
        $expectedKernelSplitNodes = 0L
    }
    elseif ($GraphKernel -eq "split_64k") {
        $expectedKernelFusedNodes = 0L
        $expectedKernelSplitNodes = $expectedCapturedNodes
    }
    else {
        $expectedKernelFusedNodes = 0L
        for ($q = 1; $q -le 6; $q++) {
            $expectedKernelFusedNodes += $expectedQHistogram[$q - 1]
        }
        $expectedKernelSplitNodes = `
            $expectedCapturedNodes - $expectedKernelFusedNodes
    }
    $expectedTransportKernelPath = $selectedTransportKernelPath
    if ($AllreduceProtocol -eq "two_slot_deferred_ack") {
        $expectedPayloadSlots = 2
        $expectedSlotReuse = if ($expected -ge 3) { "true" } else { "false" }
    }
    else {
        $expectedPayloadSlots = 1
        $expectedSlotReuse = "false"
    }
    if ($TimingMode -eq "isolated") {
        $expectedTimingScope = "device_output_ready_single_replay"
        $expectedDeviceGateMetric = `
            "p95_device_output_ready_us_per_graph"
    }
    else {
        $expectedTimingScope = `
            "device_output_ready_replay_throughput"
        $expectedDeviceGateMetric = `
            "mean_device_output_ready_us_per_collective"
    }

    foreach ($node in $nodes) {
        $name = "spark-tp4-graph-q1-r$($node.Rank)"
        $state = Get-ContainerState -Node $node
        $log = (& ssh -o BatchMode=yes -o ConnectTimeout=8 $node.Target `
            "docker logs $name 2>&1")
        $result = @($log | Where-Object { $_ -like "TP4_GRAPH_Q1*" })
        Write-Output "rank=$($node.Rank) state=$state"
        $result | Write-Output

        $gate = $result -join " "
        if ($state -ne "exited:0" `
            -or $gate -notmatch "publisher=device" `
            -or $gate -notmatch "mode=$expectedMode(?:\s|$)" `
            -or $gate -notmatch "mixed_q=$expectedMixedQ(?:\s|$)" `
            -or $gate -notmatch "fixed_q=$FixedQ(?:\s|$)" `
            -or $gate -notmatch "maximum_q=$MaximumQ(?:\s|$)" `
            -or $gate -notmatch "session_capacity_bytes=$expectedSessionCapacityBytes(?:\s|$)" `
            -or $gate -notmatch "allreduce_protocol=$AllreduceProtocol(?:\s|$)" `
            -or $gate -notmatch "wire_schedule=$WireSchedule(?:\s|$)" `
            -or $gate -notmatch "transport_kernel_path=$expectedTransportKernelPath(?:\s|$)" `
            -or $gate -notmatch "graph_kernel=$GraphKernel(?:\s|$)" `
            -or $gate -notmatch "kernel_fused_nodes=$expectedKernelFusedNodes(?:\s|$)" `
            -or $gate -notmatch "kernel_split_64k_nodes=$expectedKernelSplitNodes(?:\s|$)" `
            -or $gate -notmatch "payload_slots=$expectedPayloadSlots(?:\s|$)" `
            -or $gate -notmatch "slot_reuse_exercised=$expectedSlotReuse(?:\s|$)" `
            -or $gate -notmatch "timing_mode=$TimingMode(?:\s|$)" `
            -or $gate -notmatch "graph_a_operations=$expectedGraphAOperations(?:\s|$)" `
            -or $gate -notmatch "graph_b_operations=$expectedGraphBOperations(?:\s|$)" `
            -or $gate -notmatch "q1_nodes=$($expectedQHistogram[0])(?:\s|$)" `
            -or $gate -notmatch "q2_nodes=$($expectedQHistogram[1])(?:\s|$)" `
            -or $gate -notmatch "q3_nodes=$($expectedQHistogram[2])(?:\s|$)" `
            -or $gate -notmatch "q4_nodes=$($expectedQHistogram[3])(?:\s|$)" `
            -or $gate -notmatch "q5_nodes=$($expectedQHistogram[4])(?:\s|$)" `
            -or $gate -notmatch "q6_nodes=$($expectedQHistogram[5])(?:\s|$)" `
            -or $gate -notmatch "q40_nodes=$expectedQ40(?:\s|$)" `
            -or $gate -notmatch "q48_nodes=$expectedQ48(?:\s|$)" `
            -or $gate -notmatch "q72_nodes=$expectedQ72(?:\s|$)" `
            -or $gate -notmatch "q144_nodes=$expectedQ144(?:\s|$)" `
            -or $gate -notmatch "q512_nodes=$expectedQ512(?:\s|$)" `
            -or $gate -notmatch "active_bytes_per_graph_cycle=$expectedActiveBytesPerGraphCycle(?:\s|$)" `
            -or $gate -notmatch "validated_active_bytes_total=$expectedValidatedActiveBytesTotal(?:\s|$)" `
            -or $gate -notmatch "graph_launches=$expectedLaunches(?:\s|$)" `
            -or $gate -notmatch "input_updates=$expectedLaunches(?:\s|$)" `
            -or $gate -notmatch "captured_nodes=$expectedCapturedNodes(?:\s|$)" `
            -or $gate -notmatch "submit_affinity_verified=true(?:\s|$)" `
            -or $gate -notmatch "progress_affinity_verified=true(?:\s|$)" `
            -or $gate -notmatch "protocol_status_match=true(?:\s|$)" `
            -or $gate -notmatch "timing_scope=$expectedTimingScope(?:\s|$)" `
            -or $gate -notmatch "device_gate_metric=$expectedDeviceGateMetric(?:\s|$)" `
            -or $gate -notmatch "graph_submit_cpu=$SubmitCpu(?:\s|$)" `
            -or $gate -notmatch "graph_progress_cpu=$ProgressCpu(?:\s|$)" `
            -or $gate -notmatch "pre_replay_capture_valid=true(?:\s|$)" `
            -or $gate -notmatch "published=$expected(?:\s|$)" `
            -or $gate -notmatch "consumed=$expected(?:\s|$)" `
            -or $gate -notmatch "completed=$expected(?:\s|$)" `
            -or $gate -notmatch "overflow=0(?:\s|$)" `
            -or $gate -notmatch "mismatched_elements=0(?:\s|$)" `
            -or $gate -notmatch "monotonic_sequences=true(?:\s|$)" `
            -or $gate -notmatch "passed=true(?:\s|$)") {
            $failed = $true
            Write-Output "rank=$($node.Rank) failure_log:"
            $log | Select-Object -Last 40 | Write-Output
        }
        if ($TimingMode -eq "isolated" -and `
            ($gate -notmatch "timing_samples=$Iterations(?:\s|$)" `
                -or $gate -notmatch "device_output_ready_us_per_graph_min=[0-9.eE+-]+(?:\s|$)" `
                -or $gate -notmatch "device_output_ready_us_per_graph_p50=[0-9.eE+-]+(?:\s|$)" `
                -or $gate -notmatch "device_output_ready_us_per_graph_p95=[0-9.eE+-]+(?:\s|$)")) {
            $failed = $true
            Write-Output "rank=$($node.Rank) isolated timing receipt gate failed"
        }
        if ($MultiGraphValidation `
            -and $gate -notmatch "post_replay_capture_rejected=true(?:\s|$)") {
            $failed = $true
            Write-Output "rank=$($node.Rank) post-replay capture rejection gate failed"
        }
    }
}
finally {
    if (-not $KeepContainers) {
        foreach ($node in $nodes) {
            $name = "spark-tp4-graph-q1-r$($node.Rank)"
            Invoke-NodeSsh -Node $node `
                -Command "docker rm -f $name >/dev/null 2>&1 || true" | Out-Null
        }
    }
}

if ($timedOut) {
    throw "TP4 graph Q1 probe exceeded the $WatchdogSeconds-second watchdog"
}
if ($failed) {
    throw "one or more TP4 graph Q1 ranks failed"
}

Write-Output "gate=pass ranks=4 mode=$expectedMode protocol=$AllreduceProtocol wire_schedule=$WireSchedule transport_kernel_path=$expectedTransportKernelPath graph_kernel=$GraphKernel timing_mode=$TimingMode expected_sequence=$expected input_updates=$expectedLaunches"
