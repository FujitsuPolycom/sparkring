param(
    [ValidateRange(0, 10000000)]
    [int]$Warmup = 2,

    [ValidateRange(1, 10000000)]
    [int]$Iterations = 100,

    [ValidateRange(1024, 65534)]
    [int]$ControlPort0 = 9892,

    [ValidateRange(1024, 65534)]
    [int]$ControlPort1 = 9893,

    [ValidatePattern("^[0-9,-]+$")]
    [string]$CpuSet = "10,13",

    [ValidateRange(0, 4095)]
    [int]$SubmitCpu = 10,

    [ValidateRange(0, 4095)]
    [int]$ProgressCpu = 13,

    [ValidateSet(0, 256, 512)]
    [int]$CombineDimension = 0,

    [ValidateSet(0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40)]
    [int]$SingleQ = 0,

    [ValidateSet("adaptive-yield", "dedicated-spin")]
    [string]$PollPolicy = "adaptive-yield",

    [ValidateRange(10, 3600)]
    [int]$WatchdogSeconds = 180,

    [string]$ProbeBinary =
        "/tmp/spark_tp4_dcp_graph_probe-sparkring-v21",
    [string]$Library =
        "/tmp/libspark_transport_capi-sparkring-v21.so",
    # Replace <your-vllm-image> with your locally built vLLM container tag.
    [string]$Image = "<your-vllm-image>",
    # Rank SSH targets default from SPARKRING_TARGETS, a comma-separated
    # user@host list in rank order 0-3; -RankNTarget overrides per rank.
    [string]$Rank0Target = ($env:SPARKRING_TARGETS -split ",")[0],
    [string]$Rank0ProxyJump = "",
    [string]$Rank1Target = ($env:SPARKRING_TARGETS -split ",")[1],
    [string]$Rank2Target = ($env:SPARKRING_TARGETS -split ",")[2],
    [string]$Rank3Target = ($env:SPARKRING_TARGETS -split ",")[3],
    [string]$Rank3ProxyJump = "",
    [switch]$PhaseTrace,
    [switch]$KeepContainers
)

$ErrorActionPreference = "Stop"

if ($ControlPort0 -eq $ControlPort1) {
    throw "ControlPort0 and ControlPort1 must differ"
}
if ($SubmitCpu -eq $ProgressCpu) {
    throw "SubmitCpu and ProgressCpu must differ"
}
if (@($Rank0Target, $Rank1Target, $Rank2Target, $Rank3Target) |
        Where-Object { [string]::IsNullOrWhiteSpace($_) }) {
    throw ("missing rank SSH targets: set SPARKRING_TARGETS to a " +
        "comma-separated user@host list for ranks 0-3, or pass " +
        "-Rank0Target..-Rank3Target explicitly")
}

# Peer0/Peer1 are RFC 5737 placeholder addresses for the point-to-point
# fabric links between ring neighbours; replace them with your cluster's
# per-link fabric IPs.
$nodes = @(
    [pscustomobject]@{
        Rank = 0
        Target = $Rank0Target
        ProxyJump = $Rank0ProxyJump
        Peer0 = "192.0.2.11"
        Peer1 = "192.0.2.41"
    },
    [pscustomobject]@{
        Rank = 1
        Target = $Rank1Target
        ProxyJump = ""
        Peer0 = "192.0.2.10"
        Peer1 = "192.0.2.21"
    },
    [pscustomobject]@{
        Rank = 2
        Target = $Rank2Target
        ProxyJump = ""
        Peer0 = "192.0.2.30"
        Peer1 = "192.0.2.20"
    },
    [pscustomobject]@{
        Rank = 3
        Target = $Rank3Target
        ProxyJump = $Rank3ProxyJump
        Peer0 = "192.0.2.31"
        Peer1 = "192.0.2.40"
    }
)

function Get-NodeSshArguments {
    param(
        [Parameter(Mandatory)]
        [pscustomobject]$Node
    )

    $arguments = @(
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=8",
        "-o", "StrictHostKeyChecking=accept-new"
    )
    if (-not [string]::IsNullOrWhiteSpace($Node.ProxyJump)) {
        $arguments += @("-J", $Node.ProxyJump)
    }
    $arguments += $Node.Target
    return $arguments
}

function Invoke-NodeSsh {
    param(
        [Parameter(Mandatory)]
        [pscustomobject]$Node,

        [Parameter(Mandatory)]
        [string]$Command
    )

    $arguments = @(Get-NodeSshArguments -Node $Node)
    & ssh @arguments $Command
    return $LASTEXITCODE
}

function Get-ContainerState {
    param(
        [Parameter(Mandatory)]
        [pscustomobject]$Node
    )

    $name = "spark-tp4-dcp-graph-r$($Node.Rank)"
    $arguments = @(Get-NodeSshArguments -Node $Node)
    $state = (& ssh @arguments `
        "docker inspect $name --format '{{.State.Status}}:{{.State.ExitCode}}'" `
        2>$null)
    if ($LASTEXITCODE -ne 0) {
        return "missing"
    }
    return $state.Trim()
}

$artifactHashes = @()
foreach ($node in $nodes) {
    $sshArguments = @(Get-NodeSshArguments -Node $node)
    $runningModel = (& ssh @sshArguments `
        "docker ps --filter name=^/glm52-trace$ --format '{{.Names}}'")
    if ($LASTEXITCODE -ne 0) {
        throw "failed to inspect running containers on rank $($node.Rank)"
    }
    if (($runningModel -join "`n").Trim() -eq "glm52-trace") {
        throw "rank $($node.Rank) still runs glm52-trace; the DCP graph probe requires the model-down memory window"
    }

    $hash = (& ssh @sshArguments `
        "test -x '$ProbeBinary' && test -f '$Library' && sha256sum '$ProbeBinary' '$Library'")
    if ($LASTEXITCODE -ne 0) {
        throw "rank $($node.Rank) is missing a staged DCP graph artifact"
    }
    $artifactHashes += ,(($hash -join "`n").Trim())
}
if (@($artifactHashes | Sort-Object -Unique).Count -ne 1) {
    throw "DCP graph artifact SHA-256 values differ across ranks"
}
Write-Output "preflight=pass model_down=true identical_sha256=true"
Write-Output $artifactHashes[0]

$failed = $false
$timedOut = $false
$traceEnvironment = if ($PhaseTrace) {
    "SPARK_TRANSPORT_TRACE=1"
}
else {
    "-u SPARK_TRANSPORT_TRACE"
}
try {
    foreach ($node in $nodes) {
        $name = "spark-tp4-dcp-graph-r$($node.Rank)"
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
            "env $traceEnvironment"
            "SPARK_TP4_DCP_GRAPH_POLL_POLICY=$PollPolicy"
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
            "--warmup $Warmup"
            "--iterations $Iterations"
            "--combine-dimension $CombineDimension"
            "--single-q $SingleQ"
            ">/dev/null"
        ) -join " "

        $exitCode = Invoke-NodeSsh -Node $node -Command $command
        if ($exitCode -ne 0) {
            throw "failed to launch DCP graph rank $($node.Rank)"
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

    $capturedNodes = if ($SingleQ -eq 0) { 28L } else { 2L }
    $capturedQueryNodes = if ($SingleQ -eq 0) { 14L } else { 1L }
    $capturedCombineNodes = $capturedQueryNodes
    $expected = ([long]$Warmup + [long]$Iterations) * $capturedNodes
    foreach ($node in $nodes) {
        $name = "spark-tp4-dcp-graph-r$($node.Rank)"
        $state = Get-ContainerState -Node $node
        $sshArguments = @(Get-NodeSshArguments -Node $node)
        $log = (& ssh @sshArguments "docker logs $name 2>&1")
        $result = @($log | Where-Object { $_ -like "TP4_DCP_GRAPH*" })
        $phaseLines = @($log | Where-Object { $_ -like "DCP_PHASE*" })
        Write-Output "rank=$($node.Rank) state=$state"
        $phaseLines | Write-Output
        $result | Write-Output

        $gate = $result -join " "
        if ($state -ne "exited:0" `
            -or $gate -notmatch "captured_nodes=$capturedNodes(?:\s|$)" `
            -or $gate -notmatch "captured_query_nodes=$capturedQueryNodes(?:\s|$)" `
            -or $gate -notmatch "captured_combine_nodes=$capturedCombineNodes(?:\s|$)" `
            -or $gate -notmatch "published=$expected(?:\s|$)" `
            -or $gate -notmatch "consumed=$expected(?:\s|$)" `
            -or $gate -notmatch "completed=$expected(?:\s|$)" `
            -or $gate -notmatch "overflow=0(?:\s|$)" `
            -or $gate -notmatch "submit_cpu=$SubmitCpu(?:\s|$)" `
            -or $gate -notmatch "progress_cpu=$ProgressCpu(?:\s|$)" `
            -or $gate -notmatch (
                "combine_dimension=" +
                $(if ($CombineDimension -eq 0) {
                    "alternate"
                } else {
                    "$CombineDimension"
                }) +
                "(?:\s|$)"
            ) `
            -or $gate -notmatch (
                "single_q=" +
                $(if ($SingleQ -eq 0) {
                    "all"
                } else {
                    "$SingleQ"
                }) +
                "(?:\s|$)"
            ) `
            -or $gate -notmatch "poll_policy=$PollPolicy(?:\s|$)" `
            -or $gate -notmatch (
                "dedicated_spin=" +
                $(if ($PollPolicy -eq "dedicated-spin") {
                    "true"
                } else {
                    "false"
                }) +
                "(?:\s|$)"
            ) `
            -or $gate -notmatch "query_byte_mismatches=0(?:\s|$)" `
            -or $gate -notmatch "output_nonfinite=0(?:\s|$)" `
            -or $gate -notmatch "lse_nonfinite=0(?:\s|$)" `
            -or $gate -notmatch "passed=true(?:\s|$)") {
            $failed = $true
            Write-Output "rank=$($node.Rank) failure_log:"
            $log | Select-Object -Last 80 | Write-Output
        }
        if (
            $PhaseTrace -and
            $phaseLines.Count -ne $expected
        ) {
            $failed = $true
            Write-Output (
                "rank=$($node.Rank) phase_trace_count=" +
                "$($phaseLines.Count) expected=$expected"
            )
        }
    }
}
finally {
    if (-not $KeepContainers) {
        foreach ($node in $nodes) {
            $name = "spark-tp4-dcp-graph-r$($node.Rank)"
            Invoke-NodeSsh -Node $node `
                -Command "docker rm -f $name >/dev/null 2>&1 || true" | Out-Null
        }
    }
}

if ($timedOut) {
    throw "TP4 DCP graph probe exceeded the watchdog"
}
if ($failed) {
    throw "one or more TP4 DCP graph ranks failed"
}

Write-Output (
    "gate=pass ranks=4 buckets=" +
    $(if ($SingleQ -eq 0) { "14" } else { "1" }) +
    " captured_nodes=$capturedNodes " +
    "expected_sequence=$expected poll_policy=$PollPolicy"
)
