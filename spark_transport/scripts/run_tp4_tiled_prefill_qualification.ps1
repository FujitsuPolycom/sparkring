param(
    [ValidateSet("all", "timing", "glm", "boundary", "backpressure", "poison")]
    [string]$Suite = "all",

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

# Status: research-only. The default action is an offline plan. -Execute is an
# explicit four-rank probe request and still does not launch a serving model.
$probeRunner = Join-Path $PSScriptRoot "run_tp4_tiled_prefill_probe.ps1"
if (-not (Test-Path -LiteralPath $probeRunner -PathType Leaf)) {
    throw "missing tiled-prefill probe runner: $probeRunner"
}
$repositoryRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$qualificationModule =
    "spark_transport.experiments.tiled_prefill.qualification"

Push-Location $repositoryRoot
try {
    $planJson = (& $Python -m $qualificationModule plan --compact) -join "`n"
    if ($LASTEXITCODE -ne 0) {
        throw "failed to load the tiled-prefill qualification plan"
    }
}
finally {
    Pop-Location
}
$plan = $planJson | ConvertFrom-Json
if ($plan.runnable -ne $true -or $plan.status -ne "research-only") {
    throw "tiled-prefill matrix does not expose the native probe"
}
if ([long]$plan.geometry.registered_tile_storage_bytes_per_edge -ne 8389632L) {
    throw "tiled-prefill matrix registered storage geometry drifted"
}

$selectedArms = @($plan.arms | Where-Object {
    if ($Suite -eq "all") {
        return $true
    }
    if ($Suite -eq "poison") {
        return $_.poison -ne "none"
    }
    if ($Suite -eq "backpressure") {
        return [int]$_.credit_delay_edge -ge 0
    }
    if ($Suite -eq "glm") {
        return $_.arm_id -like "glm4096_*"
    }
    if ($Suite -eq "boundary") {
        return $_.timing_mode -eq "correctness"
    }
    return $_.poison -eq "none" -and `
        $_.arm_id -notlike "glm4096_*" -and `
        [int]$_.credit_delay_edge -lt 0 -and `
        $_.timing_mode -ne "correctness"
})
if ($selectedArms.Count -eq 0) {
    throw "suite $Suite selected no tiled-prefill arms"
}

$common = @{
    ControlPort0 = $ControlPort0
    ControlPort1 = $ControlPort1
    WatchdogSeconds = $WatchdogSeconds
    CpuSet = $CpuSet
    SubmitCpu = $SubmitCpu
    Edge0ProgressCpu = $Edge0ProgressCpu
    Edge1ProgressCpu = $Edge1ProgressCpu
    ProbeBinary = $ProbeBinary
    Image = $Image
    Python = $Python
    Targets = $Targets
    RankHosts = $RankHosts
    KeepContainers = $KeepContainers.IsPresent
}
if ($Execute) {
    $common.Execute = $true
}

$action = if ($Execute) { "execute" } else { "plan" }
foreach ($arm in $selectedArms) {
    Write-Output ("tiled_prefill_matrix_arm=$($arm.arm_id) " +
        "action=$action q=$($arm.query_rows) " +
        "timing_mode=$($arm.timing_mode) " +
        "tiles=$($arm.logical_tile_count) " +
        "tail_active_bytes=$($arm.tail_active_bytes) " +
        "credit_delay_edge=$($arm.credit_delay_edge) " +
        "poison=$($arm.poison)")
    if ($Execute) {
        & $probeRunner @common -ArmId $arm.arm_id
        if ($LASTEXITCODE -ne 0) {
            throw "tiled-prefill arm $($arm.arm_id) failed"
        }
    }
}

$state = if ($Execute) { "pass" } else { "plan" }
Write-Output ("tiled_prefill_qualification=$state suite=$Suite " +
    "arms=$($selectedArms.Count) ranks=4 " +
    "remote_contact=$($Execute.IsPresent.ToString().ToLowerInvariant()) " +
    "serving_integration=false " +
    "executor_stream_policy=single_stream_correctness_only " +
    "protocol_nodes_status=standalone-native-executor-bound " +
    "performance_claim_allowed=false " +
    "registered_tile_storage_bytes_per_edge=8389632")
