# Resolve local RDMA devices for algorithm slots rank XOR 1 and rank XOR 3.
function Resolve-Tp4DeviceMapping {
    param([string]$Preset, [string[]]$Device0, [string[]]$Device1)
    if ($Preset) {
        if ($Device0.Count -or $Device1.Count) {
            throw "Use -DevicePreset or explicit -Device0 and -Device1 arrays, not both"
        }
        if ($Preset -ne "documented-cycle") { throw "Unknown device preset: $Preset" }
        $Device0 = @("rocep1s0f0", "rocep1s0f1", "rocep1s0f0", "rocep1s0f1")
        $Device1 = @("rocep1s0f1", "rocep1s0f0", "rocep1s0f1", "rocep1s0f0")
    }
    if ($Device0.Count -ne 4 -or $Device1.Count -ne 4) {
        throw "Set -DevicePreset documented-cycle or provide four rank-ordered values for both -Device0 and -Device1"
    }
    for ($rank = 0; $rank -lt 4; $rank++) {
        foreach ($device in @($Device0[$rank], $Device1[$rank])) {
            if ($device -notmatch '^[A-Za-z0-9_][A-Za-z0-9_.-]*$') {
                throw "Invalid RDMA device name at rank $rank"
            }
        }
        if ($Device0[$rank] -eq $Device1[$rank]) {
            throw "Device0 and Device1 must differ at rank $rank"
        }
        [pscustomobject]@{ Device0 = $Device0[$rank]; Device1 = $Device1[$rank] }
    }
}
