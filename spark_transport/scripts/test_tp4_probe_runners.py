"""Offline checks for the four-rank probe runners; no SSH, Docker or CUDA.

Each runner is parsed by PowerShell's own parser, refused without a
topology before any remote command, and required to keep the container
lifecycle contract: a watchdog poll instead of a fixed sleep, forced
owned-container removal in a finally block, and rank mapping over filtered entries.
"""

from __future__ import annotations

import os
import hashlib
from pathlib import Path
import shutil
import subprocess

import pytest


SCRIPTS = Path(__file__).resolve().parent
POWERSHELL = shutil.which("pwsh") or shutil.which("powershell")
RUNNERS = (
    "run_tp4_probe.ps1",
    "run_tp4_tensor_probe.ps1",
    "run_tp4_vocab_allgather_probe.ps1",
    "run_tp4_vocab_graph_stream_switch_probe.ps1",
    "run_tp4_graph_q1_probe.ps1",
)
CPUSET_RUNNERS = (
    "run_tp4_vocab_graph_stream_switch_probe.ps1",
    "run_tp4_graph_q1_probe.ps1",
    "run_tp4_tiled_prefill_probe.ps1",
)


def _powershell(*arguments: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    if POWERSHELL is None:
        pytest.skip("PowerShell is unavailable")
    return subprocess.run(
        [POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", *arguments],
        check=False, capture_output=True, encoding="utf-8", env=env, timeout=60,
    )


@pytest.mark.parametrize("name", RUNNERS + ("run_tp4_tiled_prefill_probe.ps1",))
def test_runner_parses_without_errors(name: str) -> None:
    script = SCRIPTS / name
    command = (
        "$errors = $null; "
        "[System.Management.Automation.Language.Parser]::ParseFile("
        f"'{script.as_posix()}', [ref]$null, [ref]$errors) | Out-Null; "
        "if ($errors.Count -ne 0) { $errors | ForEach-Object { $_.Message }; exit 1 }"
    )
    result = _powershell("-Command", command, env=os.environ.copy())
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("name", RUNNERS)
def test_runner_refuses_to_run_without_a_topology(name: str) -> None:
    environment = os.environ.copy()
    environment.pop("SPARKRING_TARGETS", None)
    environment.pop("SPARKRING_RANK_HOSTS", None)
    result = _powershell("-File", str(SCRIPTS / name), env=environment)
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "SPARKRING_TARGETS" in output
    # The usage text names SSH targets; an attempted connection would surface
    # ssh's own diagnostics or a launch failure instead.
    for marker in ("BatchMode", "Connection", "failed to launch", "preflight="):
        assert marker not in output


@pytest.mark.parametrize("name", RUNNERS)
def test_runner_keeps_the_container_lifecycle_contract(name: str) -> None:
    source = (SCRIPTS / name).read_text(encoding="utf-8")
    assert "Start-Sleep -Seconds 5" not in source
    assert "$deadline = [DateTime]::UtcNow.AddSeconds($WatchdogSeconds + 15)" in source
    assert "--signal=TERM --kill-after=5s ${WatchdogSeconds}s" in source
    finally_block = source[source.index("finally {"):]
    assert "docker rm -f $name >/dev/null 2>&1 || true" in finally_block
    assert "$Targets = @($Targets | Where-Object { $_ })" in source
    assert "$RankHosts = @($RankHosts | Where-Object { $_ })" in source
    assert source.index("$Targets = @($Targets | Where-Object { $_ })") < source.index("Target = $Targets[0]")


@pytest.mark.parametrize("name", CPUSET_RUNNERS)
def test_pinned_cpus_are_validated_against_the_cpuset(name: str) -> None:
    source = (SCRIPTS / name).read_text(encoding="utf-8")
    assert "function Test-CpuInSet" in source
    assert "is outside -CpuSet" in source


def test_cpuset_membership_helper_accepts_lists_and_ranges() -> None:
    source = (SCRIPTS / "run_tp4_graph_q1_probe.ps1").read_text(encoding="utf-8")
    start = source.index("function Test-CpuInSet")
    end = source.index("# The container is confined to CpuSet")
    helper = source[start:end]
    checks = (
        "if (-not (Test-CpuInSet -Set '10,11' -Cpu 11)) { exit 11 }; "
        "if (-not (Test-CpuInSet -Set '0-7' -Cpu 7)) { exit 12 }; "
        "if (Test-CpuInSet -Set '0-7' -Cpu 8) { exit 13 }; "
        "if (Test-CpuInSet -Set '10,11' -Cpu 1) { exit 14 }; "
        "if (-not (Test-CpuInSet -Set '0-3,10' -Cpu 10)) { exit 15 }; exit 0"
    )
    result = _powershell("-Command", helper + "\n" + checks, env=os.environ.copy())
    assert result.returncode == 0, result.stdout + result.stderr


def test_tiered_gate_counts_fused_nodes_by_the_64k_boundary() -> None:
    source = (SCRIPTS / "run_tp4_graph_q1_probe.ps1").read_text(encoding="utf-8")
    assert "$tieredFusedMaximumBytes = 64L * 1024L" in source
    assert "-le $tieredFusedMaximumBytes" in source
    # Q5 (61,440 bytes) stays fused and Q6 (73,728 bytes) is split, matching
    # tp4_graph_kernel_uses_split(tiered, 64 KiB + 1).
    assert 5 * 6144 * 2 <= 64 * 1024 < 6 * 6144 * 2


@pytest.mark.parametrize("name", RUNNERS[:3])
def test_transport_failure_cleans_attempted_unique_names_only(name):
    command = r"""
$env:SPARKRING_TARGETS='host0,host1,host2,host3'
$env:SPARKRING_RANK_HOSTS='host0,host1,host2,host3'
function global:ssh {
    $cmd=$args[-1]
    Write-Host "COMMAND=$cmd"
    $global:LASTEXITCODE=0
    if ($cmd -match 'docker run .*?-r1 ') { $global:LASTEXITCODE=255 }
}
for ($i=0; $i -lt 2; $i++) {
    try { & 'SCRIPT' -Image 'fake-image' -DevicePreset documented-cycle } catch { Write-Host "EXPECTED=$($_.Exception.Message)" }
}
""".replace("SCRIPT", (SCRIPTS / name).as_posix())
    result = _powershell("-Command", command, env=os.environ.copy())
    assert result.returncode == 0, result.stdout + result.stderr
    commands = [line.removeprefix("COMMAND=") for line in result.stdout.splitlines() if line.startswith("COMMAND=")]
    launches = [line for line in commands if "docker run" in line]
    removals = [line for line in commands if "docker rm" in line]
    assert len(launches) == 4, result.stdout
    assert len(removals) == 4, result.stdout
    assert all("docker rm" not in line for line in launches)
    names = [line.split("--name ")[1].split()[0] for line in launches]
    assert len(set(names)) == 4
    assert all(name in removal for name, removal in zip(names, removals))
    assert all("-r2" not in line and "-r3" not in line for line in removals)


def test_failed_stage_copy_cleans_only_created_unique_stage():
    command = r"""
$env:SPARKRING_TARGETS='host0,host1,host2,host3'
$env:SPARKRING_RANK_HOSTS='host0,host1,host2,host3'
function global:ssh { Write-Host "COMMAND=$($args[-1])"; $global:LASTEXITCODE=0 }
function global:scp { $global:LASTEXITCODE=9 }
for ($i=0; $i -lt 2; $i++) {
    try { & 'SCRIPT' -Image 'fake-image' -DevicePreset documented-cycle } catch { Write-Host "EXPECTED=$($_.Exception.Message)" }
}
""".replace("SCRIPT", (SCRIPTS / "run_tp4_vocab_graph_stream_switch_probe.ps1").as_posix())
    result = _powershell("-Command", command, env=os.environ.copy())
    assert result.returncode == 0, result.stdout + result.stderr
    commands = [line.removeprefix("COMMAND=") for line in result.stdout.splitlines() if line.startswith("COMMAND=")]
    created = [line.removeprefix("mkdir ") for line in commands if line.startswith("mkdir ")]
    removed = [line.removeprefix("rm -rf ") for line in commands if line.startswith("rm -rf ")]
    assert len(created) == 2 and len(set(created)) == 2, result.stdout
    assert created == removed
    assert not any("docker rm" in line for line in commands)


@pytest.mark.parametrize("name", RUNNERS[:3])
@pytest.mark.parametrize("mapping,devices0,devices1", [
    ("-DevicePreset documented-cycle", ["rocep1s0f0", "rocep1s0f1"] * 2, ["rocep1s0f1", "rocep1s0f0"] * 2),
    ("-Device0 a,a,a,a -Device1 b,b,b,b", ["a"] * 4, ["b"] * 4),
])
def test_actual_launch_arguments_match_explicit_device_mapping(name, mapping, devices0, devices1):
    command = r"""
$env:SPARKRING_TARGETS='host0,host1,host2,host3'
$env:SPARKRING_RANK_HOSTS='host0,host1,host2,host3'
function global:ssh {
    $cmd=$args[-1]; Write-Host "COMMAND=$cmd"; $global:LASTEXITCODE=0
    if ($cmd -match 'docker run .*?-r3 ') { $global:LASTEXITCODE=255 }
}
try { & 'SCRIPT' -Image fake-image MAPPING } catch { Write-Host "EXPECTED=$($_.Exception.Message)" }
""".replace("SCRIPT", (SCRIPTS / name).as_posix()).replace("MAPPING", mapping)
    result = _powershell("-Command", command, env=os.environ.copy())
    launches = [line for line in result.stdout.splitlines() if line.startswith("COMMAND=") and "docker run" in line]
    assert len(launches) == 4, result.stdout + result.stderr
    for rank, line in enumerate(launches):
        assert f"--rank {rank} " in line
        assert f"--peer0 host{rank ^ 1} " in line
        assert f"--peer1 host{rank ^ 3} " in line
        assert f"--device0 {devices0[rank]} --device1 {devices1[rank]}" in line


@pytest.mark.parametrize("name", RUNNERS[:3] + (
    "run_tp4_vocab_graph_probe.ps1", "run_tp4_vocab_graph_stream_switch_probe.ps1",
))
@pytest.mark.parametrize("mapping", ["", "-Device0 a,a,a,a", "-Device0 a,a,a -Device1 b,b,b", "-Device0 a,a,a,a -Device1 a,b,b,b", "-Device0 'a;bad',a,a,a -Device1 b,b,b,b", "-DevicePreset documented-cycle -Device0 a,a,a,a -Device1 b,b,b,b"])
def test_invalid_device_mapping_fails_before_ssh(name, mapping):
    command = r"""
$env:SPARKRING_TARGETS='host0,host1,host2,host3'
$env:SPARKRING_RANK_HOSTS='host0,host1,host2,host3'
function global:ssh { Write-Host 'UNEXPECTED_SSH'; throw 'must not connect' }
try { & 'SCRIPT' -Image fake-image MAPPING; exit 2 } catch { Write-Host "EXPECTED=$($_.Exception.Message)" }
""".replace("SCRIPT", (SCRIPTS / name).as_posix()).replace("MAPPING", mapping)
    result = _powershell("-Command", command, env=os.environ.copy())
    assert result.returncode == 0, result.stdout + result.stderr
    assert "EXPECTED=" in result.stdout
    assert "UNEXPECTED_SSH" not in result.stdout


@pytest.mark.parametrize("name", [
    "run_tp4_numerical_audit.ps1", "run_tp4_vocab_graph_probe.ps1",
])
@pytest.mark.parametrize("embedded_empty", [False, True])
def test_filtered_ranks_and_lost_launch_reply_cleanup(name, embedded_empty):
    command = r"""
function global:ssh {
    $cmd=$args[-1]
    Write-Host "COMMAND=$cmd"
    Write-Host "TARGET=$($args[-2])"
    $global:LASTEXITCODE=0
    if ($cmd -match 'sha256sum') { Write-Output ('a' * 64 + '  /probe') }
    if ($cmd -match 'docker run .*?-r1 ') { $global:LASTEXITCODE=255 }
}
try {
    & 'SCRIPT' -Image fake-image `
        -Targets @('host0','','host1','host2','host3') `
        -RankHosts @('peer0','','peer1','peer2','peer3')
    exit 2
} catch { Write-Host "EXPECTED=$($_.Exception.Message)" }
""".replace("SCRIPT", (SCRIPTS / name).as_posix())
    if not embedded_empty:
        command = command.replace(",''", "")
    if name == "run_tp4_vocab_graph_probe.ps1":
        command = command.replace("-Image fake-image", "-Image fake-image -DevicePreset documented-cycle")
    result = _powershell("-Command", command, env=os.environ.copy())
    assert result.returncode == 0, result.stdout + result.stderr
    lines = result.stdout.splitlines()
    calls = [
        (line.removeprefix("COMMAND="), lines[index + 1].removeprefix("TARGET="))
        for index, line in enumerate(lines) if line.startswith("COMMAND=")
    ]
    launches = [(command, target) for command, target in calls if "docker run" in command]
    removals = [(command, target) for command, target in calls if "docker rm" in command]
    assert len(launches) == 2, result.stdout + result.stderr
    assert [target for _, target in launches] == ["host0", "host1"]
    assert len(removals) == 2, result.stdout
    for (launch, target), (removal, cleanup_target) in zip(launches, removals):
        owned_name = launch.split("--name ")[1].split()[0]
        assert removal == f"docker rm -f {owned_name} >/dev/null 2>&1 || true"
        assert cleanup_target == target
    if name == "run_tp4_vocab_graph_probe.ps1":
        assert "--peer0 peer1 --peer1 peer3" in launches[0][0]
        assert "--peer0 peer0 --peer1 peer2" in launches[1][0]
    else:
        assert all("MASTER_ADDR=peer0 " in command for command, _ in launches)


@pytest.mark.parametrize("name", [
    "run_tp4_vocab_graph_probe.ps1", "run_tp4_vocab_graph_stream_switch_probe.ps1",
])
@pytest.mark.parametrize("mapping,devices0,devices1", [
    ("-DevicePreset documented-cycle", ["rocep1s0f0", "rocep1s0f1"] * 2, ["rocep1s0f1", "rocep1s0f0"] * 2),
    ("-Device0 a,b,c,d -Device1 e,f,g,h", list("abcd"), list("efgh")),
])
def test_vocabulary_graph_launches_use_selected_device_mapping(name, mapping, devices0, devices1):
    command = r"""
$env:SPARKRING_TARGETS='host0,host1,host2,host3'
$env:SPARKRING_RANK_HOSTS='peer0,peer1,peer2,peer3'
function global:scp { $global:LASTEXITCODE=0 }
function global:ssh {
    $cmd=$args[-1]; Write-Host "COMMAND=$cmd"; $global:LASTEXITCODE=0
    if ($cmd -match 'sha256sum') {
        if ($cmd -match 'spark_tp4_query_contract.py') {
            foreach ($match in [regex]::Matches(($cmd -split 'sha256sum ')[1], "'([^']+)'")) {
                $path=$match.Groups[1].Value
                $hash=switch -Wildcard ($path) {
                    '*/probe.py' { 'PROBE_HASH' }
                    '*/adapter.py' { 'ADAPTER_HASH' }
                    '*/spark_tp4_query_contract.py' { 'CONTRACT_HASH' }
                    default { '8657306b1807c4111687f5f128725ff6726be2c8c8c4bebc67b520a730c0acca' }
                }
                Write-Output "$hash  $path"
            }
        } else { Write-Output 'identical fixture hashes' }
    }
    if ($cmd -match 'docker run .*?-r3 ') { $global:LASTEXITCODE=255 }
}
try { & 'SCRIPT' -Image fake-image MAPPING } catch { Write-Host "EXPECTED=$($_.Exception.Message)" }
""".replace("SCRIPT", (SCRIPTS / name).as_posix()).replace("MAPPING", mapping)
    integration = SCRIPTS.parent / "integrations/vllm"
    for token, filename in (
        ("PROBE_HASH", "probe_vocab_graph_stream_switch.py"),
        ("ADAPTER_HASH", "spark_tp4_vocab_allgather_backend.py"),
        ("CONTRACT_HASH", "spark_tp4_query_contract.py"),
    ):
        command = command.replace(token, hashlib.sha256((integration / filename).read_bytes()).hexdigest())
    result = _powershell("-Command", command, env=os.environ.copy())
    launches = [line for line in result.stdout.splitlines() if line.startswith("COMMAND=") and "docker run" in line]
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(launches) == 4, result.stdout + result.stderr
    for rank, launch in enumerate(launches):
        if "stream_switch" in name:
            assert f"-e SPARK_TP4_DEVICE0={devices0[rank]} " in launch
            assert f"-e SPARK_TP4_DEVICE1={devices1[rank]} " in launch
            assert f"-e SPARK_TP4_PEER0=peer{rank ^ 1} " in launch
            assert f"-e SPARK_TP4_PEER1=peer{rank ^ 3} " in launch
        else:
            assert f"--device0 {devices0[rank]} --device1 {devices1[rank]}" in launch
            assert f"--peer0 peer{rank ^ 1} --peer1 peer{rank ^ 3}" in launch
    if "stream_switch" in name:
        assert "model_container=glm52-trace model_container_running=false" in result.stdout
        assert "model_down=true" not in result.stdout
