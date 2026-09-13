"""Offline ownership checks: shadow SSH before running probe launch scripts."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("script,scenario", [
    (script, scenario)
    for script in ("run_tp4_numerical_audit.ps1", "run_tp4_vocab_graph_probe.ps1")
    for scenario in ("success", "launch_failure", "keep", "model_running", "multiple_models", "inspect_failure", "duplicate_receipt", "split_receipt")
    if "numerical" not in script or scenario not in {"duplicate_receipt", "split_receipt"}
])
def test_probe_container_ownership(script, scenario):
    shell = shutil.which("pwsh")
    if shell is None:
        pytest.skip("PowerShell is unavailable")
    program = r'''
$commands = [System.Collections.Generic.List[string]]::new()
function global:ssh {
    $line = $args -join ' '
    $commands.Add($line)
    $global:LASTEXITCODE = 0
    if ($line -match 'docker ps') {
        if ($env:PROBE_SCENARIO -eq 'model_running') { 'custom.serving' }
        if ($env:PROBE_SCENARIO -eq 'multiple_models') { 'customXserving'; 'custom.serving' }
        if ($env:PROBE_SCENARIO -eq 'inspect_failure') { $global:LASTEXITCODE = 1 }
        return
    }
    if ($line -match 'sha256sum') { 'identical fixture hashes'; return }
    if ($line -match 'docker run' -and $env:PROBE_SCENARIO -eq 'launch_failure' -and $line -match '-r1 ') {
        $global:LASTEXITCODE = 1
        return
    }
    if ($line -match 'docker inspect') { 'exited:0' }
    if ($line -match 'docker logs') {
        $record = 'TP4_VOCAB_GRAPH mtp_tokens=4 pattern=5,1,1,1,1 captured_nodes=5 published=510 consumed=510 completed=510 overflow=0 submit_cpu=10 progress_cpu=12 mismatches=0 passed=true'
        if ($env:PROBE_SCENARIO -eq 'split_receipt') {
            ($record -split ' mismatches=')[0]
            'TP4_VOCAB_GRAPH mismatches=0 passed=true'
        } else {
            $record
            if ($env:PROBE_SCENARIO -eq 'duplicate_receipt') { $record }
        }
    }
}
$failures = @()
foreach ($run in 1..2) {
    try {
        $mapping = @{}
        if ($env:PROBE_SCRIPT -like '*vocab_graph_probe.ps1') {
            $mapping.DevicePreset = 'documented-cycle'
        }
        & $env:PROBE_SCRIPT -Image fixture-image -ModelContainer custom.serving `
            -Targets node0,node1,node2,node3 -RankHosts 192.0.2.1,192.0.2.2,192.0.2.3,192.0.2.4 `
            -KeepContainers:($env:PROBE_SCENARIO -eq 'keep') @mapping
    } catch { $failures += $_.Exception.Message }
}
'CAPTURE_JSON ' + (@{calls=@($commands); failures=$failures} | ConvertTo-Json -Depth 4 -Compress)
'''
    path = Path(__file__).resolve().parents[1] / "scripts" / script
    result = subprocess.run(
        [shell, "-NoProfile", "-NonInteractive", "-Command", program],
        env={**os.environ, "PROBE_SCRIPT": str(path), "PROBE_SCENARIO": scenario},
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(next(line.removeprefix("CAPTURE_JSON ") for line in result.stdout.splitlines()
                           if line.startswith("CAPTURE_JSON ")))
    assert len(data["failures"]) == (0 if scenario in {"success", "keep"} else 2)
    calls = data["calls"]
    launches = [call for call in calls if "docker run" in call]
    removals = [call for call in calls if "docker rm -f" in call]
    if scenario in {"model_running", "multiple_models", "inspect_failure"}:
        assert not launches and not removals
        return
    assert all("docker rm" not in call for call in launches)
    names = [re.search(r"--name (\S+)", call)[1] for call in launches]
    assert len(names) == len(set(names))
    # A failed SSH reply can follow successful container creation. Every
    # invocation-specific attempted name must be included in cleanup.
    owned = {re.search(r"--name (\S+)", call)[1] for call in launches}
    removed = {re.search(r"docker rm -f (\S+)", call)[1] for call in removals}
    assert removed == (set() if scenario == "keep" else owned)
    assert all("glm52-trace" not in call for call in calls)
    assert all("custom.serving" in call for call in calls if "docker ps" in call)
    if "numerical" in script:
        assert all("&& docker run" in call and "|| true;" not in call for call in launches)
