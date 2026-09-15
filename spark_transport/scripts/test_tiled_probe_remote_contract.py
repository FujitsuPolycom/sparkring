"""Tiled probe ownership and literal commands with an injected remote executor."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest


SCRIPTS = Path(__file__).parent
POWERSHELL = shutil.which("pwsh")
pytestmark = pytest.mark.skipif(POWERSHELL is None, reason="PowerShell 7 unavailable")
LITERAL = "/tmp/Cody's folder/$(printf WRONG);`printf WRONG`"


def run_fixture(runner, scenario, failure_rank=1):
    program = r'''
$calls=[System.Collections.Generic.List[object]]::new()
$executor={
    param($FilePath,$ArgumentList,$TimeoutSeconds)
    $command=$ArgumentList[-1]
    $calls.Add([pscustomobject]@{command=$command; target=$ArgumentList[-2]; timeout=$TimeoutSeconds; program=$FilePath})
    $code=0; $timedOut=$false; $output=''
    if ($command -match 'sha256sum') {
        $output=('a' * 64)+'  fixture'
        if ($env:TILED_SCENARIO -eq 'hash_failure') { $code=1; $output='' }
    }
    if ($command -match 'docker run' -and $command -match ('-r'+$env:TILED_FAILURE_RANK+' ')) {
        $code=255
        if ($env:TILED_SCENARIO -eq 'timeout') { $code=124; $timedOut=$true }
    }
    [pscustomobject]@{ExitCode=$code; TimedOut=$timedOut; StandardOutput=$output; StandardError=''}
}.GetNewClosure()
$failures=@()
foreach ($run in 1..2) {
    $options=@{
        Execute=$true; Image=$env:TILED_LITERAL; ProbeBinary=$env:TILED_LITERAL
        Targets=@('host0','host1','host2','host3')
        RankHosts=@('peer0','peer1','peer2','peer3')
        Python=$env:TILED_PYTHON; RemoteTimeoutSeconds=13; RemoteExecutor=$executor
        KeepContainers=($env:TILED_SCENARIO -eq 'keep')
    }
    if ($env:TILED_RUNNER -like '*qualification.ps1') { $options.Suite='poison' }
    else { $options.ArmId='q40_isolated' }
    try { & $env:TILED_RUNNER @options } catch { $failures += $_.Exception.Message }
}
'CAPTURE_JSON '+(@{calls=@($calls); failures=$failures}|ConvertTo-Json -Depth 5 -Compress)
'''
    result = subprocess.run(
        [POWERSHELL, "-NoProfile", "-NonInteractive", "-Command", program],
        env={**os.environ, "TILED_RUNNER": str(SCRIPTS / runner), "TILED_SCENARIO": scenario,
             "TILED_FAILURE_RANK": str(failure_rank), "TILED_LITERAL": LITERAL, "TILED_PYTHON": sys.executable},
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(next(line.removeprefix("CAPTURE_JSON ") for line in result.stdout.splitlines()
                           if line.startswith("CAPTURE_JSON ")))


@pytest.mark.parametrize("runner", ["run_tp4_tiled_prefill_probe.ps1", "run_tp4_tiled_prefill_qualification.ps1"])
@pytest.mark.parametrize("scenario,failure_rank", [("launch_failure", 0), ("launch_failure", 1), ("timeout", 1), ("keep", 1), ("hash_failure", 0)])
def test_tiled_cleanup_owns_only_attempted_unique_names(runner, scenario, failure_rank):
    result = run_fixture(runner, scenario, failure_rank)
    assert len(result["failures"]) == 2
    calls = result["calls"]
    assert all(call["program"] == "ssh" and call["timeout"] == 13 for call in calls)
    launches = [call for call in calls if "docker run" in call["command"]]
    removals = [call for call in calls if "docker rm" in call["command"]]
    if scenario == "hash_failure":
        assert launches == removals == []
        return
    assert len(launches) == 2 * (failure_rank + 1)
    assert all("docker rm" not in call["command"] for call in launches)
    names = [re.search(r"--name (\S+)", call["command"])[1] for call in launches]
    assert len(set(names)) == len(names)
    assert all(re.fullmatch(r"spark-tp4-tiled-[a-z0-9_]+-[0-9a-f]{32}-r[0-3]", name) for name in names)
    if scenario == "keep":
        assert removals == []
    else:
        assert len(removals) == len(launches)
        for launch, removal, name in zip(launches, removals, names):
            assert removal["target"] == launch["target"]
            assert removal["command"] == f"docker rm -f {name} >/dev/null 2>&1 || true"
    # Fixed historical names represent preexisting containers and are never
    # removed, even when launch completion is unknown after a timeout.
    assert not any(re.search(r"docker rm -f spark-tp4-tiled-[a-z0-9_]+-r[0-3](?:\s|$)", call["command"]) for call in calls)


def test_tiled_command_quotes_artifact_and_image_for_local_shell(tmp_path):
    result = run_fixture("run_tp4_tiled_prefill_probe.ps1", "launch_failure", 0)
    command = next(call["command"] for call in result["calls"] if "docker run" in call["command"])
    program = "exec 3>&1\ndocker() { printf '%s\\0' \"$@\" >&3; }\n" + command
    path = tmp_path / "shell-fixture.sh"
    path.write_text(program, encoding="utf-8", newline="\n")
    if os.name == "nt":
        if shutil.which("wsl") is None:
            pytest.skip("Local WSL shell unavailable")
        absolute = path.resolve().as_posix()
        argv = ["wsl", "--exec", "sh", "/mnt/" + absolute[0].lower() + absolute[2:]]
    else:
        argv = ["sh", str(path)]
    output = subprocess.check_output(argv, timeout=30)
    arguments = [value.decode() for value in output.split(b"\0")[:-1]]
    assert LITERAL in arguments
    assert arguments[arguments.index("-v") + 1] == LITERAL + ":/probe:ro"
    assert arguments[arguments.index("--peer0") + 1] == "peer1"
    assert arguments[arguments.index("--peer1") + 1] == "peer3"
