"""Exercise PowerShell orchestration with an in-process fake SSH function."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest


@pytest.mark.parametrize('failure', [False, True])
def test_replay_only_cleans_containers_created_by_this_invocation(failure):
    shell = shutil.which('pwsh')
    if shell is None:
        pytest.skip('PowerShell is unavailable')
    program = r'''
$commands = [System.Collections.Generic.List[string]]::new()
function global:ssh {
    $line = $args -join ' '
    $commands.Add($line)
    $global:LASTEXITCODE = 0
    if ($line -match 'docker run' -and $env:REPLAY_TEST_FAIL -eq '1' -and $line -match '--rank 1 ') {
        $global:LASTEXITCODE = 1
        return
    }
    if ($line -match 'docker inspect') { 'exited:0' }
    if ($line -match 'docker logs') { 'MODEL_REPLAY synthetic-result' }
}
$failures = @()
foreach ($run in 1..2) {
    try {
        & $env:REPLAY_TEST_SCRIPT -Mode Spark -Image fixture-image `
            -Targets node0,node1,node2,node3 -RankHosts 192.0.2.1,192.0.2.2,192.0.2.3,192.0.2.4
    } catch { $failures += $_.Exception.Message }
}
'CAPTURE_JSON ' + (@{calls=@($commands); failures=$failures} | ConvertTo-Json -Depth 4 -Compress)
'''
    result = subprocess.run([shell, '-NoProfile', '-NonInteractive', '-Command', program],
                            env={**os.environ, 'REPLAY_TEST_SCRIPT': str(Path(__file__).with_name('run_model_loop_replay.ps1')),
                                 'REPLAY_TEST_FAIL': '1' if failure else '0'},
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    data = json.loads(next(line.removeprefix('CAPTURE_JSON ') for line in result.stdout.splitlines()
                           if line.startswith('CAPTURE_JSON ')))
    assert len(data['failures']) == (2 if failure else 0)
    launches = [call for call in data['calls'] if 'docker run' in call]
    assert launches and all('docker rm' not in call for call in launches)
    names = [re.search(r'--name (\S+)', call)[1] for call in launches]
    assert len(names) == len(set(names))
    owned = {re.search(r'--name (\S+)', call)[1] for call in launches
             if not (failure and '--rank 1 ' in call)}
    removed = {re.search(r'docker rm -f (\S+)', call)[1] for call in data['calls']
               if 'docker rm -f' in call}
    assert removed == owned
