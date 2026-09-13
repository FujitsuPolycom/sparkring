"""Exercise bounded child processes locally; no SSH or containers are started."""
import json
from pathlib import Path
import shutil
import subprocess
import time

import pytest

SHELL = shutil.which('pwsh')
pytestmark = pytest.mark.skipif(SHELL is None, reason='PowerShell 7 is unavailable')
HELPER = Path(__file__).with_name('probe_process.ps1')


def execute(tmp_path, body, arguments=(), timeout=10):
    child = tmp_path / 'child with spaces.ps1'
    child.write_text(body, encoding='utf-8')
    inputs = tmp_path / 'inputs.json'
    inputs.write_text(json.dumps({'arguments': list(arguments), 'timeout': timeout}), encoding='utf-8')
    driver = tmp_path / 'driver.ps1'
    driver.write_text('param($Helper,$Child,$Inputs)\n. $Helper\n'
                      '$config=Get-Content -Raw -LiteralPath $Inputs | ConvertFrom-Json\n'
                      '$argsForChild=@("-NoProfile","-NonInteractive","-File",$Child)+@($config.arguments)\n'
                      'Invoke-ProbeProcess -FilePath (Get-Process -Id $PID).Path '
                      '-ArgumentList $argsForChild -TimeoutSeconds $config.timeout | ConvertTo-Json -Compress\n',
                      encoding='utf-8')
    result = subprocess.run([SHELL, '-NoProfile', '-NonInteractive', '-File', str(driver),
                             str(HELPER), str(child), str(inputs)], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_arguments_streams_and_exit_status(tmp_path):
    literal = "a path with 'quotes' and $(literal); text"
    result = execute(tmp_path, 'param($Value)\n[Console]::Out.Write($Value)\n'
                     '[Console]::Error.Write("diagnostic")\nexit 7\n', [literal])
    assert result == {'ExitCode': 7, 'TimedOut': False,
                      'StandardOutput': literal, 'StandardError': 'diagnostic'}


def test_hung_child_returns_within_cleanup_bound(tmp_path):
    started = time.monotonic()
    result = execute(tmp_path, 'Start-Sleep -Seconds 60\n', timeout=0.2)
    assert result['TimedOut'] and result['ExitCode'] == 124
    assert time.monotonic() - started < 8


def test_full_output_pipes_do_not_deadlock(tmp_path):
    result = execute(tmp_path, '[Console]::Out.Write("o" * 200000)\n'
                     '[Console]::Error.Write("e" * 200000)\n')
    assert result['ExitCode'] == 0 and not result['TimedOut']
    assert result['StandardOutput'] == 'o' * 200000
    assert result['StandardError'] == 'e' * 200000
