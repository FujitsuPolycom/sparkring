from pathlib import Path
import os
import shutil
import subprocess

import pytest

SCRIPT = Path(__file__).with_name('run_two_spark_sweep.sh')


@pytest.mark.parametrize('failure,expected,removed', [('none', 0, 2), ('client', 7, 2), ('server', 6, 0), ('foreign', 125, 0)])
def test_sweep_cleans_only_owned_containers_on_exit(tmp_path, failure, expected, removed):
    shell = shutil.which('bash') if os.name != 'nt' else str(Path(os.environ.get('ProgramFiles', 'C:/Program Files'))/'Git/bin/bash.exe')
    if not shell or not Path(shell).is_file():
        pytest.skip('Bash is required for the isolated orchestration test')
    stub = r'''
ssh() { shift; eval "$*"; }
docker() {
  local name='' owner='' id='' side='' command="$1"
  shift
  case "$command" in
    run)
      while (( $# )); do
        case "$1" in
          --name) name="$2"; shift ;;
          --label) owner="${2#*=}"; shift ;;
        esac
        shift
      done
      printf 'name %s\n' "$name" >> "$FAKE_DIR/calls"
      if [[ "$name" == *server* ]]; then
        [[ "$FAILURE" != server ]] || return 6
        side=server
        id=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
        if [[ "$FAILURE" == foreign ]]; then owner=another-run; fi
      else
        side=client
        id=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
      fi
      printf '%s %s\n' "$id" "$owner" > "$FAKE_DIR/$side"
      [[ "$FAILURE" != foreign ]] || return 125
      if [[ "$side" == client ]]; then
        printf 'GPU_ROUNDTRIP sample\n'
        [[ "$FAILURE" != client ]] || return 7
      fi
      ;;
    inspect)
      name="${!#}"
      if [[ "$name" == *server* ]]; then side=server; else side=client; fi
      [[ -f "$FAKE_DIR/$side" ]] || return 1
      cat "$FAKE_DIR/$side"
      ;;
    logs) printf 'VERIFY passed\n' ;;
    rm) printf 'remove %s\n' "${!#}" >> "$FAKE_DIR/calls" ;;
    *) return 99 ;;
  esac
}
export -f ssh docker
source "$1"
'''
    env = {**os.environ, 'FAKE_DIR':tmp_path.as_posix(), 'FAILURE':failure,
           'SERVER_SSH':'example@server', 'PEER_FABRIC_IP':'192.0.2.2',
           'IMAGE':'example/image:test', 'SIZES':'4096'}
    result = subprocess.run([shell, '--noprofile', '--norc', '-c', stub, 'test', SCRIPT.as_posix()], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == expected, result.stderr
    calls = (tmp_path/'calls').read_text().splitlines()
    removals = [line.split()[1] for line in calls if line.startswith('remove ')]
    assert len(removals) == removed
    assert all(value in ('a'*64, 'b'*64) for value in removals)
    assert all('spark-sweep-' not in value for value in removals)
    assert any(line.startswith('name spark-sweep-server-') and line.count('-') >= 6 for line in calls)
