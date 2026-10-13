"""CPU admission tests with real competing processes and no Spark access."""
import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from runtime.common import process_lock
from runtime.host import node, placement, progress, slices

pytestmark = pytest.mark.skipif(os.name != "posix", reason="Production concurrency uses Linux flock")


@contextlib.contextmanager
def holder(root, positions, *, legacy=False):
    code = """
import json, sys
from pathlib import Path
from runtime.host import slices
from runtime.common import process_lock
root = Path(sys.argv[1])
context = (process_lock.hold(root / 'install.lock') if sys.argv[3] == 'legacy'
           else slices.hold(root, json.loads(sys.argv[2]), 'install fixture'))
with context:
    print('ready', flush=True)
    sys.stdin.readline()
"""
    child = subprocess.Popen([sys.executable, "-c", code, str(root), json.dumps(positions),
                              "legacy" if legacy else "slice"], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "ready"
        yield child
    finally:
        if child.poll() is None:
            child.communicate("done\n", timeout=10)
        else:
            child.communicate(timeout=10)


def test_disjoint_processes_overlap_and_conflicts_name_holder(tmp_path):
    with holder(tmp_path, [0, 1, 2, 3]):
        with slices.hold(tmp_path, (4, 5), "restart"):
            assert len(slices.active(tmp_path)) == 2
        with pytest.raises(ValueError, match="install fixture.*Sparks 0,1,2,3"):
            with slices.hold(tmp_path, (2, 3), "down"):
                pytest.fail("overlap entered")
        with pytest.raises(ValueError, match="Blocked"):
            with slices.hold(tmp_path, None, "setup"):
                pytest.fail("ring operation entered")
    assert slices.active(tmp_path) == []


@pytest.mark.parametrize("legacy", [False, True])
def test_ring_excludes_slice_and_legacy_installers(tmp_path, legacy):
    with holder(tmp_path, None, legacy=legacy):
        with pytest.raises(ValueError, match="whole ring"):
            with slices.hold(tmp_path, (6, 7), "install"):
                pytest.fail("ring exclusion failed")
    with slices.hold(tmp_path, (6, 7), "install"):
        with pytest.raises(ValueError):
            with process_lock.hold(tmp_path / "install.lock"):
                pytest.fail("legacy exclusion failed")


def test_crash_releases_kernel_locks_and_stale_metadata_is_ignored(tmp_path):
    with holder(tmp_path, [0, 1]) as child:
        child.kill()
        child.wait(timeout=10)
    assert list((tmp_path / "slice-locks").glob("*.lock"))
    assert slices.active(tmp_path) == []
    with slices.hold(tmp_path, (0, 1), "retry"):
        assert slices.busy(tmp_path, (0, 1))
        assert not slices.busy(tmp_path, (4, 5))


def test_replacement_covers_displaced_deployment_and_rollbacks(tmp_path):
    deployment = tmp_path / "deployment"
    node.save(deployment, "deployment.lock.json", {"site": {"placement": [4, 5, 6, 7]}})
    placement.record(tmp_path, (4, 5, 6, 7), deployment)
    with slices.hold(tmp_path, (6, 7), "swap", expand=True) as owner:
        assert owner["positions"] == [4, 5, 6, 7]
        assert slices.busy(tmp_path, (4, 5))
        assert not slices.busy(tmp_path, (0, 1))


def test_partial_acquisition_failure_leaks_no_position_locks(tmp_path):
    with holder(tmp_path, [2, 3]):
        with pytest.raises(ValueError):
            with slices.hold(tmp_path, (0, 1, 2), "install"):
                pass
        with slices.hold(tmp_path, (0, 1), "down"):
            pass


def test_operation_logs_are_selected_by_slice(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SPARKRING_LOG_DIR", str(tmp_path))
    for positions, message in (((0, 1), "first slice"), ((4, 5), "second slice")):
        with progress.run("install"):
            progress.bind_slice(positions)
            progress.say(message)
    capsys.readouterr()
    assert progress.main(["--on", "4-5", "--plain"]) == 0
    shown = capsys.readouterr().out
    assert "second slice" in shown
    assert "first slice" not in shown
    assert len(list(tmp_path.glob("operation-*.log"))) == 4


def test_admission_adds_no_threads_timers_or_network(monkeypatch, tmp_path):
    import socket
    import threading
    def refuse(*args, **kwargs):
        pytest.fail("admission started background work or network")
    monkeypatch.setattr(threading.Thread, "start", refuse)
    monkeypatch.setattr(socket, "socket", refuse)
    with slices.hold(tmp_path, (4, 5), "install"):
        assert slices.active(tmp_path)


def test_atomic_save_has_no_fixed_temporary_path(tmp_path):
    node.save(tmp_path, "state.json", {"value": 1})
    assert json.loads(Path(tmp_path, "state.json").read_text()) == {"value": 1}
    assert not list(tmp_path.glob("*.writing"))
