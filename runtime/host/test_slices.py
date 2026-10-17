"""CPU admission tests with real competing processes and no Spark access."""
import contextlib
import concurrent.futures
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
        with pytest.raises(ValueError, match="install fixture.*Sparks 0,1,2,3") as caught:
            with slices.hold(tmp_path, (2, 3), "down"):
                pytest.fail("overlap entered")
        reason = caught.value.activity["reasons"][0]
        assert reason["grade"] == "blocked" and reason["resource"] == "Sparks 0,1,2,3"
        assert reason["activity"] == "install fixture" and reason["pid"]
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
    node.save(deployment, "state.json", {"operation": "up", "complete": True})
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


def test_pair_install_ignores_stopped_group_and_reserves_only_actual_route(tmp_path):
    directory = tmp_path / "deployment"
    node.save(directory, "deployment.lock.json", {})
    node.save(directory, "state.json", {"operation": "down", "complete": True})
    placement.record(tmp_path, (4, 5, 6, 7), directory)
    with slices.hold(tmp_path, (6, 7), "install pair", expand=True) as owner:
        assert owner["positions"] == [6, 7]
        with slices.transfer(tmp_path, owner["positions"], [0, 7, 6]):
            with slices.hold(tmp_path, (4, 5), "install other", expand=True):
                pass
        with holder(tmp_path, [4, 5]):
            with pytest.raises(ValueError, match="Sparks 4,5"):
                # A repair route through another operation must be refused.
                with slices.transfer(tmp_path, owner["positions"], [0, 1, 2, 3, 4, 5, 6]):
                    pytest.fail("contended route started")
        assert not slices.busy(tmp_path, (1, 2, 3))
        with slices.transfer(tmp_path, owner["positions"], [0, 1, 2, 3, 4, 5, 6]):
            assert slices.busy(tmp_path, (4, 5))
        assert not slices.busy(tmp_path, (4, 5))


def test_activity_scan_does_not_report_another_scans_temporary_lock(tmp_path):
    # A completed operation leaves a lock file. Pause a real scanner while
    # its liveness probe owns that file, then start another scanner.
    with slices.hold(tmp_path, (7,), "completed install"):
        pass
    code = """
import contextlib, sys
from pathlib import Path
from runtime.common import process_lock
from runtime.host import slices
original = process_lock.hold
@contextlib.contextmanager
def pause(path, **options):
    with original(path, **options):
        if Path(path).name == 'spark-7.lock':
            print('probing', flush=True)
            sys.stdin.readline()
        yield
process_lock.hold = pause
slices.active(Path(sys.argv[1]))
"""
    child = subprocess.Popen([sys.executable, "-c", code, str(tmp_path)], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "probing"
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            scan = pool.submit(slices.active, tmp_path)
            try:
                result = scan.result(timeout=0.3)
            except concurrent.futures.TimeoutError:
                result = None
            finally:
                child.communicate("done\n", timeout=10)
            assert scan.result(timeout=10) == []
            assert result in (None, [])
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate(timeout=10)


@pytest.mark.parametrize("asset", ["image", "checkpoint"])
def test_transfer_conflict_keeps_holding_profile_and_deployment(tmp_path, asset):
    from runtime.host import spread
    from runtime.host.install_errors import NeedsInput
    from runtime.host.test_spread import loopback
    with holder(tmp_path, [6, 7]):
        with slices.hold(tmp_path, (4, 5), "install qwen-pair") as owner:
            runner = spread.Spread(loopback("cycle", 8), start=lambda *a: pytest.fail("hop started"),
                                   send=lambda *a: pytest.fail("copy started"), alive=lambda p: True,
                                   reserve=lambda route: slices.transfer(tmp_path, owner["positions"], route))
            with pytest.raises(NeedsInput, match="install fixture.*Sparks 6,7") as refused:
                runner.run(asset, {"file": [1, "a" * 64]}, 0, {5: ["file"]},
                           members=[0, 7, 6, 5, 4])
            assert refused.value.field == "activity"


def test_three_cached_installs_admit_together_with_real_process_locks(tmp_path):
    from runtime.common import fabric_layout
    from runtime.host import topology
    from runtime.host.test_fabric_layouts import sparks
    found = sparks(fabric_layout.layout("cycle", 8))
    plan = topology.build_spec(found, found[0]["node_id"])
    node.save(tmp_path, "cluster.json", {"name": "sparkring", "plan": plan})
    # Retained files from earlier operations must not become phantom owners.
    with slices.hold(tmp_path, tuple(range(8)), "completed install"):
        pass
    code = """
import json, sys
from pathlib import Path
from types import SimpleNamespace
from runtime.host import activity, install_assets, install_workflow, models, slices, spread
from runtime.host.test_spread import loopback
root, positions = Path(sys.argv[1]), json.loads(sys.argv[2])
profile = 'profile-' + '-'.join(map(str, positions))
models.catalog = lambda: [{'profile': profile, 'nodes': len(positions)}]
args = SimpleNamespace(on=','.join(map(str, positions)), profile=profile, yes=True, plan=False, json=True)
activity.read_metrics = lambda url: None
def refuse(*args, **kwargs):
    raise AssertionError('a cached asset started a transfer')
with install_workflow.operation_scope(args, root) as owner:
    assets = install_assets.Assets(SimpleNamespace(hosts=[{}] * 8), root / profile, run=refuse, popen=refuse)
    assets.remote = lambda rank, fn, *a: {'present': True, 'size_bytes': 1}
    assets.reserve_route = refuse
    result = assets.images({'image_id': 'sha256:' + 'a' * 64, 'image_reference': 'registry/image'}, positions)
    assert result['reused_ranks'] == positions
    copier = spread.Spread(loopback('cycle', 8), start=refuse, send=refuse, alive=lambda p: True, reserve=refuse)
    assert not copier.run('checkpoint', {'file': [1, 'a' * 64]}, positions[0], {})['remaining']
    for _ in range(10):
        owners = slices.active(root)
        assert all(row['operation'].startswith('install profile-') for row in owners), owners
    print('ready', flush=True)
    sys.stdin.readline()
"""
    children = []
    try:
        for positions in ([4, 5], [6, 7], [0, 1, 2, 3]):
            children.append(subprocess.Popen([sys.executable, "-c", code, str(tmp_path), json.dumps(positions)],
                                             stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE, text=True))
        for child in children:
            line = child.stdout.readline().strip()
            assert line == "ready", child.stderr.read() if child.poll() is not None else line
        owners = slices.active(tmp_path)
        assert sorted(row["positions"] for row in owners) == [[0, 1, 2, 3], [4, 5], [6, 7]]
    finally:
        for child in children:
            child.communicate("done\n" if child.poll() is None else None, timeout=15)
    assert all(child.returncode == 0 for child in children)


def test_transfer_route_holder_names_its_profile_and_deployment(tmp_path):
    with slices.hold(tmp_path, (4, 5), "install qwen-pair") as owner:
        with slices.transfer(tmp_path, owner["positions"], [0, 7, 6, 5], operation=owner["operation"]):
            with pytest.raises(ValueError, match="install qwen-pair on Sparks 4,5 transfer route.*Sparks 6,7"):
                with slices.hold(tmp_path, (6, 7), "install glm-pair"):
                    pytest.fail("transfer route was not held")


def test_image_admission_refusal_does_not_fall_back_to_unreserved_copy(tmp_path):
    from types import SimpleNamespace
    from runtime.host import install_assets, spread
    from runtime.host.install_errors import NeedsInput
    from runtime.host.test_spread import loopback
    with holder(tmp_path, [6, 7]):
        with slices.hold(tmp_path, (4, 5), "install qwen-pair") as owner:
            def refuse(*args, **kwargs):
                pytest.fail("refused transfer fell back to another copy")
            assets = install_assets.Assets(SimpleNamespace(hosts=[{}] * 8), tmp_path / "assets", run=refuse, popen=refuse)
            probes = []
            def probe(rank, *args):
                probes.append(rank)
                return {"present": rank == 0, "size_bytes": 1, "free_bytes": 100 * 1024**3}
            assets.remote = probe
            copier = spread.Spread(loopback("cycle", 8), start=refuse, send=refuse, alive=lambda p: True,
                                   reserve=lambda route: slices.transfer(tmp_path, owner["positions"], route,
                                                                         operation=owner["operation"]))
            assets.relay = lambda *args: copier.run("image", {"file": [1, "a" * 64]}, 0, {5: ["file"]},
                                                   members=[0, 7, 6, 5, 4])
            with pytest.raises(NeedsInput, match="install fixture.*Sparks 6,7"):
                assets.images({"image_id": "sha256:" + "a" * 64, "image_reference": "registry/image"}, [4, 5])
            assert probes == [0, 4, 5]
