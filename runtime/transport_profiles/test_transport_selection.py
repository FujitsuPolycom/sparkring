"""CPU checks of source identity, path mapping, and fail-closed Python startup."""

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sparkring_transport_selector", ROOT / "sparkring_transport_selector.py")
selector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selector)
BUNDLE = ROOT / selector.PROFILE
DIGEST = hashlib.sha256((BUNDLE / "manifest.json").read_bytes()).hexdigest()


@pytest.fixture(autouse=True)
def isolated_selector(monkeypatch):
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    monkeypatch.delenv(selector.PROFILE_ENV, raising=False)
    monkeypatch.delenv(selector.DIGEST_ENV, raising=False)


def test_source_bundle_verifies_and_selects_only_roce(monkeypatch):
    monkeypatch.setenv(selector.PROFILE_ENV, selector.PROFILE)
    monkeypatch.setenv(selector.DIGEST_ENV, DIGEST)
    selector.install_from_environment()
    selector.require_active(require_startup_hook=False)
    finder = sys.meta_path[0]
    assert Path(finder.find_spec("b12x.comm.roce").origin) == BUNDLE / "roce/__init__.py"
    assert Path(finder.find_spec("b12x.comm.roce._proxy").origin) == BUNDLE / "roce/_proxy.py"
    assert finder.find_spec("b12x.integration.vllm.loader") is None
    assert finder.find_spec("b12x.sequence.kda_prefill") is None


def test_no_selection_preserves_existing_imports():
    before = list(sys.meta_path)
    selector.install_from_environment()
    assert sys.meta_path == before


def test_late_activation_is_refused(monkeypatch):
    monkeypatch.setenv(selector.PROFILE_ENV, selector.PROFILE)
    monkeypatch.setenv(selector.DIGEST_ENV, DIGEST)
    monkeypatch.setitem(sys.modules, "b12x.comm.roce", object())
    with pytest.raises(SystemExit, match="before importing"):
        selector.install_from_environment()


@pytest.mark.parametrize("damage", ["manifest", "source", "extra"])
def test_bundle_drift_is_refused_before_hook_install(tmp_path, monkeypatch, damage):
    shutil.copytree(BUNDLE, tmp_path / selector.PROFILE)
    paths = {
        "manifest": "manifest.json", "source": "roce/_proxy.py", "extra": "roce/extra.py",
    }
    with (tmp_path / selector.PROFILE / paths[damage]).open("ab") as stream:
        stream.write(b"\n# altered\n")
    monkeypatch.setenv(selector.PROFILE_ENV, selector.PROFILE)
    monkeypatch.setenv(selector.DIGEST_ENV, DIGEST)
    before = list(sys.meta_path)
    with pytest.raises(SystemExit, match="activation failed"):
        selector.install_from_environment(tmp_path)
    assert sys.meta_path == before


def test_python_pth_failure_stops_the_consumer(tmp_path):
    hook_dir = tmp_path / "site"
    hook_dir.mkdir()
    (hook_dir / selector.HOOK_NAME).write_text(selector.startup_hook())
    code = "import site; site.addsitedir(" + repr(str(hook_dir)) + "); print('CONSUMER_REACHED')"
    environment = {**os.environ, selector.PROFILE_ENV: selector.PROFILE,
                   selector.DIGEST_ENV: "0" * 64}
    result = subprocess.run([sys.executable, "-I", "-c", code], env=environment,
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "CONSUMER_REACHED" not in result.stdout
    assert "manifest SHA-256 differs" in result.stderr


def test_python_pth_activates_in_a_fresh_worker(tmp_path):
    hook_dir = tmp_path / "site"
    hook_dir.mkdir()
    (hook_dir / selector.HOOK_NAME).write_text(selector.startup_hook())
    code = (
        "import site, sys; site.addsitedir(" + repr(str(hook_dir)) + "); "
        "import sparkring_transport_selector as s; "
        "s.require_active(require_startup_hook=False); "
        "print(sys.meta_path[0].find_spec('b12x.comm.roce').origin)"
    )
    result = subprocess.run([sys.executable, "-I", "-c", code],
                            env={**os.environ, selector.PROFILE_ENV: selector.PROFILE,
                                 selector.DIGEST_ENV: DIGEST}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()) == BUNDLE / "roce/__init__.py"


@pytest.mark.parametrize("rank", [0, 1])
def test_same_cage_paths_are_reciprocal(rank, monkeypatch):
    spec = importlib.util.spec_from_file_location("tp2_path_config", BUNDLE / "roce/_path_config.py")
    paths = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(paths)
    monkeypatch.setenv("B12X_ROCE_PAIR_PATHS", "2")
    monkeypatch.setenv("B12X_ROCE_PEER_HCA_MAP", f"{1-rank}=0/2")
    count = paths.opposite_path_count(2, 4)
    mapping = paths.peer_hca_map(2, rank, 4, count)
    assert mapping[1-rank] == (0, 2)
    assert tuple(paths.CANONICAL_HCAS[index] for index in mapping[1-rank]) == (
        "rocep1s0f0", "roceP2p1s0f0",
    )
    assert mapping[rank] == (-1, -1)


def test_adaptive_grids_have_independent_barrier_counters():
    tree = ast.parse((BUNDLE / "roce/roce_oneshot.py").read_text())
    grid = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_grid_blocks")
    klass = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RoceOneshotAllReduce")
    counters = next(node for node in klass.body if isinstance(node, ast.FunctionDef) and node.name == "_counter_addresses")
    namespace = {}
    exec(compile(ast.Module(body=[grid, counters], type_ignores=[]), "grid-contract", "exec"), namespace)
    assert [namespace["_grid_blocks"](size // 16, 512, 8)
            for size in (8192, 16384, 32768, 65536, 1048576)] == [1, 1, 2, 4, 8]
    state = type("Counters", (), {"_epoch_address": 4096, "_counter_classes": 4})()
    pairs = [namespace["_counter_addresses"](state, blocks) for blocks in (1, 2, 4, 8)]
    addresses = [address for pair in pairs for address in pair]
    assert len(set(addresses)) == 8
    assert 4096 not in addresses
    assert 4096 + 4 * 9 not in addresses  # poison word must not alias a barrier


def test_snapshot_is_distinct_from_weighted_tp4_bundle():
    root = ROOT.parents[1]
    mesh = root / "runtime/glm53-spark-mtp3-mesh/performance/transport/bundle-source/b12x_overlay/b12x/comm/roce/roce_oneshot.py"
    assert hashlib.sha256(mesh.read_bytes()).hexdigest() != json.loads(
        (BUNDLE / "manifest.json").read_text())["files"]["roce/roce_oneshot.py"]


def test_packaged_hook_uses_absolute_posix_image_path():
    assert selector.startup_hook(Path("/opt/sparkring/transports")).splitlines()[0] == "/opt/sparkring/transports"


def test_package_copies_verified_files_and_refuses_existing_output(tmp_path):
    destination = tmp_path / "staging"
    command = [sys.executable, str(ROOT / "package.py"), "--destination", str(destination)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    assert receipt["manifest_sha256"] == DIGEST
    assert not receipt["builds_image"] and not receipt["starts_serving"]
    selector.verify_bundle(selector.PROFILE, DIGEST, destination)
    assert (destination / selector.HOOK_NAME).read_text().splitlines()[0] == "/opt/sparkring/transports"
    assert subprocess.run(command, capture_output=True).returncode != 0
