"""Contract tests for the offline LMCache group-tracking verifier.

The fixtures below are synthetic package trees, not vendored LMCache source.
They carry only the symbols the verifier decides on, so the tests state the
decision rule rather than pinning an upstream file's contents.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import verify_lmcache_group_tracking as checker  # noqa: E402

SCRIPT = Path(__file__).resolve().parent / "verify_lmcache_group_tracking.py"

CONNECTOR_MODULE = "integration/vllm/lmcache_mp_connector.py"

GROUP_MODULE_SOURCE = '''"""Kernel-group partitioning."""


class KernelGroupIdentity(NamedTuple):
    kv_size: int
    num_heads: int
    head_size: int
    block_size: int
    engine_group_idx: int


class PageBufferShapeDesc:
    def __init__(self, bs, block_stride_elems):
        self.bs = bs
        self.block_stride_elems = block_stride_elems


class KVLayerGroupsManager:
    pass
'''

CONNECTOR_SOURCE = '''class LMCacheMPConnector:
    def register(self, payload):
        payload["vllm_block_size"] = self.block_size
'''

HEALTHY_HEARTBEAT = '''class Server:
    def start(self):
        if self._heartbeats:
            self._spawn()
'''

DEFECTIVE_HEARTBEAT = '''class Server:
    def start(self):
        if self._heartbeats is not None:
            self._spawn()
'''


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_package(
    root: Path,
    *,
    version: str = "0.5.3",
    group_tracking: bool = True,
    connector_class: str = "LMCacheMPConnector",
    heartbeat_source: str = HEALTHY_HEARTBEAT,
    single_server_only: bool = False,
    local_topology_patch: bool = False,
) -> Path:
    """Materialize a synthetic lmcache package tree and return its directory."""
    package = root / "lmcache"
    _write(package / "__init__.py", f'__version__ = "{version}"\n')

    if group_tracking:
        _write(package / checker.GROUP_MODULE, GROUP_MODULE_SOURCE)

    connector = CONNECTOR_SOURCE.replace("LMCacheMPConnector", connector_class, 1)
    if not group_tracking:
        connector = connector.replace('payload["vllm_block_size"] = self.block_size', "pass")
    _write(package / CONNECTOR_MODULE, connector)

    _write(package / "v1/multiprocess/server.py", heartbeat_source)

    adapter = "class ParallelStrategy:\n    pass\n"
    if single_server_only:
        adapter = (
            "class ParallelStrategy:\n"
            "    def check(self):\n"
            '        raise ValueError("LMCache MLA+DCP currently '
            'requires one LMCache server")\n'
        )
    if local_topology_patch:
        adapter += (
            "\n\ndef local_server_url_for_worker(server_urls, parallel_strategy):\n"
            "    return server_urls[0]\n"
        )
    _write(package / "integration/vllm/vllm_multi_process_adapter.py", adapter)

    return package


def test_post_change_package_passes(tmp_path):
    package = build_package(tmp_path)
    report = checker.verify(package)
    assert report["verdict"] == "pass"
    assert report["failed_checks"] == []
    assert report["declared_version"] == "0.5.3"


def test_package_without_group_module_fails(tmp_path):
    package = build_package(tmp_path, version="0.5.2", group_tracking=False)
    report = checker.verify(package)
    assert report["verdict"] == "fail"
    assert "group_module_present" in report["failed_checks"]
    assert "kernel_group_identity_defined" in report["failed_checks"]
    assert "identity_separates_block_size" in report["failed_checks"]
    assert "padded_stride_described" in report["failed_checks"]
    assert "registration_carries_engine_block_size" in report["failed_checks"]


def test_renamed_connector_is_a_failure(tmp_path):
    """A package that renames the connector breaks the serving recipe's binding."""
    package = build_package(tmp_path, connector_class="LMCacheMPConnectorDynamic")
    report = checker.verify(package)
    assert report["verdict"] == "fail"
    assert "recipe_connector_symbol_present" in report["failed_checks"]

    # The same tree passes when the recipe is repointed at the new symbol.
    repointed = checker.verify(package, connector_class="LMCacheMPConnectorDynamic")
    assert repointed["verdict"] == "pass"


def test_defective_heartbeat_guard_is_located(tmp_path):
    package = build_package(tmp_path, heartbeat_source=DEFECTIVE_HEARTBEAT)
    report = checker.verify(package)
    assert report["verdict"] == "fail"
    assert "heartbeat_guard_tests_contents" in report["failed_checks"]
    detail = next(
        check["detail"]
        for check in report["checks"]
        if check["name"] == "heartbeat_guard_tests_contents"
    )
    assert detail["defective_locations"] == ["v1/multiprocess/server.py:3"]


def test_single_server_rejection_fails_multi_server_topology(tmp_path):
    package = build_package(tmp_path, single_server_only=True)
    report = checker.verify(package)
    assert report["verdict"] == "fail"
    assert "multi_server_topology_permitted" in report["failed_checks"]


def test_local_topology_patch_is_reported_not_required(tmp_path):
    """Absence of the local patch is recorded but never fails the verdict."""
    without = checker.verify(build_package(tmp_path / "a"))
    assert without["verdict"] == "pass"
    reported = next(
        check
        for check in without["checks"]
        if check["name"] == "local_topology_patch_applied"
    )
    assert reported["required"] is False
    assert reported["passed"] is False

    with_patch = checker.verify(
        build_package(tmp_path / "b", local_topology_patch=True)
    )
    assert with_patch["verdict"] == "pass"
    applied = next(
        check
        for check in with_patch["checks"]
        if check["name"] == "local_topology_patch_applied"
    )
    assert applied["passed"] is True


def test_expected_version_mismatch_fails(tmp_path):
    package = build_package(tmp_path, version="0.5.3")
    assert checker.verify(package, expect_version="0.5.3")["verdict"] == "pass"
    mismatch = checker.verify(package, expect_version="0.5.2+glm52dcp4.1")
    assert mismatch["verdict"] == "fail"
    assert "declared_version_matches" in mismatch["failed_checks"]


def test_local_version_suffix_is_parsed(tmp_path):
    package = build_package(tmp_path, version="0.5.3+glm52dcp4.2")
    assert checker.verify(package)["declared_version"] == "0.5.3+glm52dcp4.2"


def test_missing_version_is_reported_as_none(tmp_path):
    package = build_package(tmp_path)
    (package / "__init__.py").write_text("", encoding="utf-8")
    assert checker.verify(package)["declared_version"] is None


@pytest.mark.parametrize(
    "factory",
    [
        lambda root: root / "absent",
        lambda root: root,
    ],
)
def test_non_package_directories_are_config_errors(tmp_path, factory):
    build_package(tmp_path)
    with pytest.raises(checker.ConfigError):
        checker.verify(factory(tmp_path))


def test_cli_exit_codes(tmp_path):
    good = build_package(tmp_path / "good")
    bad = build_package(tmp_path / "bad", group_tracking=False)

    ok = subprocess.run(
        [sys.executable, str(SCRIPT), "--package-dir", str(good)],
        capture_output=True,
        text=True,
    )
    assert ok.returncode == checker.EXIT_OK
    assert json.loads(ok.stdout)["verdict"] == "pass"

    fail = subprocess.run(
        [sys.executable, str(SCRIPT), "--package-dir", str(bad)],
        capture_output=True,
        text=True,
    )
    assert fail.returncode == checker.EXIT_FAIL
    assert json.loads(fail.stdout)["verdict"] == "fail"

    config = subprocess.run(
        [sys.executable, str(SCRIPT), "--package-dir", str(tmp_path / "absent")],
        capture_output=True,
        text=True,
    )
    assert config.returncode == checker.EXIT_CONFIG_ERROR
    assert "CONFIG ERROR" in config.stderr
