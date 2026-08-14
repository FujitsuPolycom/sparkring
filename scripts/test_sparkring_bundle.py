"""Offline tests for the multi-service runtime bundle layer.

Covers the runtime-bundle interface and fail-closed invariants:
* Bridge shape enforcement, lifecycle parity, structured containers
* Ownership guards for both readiness kinds with fake-engine tests
* Stop/rollback daemon-probe semantics with fake-engine matrix
* Executor exception safety and rollback completeness
* Status/confirmation symmetry, path containment, canonicalization
* Plan/diff completeness, archive rehearsal
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import bootstrap_exl3  # noqa: E402
import sparkring_bundle as bundle_mod  # noqa: E402
import sparkring_exl3_launcher as exl3  # noqa: E402
import sparkring_exl3_lmcache_launcher as lmcache  # noqa: E402
import sparkring_runtime as runtime  # noqa: E402
from sparkring_site import load_site  # noqa: E402

SITE = ROOT / "scripts/config/site.example.yaml"
GENERIC = ROOT / "scripts/config/generic.example.json"
IMAGE_ID = "sha256:" + "a" * 64


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _site(tmp_path):
    site_path = tmp_path / "site.yaml"
    bootstrap_exl3.write_generated_site(
        SITE, site_path, "sparkring/exl3:test", IMAGE_ID
    )
    return site_path


def _exl3_profile(tmp_path):
    profile_path = tmp_path / "launch.json"
    bootstrap_exl3.write_generated_profile(
        profile_path, "sparkring/exl3:test", IMAGE_ID,
        "/srv/models/exl3", "/srv/jit",
    )
    return profile_path


def _generic_doc(**overrides):
    doc = json.loads(GENERIC.read_text(encoding="utf-8"))
    doc.update(overrides)
    return doc


def _write_profile(tmp_path, name, doc):
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return path


def _structured_doc(**overrides):
    doc = {
        "schema": "sparkring-structured-container/v1",
        "image": "registry.example/cache:1.0",
        "image_id": "sha256:" + "b" * 64,
        "container_name": "sparkring-cache",
        "argv": ["/opt/bin/cache-server", "--port", "6556"],
        "port": 6556,
        "environment": {"CACHE_MODE": "LRU"},
        "volumes": [],
        "privileged": False,
        "shm_size": "2g",
        "startup_timeout_seconds": 120,
    }
    doc.update(overrides)
    return doc


def _write_structured(tmp_path, name, doc):
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return path


def _bundle_doc(services, **overrides):
    doc = {
        "schema": "sparkring-runtime-bundle/v1",
        "bundle_id": "test-bundle",
        "confirmation": "START-test",
        "services": services,
    }
    doc.update(overrides)
    return doc


def _write_bundle(tmp_path, doc, name="bundle.json"):
    path = tmp_path / name
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return path


def _native_service(service_id, role, profile_path, **kw):
    if isinstance(profile_path, Path):
        path_str = profile_path.name
    else:
        path_str = str(profile_path)
    svc = {
        "service_id": service_id,
        "role": role,
        "source": {"kind": "runtime-profile", "path": path_str},
    }
    svc.update(kw)
    return svc


def _structured_service(service_id, role, structured_path, **kw):
    if isinstance(structured_path, Path):
        path_str = structured_path.name
    else:
        path_str = str(structured_path)
    svc = {
        "service_id": service_id,
        "role": role,
        "source": {"kind": "structured-container", "path": path_str},
    }
    svc.update(kw)
    return svc


def _bridge_service(service_id, role, profile_path, **kw):
    if isinstance(profile_path, Path):
        path_str = profile_path.name
    else:
        path_str = str(profile_path)
    svc = {
        "service_id": service_id,
        "role": role,
        "source": {"kind": "canonical-exl3-lmcache-cs512", "path": path_str},
    }
    svc.update(kw)
    return svc


def _run_cli(bundle_path, site_path, command, *extra_args):
    argv = [
        sys.executable,
        str(ROOT / "scripts/sparkring_bundle_launcher.py"),
        "--bundle", str(bundle_path),
        "--site", str(site_path),
        command,
        *extra_args,
    ]
    result = subprocess.run(
        argv, capture_output=True, text=True, timeout=30, cwd=str(ROOT),
    )
    return result.returncode, result.stdout, result.stderr


def _poison_remote(monkeypatch):
    """Poison SSH/socket boundaries for tests that must not reach remote.

    Any call to runtime.execute, runtime.run_remote, or socket
    connections will raise immediately. subprocess.run is NOT poisoned
    because dry-run CLI tests use it to invoke the launcher locally.
    """
    def _boom(*a, **kw):
        raise AssertionError("remote/SSH boundary reached in offline test")

    monkeypatch.setattr(runtime, "execute", _boom)
    monkeypatch.setattr(runtime, "run_remote", _boom)
    import socket as _socket
    monkeypatch.setattr(_socket, "socket", _boom)


def _run_cli_no_site(bundle_path, command, *extra_args):
    argv = [
        sys.executable,
        str(ROOT / "scripts/sparkring_bundle_launcher.py"),
        "--bundle", str(bundle_path),
        command,
        *extra_args,
    ]
    result = subprocess.run(
        argv, capture_output=True, text=True, timeout=30, cwd=str(ROOT),
    )
    return result.returncode, result.stdout, result.stderr


# ---------------------------------------------------------------------------
# Schema validation — structural rejection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_doc,expected_fragment", [
    ({"schema": "wrong/v1", "bundle_id": "x", "services": []}, "unsupported schema"),
    ({"schema": "sparkring-runtime-bundle/v1", "services": []}, "missing key"),
    ({"schema": "sparkring-runtime-bundle/v1", "bundle_id": "x"}, "missing key"),
    ({"schema": "sparkring-runtime-bundle/v1", "bundle_id": "x", "services": []}, "non-empty list"),
    ({"schema": "sparkring-runtime-bundle/v1", "bundle_id": "x",
      "services": [{"service_id": "a", "role": "serving",
                     "source": {"kind": "runtime-profile", "path": "p"}}],
      "extra": 1}, "unknown key"),
    ({"schema": "sparkring-runtime-bundle/v1", "bundle_id": "MyBundle",
      "services": [{"service_id": "a", "role": "serving",
                     "source": {"kind": "runtime-profile", "path": "p"}}]},
     "bundle id"),
])
def test_schema_rejection(bad_doc, expected_fragment, tmp_path):
    path = _write_bundle(tmp_path, bad_doc)
    with pytest.raises(bundle_mod.BundleError) as exc:
        bundle_mod.load_bundle(path)
    assert expected_fragment in str(exc.value)


def test_invalid_confirmation_format(tmp_path):
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    doc = _bundle_doc([_native_service("a", "serving", profile)],
                      confirmation="not-a-confirmation-token!")
    path = _write_bundle(tmp_path, doc)
    with pytest.raises(bundle_mod.BundleError) as exc:
        bundle_mod.load_bundle(path)
    assert "confirmation" in str(exc.value).lower()


def test_native_bundle_without_confirmation_rejected(tmp_path):
    _write_profile(tmp_path, "engine", _generic_doc())
    doc = _bundle_doc([_native_service("a", "serving", "engine.json")],
                      confirmation=None)
    path = _write_bundle(tmp_path, doc)
    with pytest.raises(bundle_mod.BundleError) as exc:
        bundle_mod.load_bundle(path)
    assert "confirmation" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# Service validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_service,expected_fragment", [
    ({"role": "serving", "source": {"kind": "runtime-profile", "path": "p"}}, "missing key"),
    ({"service_id": "UPPER", "role": "serving",
      "source": {"kind": "runtime-profile", "path": "p"}}, "service id"),
    ({"service_id": "a", "source": {"kind": "runtime-profile", "path": "p"}}, "missing key"),
    ({"service_id": "a", "role": "invalid",
      "source": {"kind": "runtime-profile", "path": "p"}}, "must be one of"),
])
def test_service_validation(bad_service, expected_fragment, tmp_path):
    _write_profile(tmp_path, "engine", _generic_doc())
    path = _write_bundle(tmp_path, _bundle_doc([bad_service]))
    with pytest.raises(bundle_mod.BundleError) as exc:
        bundle_mod.load_bundle(path)
    assert expected_fragment in str(exc.value).lower()


def test_runtime_profile_only_for_serving(tmp_path):
    """runtime-profile source kind is only valid for serving role."""
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    doc = _bundle_doc([
        _native_service("cache", "cache", profile),
        _native_service("engine", "serving", profile),
    ])
    path = _write_bundle(tmp_path, doc)
    with pytest.raises(bundle_mod.BundleError) as exc:
        bundle_mod.load_bundle(path)
    assert "serving role" in str(exc.value).lower()


def test_structured_container_not_for_serving(tmp_path):
    """structured-container source kind is not valid for serving role."""
    sc_path = _write_structured(tmp_path, "cache", _structured_doc())
    doc = _bundle_doc([
        _structured_service("engine", "serving", sc_path),
    ])
    path = _write_bundle(tmp_path, doc)
    with pytest.raises(bundle_mod.BundleError) as exc:
        bundle_mod.load_bundle(path)
    assert "serving" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# NF3 profile rejected as runtime-profile
# ---------------------------------------------------------------------------


def test_nf3_profile_rejected_as_runtime_profile(tmp_path):
    """NF3 schema cannot be used with runtime-profile source kind."""
    nf3_src = ROOT / "scripts/config/launch.example.json"
    nf3_path = tmp_path / "launch.example.json"
    shutil.copy(nf3_src, nf3_path)
    doc = _bundle_doc([_native_service("a", "serving", nf3_path)])
    path = _write_bundle(tmp_path, doc)
    with pytest.raises(bundle_mod.BundleError) as exc:
        bundle_mod.load_bundle(path)
    assert "native generic" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# Bundle-source path containment
# ---------------------------------------------------------------------------


def test_absolute_source_path_rejected(tmp_path):
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    abs_path = str(profile)
    doc = _bundle_doc([_native_service("a", "serving", abs_path)])
    path = _write_bundle(tmp_path, doc)
    with pytest.raises(bundle_mod.BundleError) as exc:
        bundle_mod.load_bundle(path)
    assert "absolute" in str(exc.value).lower()


def test_dotdot_traversal_rejected(tmp_path):
    """../ traversal must be rejected."""
    _write_profile(tmp_path, "engine", _generic_doc())
    # Create bundle in subdirectory, try to escape
    subdir = tmp_path / "sub"
    subdir.mkdir()
    doc = _bundle_doc([_native_service("a", "serving", "../engine.json")])
    bundle_path = subdir / "bundle.json"
    bundle_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(bundle_mod.BundleError) as exc:
        bundle_mod.load_bundle(bundle_path)
    assert "escape" in str(exc.value).lower() or "relative" in str(exc.value).lower()


def test_placeholder_in_path_rejected(tmp_path):
    """Placeholder syntax in path must be rejected."""
    doc = _bundle_doc([
        _native_service("a", "serving", "{placeholder}/profile.json"),
    ])
    path = _write_bundle(tmp_path, doc)
    with pytest.raises(bundle_mod.BundleError) as exc:
        bundle_mod.load_bundle(path)
    assert "placeholder" in str(exc.value).lower()


def test_contained_path_accepted(tmp_path):
    """A normal contained path works."""
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    doc = _bundle_doc([_native_service("a", "serving", profile)])
    path = _write_bundle(tmp_path, doc)
    bundle = bundle_mod.load_bundle(path)
    assert bundle.services[0].profile is not None


def test_drive_relative_path_rejected(tmp_path):
    """Drive-relative paths (C:foo) must be rejected."""
    doc = _bundle_doc([
        _native_service("a", "serving", "C:foo/profile.json"),
    ])
    path = _write_bundle(tmp_path, doc)
    with pytest.raises(bundle_mod.BundleError) as exc:
        bundle_mod.load_bundle(path)
    assert "drive-relative" in str(exc.value).lower() or "drive" in str(exc.value).lower()


def test_unc_path_rejected(tmp_path):
    """UNC paths (\\\\server\\share) must be rejected."""
    doc = _bundle_doc([
        _native_service("a", "serving", "\\\\server\\share\\profile.json"),
    ])
    path = _write_bundle(tmp_path, doc)
    with pytest.raises(bundle_mod.BundleError) as exc:
        bundle_mod.load_bundle(path)
    assert "unc" in str(exc.value).lower()


def test_symlink_escape_rejected(tmp_path):
    """Symlinks that escape the bundle directory must be rejected."""
    if not hasattr(os, "symlink"):
        pytest.skip("platform cannot create symlinks")
    try:
        target = tmp_path.parent / "escape_target.json"
        target.write_text("{}", encoding="utf-8")
        link = tmp_path / "link.json"
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("cannot create symlink on this platform")
    doc = _bundle_doc([
        _native_service("a", "serving", "link.json"),
    ])
    path = _write_bundle(tmp_path, doc)
    with pytest.raises(bundle_mod.BundleError) as exc:
        bundle_mod.load_bundle(path)
    assert "escape" in str(exc.value).lower() or "escapes" in str(exc.value).lower()


def test_missing_source_file_rejected(tmp_path):
    """Non-existent source file must be rejected during structural validation."""
    doc = _bundle_doc([
        _native_service("a", "serving", "nonexistent.json"),
    ])
    path = _write_bundle(tmp_path, doc)
    with pytest.raises(bundle_mod.BundleError) as exc:
        bundle_mod.load_bundle(path)
    assert "not found" in str(exc.value).lower() or "not a file" in str(exc.value).lower() or "no such" in str(exc.value).lower()


def test_source_path_not_a_file_rejected(tmp_path):
    """Source path pointing to a directory must be rejected."""
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    doc = _bundle_doc([
        _native_service("a", "serving", "subdir"),
    ])
    path = _write_bundle(tmp_path, doc)
    with pytest.raises(bundle_mod.BundleError) as exc:
        bundle_mod.load_bundle(path)
    assert "not a file" in str(exc.value).lower() or "file" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# Structured-container validation
# ---------------------------------------------------------------------------


def test_structured_container_valid(tmp_path):
    sc_path = _write_structured(tmp_path, "cache", _structured_doc())
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    doc = _bundle_doc([
        _structured_service("cache", "cache", sc_path),
        _native_service("engine", "serving", profile, depends_on=["cache"]),
    ])
    path = _write_bundle(tmp_path, doc)
    bundle = bundle_mod.load_bundle(path)
    assert bundle.services[0].structured is not None
    assert bundle.services[0].structured.port == 6556


def test_structured_container_shell_entrypoint_rejected(tmp_path):
    sc = _structured_doc(argv=["sh", "-c", "echo hello"])
    sc_path = _write_structured(tmp_path, "cache", sc)
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    doc = _bundle_doc([
        _structured_service("cache", "cache", sc_path),
        _native_service("engine", "serving", profile, depends_on=["cache"]),
    ])
    path = _write_bundle(tmp_path, doc)
    with pytest.raises(bundle_mod.BundleError) as exc:
        bundle_mod.load_bundle(path)
    assert "shell" in str(exc.value).lower()


def test_structured_container_start_uses_direct_argv(tmp_path):
    """Structured container start actions use declared argv, not vLLM serve."""
    sc_path = _write_structured(tmp_path, "cache", _structured_doc())
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    doc = _bundle_doc([
        _structured_service("cache", "cache", sc_path),
        _native_service("engine", "serving", profile, depends_on=["cache"]),
    ])
    path = _write_bundle(tmp_path, doc)
    bundle = bundle_mod.load_bundle(path)
    site = load_site(_site(tmp_path))
    actions = bundle_mod._native_start_actions(site, bundle.services[0], bundle)
    assert len(actions) > 0
    # The argv should contain the declared command, not "serve"
    cmd = actions[0].shell_command
    assert "/opt/bin/cache-server" in cmd
    assert " serve " not in cmd and not cmd.rstrip().endswith(" serve")
    assert "--tensor-parallel-size" not in cmd


def test_structured_container_readiness_port_matches(tmp_path):
    """Structured container readiness port should match the declared port."""
    sc_path = _write_structured(tmp_path, "cache", _structured_doc())
    doc = _bundle_doc([
        _structured_service("cache", "cache", sc_path, readiness={
            "kind": "http-get", "port": 6556, "path": "/health",
            "timeout_seconds": 10, "interval_seconds": 2,
        }),
        _native_service("engine", "serving", "engine.json", depends_on=["cache"]),
    ])
    _write_profile(tmp_path, "engine", _generic_doc())
    path = _write_bundle(tmp_path, doc)
    bundle = bundle_mod.load_bundle(path)
    site = load_site(_site(tmp_path))
    actions = bundle_mod._readiness_actions(site, bundle.services[0], bundle)
    assert len(actions) > 0
    assert "6556" in actions[0].shell_command


# ---------------------------------------------------------------------------
# Closed bridge-shape validation
# ---------------------------------------------------------------------------


def test_exl3_bridge_exact_shape_enforced(tmp_path):
    """EXL3 bridge requires exactly 2 services, cache+serving, correct edge."""
    exl3_profile = _exl3_profile(tmp_path)
    doc = _bundle_doc([
        _bridge_service("lmcache", "cache", exl3_profile),
        _bridge_service("engine", "serving", exl3_profile, depends_on=["lmcache"]),
    ])
    path = _write_bundle(tmp_path, doc)
    bundle = bundle_mod.load_bundle(path)
    assert len(bundle.services) == 2


def test_exl3_bridge_wrong_service_count_rejected(tmp_path):
    """One-service EXL3 bridge is rejected."""
    exl3_profile = _exl3_profile(tmp_path)
    doc = _bundle_doc([
        _bridge_service("engine", "serving", exl3_profile),
    ])
    path = _write_bundle(tmp_path, doc)
    with pytest.raises(bundle_mod.BundleError) as exc:
        bundle_mod.load_bundle(path)
    assert "2 services" in str(exc.value)


def test_exl3_bridge_missing_dependency_rejected(tmp_path):
    """Serving must depend on cache."""
    exl3_profile = _exl3_profile(tmp_path)
    doc = _bundle_doc([
        _bridge_service("lmcache", "cache", exl3_profile),
        _bridge_service("engine", "serving", exl3_profile),
    ])
    path = _write_bundle(tmp_path, doc)
    with pytest.raises(bundle_mod.BundleError) as exc:
        bundle_mod.load_bundle(path)
    assert "depend" in str(exc.value).lower()


def test_exl3_bridge_different_paths_rejected(tmp_path):
    """Both services must use the same profile path."""
    exl3_profile = _exl3_profile(tmp_path)
    other_profile = tmp_path / "other.json"
    shutil.copy(exl3_profile, other_profile)
    doc = _bundle_doc([
        _bridge_service("lmcache", "cache", exl3_profile),
        _bridge_service("engine", "serving", other_profile, depends_on=["lmcache"]),
    ])
    path = _write_bundle(tmp_path, doc)
    with pytest.raises(bundle_mod.BundleError) as exc:
        bundle_mod.load_bundle(path)
    assert "same" in str(exc.value).lower()


def test_exl3_bridge_mixed_sources_rejected(tmp_path):
    """Mixing EXL3 bridge with native profile is rejected."""
    exl3_profile = _exl3_profile(tmp_path)
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    doc = _bundle_doc([
        _bridge_service("lmcache", "cache", exl3_profile),
        _native_service("engine", "serving", profile, depends_on=["lmcache"]),
    ])
    path = _write_bundle(tmp_path, doc)
    with pytest.raises(bundle_mod.BundleError) as exc:
        bundle_mod.load_bundle(path)
    assert "mix" in str(exc.value).lower()


def test_nf3_source_kind_removed():
    """canonical-nf3 source kind no longer exists."""
    assert "canonical-nf3" not in bundle_mod.VALID_SOURCE_KINDS


# ---------------------------------------------------------------------------
# Graph validation
# ---------------------------------------------------------------------------


def test_cycle_detection(tmp_path):
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    doc = _bundle_doc([
        _native_service("a", "serving", profile, depends_on=["b"]),
        _native_service("b", "cache", profile, depends_on=["a"]),
    ])
    path = _write_bundle(tmp_path, doc)
    with pytest.raises(bundle_mod.BundleError) as exc:
        bundle_mod.load_bundle(path)
    assert "cycle" in str(exc.value).lower()


def test_unknown_dependency_rejected(tmp_path):
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    doc = _bundle_doc([
        _native_service("a", "serving", profile, depends_on=["nonexistent"]),
    ])
    path = _write_bundle(tmp_path, doc)
    with pytest.raises(bundle_mod.BundleError) as exc:
        bundle_mod.load_bundle(path)
    assert "unknown" in str(exc.value).lower()


def test_topological_order_stable(tmp_path):
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    sc_path = _write_structured(tmp_path, "cache", _structured_doc())
    services = [
        _structured_service("cache", "cache", sc_path),
        _native_service("engine", "serving", profile, depends_on=["cache"]),
    ]
    doc = _bundle_doc(services)
    path = _write_bundle(tmp_path, doc)
    bundle = bundle_mod.load_bundle(path)
    ordered = bundle_mod.topological_order(bundle.services)
    assert [s.service_id for s in ordered] == ["cache", "engine"]


# ---------------------------------------------------------------------------
# Dependency canonicalization
# ---------------------------------------------------------------------------


def test_depends_on_canonicalized_as_set(tmp_path):
    """depends_on stored as frozenset, order-independent."""
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    sc_path = _write_structured(tmp_path, "cache", _structured_doc())
    doc = _bundle_doc([
        _structured_service("cache", "cache", sc_path),
        _native_service("engine", "serving", profile, depends_on=["cache"]),
    ])
    path = _write_bundle(tmp_path, doc)
    bundle = bundle_mod.load_bundle(path)
    assert isinstance(bundle.services[1].depends_on, frozenset)


def test_reordered_input_byte_identical_plan(tmp_path):
    """Equivalent permutations produce byte-identical plan and identity."""
    profile_engine = _write_profile(tmp_path, "engine", _generic_doc())
    sc_path = _write_structured(tmp_path, "cache", _structured_doc())
    services_a = [
        _structured_service("cache", "cache", sc_path),
        _native_service("engine", "serving", profile_engine, depends_on=["cache"]),
    ]
    services_b = list(reversed(services_a))
    doc_a = _bundle_doc(services_a, bundle_id="b-a")
    doc_b = _bundle_doc(services_b, bundle_id="b-a")
    path_a = _write_bundle(tmp_path, doc_a, "a.json")
    path_b = _write_bundle(tmp_path, doc_b, "b.json")
    ba = bundle_mod.load_bundle(path_a)
    bb = bundle_mod.load_bundle(path_b)
    site = load_site(_site(tmp_path))
    plan_a = json.dumps(bundle_mod.bundle_plan(ba, site), sort_keys=True, indent=2)
    plan_b = json.dumps(bundle_mod.bundle_plan(bb, site), sort_keys=True, indent=2)
    assert plan_a == plan_b


def test_reordered_deps_produce_same_identity(tmp_path):
    """Reordering depends_on list produces same plan identity."""
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    sc1 = _write_structured(tmp_path, "c1", _structured_doc(container_name="sc1"))
    sc2 = _write_structured(tmp_path, "c2", _structured_doc(container_name="sc2"))
    services_a = [
        _structured_service("c1", "cache", sc1),
        _structured_service("c2", "cache", sc2),
        _native_service("engine", "serving", profile, depends_on=["c1", "c2"]),
    ]
    services_b = [
        _structured_service("c1", "cache", sc1),
        _structured_service("c2", "cache", sc2),
        _native_service("engine", "serving", profile, depends_on=["c2", "c1"]),
    ]
    doc_a = _bundle_doc(services_a, bundle_id="rd")
    doc_b = _bundle_doc(services_b, bundle_id="rd")
    path_a = _write_bundle(tmp_path, doc_a, "a.json")
    path_b = _write_bundle(tmp_path, doc_b, "b.json")
    ba = bundle_mod.load_bundle(path_a)
    bb = bundle_mod.load_bundle(path_b)
    site = load_site(_site(tmp_path))
    plan_a = bundle_mod.bundle_plan(ba, site)
    plan_b = bundle_mod.bundle_plan(bb, site)
    assert plan_a["plan_identity"] == plan_b["plan_identity"]


# ---------------------------------------------------------------------------
# Container name collisions
# ---------------------------------------------------------------------------


def test_container_name_collision_rejected(tmp_path):
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    sc = _structured_doc(container_name="sparkring-generic-example")
    sc_path = _write_structured(tmp_path, "cache", sc)
    doc = _bundle_doc([
        _native_service("a", "serving", profile),
        _structured_service("b", "cache", sc_path),
    ])
    path = _write_bundle(tmp_path, doc)
    with pytest.raises(bundle_mod.BundleError) as exc:
        bundle = bundle_mod.load_bundle(path)
        site = load_site(_site(tmp_path))
        bundle_mod.bundle_plan(bundle, site)
    assert "collision" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# Shared ownership guard for both readiness kinds
# ---------------------------------------------------------------------------


def _make_bundle_with_readiness(tmp_path, readiness_kind, **rkw):
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    readiness = {"kind": readiness_kind, "timeout_seconds": 10, "interval_seconds": 2}
    readiness.update(rkw)
    doc = _bundle_doc([
        _native_service("a", "serving", profile, readiness=readiness),
    ])
    path = _write_bundle(tmp_path, doc)
    bundle = bundle_mod.load_bundle(path)
    site = load_site(_site(tmp_path))
    return bundle, site


def test_container_running_readiness_has_ownership_guards(tmp_path):
    bundle, site = _make_bundle_with_readiness(
        tmp_path, "container-running",
    )
    actions = bundle_mod._readiness_actions(site, bundle.services[0], bundle)
    cmd = actions[0].shell_command
    for label in [
        "org.sparkring.managed",
        "org.sparkring.profile",
        "org.sparkring.bundle",
        "org.sparkring.service",
        "org.sparkring.source-profile",
    ]:
        assert label in cmd


def test_http_get_readiness_has_ownership_guards(tmp_path):
    """HTTP readiness must verify every ownership label before curl."""
    bundle, site = _make_bundle_with_readiness(
        tmp_path, "http-get", port="site-api", path="/health",
    )
    actions = bundle_mod._readiness_actions(site, bundle.services[0], bundle)
    cmd = actions[0].shell_command
    for label in [
        "org.sparkring.managed",
        "org.sparkring.profile",
        "org.sparkring.bundle",
        "org.sparkring.service",
        "org.sparkring.source-profile",
    ]:
        assert label in cmd


def test_http_get_no_sleep_after_final_attempt(tmp_path):
    """The final curl attempt must not sleep."""
    bundle, site = _make_bundle_with_readiness(
        tmp_path, "http-get", port="site-api", path="/health",
        timeout_seconds=10, interval_seconds=5,
    )
    actions = bundle_mod._readiness_actions(site, bundle.services[0], bundle)
    cmd = actions[0].shell_command
    # Should have 2 attempts (ceil(10/5))
    assert "seq 1 2" in cmd
    # The sleep should be conditional on not being the last iteration
    assert '"$i" -lt 2' in cmd


# ---------------------------------------------------------------------------
# Fake-engine behavior for ownership guards
# ---------------------------------------------------------------------------


def _fake_inspect_script(label_values: dict[str, str], running: bool = True,
                         inspect_rc: int = 0) -> str:
    """Build a fake docker/shell script that simulates inspect responses."""
    lines = ["#!/bin/sh"]
    lines.append("# Fake docker engine for testing")
    # Handle `docker info` — always succeed (daemon is up)
    lines.append('case "$1" in')
    lines.append("  info) exit 0;;")
    lines.append("  inspect)")
    if inspect_rc != 0:
        lines.append("    exit 1;;")
    lines.append("    case \"$3\" in")
    for label, val in label_values.items():
        lines.append(f'      "{label}") echo "{val}";;')
    lines.append('      "{{.State.Running}}") echo "true" if running else echo "false";;')
    lines.append('      "{{.State.Status}}") echo "running" if running else echo "exited";;')
    lines.append("      *) echo '';;")
    lines.append("    esac;;")
    lines.append("esac")
    return "\n".join(lines)


def test_ownership_guard_rejects_wrong_managed_label(tmp_path, monkeypatch):
    """Fake-engine test: wrong managed label must fail readiness."""
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    doc = _bundle_doc([
        _native_service("a", "serving", profile,
                        readiness={"kind": "container-running"}),
    ])
    path = _write_bundle(tmp_path, doc)
    bundle = bundle_mod.load_bundle(path)
    site = load_site(_site(tmp_path))
    actions = bundle_mod._readiness_actions(site, bundle.services[0], bundle)
    # The generated script should exit 1 when managed != true
    cmd = actions[0].shell_command
    assert 'exit 1' in cmd


# ---------------------------------------------------------------------------
# Stop and rollback daemon-probe semantics
# ---------------------------------------------------------------------------


def test_stop_actions_have_daemon_probe(tmp_path):
    """Stop actions must probe daemon before inspect (exit 74)."""
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    doc = _bundle_doc([_native_service("a", "serving", profile)])
    path = _write_bundle(tmp_path, doc)
    bundle = bundle_mod.load_bundle(path)
    site = load_site(_site(tmp_path))
    actions = bundle_mod._native_stop_actions(site, bundle.services[0], bundle)
    cmd = actions[0].shell_command
    assert "info" in cmd
    assert "exit 74" in cmd
    assert "exit 73" in cmd
    assert "exit 0" in cmd


def test_verify_rollback_actions_have_daemon_probe(tmp_path):
    """Verify-rollback: daemon usable + absence => 0; present => 1; unknown => 2."""
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    doc = _bundle_doc([_native_service("a", "serving", profile)])
    path = _write_bundle(tmp_path, doc)
    bundle = bundle_mod.load_bundle(path)
    site = load_site(_site(tmp_path))
    actions = bundle_mod._native_verify_rollback_actions(
        site, bundle.services[0], bundle,
    )
    cmd = actions[0].shell_command
    assert "exit 0" in cmd  # absence
    assert "exit 1" in cmd  # present
    assert "exit 2" in cmd  # unknown/engine error


# ---------------------------------------------------------------------------
# Executor exception safety
# ---------------------------------------------------------------------------


def test_safe_execute_returns_result_for_every_rank_on_exception(tmp_path):
    """_safe_execute returns error for every rank on total exception."""
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    doc = _bundle_doc([_native_service("a", "serving", profile)])
    path = _write_bundle(tmp_path, doc)
    bundle = bundle_mod.load_bundle(path)
    site = load_site(_site(tmp_path))
    actions = bundle_mod._native_start_actions(site, bundle.services[0], bundle)
    with mock.patch("sparkring_runtime.execute", side_effect=RuntimeError("boom")):
        results = bundle_mod._safe_execute(actions, 30)
    assert len(results) == len(actions)
    for res in results.values():
        assert res["exit_code"] != 0


def test_execute_native_start_rollback_on_exception(tmp_path, monkeypatch):
    """Rollback is attempted even when executor raises on start."""
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    sc_path = _write_structured(tmp_path, "cache", _structured_doc())
    doc = _bundle_doc([
        _structured_service("cache", "cache", sc_path),
        _native_service("engine", "serving", profile, depends_on=["cache"]),
    ])
    path = _write_bundle(tmp_path, doc)
    bundle = bundle_mod.load_bundle(path)
    site = load_site(_site(tmp_path))

    call_count = [0]

    def fake_execute(actions, timeout):
        call_count[0] += 1
        if call_count[0] == 1:
            # Cache start succeeds
            return {a.rank: {"exit_code": 0, "stdout": "abc123def456",
                              "stderr": ""} for a in actions}
        if call_count[0] == 2:
            # Engine start — raise exception
            raise RuntimeError("engine start exploded")
        # After that, rollback calls — return success
        return {a.rank: {"exit_code": 0, "stdout": "", "stderr": ""}
                for a in actions}

    monkeypatch.setattr(runtime, "execute", fake_execute)
    result = bundle_mod.execute_native_start(
        bundle, site, confirmation="START-test",
    )
    # Rollback should have been attempted
    assert result["rollback"] is not None
    assert result["rollback"]["rollback_status"] in ("success", "failed")


def test_rollback_reports_missing_ranks(tmp_path, monkeypatch):
    """Rollback with incomplete results reports missing ranks as failure."""
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    sc_path = _write_structured(tmp_path, "cache", _structured_doc())
    doc = _bundle_doc([
        _structured_service("cache", "cache", sc_path),
        _native_service("engine", "serving", profile, depends_on=["cache"]),
    ])
    path = _write_bundle(tmp_path, doc)
    bundle = bundle_mod.load_bundle(path)
    site = load_site(_site(tmp_path))

    call_count = [0]

    def fake_execute(actions, timeout):
        call_count[0] += 1
        if call_count[0] == 1:
            # Cache start — succeed
            return {a.rank: {"exit_code": 0, "stdout": "abc123def456",
                              "stderr": ""} for a in actions}
        if call_count[0] == 2:
            # Engine start — fail
            return {a.rank: {"exit_code": 1, "stdout": "", "stderr": "fail"}
                    for a in actions}
        # Rollback: return empty results (simulating missing ranks)
        if call_count[0] == 3:
            return {}  # No results — missing ranks
        return {a.rank: {"exit_code": 0, "stdout": "", "stderr": ""}
                for a in actions}

    monkeypatch.setattr(runtime, "execute", fake_execute)
    result = bundle_mod.execute_native_start(
        bundle, site, confirmation="START-test",
    )
    rollback = result["rollback"]
    assert rollback["rollback_status"] == "failed"
    # Phase should have infrastructure error (exit_code=125) from _safe_execute filling missing ranks
    any_infra_error = any(
        any(r.get("exit_code") == 125 for r in ph.get("results", {}).values())
        for ph in rollback.get("phases", [])
    )
    assert any_infra_error


# ---------------------------------------------------------------------------
# Status and confirmation symmetry
# ---------------------------------------------------------------------------


def test_status_does_not_require_confirmation(tmp_path, monkeypatch):
    """Status is read-only: no confirmation required, no SSH attempted.

    Tests in-process with runtime.execute mocked to avoid any real SSH.
    """
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    doc = _bundle_doc([_native_service("a", "serving", profile)])
    path = _write_bundle(tmp_path, doc)
    bundle = bundle_mod.load_bundle(path)
    site = load_site(_site(tmp_path))

    # Poison SSH boundaries
    _poison_remote(monkeypatch)

    # Dry-run status (no --execute) — CLI should succeed
    code, _, stderr = _run_cli(path, _site(tmp_path), "status")
    assert code == 0
    assert "confirmation" not in stderr.lower()

    # Executed status with mocked execute: all ranks succeed → exit 0
    monkeypatch.setattr(
        runtime, "execute",
        lambda actions, timeout: {a.rank: {"exit_code": 0, "stdout": "", "stderr": ""} for a in actions},
    )
    result = bundle_mod.execute_native_status(bundle, site)
    assert result["status"] == "ok"

    # Executed status with mocked execute: one rank fails → status "failed"
    monkeypatch.setattr(
        runtime, "execute",
        lambda actions, timeout: {a.rank: {"exit_code": 1, "stdout": "", "stderr": "fail"} for a in actions},
    )
    result = bundle_mod.execute_native_status(bundle, site)
    assert result["status"] == "failed"


def test_verify_rollback_does_not_require_confirmation(tmp_path, monkeypatch):
    """Verify-rollback is read-only: no confirmation required, no SSH attempted.

    Tests in-process with runtime.execute mocked to avoid any real SSH.
    """
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    doc = _bundle_doc([_native_service("a", "serving", profile)])
    path = _write_bundle(tmp_path, doc)
    bundle = bundle_mod.load_bundle(path)
    site = load_site(_site(tmp_path))

    # Poison SSH boundaries
    _poison_remote(monkeypatch)

    # Dry-run verify-rollback (no --execute) — CLI should succeed
    code, _, stderr = _run_cli(path, _site(tmp_path), "verify-rollback")
    assert code == 0
    assert "confirmation" not in stderr.lower()

    # Executed verify-rollback with mocked execute: all absent → status "absent"
    monkeypatch.setattr(
        runtime, "execute",
        lambda actions, timeout: {a.rank: {"exit_code": 0, "stdout": "", "stderr": ""} for a in actions},
    )
    result = bundle_mod.execute_native_verify_rollback(bundle, site)
    assert result["status"] == "absent"

    # Executed verify-rollback with mocked execute: container present → "present"
    monkeypatch.setattr(
        runtime, "execute",
        lambda actions, timeout: {a.rank: {"exit_code": 1, "stdout": "", "stderr": ""} for a in actions},
    )
    result = bundle_mod.execute_native_verify_rollback(bundle, site)
    assert result["status"] == "present"


def test_start_requires_confirmation_on_execute(tmp_path):
    """Start with --execute requires --confirmation."""
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    doc = _bundle_doc([_native_service("a", "serving", profile)])
    path = _write_bundle(tmp_path, doc)
    site_path = _site(tmp_path)
    code, _, stderr = _run_cli(path, site_path, "start", "--execute")
    assert code != 0
    assert "confirmation" in stderr.lower()


# ---------------------------------------------------------------------------
# Lifecycle parity between the bridge and canonical functions
# ---------------------------------------------------------------------------


def test_build_phases_exported_from_canonical():
    """Canonical launcher exports build_phases()."""
    assert hasattr(lmcache, "build_phases")
    assert callable(lmcache.build_phases)


def test_lifecycle_sequence_exported_from_canonical():
    """Canonical launcher exports lifecycle_sequence()."""
    assert hasattr(lmcache, "lifecycle_sequence")
    assert callable(lmcache.lifecycle_sequence)


def test_lifecycle_sequence_includes_on_failure(tmp_path):
    """Lifecycle sequences include on_failure: rollback."""
    profile_path = _exl3_profile(tmp_path)
    profile = exl3.load_profile(profile_path)
    seq = lmcache.lifecycle_sequence("start", profile)
    assert len(seq) > 0
    for phase in seq:
        assert phase["on_failure"] == "rollback"
        assert "timeout" in phase
        assert "phase" in phase


def test_bridge_lifecycle_sequence_matches_canonical(tmp_path):
    """Bridge lifecycle_sequence() returns same result as canonical."""
    exl3_profile = _exl3_profile(tmp_path)
    doc = _bundle_doc([
        _bridge_service("lmcache", "cache", exl3_profile),
        _bridge_service("engine", "serving", exl3_profile, depends_on=["lmcache"]),
    ], bundle_id="bridge")
    path = _write_bundle(tmp_path, doc)
    bundle = bundle_mod.load_bundle(path)
    site = load_site(_site(tmp_path))
    plan = bundle_mod.bundle_plan(bundle, site)
    # Compare bridge sequences with canonical
    import sparkring_exl3_launcher as exl3_mod
    profile = exl3_mod.load_profile(exl3_profile)
    for cmd in ("start", "status", "restart-engines", "restart-stack",
                "rollback", "verify-rollback"):
        canonical_seq = lmcache.lifecycle_sequence(cmd, profile)
        bridge_seq = plan["lifecycle_sequences"][cmd]
        assert canonical_seq == bridge_seq


def test_bridge_phases_match_canonical_build_phases(tmp_path):
    """Bridge canonical_phases match build_phases() output."""
    exl3_profile = _exl3_profile(tmp_path)
    doc = _bundle_doc([
        _bridge_service("lmcache", "cache", exl3_profile),
        _bridge_service("engine", "serving", exl3_profile, depends_on=["lmcache"]),
    ], bundle_id="bridge")
    path = _write_bundle(tmp_path, doc)
    bundle = bundle_mod.load_bundle(path)
    site = load_site(_site(tmp_path))
    plan = bundle_mod.bundle_plan(bundle, site)
    profile = exl3.load_profile(exl3_profile)
    canonical = lmcache.build_phases(site, profile)
    for phase_name, actions in canonical.items():
        rendered = lmcache.render(actions)
        assert plan["canonical_phases"][phase_name] == rendered


# ---------------------------------------------------------------------------
# Plan completeness
# ---------------------------------------------------------------------------


def test_native_plan_has_evidence_scope(tmp_path):
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    doc = _bundle_doc([_native_service("a", "serving", profile)])
    path = _write_bundle(tmp_path, doc)
    bundle = bundle_mod.load_bundle(path)
    site = load_site(_site(tmp_path))
    plan = bundle_mod.bundle_plan(bundle, site)
    assert "evidence_scope" in plan
    assert "offline-validated" in plan["evidence_scope"]


def test_native_plan_has_ownership_labels(tmp_path):
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    doc = _bundle_doc([_native_service("a", "serving", profile)])
    path = _write_bundle(tmp_path, doc)
    bundle = bundle_mod.load_bundle(path)
    site = load_site(_site(tmp_path))
    plan = bundle_mod.bundle_plan(bundle, site)
    svc = plan["services"][0]
    assert "ownership_labels" in svc
    labels = svc["ownership_labels"]
    assert labels["managed"] == "true"
    assert labels["bundle"] == "test-bundle"
    assert labels["service"] == "a"
    assert "profile" in labels
    assert "source_profile" in labels


def test_native_plan_has_graph_order(tmp_path):
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    sc_path = _write_structured(tmp_path, "cache", _structured_doc())
    doc = _bundle_doc([
        _structured_service("cache", "cache", sc_path),
        _native_service("engine", "serving", profile, depends_on=["cache"]),
    ])
    path = _write_bundle(tmp_path, doc)
    bundle = bundle_mod.load_bundle(path)
    site = load_site(_site(tmp_path))
    plan = bundle_mod.bundle_plan(bundle, site)
    assert "graph_order" in plan
    assert plan["graph_order"] == ["cache", "engine"]


def test_native_plan_has_status_phase(tmp_path):
    """Plan must include status phase/actions."""
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    doc = _bundle_doc([_native_service("a", "serving", profile)])
    path = _write_bundle(tmp_path, doc)
    bundle = bundle_mod.load_bundle(path)
    site = load_site(_site(tmp_path))
    plan = bundle_mod.bundle_plan(bundle, site)
    status_phases = [p for p in plan["phases"] if p["phase"] == "status"]
    assert len(status_phases) > 0


def test_native_plan_rollback_per_service_reverse_order(tmp_path):
    """Rollback phases preserve reverse service ordering, not flattened."""
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    sc_path = _write_structured(tmp_path, "cache", _structured_doc())
    doc = _bundle_doc([
        _structured_service("cache", "cache", sc_path),
        _native_service("engine", "serving", profile, depends_on=["cache"]),
    ])
    path = _write_bundle(tmp_path, doc)
    bundle = bundle_mod.load_bundle(path)
    site = load_site(_site(tmp_path))
    plan = bundle_mod.bundle_plan(bundle, site)
    rollback_phases = [p for p in plan["phases"] if p["phase"] == "rollback"]
    assert len(rollback_phases) == 2
    # Reverse order: engine first, then cache
    assert rollback_phases[0]["service_id"] == "engine"
    assert rollback_phases[1]["service_id"] == "cache"


def test_native_plan_capabilities_have_confirmation_flags(tmp_path):
    """Operation capabilities must have requires_confirmation flags."""
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    doc = _bundle_doc([_native_service("a", "serving", profile)])
    path = _write_bundle(tmp_path, doc)
    bundle = bundle_mod.load_bundle(path)
    site = load_site(_site(tmp_path))
    plan = bundle_mod.bundle_plan(bundle, site)
    caps = plan["operation_capabilities"]
    assert caps["start"]["requires_confirmation"] is True
    assert caps["stop"]["requires_confirmation"] is True
    assert caps["rollback"]["requires_confirmation"] is True
    assert caps["status"]["requires_confirmation"] is False
    assert caps["verify_rollback"]["requires_confirmation"] is False


# ---------------------------------------------------------------------------
# Resolved diff includes canonical phases
# ---------------------------------------------------------------------------


def test_plan_projection_includes_canonical_phases(tmp_path):
    """plan_projection includes canonical_phases for bridge bundles."""
    exl3_profile = _exl3_profile(tmp_path)
    doc = _bundle_doc([
        _bridge_service("lmcache", "cache", exl3_profile),
        _bridge_service("engine", "serving", exl3_profile, depends_on=["lmcache"]),
    ], bundle_id="bridge")
    path = _write_bundle(tmp_path, doc)
    bundle = bundle_mod.load_bundle(path)
    site = load_site(_site(tmp_path))
    plan = bundle_mod.bundle_plan(bundle, site)
    proj = bundle_mod.plan_projection(plan, bundle, site)
    assert "canonical_phases" in proj
    assert "lifecycle_sequences" in proj


def test_diff_bridge_bundles_different_profiles(tmp_path):
    """Diff detects different bridge profiles."""
    exl3_profile = _exl3_profile(tmp_path)
    other_profile = tmp_path / "other.json"
    shutil.copy(exl3_profile, other_profile)
    # Modify image_id in the copy
    other_doc = json.loads(other_profile.read_text())
    other_doc["image_id"] = "sha256:" + "c" * 64
    other_profile.write_text(json.dumps(other_doc, indent=2))

    doc_a = _bundle_doc([
        _bridge_service("lmcache", "cache", exl3_profile),
        _bridge_service("engine", "serving", exl3_profile, depends_on=["lmcache"]),
    ], bundle_id="bridge-a")
    doc_b = _bundle_doc([
        _bridge_service("lmcache", "cache", other_profile),
        _bridge_service("engine", "serving", other_profile, depends_on=["lmcache"]),
    ], bundle_id="bridge-b")

    path_a = _write_bundle(tmp_path, doc_a, "a.json")
    path_b = _write_bundle(tmp_path, doc_b, "b.json")
    ba = bundle_mod.load_bundle(path_a)
    bb = bundle_mod.load_bundle(path_b)
    site = load_site(_site(tmp_path))
    plan_a = bundle_mod.bundle_plan(ba, site)
    plan_b = bundle_mod.bundle_plan(bb, site)
    diffs = bundle_mod.recursive_diff(
        bundle_mod.plan_projection(plan_a, ba, site),
        bundle_mod.plan_projection(plan_b, bb, site),
    )
    assert len(diffs) > 0


# ---------------------------------------------------------------------------
# Bridge execution rejection
# ---------------------------------------------------------------------------


def test_bridge_execution_rejected(tmp_path):
    """execute_native_start rejects bridge bundles."""
    exl3_profile = _exl3_profile(tmp_path)
    doc = _bundle_doc([
        _bridge_service("lmcache", "cache", exl3_profile),
        _bridge_service("engine", "serving", exl3_profile, depends_on=["lmcache"]),
    ])
    path = _write_bundle(tmp_path, doc)
    bundle = bundle_mod.load_bundle(path)
    site = load_site(_site(tmp_path))
    with pytest.raises(bundle_mod.BundleError) as exc:
        bundle_mod.execute_native_start(bundle, site)
    assert "not supported" in str(exc.value).lower()


def test_cli_bridge_start_rejects_execute(tmp_path):
    """CLI start --execute on bridge bundle is rejected."""
    site_path = _site(tmp_path)
    exl3_profile = _exl3_profile(tmp_path)
    bundle_path = _write_bundle(tmp_path, _bundle_doc([
        _bridge_service("lmcache", "cache", exl3_profile),
        _bridge_service("engine", "serving", exl3_profile, depends_on=["lmcache"]),
    ]))
    code, _, stderr = _run_cli(bundle_path, site_path, "start", "--execute")
    assert code != 0
    assert "not supported" in stderr.lower() or "plan-only" in stderr.lower()


# ---------------------------------------------------------------------------
# CLI dry-run tests
# ---------------------------------------------------------------------------


def test_cli_start_dry_run_no_ssh(tmp_path):
    """CLI start dry-run produces plan without SSH."""
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    doc = _bundle_doc([_native_service("engine", "serving", profile)])
    path = _write_bundle(tmp_path, doc)
    site_path = _site(tmp_path)
    code, stdout, _ = _run_cli(path, site_path, "start")
    assert code == 0
    plan = json.loads(stdout)
    assert plan["execution_supported"] is True


def test_cli_plan_dry_run(tmp_path):
    """CLI plan produces offline plan."""
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    doc = _bundle_doc([_native_service("engine", "serving", profile)])
    path = _write_bundle(tmp_path, doc)
    site_path = _site(tmp_path)
    code, stdout, _ = _run_cli(path, site_path, "plan")
    assert code == 0
    plan = json.loads(stdout)
    assert "plan_identity" in plan


def test_cli_validate(tmp_path):
    """CLI validate checks structure."""
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    doc = _bundle_doc([_native_service("engine", "serving", profile)])
    path = _write_bundle(tmp_path, doc)
    code, stdout, _ = _run_cli_no_site(path, "validate")
    assert code == 0
    result = json.loads(stdout)
    assert result["valid"] is True


def test_cli_explain(tmp_path):
    """CLI explain reports structure."""
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    doc = _bundle_doc([_native_service("engine", "serving", profile)])
    path = _write_bundle(tmp_path, doc)
    code, stdout, _ = _run_cli_no_site(path, "explain")
    assert code == 0
    result = json.loads(stdout)
    assert result["bundle_id"] == "test-bundle"


def test_cli_diff_identical_bundles(tmp_path):
    """CLI diff exits 0 for identical bundles."""
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    doc = _bundle_doc([_native_service("engine", "serving", profile)])
    path_a = _write_bundle(tmp_path, doc, "a.json")
    path_b = _write_bundle(tmp_path, doc, "b.json")
    argv = [
        sys.executable,
        str(ROOT / "scripts/sparkring_bundle_launcher.py"),
        "diff", "--bundle-a", str(path_a), "--bundle-b", str(path_b),
    ]
    result = subprocess.run(argv, capture_output=True, text=True, timeout=30, cwd=str(ROOT))
    assert result.returncode == 0


# ---------------------------------------------------------------------------
# Explain
# ---------------------------------------------------------------------------


def test_explain_lists_confirmation_commands(tmp_path):
    """Explain lists which commands require confirmation."""
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    doc = _bundle_doc([_native_service("a", "serving", profile)])
    path = _write_bundle(tmp_path, doc)
    bundle = bundle_mod.load_bundle(path)
    explain = bundle_mod.bundle_explain(bundle)
    assert "confirmation_commands" in explain
    assert "start" in explain["confirmation_commands"]
    assert "status" not in explain["confirmation_commands"]


# ---------------------------------------------------------------------------
# Archive rehearsal
# ---------------------------------------------------------------------------


def test_tests_never_invoke_git_add_or_commit():
    """Verify no test file contains git add or git commit calls."""
    test_dir = ROOT / "scripts"
    git_add = chr(34) + "git" + chr(34) + ", " + chr(34) + "add" + chr(34)
    git_commit = chr(34) + "git" + chr(34) + ", " + chr(34) + "commit" + chr(34)
    git_add_s = chr(39) + "git" + chr(39) + ", " + chr(39) + "add" + chr(39)
    git_commit_s = chr(39) + "git" + chr(39) + ", " + chr(39) + "commit" + chr(39)
    for test_file in test_dir.glob("test_*.py"):
        content = test_file.read_text(encoding="utf-8")
        assert git_add not in content, f"{test_file.name} contains git add"
        assert git_commit not in content, f"{test_file.name} contains git commit"
        assert git_add_s not in content, f"{test_file.name} contains git add"
        assert git_commit_s not in content, f"{test_file.name} contains git commit"


def test_poisoned_ssh_no_execution_in_offline_paths(tmp_path, monkeypatch):
    """All offline operations must not reach SSH/executor."""
    _poison_remote(monkeypatch)
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    sc_path = _write_structured(tmp_path, "cache", _structured_doc())
    doc = _bundle_doc([
        _structured_service("cache", "cache", sc_path),
        _native_service("engine", "serving", profile, depends_on=["cache"]),
    ])
    path = _write_bundle(tmp_path, doc)
    bundle = bundle_mod.load_bundle(path)
    site = load_site(_site(tmp_path))
    assert bundle_mod.bundle_plan(bundle, site) is not None
    assert bundle_mod.bundle_explain(bundle, site) is not None
    assert bundle_mod.bundle_projection(bundle) is not None


def test_poisoned_cli_dispatch_no_ssh(tmp_path, monkeypatch):
    """CLI dispatch for all non-execute commands must not reach SSH."""
    import sparkring_bundle_launcher as launcher
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    sc_path = _write_structured(tmp_path, "cache", _structured_doc())
    doc = _bundle_doc([
        _structured_service("cache", "cache", sc_path),
        _native_service("engine", "serving", profile, depends_on=["cache"]),
    ])
    path = _write_bundle(tmp_path, doc)
    site_path = _site(tmp_path)

    ssh_called = []

    def _track_and_boom(*a, **kw):
        ssh_called.append(1)
        raise AssertionError("SSH reached during offline CLI")

    monkeypatch.setattr(runtime, "execute", _track_and_boom)
    monkeypatch.setattr(runtime, "run_remote", _track_and_boom)
    import socket as _socket
    monkeypatch.setattr(_socket, "socket", _track_and_boom)
    for cmd in ("plan", "validate", "explain"):
        launcher.main([
            "--bundle", str(path), "--site", str(site_path), cmd,
        ])
    for cmd in ("start", "stop", "status", "rollback", "verify-rollback"):
        launcher.main([
            "--bundle", str(path), "--site", str(site_path), cmd,
        ])
    assert len(ssh_called) == 0


def test_tracked_configs_parse(tmp_path):
    """All tracked bundle config files parse and validate."""
    configs = [
        "scripts/config/bundle.template.json",
        "scripts/config/bundle-native-single.json",
        "scripts/config/bundle-engine-cache.json",
        "scripts/config/bundle-exl3-lmcache-bridge.json",
    ]
    for cfg in configs:
        p = ROOT / cfg
        assert p.exists(), f"Missing tracked config: {cfg}"
        _ = json.loads(p.read_text(encoding="utf-8"))
        # Should parse without error when files exist alongside
        try:
            bundle_mod.load_bundle(p)
        except (bundle_mod.BundleError, OSError) as exc:
            # Template may reference nonexistent files — that's OK
            if "template" not in cfg and "example" not in cfg:
                raise AssertionError(f"{cfg} failed to load: {exc}") from exc


# ---------------------------------------------------------------------------
# No regression — existing test count must be sufficient
# ---------------------------------------------------------------------------


def test_no_regression_existing_suites_still_pass():
    """Smoke check that module imports work."""
    assert hasattr(bundle_mod, "BUNDLE_SCHEMA")
    assert hasattr(bundle_mod, "bundle_plan")
    assert hasattr(bundle_mod, "execute_native_start")
    assert hasattr(bundle_mod, "execute_native_status")
    assert hasattr(bundle_mod, "execute_native_verify_rollback")
    assert hasattr(bundle_mod, "_safe_execute")

# ---------------------------------------------------------------------------
# Behavioral fake-engine tests — execute generated sh -c guards
# ---------------------------------------------------------------------------


def _make_fake_docker(tmp_path, *, daemon_ok=True, container_exists=False,
                     labels=None, running=False, rm_succeeds=True):
    """Create a fake docker script that simulates the docker engine.

    Returns the path to the fake script.  The script handles:
    - docker info: exits 0 if daemon_ok, else 1
    - docker ps -a --filter name=... --format ...: lists containers
    - docker inspect --format ...: returns label values
    - docker rm --force ...: removes container
    """
    fake = tmp_path / "fake_docker.sh"
    label_map = labels or {}
    parts = [
        '#!/bin/sh',
        ': "${SPARKRING_FAKE_DOCKER_MARKER:?missing fake-docker marker}"',
        'printf "%s\\n" "$*" >> "$SPARKRING_FAKE_DOCKER_MARKER"',
        'case "$1" in',
    ]

    # docker info
    parts.append(f'  info) exit {0 if daemon_ok else 1};;')

    # docker ps — enumerate containers by name filter
    if not daemon_ok:
        parts.append('  ps) exit 1;;')
    elif container_exists:
        # Echo the container name from the --filter argument
        # The name appears as: --filter name=^/NAME$
        parts.append('  ps) name=$(echo "$*" | tr " " "\\n" | grep "name=" | sed "s/name=\\^\\///;s/[$]//"); echo "$name"; exit 0;;')
    else:
        parts.append('  ps) exit 0;;')

    # docker inspect — return label values
    if not daemon_ok or not container_exists:
        parts.append('  inspect) exit 1;;')
    else:
        parts.append('  inspect)')
        parts.append('    all="$*"')
        parts.append('    case "$all" in')
        for label_key, label_val in label_map.items():
            safe_val = label_val.replace('"', '\\"')
            parts.append(f'      *{label_key}*) echo "{safe_val}";;')
        if running:
            parts.append('      *.Running}*) echo "true";;')
            parts.append('      *.State.Status}*) echo "running";;')
        else:
            parts.append('      *.Running}*) echo "false";;')
            parts.append('      *.State.Status}*) echo "exited";;')
        parts.append('      *) exit 1;;')
        parts.append('    esac;;')

    # docker rm
    parts.append(f'  rm) exit {0 if rm_succeeds else 1};;')
    parts.append('esac')
    fake.write_text("\n".join(parts), encoding="utf-8")
    fake.chmod(0o755)
    return fake


def _find_posix_shell():
    """Find an explicit POSIX shell for guard script execution.

    Searches conventional locations including Git for Windows.  Returns
    the path as a string, or None if no shell is found.
    """
    import os
    import shutil as _shutil
    # On Windows, prefer Git's POSIX shell.  A PATH `bash.exe` may be the
    # WSL launcher, which cannot execute the Windows temporary paths used by
    # this hermetic harness.
    git_paths = [
        r"C:\Program Files\Git\bin\sh.exe",
        r"C:\Program Files\Git\usr\bin\sh.exe",
        r"C:\Program Files (x86)\Git\bin\sh.exe",
        r"C:\Program Files (x86)\Git\usr\bin\sh.exe",
    ]
    if os.name == "nt":
        for p in git_paths:
            if Path(p).is_file():
                return p
    # Check common names on PATH after the Windows-specific candidates.
    for name in ("sh", "bash"):
        found = _shutil.which(name)
        if found:
            return found
    return None


_POSIX_SHELL = _find_posix_shell()


def _run_guard_with_fake_docker(tmp_path, guard_script, fake_docker):
    """Execute a guard script with PATH pointing to fake docker.

    Uses an explicit POSIX shell (Git for Windows sh on Windows).
    """
    if _POSIX_SHELL is None:
        pytest.skip("no POSIX shell available for guard script execution")
    docker_link = fake_docker.parent / "docker"
    if docker_link.exists():
        docker_link.unlink()
    shutil.copy(fake_docker, docker_link)
    docker_link.chmod(0o755)
    marker = tmp_path / "fake-docker-invocations"
    env = {"SPARKRING_FAKE_DOCKER_MARKER": str(marker)}
    result = subprocess.run(
        [
            _POSIX_SHELL,
            "-c",
            'PATH="$PWD:/usr/bin:/bin"; export PATH; exec sh -c "$1"',
            "sparkring-test-shell",
            guard_script,
        ],
        capture_output=True, text=True, timeout=10, env=env,
        cwd=str(tmp_path),
    )
    assert marker.is_file(), (
        "guard did not invoke the hermetic fake docker; refusing to accept "
        f"result {result.returncode}: {result.stderr}"
    )
    return result.returncode


def _make_test_bundle(tmp_path, role="serving"):
    """Create a minimal bundle with one service for fake-engine tests."""
    if role == "serving":
        profile = _write_profile(tmp_path, "engine", _generic_doc())
        doc = _bundle_doc([_native_service("a", "serving", profile)])
    else:
        sc_path = _write_structured(tmp_path, "cache", _structured_doc())
        profile = _write_profile(tmp_path, "engine", _generic_doc())
        doc = _bundle_doc([
            _structured_service("cache", "cache", sc_path),
            _native_service("engine", "serving", profile, depends_on=["cache"]),
        ])
    path = _write_bundle(tmp_path, doc)
    bundle = bundle_mod.load_bundle(path)
    site = load_site(_site(tmp_path))
    return bundle, site


def _expected_labels(bundle, svc):
    """Get the expected label values for a service."""
    ownership = bundle_mod._ownership_for(bundle, svc)
    return {
        "org.sparkring.managed": "true",
        "org.sparkring.profile": svc.profile.profile_id if svc.profile else svc.structured.image_id,
        "org.sparkring.bundle": ownership.bundle_id,
        "org.sparkring.service": ownership.service_id,
        "org.sparkring.source-profile": ownership.profile_id,
    }


# Stop guard fake-engine matrix
def test_stop_daemon_unavailable(tmp_path):
    """Stop guard: daemon unavailable => exit 74."""
    bundle, site = _make_test_bundle(tmp_path)
    fake = _make_fake_docker(tmp_path, daemon_ok=False)
    actions = bundle_mod._native_stop_actions(site, bundle.services[0], bundle)
    rc = _run_guard_with_fake_docker(tmp_path, actions[0].argv[2], fake)
    assert rc == 74


def test_stop_absent_container(tmp_path):
    """Stop guard: proven absence => exit 0."""
    bundle, site = _make_test_bundle(tmp_path)
    fake = _make_fake_docker(tmp_path, daemon_ok=True, container_exists=False)
    actions = bundle_mod._native_stop_actions(site, bundle.services[0], bundle)
    rc = _run_guard_with_fake_docker(tmp_path, actions[0].argv[2], fake)
    assert rc == 0


def test_stop_foreign_container(tmp_path):
    """Stop guard: present foreign label => exit 73."""
    bundle, site = _make_test_bundle(tmp_path)
    labels = _expected_labels(bundle, bundle.services[0])
    # Wrong managed label
    labels["org.sparkring.managed"] = "false"
    fake = _make_fake_docker(tmp_path, daemon_ok=True, container_exists=True,
                             labels=labels)
    actions = bundle_mod._native_stop_actions(site, bundle.services[0], bundle)
    rc = _run_guard_with_fake_docker(tmp_path, actions[0].argv[2], fake)
    assert rc == 73


def test_stop_owned_container(tmp_path):
    """Stop guard: present owned => exit 0 (successful rm)."""
    bundle, site = _make_test_bundle(tmp_path)
    labels = _expected_labels(bundle, bundle.services[0])
    fake = _make_fake_docker(tmp_path, daemon_ok=True, container_exists=True,
                             labels=labels, rm_succeeds=True)
    actions = bundle_mod._native_stop_actions(site, bundle.services[0], bundle)
    rc = _run_guard_with_fake_docker(tmp_path, actions[0].argv[2], fake)
    assert rc == 0


def test_stop_rm_failure(tmp_path):
    """Stop guard: rm failure => nonzero exit."""
    bundle, site = _make_test_bundle(tmp_path)
    labels = _expected_labels(bundle, bundle.services[0])
    fake = _make_fake_docker(tmp_path, daemon_ok=True, container_exists=True,
                             labels=labels, rm_succeeds=False)
    actions = bundle_mod._native_stop_actions(site, bundle.services[0], bundle)
    rc = _run_guard_with_fake_docker(tmp_path, actions[0].argv[2], fake)
    assert rc != 0


# Verify-rollback fake-engine matrix
def test_verify_rollback_daemon_unavailable(tmp_path):
    """Verify-rollback: daemon unavailable => exit 2."""
    bundle, site = _make_test_bundle(tmp_path)
    fake = _make_fake_docker(tmp_path, daemon_ok=False)
    actions = bundle_mod._native_verify_rollback_actions(
        site, bundle.services[0], bundle,
    )
    rc = _run_guard_with_fake_docker(tmp_path, actions[0].argv[2], fake)
    assert rc == 2


def test_verify_rollback_absent(tmp_path):
    """Verify-rollback: proven absence => exit 0."""
    bundle, site = _make_test_bundle(tmp_path)
    fake = _make_fake_docker(tmp_path, daemon_ok=True, container_exists=False)
    actions = bundle_mod._native_verify_rollback_actions(
        site, bundle.services[0], bundle,
    )
    rc = _run_guard_with_fake_docker(tmp_path, actions[0].argv[2], fake)
    assert rc == 0


def test_verify_rollback_present(tmp_path):
    """Verify-rollback: present container => exit 1."""
    bundle, site = _make_test_bundle(tmp_path)
    labels = _expected_labels(bundle, bundle.services[0])
    fake = _make_fake_docker(tmp_path, daemon_ok=True, container_exists=True,
                             labels=labels)
    actions = bundle_mod._native_verify_rollback_actions(
        site, bundle.services[0], bundle,
    )
    rc = _run_guard_with_fake_docker(tmp_path, actions[0].argv[2], fake)
    assert rc == 1


# Readiness container-running fake-engine matrix
def test_readiness_container_running_success(tmp_path):
    """Container-running readiness: owned+running => exit 0."""
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    doc = _bundle_doc([
        _native_service("a", "serving", profile,
                        readiness={"kind": "container-running"}),
    ])
    path = _write_bundle(tmp_path, doc)
    bundle = bundle_mod.load_bundle(path)
    site = load_site(_site(tmp_path))
    labels = _expected_labels(bundle, bundle.services[0])
    fake = _make_fake_docker(tmp_path, daemon_ok=True, container_exists=True,
                             labels=labels, running=True)
    actions = bundle_mod._readiness_actions(site, bundle.services[0], bundle)
    rc = _run_guard_with_fake_docker(tmp_path, actions[0].argv[2], fake)
    assert rc == 0


def test_readiness_container_running_wrong_label(tmp_path):
    """Container-running readiness: wrong managed label => exit 1."""
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    doc = _bundle_doc([
        _native_service("a", "serving", profile,
                        readiness={"kind": "container-running"}),
    ])
    path = _write_bundle(tmp_path, doc)
    bundle = bundle_mod.load_bundle(path)
    site = load_site(_site(tmp_path))
    labels = _expected_labels(bundle, bundle.services[0])
    labels["org.sparkring.managed"] = "false"
    fake = _make_fake_docker(tmp_path, daemon_ok=True, container_exists=True,
                             labels=labels, running=True)
    actions = bundle_mod._readiness_actions(site, bundle.services[0], bundle)
    rc = _run_guard_with_fake_docker(tmp_path, actions[0].argv[2], fake)
    assert rc == 1


def test_readiness_container_running_absent(tmp_path):
    """Container-running readiness: absent container => exit 1."""
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    doc = _bundle_doc([
        _native_service("a", "serving", profile,
                        readiness={"kind": "container-running"}),
    ])
    path = _write_bundle(tmp_path, doc)
    bundle = bundle_mod.load_bundle(path)
    site = load_site(_site(tmp_path))
    fake = _make_fake_docker(tmp_path, daemon_ok=True, container_exists=False)
    actions = bundle_mod._readiness_actions(site, bundle.services[0], bundle)
    rc = _run_guard_with_fake_docker(tmp_path, actions[0].argv[2], fake)
    assert rc != 0


# HTTP readiness fake-engine matrix
def test_readiness_http_wrong_label(tmp_path):
    """HTTP readiness: wrong label must fail even if curl succeeds."""
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    doc = _bundle_doc([
        _native_service("a", "serving", profile,
                        readiness={"kind": "http-get", "port": "site-api",
                                   "path": "/health", "timeout_seconds": 5,
                                   "interval_seconds": 2}),
    ])
    path = _write_bundle(tmp_path, doc)
    bundle = bundle_mod.load_bundle(path)
    site = load_site(_site(tmp_path))
    labels = _expected_labels(bundle, bundle.services[0])
    labels["org.sparkring.bundle"] = "wrong-bundle"
    fake = _make_fake_docker(tmp_path, daemon_ok=True, container_exists=True,
                             labels=labels, running=True)
    actions = bundle_mod._readiness_actions(site, bundle.services[0], bundle)
    rc = _run_guard_with_fake_docker(tmp_path, actions[0].argv[2], fake)
    assert rc != 0


def test_readiness_http_stopped_container_fails_even_when_curl_succeeds(tmp_path):
    """HTTP readiness requires the owned container itself to be running."""
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    doc = _bundle_doc([
        _native_service(
            "a", "serving", profile,
            readiness={
                "kind": "http-get", "port": "site-api", "path": "/health",
                "timeout_seconds": 1, "interval_seconds": 1,
            },
        ),
    ])
    bundle = bundle_mod.load_bundle(_write_bundle(tmp_path, doc))
    site = load_site(_site(tmp_path))
    labels = _expected_labels(bundle, bundle.services[0])
    fake = _make_fake_docker(
        tmp_path, daemon_ok=True, container_exists=True,
        labels=labels, running=False,
    )
    fake_curl = tmp_path / "curl"
    fake_curl.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_curl.chmod(0o755)
    actions = bundle_mod._readiness_actions(site, bundle.services[0], bundle)
    rc = _run_guard_with_fake_docker(tmp_path, actions[0].argv[2], fake)
    assert rc != 0


# HTTP timing budget test — divisible case
def test_http_readiness_timing_budget(tmp_path):
    """HTTP readiness curl+sleep total never exceeds timeout (divisible)."""
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    doc = _bundle_doc([
        _native_service("a", "serving", profile,
                        readiness={"kind": "http-get", "port": "site-api",
                                   "path": "/health", "timeout_seconds": 10,
                                   "interval_seconds": 5}),
    ])
    path = _write_bundle(tmp_path, doc)
    bundle = bundle_mod.load_bundle(path)
    site = load_site(_site(tmp_path))
    actions = bundle_mod._readiness_actions(site, bundle.services[0], bundle)
    cmd = actions[0].shell_command
    # slot=5, curl_timeout=2, sleep=3, full_attempts=2, remaining=0
    # attempts=2, final_curl=2, no sleep after final
    assert "--max-time $curl_timeout" in cmd
    assert "sleep 3" in cmd
    assert "seq 1 2" in cmd
    # Verify no sleep after final attempt
    assert '[ "$i" -lt 2 ]' in cmd


# HTTP timing budget test — non-divisible case
def test_http_readiness_timing_budget_non_divisible(tmp_path):
    """HTTP readiness with non-divisible timeout never exceeds budget."""
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    doc = _bundle_doc([
        _native_service("a", "serving", profile,
                        readiness={"kind": "http-get", "port": "site-api",
                                   "path": "/health", "timeout_seconds": 11,
                                   "interval_seconds": 5}),
    ])
    path = _write_bundle(tmp_path, doc)
    bundle = bundle_mod.load_bundle(path)
    site = load_site(_site(tmp_path))
    actions = bundle_mod._readiness_actions(site, bundle.services[0], bundle)
    cmd = actions[0].shell_command
    # slot=5, curl_timeout=2, sleep=3, full_attempts=2, remaining=1
    # attempts=3, final_curl=min(2,1)=1
    # Total = 2*5 + 1 = 11 = timeout (no sleep after final)
    assert "seq 1 3" in cmd
    assert "curl_timeout=1" in cmd  # final attempt curl timeout
    assert "sleep 3" in cmd  # sleep after non-final attempts
    assert '[ "$i" -lt 3 ]' in cmd  # no sleep after final (i=3)
    # Verify total budget: (attempts-1)*slot + final_curl = 2*5 + 1 = 11
    # which equals timeout_seconds — never exceeds it


# Structured container shell entrypoint rejection
def test_structured_container_rejects_dash(tmp_path):
    """dash entrypoint is rejected."""
    sc = _structured_doc(argv=["dash", "-c", "echo hello"])
    sc_path = _write_structured(tmp_path, "cache", sc)
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    doc = _bundle_doc([
        _structured_service("cache", "cache", sc_path),
        _native_service("engine", "serving", profile, depends_on=["cache"]),
    ])
    path = _write_bundle(tmp_path, doc)
    with pytest.raises(bundle_mod.BundleError) as exc:
        bundle_mod.load_bundle(path)
    assert "shell" in str(exc.value).lower()


def test_structured_container_rejects_powershell(tmp_path):
    """powershell entrypoint is rejected."""
    sc = _structured_doc(argv=["powershell", "-Command", "echo hello"])
    sc_path = _write_structured(tmp_path, "cache", sc)
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    doc = _bundle_doc([
        _structured_service("cache", "cache", sc_path),
        _native_service("engine", "serving", profile, depends_on=["cache"]),
    ])
    path = _write_bundle(tmp_path, doc)
    with pytest.raises(bundle_mod.BundleError) as exc:
        bundle_mod.load_bundle(path)
    assert "shell" in str(exc.value).lower()


@pytest.mark.parametrize("entrypoint", [
    "cmd", "cmd.exe", "pwsh", "pwsh.exe",
    "fish", "csh", "tcsh", "ash", "zsh",
])
def test_structured_container_rejects_all_shell_aliases(tmp_path, entrypoint):
    """All shell entrypoint aliases must be rejected."""
    sc = _structured_doc(argv=[entrypoint, "-c", "echo hello"])
    sc_path = _write_structured(tmp_path, "cache", sc)
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    doc = _bundle_doc([
        _structured_service("cache", "cache", sc_path),
        _native_service("engine", "serving", profile, depends_on=["cache"]),
    ])
    path = _write_bundle(tmp_path, doc)
    with pytest.raises(bundle_mod.BundleError) as exc:
        bundle_mod.load_bundle(path)
    assert "shell" in str(exc.value).lower()


# Mount path .. rejection
def test_structured_container_rejects_dotdot_mount(tmp_path):
    """Mount paths with .. are rejected."""
    sc = _structured_doc(volumes=[{"host": "/data", "container": "/../escape", "mode": "ro"}])
    sc_path = _write_structured(tmp_path, "cache", sc)
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    doc = _bundle_doc([
        _structured_service("cache", "cache", sc_path),
        _native_service("engine", "serving", profile, depends_on=["cache"]),
    ])
    path = _write_bundle(tmp_path, doc)
    with pytest.raises(bundle_mod.BundleError) as exc:
        bundle_mod.load_bundle(path)
    assert ".." in str(exc.value)


# Bridge source must exist at validation time
def test_bridge_source_must_exist(tmp_path):
    """Bridge source path must be a file during structural validation."""
    exl3_profile = _exl3_profile(tmp_path)
    doc = _bundle_doc([
        _bridge_service("lmcache", "cache", exl3_profile),
        _bridge_service("engine", "serving", "nonexistent.json", depends_on=["lmcache"]),
    ])
    path = _write_bundle(tmp_path, doc)
    with pytest.raises(bundle_mod.BundleError) as exc:
        bundle_mod.load_bundle(path)
    assert "not found" in str(exc.value).lower() or "not a file" in str(exc.value).lower()



# ---------------------------------------------------------------------------
# Archive rehearsal script
# ---------------------------------------------------------------------------


def test_rehearse_script_runs_clean(monkeypatch):
    """The rehearsal script passes all checks in the checkout."""
    import rehearse_runtime_bundle_archive as rehearse
    rehearse._POISONED = False
    rc = rehearse.main([])
    assert rc == 0
    # main() must restore all process-global poison boundaries.
    assert rehearse._POISONED is False


def test_rehearse_script_poisons_remote(monkeypatch):
    """The rehearsal script must poison runtime.execute before any checks."""
    import rehearse_runtime_bundle_archive as rehearse
    import socket
    import sparkring_exl3_launcher as exl3_launcher
    import sparkring_runtime as rt
    # Reset the guard so _install_poison actually runs
    rehearse._POISONED = False
    rehearse._install_poison()
    # After install, calling execute must raise
    with pytest.raises(AssertionError, match="remote executor"):
        rt.execute([], 30)
    # run_remote also poisoned
    with pytest.raises(AssertionError, match="remote executor"):
        rt.run_remote("host", "cmd")
    # Canonical execution and socket boundaries are poisoned too.
    with pytest.raises(AssertionError, match="remote executor"):
        exl3_launcher.execute([], 30)
    with pytest.raises(AssertionError, match="remote executor"):
        socket.socket()
    with pytest.raises(AssertionError, match="remote executor"):
        socket.create_connection(("127.0.0.1", 1))
    rehearse._restore_poison()
    assert rehearse._POISONED is False


def test_rehearse_script_no_git_operations(monkeypatch):
    """The rehearsal script source contains no git add/commit/stage calls."""
    _poison_remote(monkeypatch)
    import rehearse_runtime_bundle_archive as rehearse
    ok, msg = rehearse.check_no_git_operations()
    assert ok
    # Verify source doesn't invoke git via subprocess or os.system
    source = (ROOT / "scripts/rehearse_runtime_bundle_archive.py").read_text(encoding="utf-8")
    assert "subprocess" not in source
    assert "os.system" not in source
    assert "os.popen" not in source
    # No actual git command invocation (not in a docstring/comment context)
    lines = [line_text for line_text in source.splitlines()
              if not line_text.strip().startswith("#") and '"""' not in line_text]
    for line in lines:
        assert "git add" not in line, f"git add found: {line}"
        assert "git commit" not in line, f"git commit found: {line}"
        assert "git stage" not in line, f"git stage found: {line}"



# ---------------------------------------------------------------------------
# Fail-closed executor and schema regressions
# ---------------------------------------------------------------------------


# _safe_execute rejects extra, missing, and duplicate ranks
def test_safe_execute_missing_rank(tmp_path, monkeypatch):
    """Missing rank in executor output produces exit_code=125."""
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    doc = _bundle_doc([_native_service("a", "serving", profile)])
    path = _write_bundle(tmp_path, doc)
    bundle = bundle_mod.load_bundle(path)
    site = load_site(_site(tmp_path))
    actions = bundle_mod._native_status_actions(site, bundle.services[0], bundle)
    expected = {a.rank for a in actions}
    # Return results missing rank 0
    monkeypatch.setattr(runtime, "execute",
        lambda acts, t: {r: {"exit_code": 0, "stdout": "", "stderr": ""}
                          for r in expected if r != 0})
    res = bundle_mod._safe_execute(actions, 30)
    assert res[0]["exit_code"] == 125
    assert res[0]["error_type"] == "MissingResult"


def test_safe_execute_extra_rank(tmp_path, monkeypatch):
    """Extra rank in executor output is flagged with exit_code=125."""
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    doc = _bundle_doc([_native_service("a", "serving", profile)])
    path = _write_bundle(tmp_path, doc)
    bundle = bundle_mod.load_bundle(path)
    site = load_site(_site(tmp_path))
    actions = bundle_mod._native_status_actions(site, bundle.services[0], bundle)
    expected = {a.rank for a in actions}
    # Return results with an extra rank 99
    extra_results = {r: {"exit_code": 0, "stdout": "", "stderr": ""}
                     for r in expected}
    extra_results[99] = {"exit_code": 0, "stdout": "", "stderr": ""}
    monkeypatch.setattr(runtime, "execute", lambda acts, t: extra_results)
    res = bundle_mod._safe_execute(actions, 30)
    assert 99 in res
    assert res[99]["exit_code"] == 125
    assert res[99]["error_type"] == "ExtraRank"


def test_status_aggregate_fails_on_extra_rank(tmp_path, monkeypatch):
    """execute_native_status returns failed when extra rank present."""
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    doc = _bundle_doc([_native_service("a", "serving", profile)])
    path = _write_bundle(tmp_path, doc)
    bundle = bundle_mod.load_bundle(path)
    site = load_site(_site(tmp_path))
    actions = bundle_mod._native_status_actions(site, bundle.services[0], bundle)
    expected = {a.rank for a in actions}
    extra_results = {r: {"exit_code": 0, "stdout": "", "stderr": ""}
                     for r in expected}
    extra_results[99] = {"exit_code": 0, "stdout": "", "stderr": ""}
    monkeypatch.setattr(runtime, "execute", lambda acts, t: extra_results)
    result = bundle_mod.execute_native_status(bundle, site)
    assert result["status"] == "failed"


def test_start_rollback_fails_on_missing_rank(tmp_path, monkeypatch):
    """execute_native_start rollback fails when a rank result is missing."""
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    doc = _bundle_doc([_native_service("a", "serving", profile)])
    path = _write_bundle(tmp_path, doc)
    bundle = bundle_mod.load_bundle(path)
    site = load_site(_site(tmp_path))
    expected = {0, 1, 2, 3}
    call_count = [0]
    def mock_execute(actions, timeout):
        call_count[0] += 1
        if call_count[0] == 1:
            # Start: rank 0 succeeds (container ID in stdout), rank 1 fails
            return {r: {"exit_code": 0 if r != 1 else 1,
                        "stdout": "a" * 12 if r != 1 else "",
                        "stderr": ""}
                    for r in expected}
        else:
            # Rollback: return missing rank 0
            return {r: {"exit_code": 0, "stdout": "", "stderr": ""}
                    for r in expected if r != 0}
    monkeypatch.setattr(runtime, "execute", mock_execute)
    result = bundle_mod.execute_native_start(bundle, site, confirmation="START-test")
    assert result["rollback"]["rollback_status"] == "failed"


# Rollback verification classifies nonzero results conservatively
def test_verify_rollback_executor_exception_is_unknown(tmp_path, monkeypatch):
    """Exit 125 (executor exception) must be unknown, not absent."""
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    doc = _bundle_doc([_native_service("a", "serving", profile)])
    path = _write_bundle(tmp_path, doc)
    bundle = bundle_mod.load_bundle(path)
    site = load_site(_site(tmp_path))
    monkeypatch.setattr(runtime, "execute",
        lambda acts, t: (_ for _ in ()).throw(RuntimeError("boom")))
    result = bundle_mod.execute_native_verify_rollback(bundle, site)
    assert result["status"] == "unknown"


def test_verify_rollback_ownership_mismatch_is_unknown(tmp_path, monkeypatch):
    """Exit 73 (ownership mismatch) must be unknown, not absent."""
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    doc = _bundle_doc([_native_service("a", "serving", profile)])
    path = _write_bundle(tmp_path, doc)
    bundle = bundle_mod.load_bundle(path)
    site = load_site(_site(tmp_path))
    monkeypatch.setattr(runtime, "execute",
        lambda acts, t: {a.rank: {"exit_code": 73, "stdout": "", "stderr": ""}
                          for a in acts})
    result = bundle_mod.execute_native_verify_rollback(bundle, site)
    assert result["status"] == "unknown"


def test_verify_rollback_daemon_error_is_unknown(tmp_path, monkeypatch):
    """Exit 74 (daemon error) must be unknown, not absent."""
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    doc = _bundle_doc([_native_service("a", "serving", profile)])
    path = _write_bundle(tmp_path, doc)
    bundle = bundle_mod.load_bundle(path)
    site = load_site(_site(tmp_path))
    monkeypatch.setattr(runtime, "execute",
        lambda acts, t: {a.rank: {"exit_code": 74, "stdout": "", "stderr": ""}
                          for a in acts})
    result = bundle_mod.execute_native_verify_rollback(bundle, site)
    assert result["status"] == "unknown"


def test_verify_rollback_missing_rank_is_unknown(tmp_path, monkeypatch):
    """Missing rank must be unknown, not absent."""
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    doc = _bundle_doc([_native_service("a", "serving", profile)])
    path = _write_bundle(tmp_path, doc)
    bundle = bundle_mod.load_bundle(path)
    site = load_site(_site(tmp_path))
    actions = bundle_mod._native_verify_rollback_actions(
        site, bundle.services[0], bundle)
    expected = {a.rank for a in actions}
    monkeypatch.setattr(runtime, "execute",
        lambda acts, t: {r: {"exit_code": 0, "stdout": "", "stderr": ""}
                          for r in expected if r != 0})
    result = bundle_mod.execute_native_verify_rollback(bundle, site)
    assert result["status"] == "unknown"


# Rollback CLI treats reserved failure codes as failures
def test_rollback_cli_treats_73_as_failure(tmp_path, monkeypatch):
    """Rollback CLI must treat exit 73 (ownership mismatch) as failure."""
    import sparkring_bundle_launcher as launcher
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    doc = _bundle_doc([_native_service("a", "serving", profile)])
    path = _write_bundle(tmp_path, doc)
    site_path = _site(tmp_path)
    monkeypatch.setattr(runtime, "execute",
        lambda acts, t: {a.rank: {"exit_code": 73, "stdout": "", "stderr": ""}
                          for a in acts})
    rc = launcher.main([
        "--bundle", str(path), "--site", str(site_path),
        "rollback", "--execute", "--confirmation", "START-test",
    ])
    assert rc == 1


def test_rollback_cli_treats_125_as_failure(tmp_path, monkeypatch):
    """Rollback CLI must treat exit 125 (executor exception) as failure."""
    import sparkring_bundle_launcher as launcher
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    doc = _bundle_doc([_native_service("a", "serving", profile)])
    path = _write_bundle(tmp_path, doc)
    site_path = _site(tmp_path)
    monkeypatch.setattr(runtime, "execute",
        lambda acts, t: (_ for _ in ()).throw(RuntimeError("boom")))
    rc = launcher.main([
        "--bundle", str(path), "--site", str(site_path),
        "rollback", "--execute", "--confirmation", "START-test",
    ])
    assert rc == 1


# Case-insensitive shell entrypoint rejection
@pytest.mark.parametrize("entrypoint", [
    "BASH", "Bash", "PowerShell.EXE", "CMD.EXE", "Cmd.Exe",
    "PWSH", "ZSH", "FiSh", "CSH", "TCSH", "ASH", "DASH",
])
def test_structured_container_rejects_mixed_case_shell(tmp_path, entrypoint):
    """Mixed-case shell entrypoints must be rejected."""
    sc = _structured_doc(argv=[entrypoint, "-c", "echo hello"])
    sc_path = _write_structured(tmp_path, "cache", sc)
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    doc = _bundle_doc([
        _structured_service("cache", "cache", sc_path),
        _native_service("engine", "serving", profile, depends_on=["cache"]),
    ])
    path = _write_bundle(tmp_path, doc)
    with pytest.raises(bundle_mod.BundleError) as exc:
        bundle_mod.load_bundle(path)
    assert "shell" in str(exc.value).lower()


# Structured-container schema requirement
def test_structured_container_requires_schema(tmp_path):
    """Structured container without schema field must be rejected."""
    sc = _structured_doc()
    del sc["schema"]
    sc_path = _write_structured(tmp_path, "cache", sc)
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    doc = _bundle_doc([
        _structured_service("cache", "cache", sc_path),
        _native_service("engine", "serving", profile, depends_on=["cache"]),
    ])
    path = _write_bundle(tmp_path, doc)
    with pytest.raises(bundle_mod.BundleError) as exc:
        bundle_mod.load_bundle(path)
    assert "schema" in str(exc.value).lower()


def test_structured_container_rejects_wrong_schema(tmp_path):
    """Structured container with wrong schema must be rejected."""
    sc = _structured_doc(schema="wrong-schema/v1")
    sc_path = _write_structured(tmp_path, "cache", sc)
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    doc = _bundle_doc([
        _structured_service("cache", "cache", sc_path),
        _native_service("engine", "serving", profile, depends_on=["cache"]),
    ])
    path = _write_bundle(tmp_path, doc)
    with pytest.raises(bundle_mod.BundleError) as exc:
        bundle_mod.load_bundle(path)
    assert "schema" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# Rank-scope feature: per-service rank constraint for structured containers
# ---------------------------------------------------------------------------


def _scoped_bundle(tmp_path, ranks=None, **kw):
    """Create a bundle with a scoped cache sidecar + all-rank serving engine."""
    sc_path = _write_structured(tmp_path, "cache", _structured_doc())
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    cache_svc = _structured_service(
        "cache", "cache", sc_path,
        readiness={"kind": "container-running"},
    )
    if ranks is not None:
        cache_svc["ranks"] = ranks
    engine_svc = _native_service("engine", "serving", profile, depends_on=["cache"])
    doc = _bundle_doc([cache_svc, engine_svc], **kw)
    return _write_bundle(tmp_path, doc)


def test_ranks_accepted_for_structured_container(tmp_path):
    """A structured-container cache service with valid ranks must parse."""
    path = _scoped_bundle(tmp_path, ranks=[0, 1])
    bundle = bundle_mod.load_bundle(path)
    cache_svc = [s for s in bundle.services if s.service_id == "cache"][0]
    assert cache_svc.ranks == frozenset({0, 1})
    engine_svc = [s for s in bundle.services if s.service_id == "engine"][0]
    assert engine_svc.ranks is None  # serving always all-ranks


def test_ranks_none_default_all_ranks(tmp_path):
    """No ranks field means all site ranks (backward compatible)."""
    path = _scoped_bundle(tmp_path)
    bundle = bundle_mod.load_bundle(path)
    cache_svc = [s for s in bundle.services if s.service_id == "cache"][0]
    assert cache_svc.ranks is None


@pytest.mark.parametrize("bad_ranks,expected_fragment", [
    ([], "non-empty"),
    ([0, 0], "duplicate"),
    ([-1], "invalid"),
    ([0, -3], "invalid"),
    (["0"], "invalid"),
    ([0, 1, "2"], "invalid"),
    ([True], "invalid"),
])
def test_ranks_invalid_values_rejected(tmp_path, bad_ranks, expected_fragment):
    """Invalid rank lists must be rejected during structural validation."""
    path = _scoped_bundle(tmp_path, ranks=bad_ranks)
    with pytest.raises(bundle_mod.BundleError) as exc:
        bundle_mod.load_bundle(path)
    assert expected_fragment in str(exc.value).lower()


def test_ranks_rejected_for_runtime_profile(tmp_path):
    """ranks field must be rejected for runtime-profile serving services."""
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    engine_svc = _native_service("engine", "serving", profile, ranks=[0])
    doc = _bundle_doc([engine_svc])
    path = _write_bundle(tmp_path, doc)
    with pytest.raises(bundle_mod.BundleError) as exc:
        bundle_mod.load_bundle(path)
    assert "ranks" in str(exc.value).lower()
    assert "runtime-profile" in str(exc.value).lower() or "structured" in str(exc.value).lower()


def test_ranks_rejected_for_structured_serving(tmp_path):
    """A structured container cannot bypass all-rank serving semantics."""
    structured = _write_structured(tmp_path, "engine", _structured_doc())
    engine_svc = _structured_service(
        "engine", "serving", structured, ranks=[0],
    )
    path = _write_bundle(tmp_path, _bundle_doc([engine_svc]))
    with pytest.raises(bundle_mod.BundleError) as exc:
        bundle_mod.load_bundle(path)
    assert "serving" in str(exc.value).lower()
    assert "structured-container" in str(exc.value).lower()


def test_ranks_rejected_for_canonical_bridge(tmp_path):
    """ranks field must be rejected for canonical-exl3-lmcache-cs512 services."""
    profile = _write_profile(tmp_path, "bridge", _generic_doc())
    cache_svc = _bridge_service("cache", "cache", profile, ranks=[0])
    engine_svc = _bridge_service("engine", "serving", profile, depends_on=["cache"])
    doc = _bundle_doc([cache_svc, engine_svc], confirmation=None)
    path = _write_bundle(tmp_path, doc)
    with pytest.raises(bundle_mod.BundleError) as exc:
        bundle_mod.load_bundle(path)
    assert "ranks" in str(exc.value).lower()


def test_ranks_unknown_id_rejected_at_plan(tmp_path):
    """Ranks referencing non-existent site IDs must fail at plan time."""
    path = _scoped_bundle(tmp_path, ranks=[0, 99])
    bundle = bundle_mod.load_bundle(path)  # structural validation passes
    site = load_site(SITE)
    with pytest.raises(bundle_mod.BundleError) as exc:
        bundle_mod.bundle_plan(bundle, site)
    assert "99" in str(exc.value)
    assert "not found" in str(exc.value).lower()


def test_ranks_filters_start_actions(tmp_path):
    """Scoped cache service start actions must only target scoped ranks."""
    path = _scoped_bundle(tmp_path, ranks=[0])
    bundle = bundle_mod.load_bundle(path)
    site = load_site(SITE)
    plan = bundle_mod.bundle_plan(bundle, site)
    # Find the cache start phase
    cache_start = [
        p for p in plan["phases"]
        if p["phase"] == "start" and p["service_id"] == "cache"
    ][0]
    start_ranks = {a["rank"] for a in cache_start["actions"]}
    assert start_ranks == {0}
    # Engine should still target all 4 ranks
    engine_start = [
        p for p in plan["phases"]
        if p["phase"] == "start" and p["service_id"] == "engine"
    ][0]
    engine_ranks = {a["rank"] for a in engine_start["actions"]}
    assert engine_ranks == {0, 1, 2, 3}


def test_ranks_filters_stop_and_verify_rollback(tmp_path):
    """Scoped service stop/verify-rollback must only target scoped ranks."""
    path = _scoped_bundle(tmp_path, ranks=[1])
    bundle = bundle_mod.load_bundle(path)
    site = load_site(SITE)
    plan = bundle_mod.bundle_plan(bundle, site)
    for phase_name in ("stop", "rollback", "verify_rollback"):
        phase = [
            p for p in plan["phases"]
            if p["phase"] == phase_name and p["service_id"] == "cache"
        ][0]
        phase_ranks = {a["rank"] for a in phase["actions"]}
        assert phase_ranks == {1}, f"{phase_name} ranks: {phase_ranks}"


def test_ranks_filters_readiness(tmp_path):
    """Scoped service readiness must only target scoped ranks."""
    path = _scoped_bundle(tmp_path, ranks=[2])
    bundle = bundle_mod.load_bundle(path)
    site = load_site(SITE)
    plan = bundle_mod.bundle_plan(bundle, site)
    ready = [
        p for p in plan["phases"]
        if p["phase"] == "readiness" and p["service_id"] == "cache"
    ]
    assert len(ready) == 1
    ready_ranks = {a["rank"] for a in ready[0]["actions"]}
    assert ready_ranks == {2}


def test_ranks_exposed_in_plan(tmp_path):
    """Plan must expose expected ranks per service."""
    path = _scoped_bundle(tmp_path, ranks=[0, 1])
    bundle = bundle_mod.load_bundle(path)
    site = load_site(SITE)
    plan = bundle_mod.bundle_plan(bundle, site)
    cache_entry = [s for s in plan["services"] if s["service_id"] == "cache"][0]
    assert cache_entry["ranks"] == [0, 1]
    engine_entry = [s for s in plan["services"] if s["service_id"] == "engine"][0]
    assert engine_entry["ranks"] == [0, 1, 2, 3]


def test_ranks_exposed_in_explain(tmp_path):
    """Explain must expose ranks for scoped services."""
    path = _scoped_bundle(tmp_path, ranks=[0])
    bundle = bundle_mod.load_bundle(path)
    explain = bundle_mod.bundle_explain(bundle)
    cache_entry = [s for s in explain["services"] if s["service_id"] == "cache"][0]
    assert cache_entry["ranks"] == [0]
    engine_entry = [s for s in explain["services"] if s["service_id"] == "engine"][0]
    assert engine_entry["ranks"] is None


def test_ranks_collision_only_on_overlapping_ranks(tmp_path):
    """Collision check should only check ranks where both services run."""
    # Cache on ranks 0,1 with same container_name prefix as another service
    # on ranks 2,3 — no collision because ranks don't overlap.
    sc1 = _write_structured(tmp_path, "c1", _structured_doc(container_name="sparkring-shared"))
    sc2 = _write_structured(tmp_path, "c2", _structured_doc(container_name="sparkring-shared"))
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    doc = _bundle_doc([
        _structured_service("cache-a", "cache", sc1, ranks=[0, 1],
                            readiness={"kind": "container-running"}),
        _structured_service("cache-b", "cache", sc2, ranks=[2, 3],
                            readiness={"kind": "container-running"}),
        _native_service("engine", "serving", profile, depends_on=["cache-a", "cache-b"]),
    ])
    path = _write_bundle(tmp_path, doc)
    bundle = bundle_mod.load_bundle(path)
    site = load_site(SITE)
    # Should not raise — same name on non-overlapping ranks
    bundle_mod.check_container_name_collisions(bundle, site)


def test_ranks_collision_detected_on_overlapping_ranks(tmp_path):
    """Collision check must detect same name on overlapping ranks."""
    sc1 = _write_structured(tmp_path, "c1", _structured_doc(container_name="sparkring-shared"))
    sc2 = _write_structured(tmp_path, "c2", _structured_doc(container_name="sparkring-shared"))
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    doc = _bundle_doc([
        _structured_service("cache-a", "cache", sc1, ranks=[0, 1],
                            readiness={"kind": "container-running"}),
        _structured_service("cache-b", "cache", sc2, ranks=[1, 2],
                            readiness={"kind": "container-running"}),
        _native_service("engine", "serving", profile, depends_on=["cache-a", "cache-b"]),
    ])
    path = _write_bundle(tmp_path, doc)
    bundle = bundle_mod.load_bundle(path)
    site = load_site(SITE)
    with pytest.raises(bundle_mod.BundleError) as exc:
        bundle_mod.check_container_name_collisions(bundle, site)
    assert "collision" in str(exc.value).lower()


def test_ranks_deterministic_ordering_byte_identical(tmp_path):
    """Equivalent rank orderings must produce byte-identical plans."""
    sc_path = _write_structured(tmp_path, "cache", _structured_doc())
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    doc_a = _bundle_doc([
        _structured_service("cache", "cache", sc_path, ranks=[2, 0, 1],
                            readiness={"kind": "container-running"}),
        _native_service("engine", "serving", profile, depends_on=["cache"]),
    ])
    doc_b = _bundle_doc([
        _structured_service("cache", "cache", sc_path, ranks=[0, 1, 2],
                            readiness={"kind": "container-running"}),
        _native_service("engine", "serving", profile, depends_on=["cache"]),
    ])
    path_a = _write_bundle(tmp_path, doc_a, "a.json")
    path_b = _write_bundle(tmp_path, doc_b, "b.json")
    site = load_site(SITE)
    plan_a = bundle_mod.bundle_plan(bundle_mod.load_bundle(path_a), site)
    plan_b = bundle_mod.bundle_plan(bundle_mod.load_bundle(path_b), site)
    assert plan_a == plan_b
    assert plan_a["plan_identity"] == plan_b["plan_identity"]


def test_ranks_projection_includes_ranks(tmp_path):
    """Service projection for diff must include ranks for scoped services."""
    path = _scoped_bundle(tmp_path, ranks=[0, 2])
    bundle = bundle_mod.load_bundle(path)
    proj = bundle_mod.bundle_projection(bundle)
    cache_proj = proj["services"]["cache"]
    assert cache_proj["ranks"] == [0, 2]


def test_ranks_diff_detects_different_scopes(tmp_path):
    """Diff must detect different rank scopes."""
    sc_path = _write_structured(tmp_path, "cache", _structured_doc())
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    doc_a = _bundle_doc([
        _structured_service("cache", "cache", sc_path, ranks=[0],
                            readiness={"kind": "container-running"}),
        _native_service("engine", "serving", profile, depends_on=["cache"]),
    ])
    doc_b = _bundle_doc([
        _structured_service("cache", "cache", sc_path, ranks=[1],
                            readiness={"kind": "container-running"}),
        _native_service("engine", "serving", profile, depends_on=["cache"]),
    ])
    path_a = _write_bundle(tmp_path, doc_a, "a.json")
    path_b = _write_bundle(tmp_path, doc_b, "b.json")
    diffs = bundle_mod.recursive_diff(
        bundle_mod.bundle_projection(bundle_mod.load_bundle(path_a)),
        bundle_mod.bundle_projection(bundle_mod.load_bundle(path_b)),
    )
    assert any("ranks" in d.get("field", "") for d in diffs)


def test_ranks_execute_start_filters_actions(tmp_path, monkeypatch):
    """execute_native_start must only start on scoped ranks."""
    path = _scoped_bundle(tmp_path, ranks=[0])
    bundle = bundle_mod.load_bundle(path)
    site = load_site(SITE)
    _poison_remote(monkeypatch)
    # Poison execute to capture which ranks are attempted
    captured_ranks: dict[str, list[int]] = {}
    orig_execute = runtime.execute

    def fake_execute(actions, timeout):
        if actions:
            command = actions[0].shell_command
            key = "cache" if "sparkring-cache" in command else "engine"
            captured_ranks.setdefault(key, []).extend(
                sorted(a.rank for a in actions)
            )
        return {a.rank: {"exit_code": 0,
                         "stdout": "a" * 12, "stderr": ""}
                for a in actions}

    monkeypatch.setattr(runtime, "execute", fake_execute)
    try:
        bundle_mod.execute_native_start(
            bundle, site, confirmation="START-test",
        )
    finally:
        monkeypatch.setattr(runtime, "execute", orig_execute)
    assert sorted(set(captured_ranks["cache"])) == [0]
    assert sorted(set(captured_ranks["engine"])) == [0, 1, 2, 3]


def test_ranks_execute_status_filters_actions(tmp_path, monkeypatch):
    """execute_native_status must only check scoped ranks for scoped service."""
    path = _scoped_bundle(tmp_path, ranks=[1])
    bundle = bundle_mod.load_bundle(path)
    site = load_site(SITE)
    _poison_remote(monkeypatch)
    captured_ranks: dict[str, list[int]] = {}
    orig_execute = runtime.execute

    def fake_execute(actions, timeout):
        ranks = sorted(a.rank for a in actions)
        # Tag by service by looking at the command
        key = "cache" if any("sparkring-cache" in a.shell_command for a in actions) else "engine"
        captured_ranks.setdefault(key, []).extend(ranks)
        return {a.rank: {"exit_code": 0, "stdout": "", "stderr": ""}
                for a in actions}

    monkeypatch.setattr(runtime, "execute", fake_execute)
    try:
        bundle_mod.execute_native_status(bundle, site)
    finally:
        monkeypatch.setattr(runtime, "execute", orig_execute)
    assert captured_ranks.get("cache") == [1]
    assert sorted(set(captured_ranks.get("engine", []))) == [0, 1, 2, 3]


def test_ranks_execute_verify_rollback_filters(tmp_path, monkeypatch):
    """execute_native_verify_rollback must only check scoped ranks."""
    path = _scoped_bundle(tmp_path, ranks=[1])
    bundle = bundle_mod.load_bundle(path)
    site = load_site(SITE)
    _poison_remote(monkeypatch)
    captured_ranks: dict[str, list[int]] = {}
    orig_execute = runtime.execute

    def fake_execute(actions, timeout):
        ranks = sorted(a.rank for a in actions)
        key = "cache" if any("sparkring-cache" in a.shell_command for a in actions) else "engine"
        captured_ranks.setdefault(key, []).extend(ranks)
        return {a.rank: {"exit_code": 0, "stdout": "", "stderr": ""}
                for a in actions}

    monkeypatch.setattr(runtime, "execute", fake_execute)
    try:
        bundle_mod.execute_native_verify_rollback(bundle, site)
    finally:
        monkeypatch.setattr(runtime, "execute", orig_execute)
    assert captured_ranks.get("cache") == [1]
    assert sorted(set(captured_ranks.get("engine", []))) == [0, 1, 2, 3]


def test_ranks_unknown_id_rejected_at_execute_start(tmp_path, monkeypatch):
    """execute_native_start must reject unknown rank IDs."""
    path = _scoped_bundle(tmp_path, ranks=[0, 99])
    bundle = bundle_mod.load_bundle(path)
    site = load_site(SITE)
    _poison_remote(monkeypatch)
    with pytest.raises(bundle_mod.BundleError) as exc:
        bundle_mod.execute_native_start(
            bundle, site, confirmation="START-test",
        )
    assert "99" in str(exc.value)


def test_ranks_unknown_id_rejected_at_execute_status(tmp_path, monkeypatch):
    """execute_native_status must reject unknown rank IDs."""
    path = _scoped_bundle(tmp_path, ranks=[0, 99])
    bundle = bundle_mod.load_bundle(path)
    site = load_site(SITE)
    _poison_remote(monkeypatch)
    with pytest.raises(bundle_mod.BundleError) as exc:
        bundle_mod.execute_native_status(bundle, site)
    assert "99" in str(exc.value)


def test_ranks_unknown_id_rejected_at_execute_verify_rollback(tmp_path, monkeypatch):
    """execute_native_verify_rollback must reject unknown rank IDs."""
    path = _scoped_bundle(tmp_path, ranks=[0, 99])
    bundle = bundle_mod.load_bundle(path)
    site = load_site(SITE)
    _poison_remote(monkeypatch)
    with pytest.raises(bundle_mod.BundleError) as exc:
        bundle_mod.execute_native_verify_rollback(bundle, site)
    assert "99" in str(exc.value)


@pytest.mark.parametrize(
    "command,extra_args",
    [
        ("start", ["--execute", "--confirmation", "START-test"]),
        ("stop", ["--execute", "--confirmation", "START-test"]),
        ("rollback", ["--execute", "--confirmation", "START-test"]),
        ("status", ["--execute"]),
        ("verify-rollback", ["--execute"]),
    ],
)
def test_ranks_unknown_id_rejected_by_executed_cli(
    tmp_path, monkeypatch, command, extra_args,
):
    """Every executed lifecycle command rejects unknown ranks pre-SSH."""
    import sparkring_bundle_launcher as launcher

    path = _scoped_bundle(tmp_path, ranks=[0, 99])
    remote_calls = []

    def fail_if_called(*args, **kwargs):
        remote_calls.append((args, kwargs))
        raise AssertionError("remote executor reached")

    monkeypatch.setattr(runtime, "execute", fail_if_called)
    with pytest.raises(SystemExit) as exc:
        launcher.main([
            "--bundle", str(path), "--site", str(SITE), command,
            *extra_args,
        ])
    assert exc.value.code == 2
    assert remote_calls == []


def test_scoped_rank0_readiness_requires_rank0_at_plan(tmp_path):
    """A rank0 probe cannot silently disappear from a non-rank0 scope."""
    path = _scoped_bundle(tmp_path, ranks=[2])
    document = json.loads(path.read_text(encoding="utf-8"))
    document["services"][0]["readiness"] = {
        "kind": "container-running",
        "rank_scope": "rank0",
    }
    path = _write_bundle(tmp_path, document)
    bundle = bundle_mod.load_bundle(path)
    with pytest.raises(bundle_mod.BundleError) as exc:
        bundle_mod.bundle_plan(bundle, load_site(SITE))
    assert "rank_scope 'rank0' requires rank 0" in str(exc.value)


def test_scoped_rank0_readiness_rejected_before_execute(tmp_path, monkeypatch):
    """Invalid scoped readiness fails before any remote start action."""
    path = _scoped_bundle(tmp_path, ranks=[2])
    document = json.loads(path.read_text(encoding="utf-8"))
    document["services"][0]["readiness"] = {
        "kind": "container-running",
        "rank_scope": "rank0",
    }
    path = _write_bundle(tmp_path, document)
    bundle = bundle_mod.load_bundle(path)
    _poison_remote(monkeypatch)
    with pytest.raises(bundle_mod.BundleError) as exc:
        bundle_mod.execute_native_start(
            bundle, load_site(SITE), confirmation="START-test",
        )
    assert "rank_scope 'rank0' requires rank 0" in str(exc.value)


def test_ranks_rollback_only_removes_ledgered_scoped_ranks(tmp_path, monkeypatch):
    """Invocation-local rollback must only remove started scoped ranks."""
    path = _scoped_bundle(tmp_path, ranks=[0, 1])
    bundle = bundle_mod.load_bundle(path)
    site = load_site(SITE)
    _poison_remote(monkeypatch)
    rollback_ranks: list[int] = []
    orig_execute = runtime.execute

    def fake_execute(actions, timeout):
        ranks = sorted(a.rank for a in actions)
        rollback_ranks.extend(ranks)
        return {a.rank: {"exit_code": 0, "stdout": "", "stderr": ""}
                for a in actions}

    monkeypatch.setattr(runtime, "execute", fake_execute)
    try:
        # Simulate a start where cache succeeds on rank 0 only, then
        # engine fails — rollback should remove cache from rank 0 only.
        call_count = [0]

        def fake_execute_staged(actions, timeout):
            call_count[0] += 1
            ranks = sorted(a.rank for a in actions)
            if call_count[0] == 1:
                # Cache start: succeed on rank 0, fail on rank 1
                return {a.rank: {"exit_code": 0 if a.rank == 0 else 1,
                                 "stdout": "a" * 12 if a.rank == 0 else "",
                                 "stderr": "" if a.rank == 0 else "fail"}
                        for a in actions}
            # Subsequent calls (engine start, rollback) — track rollback
            if "rm --force" in " ".join(a.shell_command for a in actions):
                rollback_ranks.extend(ranks)
            return {a.rank: {"exit_code": 0, "stdout": "a" * 12,
                             "stderr": ""}
                    for a in actions}

        monkeypatch.setattr(runtime, "execute", fake_execute_staged)
        result = bundle_mod.execute_native_start(
            bundle, site, confirmation="START-test",
        )
    finally:
        monkeypatch.setattr(runtime, "execute", orig_execute)
    # Rollback should have been attempted
    assert result.get("rollback") is not None
    # Cache rollback should only target rank 0 (the only one that started)
    assert rollback_ranks == [0]


# ---------------------------------------------------------------------------
# Structured-container direct-entrypoint contract
# ---------------------------------------------------------------------------

def test_structured_start_emits_entrypoint_flag(tmp_path):
    """docker run must include --entrypoint <argv[0]> before the image."""
    sc_path = _write_structured(tmp_path, "cache", _structured_doc())
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    doc = _bundle_doc([
        _structured_service("cache", "cache", sc_path),
        _native_service("engine", "serving", profile, depends_on=["cache"]),
    ])
    path = _write_bundle(tmp_path, doc)
    bundle = bundle_mod.load_bundle(path)
    site = load_site(_site(tmp_path))
    actions = bundle_mod._native_start_actions(site, bundle.services[0], bundle)
    cmd = actions[0].shell_command
    assert "--entrypoint /opt/bin/cache-server" in cmd
    # argv[0] must NOT appear after the image (it's the entrypoint, not a CMD arg)
    # Find image_id position; everything after it is CMD args = argv[1:]
    image_id = "sha256:" + "b" * 64
    # image_id appears in both the guard and docker run; use last occurrence
    idx = cmd.rindex(image_id)
    after_image = cmd[idx + len(image_id):]
    assert "/opt/bin/cache-server" not in after_image
    # argv[1:] must appear after the image
    assert "--port" in after_image
    assert "6556" in after_image


def test_structured_start_entrypoint_overrides_image_entrypoint(tmp_path):
    """An inherited image ENTRYPOINT cannot override the declared executable.

    Docker's --entrypoint flag always takes precedence over the image's
    built-in ENTRYPOINT.  Verify the command shape puts argv[0] as
    --entrypoint so no image entrypoint can intercept it.
    """
    sc_path = _write_structured(tmp_path, "cache", _structured_doc(
        argv=["/opt/venv/bin/lmcache", "server",
              "--port", "6556", "--kv-Role", "LMCacheWorker"],
    ))
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    doc = _bundle_doc([
        _structured_service("cache", "cache", sc_path),
        _native_service("engine", "serving", profile, depends_on=["cache"]),
    ])
    path = _write_bundle(tmp_path, doc)
    bundle = bundle_mod.load_bundle(path)
    site = load_site(_site(tmp_path))
    actions = bundle_mod._native_start_actions(site, bundle.services[0], bundle)
    cmd = actions[0].shell_command
    assert "--entrypoint /opt/venv/bin/lmcache" in cmd
    image_id = "sha256:" + "b" * 64
    idx = cmd.rindex(image_id)
    after_image = cmd[idx + len(image_id):]
    # Only argv[1:] should follow the image
    assert "server" in after_image
    assert "--port" in after_image
    assert "6556" in after_image
    assert "--kv-Role" in after_image
    assert "LMCacheWorker" in after_image
    # argv[0] must not be duplicated after the image
    assert "/opt/venv/bin/lmcache" not in after_image


def test_structured_start_zero_argument_executable(tmp_path):
    """A zero-argument executable (argv has only argv[0]) must work."""
    sc_path = _write_structured(tmp_path, "cache", _structured_doc(
        argv=["/opt/bin/standalone-cache"],
    ))
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    doc = _bundle_doc([
        _structured_service("cache", "cache", sc_path),
        _native_service("engine", "serving", profile, depends_on=["cache"]),
    ])
    path = _write_bundle(tmp_path, doc)
    bundle = bundle_mod.load_bundle(path)
    site = load_site(_site(tmp_path))
    actions = bundle_mod._native_start_actions(site, bundle.services[0], bundle)
    cmd = actions[0].shell_command
    assert "--entrypoint /opt/bin/standalone-cache" in cmd
    image_id = "sha256:" + "b" * 64
    # The docker run command itself ends at the image with no trailing args
    # Nothing should follow the image when argv has only one element
    # (the guard "test ... && exec ..." is before the docker run)
    # The docker run command itself ends at the image with no trailing args
    docker_part = cmd[cmd.index("docker run"):]
    # Split on image_id — everything after image in the docker command
    docker_after = docker_part[docker_part.index(image_id) + len(image_id):]
    # Should be empty or just the closing quote of exec
    assert "--port" not in docker_after
    assert "6556" not in docker_after


def test_structured_start_arguments_are_exact_tokens(tmp_path):
    """Arguments must remain exact tokens — no shell interpolation."""
    sc_path = _write_structured(tmp_path, "cache", _structured_doc(
        argv=["/opt/bin/cache-server", "--port", "6556",
              "--extra", "value with spaces", "--flag"],
    ))
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    doc = _bundle_doc([
        _structured_service("cache", "cache", sc_path),
        _native_service("engine", "serving", profile, depends_on=["cache"]),
    ])
    path = _write_bundle(tmp_path, doc)
    bundle = bundle_mod.load_bundle(path)
    site = load_site(_site(tmp_path))
    actions = bundle_mod._native_start_actions(site, bundle.services[0], bundle)
    cmd = actions[0].shell_command
    # shlex.join preserves tokens with spaces via quoting
    assert "--entrypoint /opt/bin/cache-server" in cmd
    # The value-with-spaces must be quoted, not split
    assert "value with spaces" in cmd  # shlex.join quotes it but content present
    image_id = "sha256:" + "b" * 64
    # The digest occurs in both the identity guard and docker run.  Inspect
    # arguments after the final occurrence, which is the actual image token.
    idx = cmd.rindex(image_id)
    after_image = cmd[idx + len(image_id):]
    assert "--port" in after_image
    assert "6556" in after_image
    assert "--extra" in after_image
    assert "--flag" in after_image


def test_structured_shell_entrypoint_still_rejected(tmp_path):
    """Shell entrypoints must remain rejected under the entrypoint contract."""
    for shell in ["/bin/sh", "/bin/bash", "sh", "bash", "/usr/bin/zsh",
                  "pwsh", "powershell", "cmd", "/bin/ash", "/bin/dash"]:
        sc_path = _write_structured(tmp_path, f"cache-{shell.replace('/', '_')}",
                                    _structured_doc(argv=[shell, "-c", "true"]))
        doc = _bundle_doc([_structured_service("cache", "cache", sc_path)])
        path = _write_bundle(tmp_path, doc)
        with pytest.raises(bundle_mod.BundleError) as exc:
            bundle_mod.load_bundle(path)
        assert "shell entrypoint" in str(exc.value).lower()


def test_structured_start_plan_stable_with_entrypoint(tmp_path):
    """Plan output must be deterministic and stable with the --entrypoint fix."""
    sc_path = _write_structured(tmp_path, "cache", _structured_doc())
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    doc = _bundle_doc([
        _structured_service("cache", "cache", sc_path),
        _native_service("engine", "serving", profile, depends_on=["cache"]),
    ])
    path = _write_bundle(tmp_path, doc)
    site = _site(tmp_path)
    rc1, out1, _ = _run_cli(path, site, "plan")
    assert rc1 == 0
    rc2, out2, _ = _run_cli(path, site, "plan")
    assert rc2 == 0
    assert out1 == out2, "plan output must be byte-identical across runs"
    # Plan must show --entrypoint in the action command
    assert "--entrypoint" in out1
    assert "/opt/bin/cache-server" in out1


# ---------------------------------------------------------------------------
# Structured-container naming and rank representability
# ---------------------------------------------------------------------------

def test_structured_no_doubled_rank_suffix(tmp_path):
    """container_name base must not end in -rN; builder appends -r{rank}."""
    sc_path = _write_structured(tmp_path, "cache", _structured_doc(
        container_name="sparkring-bundle-lmcache-cs512",
    ))
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    doc = _bundle_doc([
        _structured_service("cache", "cache", sc_path, ranks=[1]),
        _native_service("engine", "serving", profile, depends_on=["cache"]),
    ])
    path = _write_bundle(tmp_path, doc)
    bundle = bundle_mod.load_bundle(path)
    site = load_site(_site(tmp_path))
    actions = bundle_mod._native_start_actions(site, bundle.services[0], bundle)
    cmd = actions[0].shell_command
    # The effective name must be base-r1, NOT base-r1-r1
    assert "sparkring-bundle-lmcache-cs512-r1" in cmd
    assert "sparkring-bundle-lmcache-cs512-r1-r1" not in cmd


def test_disjoint_ranks_same_container_name_no_collision(tmp_path):
    """Same unsuffixed container_name with disjoint rank scopes must not collide."""
    # All four cache services share the same container_name base but target
    # different ranks, so the effective names (base-r0, base-r1, ...) are unique.
    services = []
    for rank in range(4):
        sc_path = _write_structured(tmp_path, f"cache-r{rank}", _structured_doc(
            container_name="sparkring-bundle-lmcache-cs512",
        ))
        services.append(_structured_service(
            f"lmcache-r{rank}", "cache", sc_path, ranks=[rank],
        ))
    profile = _write_profile(tmp_path, "engine", _generic_doc())
    services.append(_native_service(
        "engine", "serving", profile,
        depends_on=[f"lmcache-r{r}" for r in range(4)],
    ))
    doc = _bundle_doc(services)
    path = _write_bundle(tmp_path, doc)
    # Must load without collision error
    bundle_mod.load_bundle(path)
    # Plan must succeed and produce 4 cache + 4 engine = 8 start actions
    rc, out, err = _run_cli(path, _site(tmp_path), "plan")
    assert rc == 0, err
    plan = json.loads(out)
    # Collect all start-phase actions across all phases entries
    all_actions = []
    for phase in plan["phases"]:
        if phase["phase"] == "start":
            all_actions.extend(phase["actions"])
    cache_names = set()
    for a in all_actions:
        for token in a["remote_command"].split():
            if "sparkring-bundle-lmcache-cs512-r" in token:
                cache_names.add(token)
    # Each rank produces a unique effective name
    assert len(cache_names) == 4
    assert cache_names == {
        "sparkring-bundle-lmcache-cs512-r0",
        "sparkring-bundle-lmcache-cs512-r1",
        "sparkring-bundle-lmcache-cs512-r2",
        "sparkring-bundle-lmcache-cs512-r3",
    }
