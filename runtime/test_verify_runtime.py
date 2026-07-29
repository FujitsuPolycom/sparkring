"""Tests for runtime/verify-runtime.py (stdlib + pytest only)."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SPEC = importlib.util.spec_from_file_location(
    "verify_runtime", os.path.join(_HERE, "verify-runtime.py")
)
vr = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(vr)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _seal(manifest: dict) -> dict:
    manifest["self_hash"] = vr.manifest_self_hash(manifest)
    return manifest


@pytest.fixture
def runtime_fixture(tmp_path):
    """Build a tiny fake installed runtime plus a sealed manifest for it."""
    src_root = tmp_path / "site-packages" / "vllm"
    src_root.mkdir(parents=True)
    tree_file = src_root / "version.py"
    tree_file.write_bytes(b"__version__ = '0.11.2.dev279'\n")
    patched_file = src_root / "worker.py"
    patched_file.write_bytes(b"# patched worker postimage\n")

    lib = tmp_path / "libnccl.so.2.30.7"
    lib.write_bytes(b"\x7fELF fake nccl payload")

    manifest = {
        "schema_version": 1,
        "runtime_id": "sparkring-gb10-20260728",
        "components": [
            {
                "name": "vllm",
                "repo": "vllm-project/vllm",
                "commit": "fcc614141e5e9ab18cb304c476f7feed2a9552e3",
                "verification": {
                    "mode": "tree",
                    "root": str(src_root),
                    "files": {"version.py": _sha256(tree_file.read_bytes())},
                },
            },
            {
                "name": "vllm-patch-overlay",
                "verification": {
                    "mode": "patched_files",
                    "root": str(src_root),
                    "files": {
                        "worker.py": {
                            "preimage": _sha256(b"# stock worker preimage\n"),
                            "postimage": _sha256(patched_file.read_bytes()),
                        }
                    },
                },
            },
        ],
        "native_libs": [{"path": str(lib), "sha256": _sha256(lib.read_bytes())}],
        "image": {"digest": "sha256:" + "ab" * 32},
    }
    _seal(manifest)

    manifest_path = tmp_path / "runtime-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return {
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "patched_file": patched_file,
        "lib": lib,
        "tmp_path": tmp_path,
    }


def _rewrite(fx, manifest, reseal=True):
    if reseal:
        _seal(manifest)
    with open(fx["manifest_path"], "w", encoding="utf-8") as f:
        json.dump(manifest, f)


def _run(fx, *extra, env_digest="sha256:" + "ab" * 32, monkeypatch=None):
    argv = ["--manifest", fx["manifest_path"], *extra]
    if monkeypatch is not None:
        if env_digest is None:
            monkeypatch.delenv(vr.IMAGE_DIGEST_ENV, raising=False)
        else:
            monkeypatch.setenv(vr.IMAGE_DIGEST_ENV, env_digest)
    return vr.main(argv)


def test_valid_manifest_passes(runtime_fixture, monkeypatch, capsys):
    rc = _run(runtime_fixture, "--json", monkeypatch=monkeypatch)
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["ok"] is True
    assert all(c["status"] == "pass" for c in out["checks"])


def test_self_hash_tamper_fails(runtime_fixture, monkeypatch, capsys):
    m = dict(runtime_fixture["manifest"])
    m["runtime_id"] = "tampered-after-freeze"
    _rewrite(runtime_fixture, m, reseal=False)  # keep the stale self_hash
    rc = _run(runtime_fixture, "--json", monkeypatch=monkeypatch)
    out = json.loads(capsys.readouterr().out)
    assert rc == 2
    check = {c["name"]: c for c in out["checks"]}["manifest_self_hash"]
    assert check["status"] == "fail"
    assert "self-hash mismatch" in check["detail"]


def test_postimage_mismatch_fails(runtime_fixture, monkeypatch, capsys):
    runtime_fixture["patched_file"].write_bytes(b"# drifted from postimage\n")
    rc = _run(runtime_fixture, "--json", monkeypatch=monkeypatch)
    out = json.loads(capsys.readouterr().out)
    assert rc == 2
    check = {c["name"]: c for c in out["checks"]}["installed_source"]
    assert check["status"] == "fail"
    assert "postimage" in check["detail"]


def test_missing_native_lib_fails(runtime_fixture, monkeypatch, capsys):
    runtime_fixture["lib"].unlink()
    rc = _run(runtime_fixture, "--json", monkeypatch=monkeypatch)
    out = json.loads(capsys.readouterr().out)
    assert rc == 2
    check = {c["name"]: c for c in out["checks"]}["native_libs"]
    assert check["status"] == "fail"
    assert "missing" in check["detail"]


def test_runtime_id_mismatch_fails(runtime_fixture, monkeypatch, capsys):
    rc = _run(
        runtime_fixture,
        "--expect-runtime-id",
        "some-other-runtime",
        "--json",
        monkeypatch=monkeypatch,
    )
    out = json.loads(capsys.readouterr().out)
    assert rc == 2
    check = {c["name"]: c for c in out["checks"]}["runtime_id"]
    assert check["status"] == "fail"
    assert "mismatch" in check["detail"]


def test_absent_image_digest_warns_not_fails(runtime_fixture, monkeypatch, capsys):
    rc = _run(runtime_fixture, "--json", env_digest=None, monkeypatch=monkeypatch)
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    check = {c["name"]: c for c in out["checks"]}["image_digest"]
    assert check["status"] == "skip"
    assert "WARNING" in check["detail"]


def test_image_digest_mismatch_fails(runtime_fixture, monkeypatch, capsys):
    rc = _run(
        runtime_fixture,
        "--json",
        env_digest="sha256:" + "cd" * 32,
        monkeypatch=monkeypatch,
    )
    out = json.loads(capsys.readouterr().out)
    assert rc == 2
    check = {c["name"]: c for c in out["checks"]}["image_digest"]
    assert check["status"] == "fail"


def test_external_image_digest_requires_launcher_binding(
    runtime_fixture, monkeypatch, capsys
):
    manifest = runtime_fixture["manifest"]
    manifest["image"]["digest"] = "external"
    manifest["self_hash"] = vr.manifest_self_hash(manifest)
    Path(runtime_fixture["manifest_path"]).write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    rc = _run(
        runtime_fixture,
        "--json",
        env_digest="sha256:" + "ab" * 32,
        monkeypatch=monkeypatch,
    )
    out = json.loads(capsys.readouterr().out)
    check = {c["name"]: c for c in out["checks"]}["image_digest"]
    assert rc == 0
    assert check["status"] == "pass"
    assert "host-side preflight" in check["detail"]


def test_external_image_digest_rejects_malformed_binding(
    runtime_fixture, monkeypatch, capsys
):
    manifest = runtime_fixture["manifest"]
    manifest["image"]["digest"] = "external"
    manifest["self_hash"] = vr.manifest_self_hash(manifest)
    Path(runtime_fixture["manifest_path"]).write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    rc = _run(
        runtime_fixture,
        "--json",
        env_digest="sha256:not-a-digest",
        monkeypatch=monkeypatch,
    )
    out = json.loads(capsys.readouterr().out)
    check = {c["name"]: c for c in out["checks"]}["image_digest"]
    assert rc == 2
    assert check["status"] == "fail"


def test_unreadable_manifest_exits_3(tmp_path, monkeypatch, capsys):
    missing = str(tmp_path / "nope.json")
    rc = vr.main(["--manifest", missing, "--json"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 3
    assert out["ok"] is False


def test_attestation_line(runtime_fixture, monkeypatch, capsys):
    rc = _run(runtime_fixture, "--emit-attestation", monkeypatch=monkeypatch)
    assert rc == 0
    last = capsys.readouterr().out.strip().splitlines()[-1]
    att = json.loads(last)
    assert att == {
        "runtime_id": runtime_fixture["manifest"]["runtime_id"],
        "manifest_self_hash": runtime_fixture["manifest"]["self_hash"],
    }
