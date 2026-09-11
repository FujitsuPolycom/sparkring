"""Offline integrity and output-boundary checks for transport packaging."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("transport_package", ROOT / "package.py")
package = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package)


def test_source_inventory_and_correctness_repairs():
    receipt, source, _ = package.verified_sources()
    assert receipt["reference_native_sha256"].startswith("056243fa")
    kernel = source["spark_transport/experiments/tiled_prefill/fused_prefill_kernels.cu"].decode()
    barrier = kernel.split("__device__ bool flow_barrier", 1)[1]
    assert barrier.index("__syncthreads();") < barrier.index("atomicAdd(")
    worker = source["spark_transport/src/tp4_fused_prefill_session.cpp"].decode()
    assert "thread_ = std::thread([this] { loop(); });" in worker
    assert "write_payload_and_doorbell" in source["spark_transport/src/verbs_endpoint.cpp"].decode()


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a\\b", "a/../b", "C:/escape", "C:relative"])
def test_unsafe_paths_refused(name):
    with pytest.raises(ValueError):
        package.checked_name(name)


def test_tampering_is_rejected_before_output(tmp_path):
    copied = tmp_path / "package"
    shutil.copytree(ROOT, copied, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    with (copied / "native-source.tar.gz").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="archive digest"):
        package.verified_sources(copied)


def test_existing_destination_is_not_modified(tmp_path):
    sentinel = tmp_path / "sentinel"
    sentinel.write_text("preserve")
    with pytest.raises(ValueError, match="must not exist"):
        package.write_tree(tmp_path, {"sentinel": b"replace"})
    assert sentinel.read_text() == "preserve"


def test_rebuilt_library_has_distinct_manifest_and_no_qualification_claim(tmp_path):
    library = tmp_path / "fixture.so"
    library.write_bytes(b"offline fixture, not a loadable library")
    digest = hashlib.sha256(library.read_bytes()).hexdigest()
    result = package.prepare_bundle(library, digest, tmp_path / "bundle")
    assert result["status"] == "research-only"
    assert not result["reference_artifact_match"]
    assert not result["hardware_qualification_performed"]
    manifest = json.loads((tmp_path / "bundle" / package.MANIFEST).read_bytes())
    rows = {row["path"]: row["sha256"] for row in manifest["files"]}
    assert rows[package.LIBRARY] == digest
    assert result["bundle_manifest_sha256"] != package.verified_sources()[0]["reference_bundle_manifest_sha256"]


def test_incorrect_library_digest_creates_no_output(tmp_path):
    library = tmp_path / "fixture.so"
    library.write_bytes(b"fixture")
    with pytest.raises(ValueError, match="library digest differs"):
        package.prepare_bundle(library, "0" * 64, tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()
