"""Check prebuilt toolchain assembly without Docker, a GPU or real artifacts."""
import hashlib
import io
import json
import tarfile

import pytest

from runtime.images import toolchain_assembly as assembly

PARENT = "sha256:" + "1" * 64
LIBRARY = b"patched nccl 2.32.3"


def add(archive, name, data=None, link=None):
    entry = tarfile.TarInfo(name)
    if link is not None:
        entry.type, entry.linkname = tarfile.SYMTYPE, link
        archive.addfile(entry)
    else:
        entry.size = len(data)
        archive.addfile(entry, io.BytesIO(data))


@pytest.fixture
def inputs(tmp_path):
    artifacts = tmp_path / "nccl.tar.gz"
    with tarfile.open(artifacts, "w:gz") as archive:
        add(archive, "./lib/libnccl.so.2.32.3", LIBRARY)
        add(archive, "./lib/libnccl.so.2", link="libnccl.so.2.32.3")
        add(archive, "./include/nccl.h", b"// header\n")
        add(archive, "./licenses/LICENSE.txt", b"license\n")
    lock = json.loads((assembly.HERE / "cuda134-nccl232-installer.json").read_text())
    lock["prebuilt_artifacts"]["nccl_archive_sha256"] = assembly.digest(artifacts)
    lock["prebuilt_artifacts"]["nccl_library_sha256"] = hashlib.sha256(LIBRARY).hexdigest()
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps(lock))
    receipt = tmp_path / "external-base-installed.json"
    receipt.write_bytes(b'{"files": {}}\n')
    return {"lock_path": lock_path, "parent_release": "software-layer", "parent_image": PARENT,
            "parent_receipt": receipt, "parent_tag": "sparkring:software-layer",
            "nccl_artifacts": artifacts, "output": tmp_path / "context", "version_label": "fixture-1"}


def test_recorded_installer_lock_names_prebuilt_identities():
    lock = json.loads((assembly.HERE / "cuda134-nccl232-installer.json").read_text())
    base = json.loads((assembly.HERE / "cuda134-nccl232.json").read_text())
    assert lock["parent"] is None and lock["variant"] == "combined"
    assert lock["cuda"] == base["cuda"] and lock["nccl"] == base["nccl"]
    assert set(lock["prebuilt_artifacts"]) == {"toolkit_image_config", "toolkit_export_sha256",
                                               "nccl_archive_sha256", "nccl_library_sha256",
                                               "toolchain_source_commit"}


def test_assembly_binds_parent_receipt_and_artifacts(inputs):
    result = assembly.prepare(**inputs)
    context = inputs["output"]
    lock = json.loads((context / "toolchain.json").read_text())
    receipt_sha = assembly.digest(inputs["parent_receipt"])
    assert lock["parent"] == {"release": "software-layer", "reference": PARENT,
                              "receipt_sha256": receipt_sha, "publication_kind": "local_candidate"}
    assert list(lock) == list(json.loads(inputs["lock_path"].read_text()))
    dockerfile = (context / "Dockerfile").read_text()
    lines = dockerfile.splitlines()
    assert lines[0] == "FROM sparkring:installer-toolkit-70b4f8d4 AS toolkit"
    assert lines[1] == "FROM sparkring:software-layer"
    assert f'RUN echo "{receipt_sha}  /opt/sparkring/receipts/external-base-installed.json" | sha256sum -c -' in lines
    assert lines.index("RUN python3 /opt/sparkring/toolchain/toolchain.py seal") > max(
        index for index, line in enumerate(lines) if "sha256sum -c" in line)
    assert 'LABEL org.sparkring.version="fixture-1" org.sparkring.toolchain.status="research-only"' in lines
    assert (context / "nccl-artifacts/lib/libnccl.so.2").is_symlink()
    assert (context / "gpu_smoke.py").read_bytes() == (assembly.HERE / "toolchain_gpu_smoke.py").read_bytes()
    assert (context / "toolchain_runtime.py").read_bytes() == (assembly.HERE / "toolchain_runtime.py").read_bytes()
    assert result["dockerfile_sha256"] == assembly.digest(context / "Dockerfile")
    with pytest.raises(ValueError, match="already exists"):
        assembly.prepare(**inputs)


def test_assembly_rejects_unrecorded_archive_without_output(inputs):
    inputs["nccl_artifacts"].write_bytes(b"other")
    with pytest.raises(ValueError, match="archive hash differs"):
        assembly.prepare(**inputs)
    assert not inputs["output"].exists()


def test_assembly_rejects_escaping_links_and_removes_output(inputs, tmp_path):
    with tarfile.open(inputs["nccl_artifacts"], "w:gz") as archive:
        add(archive, "lib/libnccl.so.2.32.3", LIBRARY)
        add(archive, "lib/libnccl.so", link="../../etc/passwd")
    lock = json.loads(inputs["lock_path"].read_text())
    lock["prebuilt_artifacts"]["nccl_archive_sha256"] = assembly.digest(inputs["nccl_artifacts"])
    inputs["lock_path"].write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="escapes its directory"):
        assembly.prepare(**inputs)
    assert not inputs["output"].exists()


@pytest.mark.parametrize("field, value", [("parent_image", "sparkring:software-layer"),
                                          ("parent_tag", "sha256:" + "2" * 64),
                                          ("version_label", "two words")])
def test_assembly_rejects_ambiguous_identities(inputs, field, value):
    inputs[field] = value
    with pytest.raises(ValueError):
        assembly.prepare(**inputs)
    assert not inputs["output"].exists()
