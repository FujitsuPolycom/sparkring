"""The CSF source layer of the kraken line and its release selection; offline.

A synthetic parent stands in for dev-20261004-kraken: its receipts record a
few vLLM and B12X sources, and a small manifest pins their merged bytes.
"""
import copy
import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest

from runtime.common import image_lock, installer_image
from runtime.images import derive_kraken_csf_sources as csf
from runtime.images import derived_layer

ROOT = Path(__file__).resolve().parents[2]
SITE = derived_layer.SITE
PARENT = "sha256:" + "a" * 64
RELEASE = "dev-20261007-kraken-csf-sircl-cuda1342-nccl2323-status034"
PARENT_RELEASE = "dev-20261004-kraken-cuda1342-nccl2323-status034"
PARENT_FILES = {"vllm/model_executor/model_loader/nvfp4_csf_loader.py": b"OLD_LOADER = 1\n",
                "b12x/_lib/quant/nvfp4_csf.py": b"OLD_SCALES = 1\n",
                "vllm/envs.py": b"UNCHANGED = 1\n"}
MERGED = {"vllm/model_executor/model_loader/nvfp4_csf_loader.py": b"CSF_LOADER = 2\n",
          "b12x/_lib/quant/nvfp4_csf.py": b"CSF_SCALES = 2\n",
          "vllm/model_executor/layers/l2_prefetch.py": b"PREFETCH = 1\n"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def manifest_record():
    files = []
    for path, data in sorted(MERGED.items()):
        inherited = PARENT_FILES.get(path)
        files.append({"path": path, "package": path.split("/")[0],
                      "inherited_sha256": sha(inherited) if inherited is not None else None,
                      "sha256": sha(data), "bytes": len(data)})
    payload = payload_bytes(MERGED)
    commits = {name: {"commit": name[0] * 40, "branch": "b", "image_commit": "c" * 40, "image_branch": "b",
                      "upstream_commit": "d" * 40, "upstream_branch": "b"} for name in csf.PACKAGES}
    return {"schema": csf.SCHEMA, "purpose": "fixture", "parent_release": PARENT_RELEASE, "parent_image_id": PARENT,
            "site_packages": SITE, "payload": {"file": "csf_payload.tar", "sha256": sha(payload),
                                               "bytes": len(payload)},
            "sources": commits, "transport_sources": [], "files": files}


def payload_bytes(files):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as bundle:
        for path, data in sorted(files.items()):
            entry = tarfile.TarInfo(path)
            entry.size, entry.mode = len(data), 0o644
            bundle.addfile(entry, io.BytesIO(data))
    return buffer.getvalue()


def parent():
    """The parent lock and a reader of its receipts and files."""
    receipt = {"schema": "sparkring-external-installed/v1", "composition_sha256": "c" * 64,
               "capabilities": {"transport_profile": "tp2-rocenante-adaptive-prepared",
                                "transport_manifest_sha256": "3" * 64, "runtime_status": {"version": "0.3.4"}},
               "files": {SITE + path: sha(data) for path, data in PARENT_FILES.items()}}
    base_raw = derived_layer.canonical_json(receipt)
    toolchain_raw = derived_layer.canonical_json({"variant": "combined", "parent_receipt_sha256": sha(base_raw)},
                                                 sort_keys=False)
    lock = dict(installer_image.default_lock(), image_id=PARENT, image_reference=PARENT,
                parent_receipt_sha256=sha(base_raw), toolchain_receipt_sha256=sha(toolchain_raw),
                composition_sha256="c" * 64, transport_manifest_sha256="3" * 64)
    files = {derived_layer.BASE_RECEIPT: base_raw, derived_layer.TOOLCHAIN_RECEIPT: toolchain_raw,
             **{SITE + path: data for path, data in PARENT_FILES.items()}}
    return lock, files.__getitem__, receipt


def merged_directory(tmp_path, files=MERGED):
    root = tmp_path / "overlay"
    for path, data in files.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return root


def test_the_committed_manifest_pins_only_vllm_and_b12x_python_sources():
    record = csf.manifest()
    assert record["parent_release"] == PARENT_RELEASE
    parent_lock = json.loads((ROOT / "runtime/releases" / PARENT_RELEASE / "installer-image.json").read_text())
    assert record["parent_image_id"] == parent_lock["image_id"]
    rows = record["files"]
    assert len(rows) == 49 and sum(row["inherited_sha256"] is None for row in rows) == 2
    assert sum(row["bytes"] for row in rows) == 3594218
    assert {row["package"] for row in rows} == {"vllm", "b12x"}
    # The six B12X sources the prepared transport hashes are left as the parent has them.
    assert {SITE + row["path"] for row in record["transport_sources"]} == derived_layer.TRANSPORT_SOURCES
    assert not {SITE + row["path"] for row in rows} & derived_layer.TRANSPORT_SOURCES
    assert record["sources"]["vllm"]["commit"].startswith("bc9ea774")
    assert record["sources"]["b12x"]["commit"].startswith("cc36aa6f")


def test_the_merged_vllm_files_are_those_of_the_pinned_build_sircl_names_for_the_csf_checkpoint():
    from spark_transport.sircl.sparkring_sircl.vllm import pins
    rows = {row["path"]: row for row in csf.manifest()["files"]}
    builds = {build.name: build for build in pins.SUPPORTED}
    image, merged = builds["lil-image-aba309e4610c"], builds["sparkring-kraken-beta-20261007-bc9ea774"]
    for name, digest in merged.files.items():
        row = rows.get("vllm/" + name)
        if row is None:
            # A file the layer does not write keeps the parent image's bytes.
            assert image.files[name] == digest, name
        else:
            assert (row["inherited_sha256"], row["sha256"]) == (image.files[name], digest), name
    checkpoint = "local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD@dec48abd33efa73c3bb7c95b74eee10cad34f9be"
    assert image_lock.CHECKPOINT_BUILDS[checkpoint] == (merged.name,)


@pytest.mark.parametrize("edit, message", [
    (lambda r: r["files"].append(dict(r["files"][0], path="vllm/zz.so")), "Python file"),
    (lambda r: (r["files"].append(dict(r["files"][0], path="b12x/preparation/session.py", package="b12x")),
                r["files"].sort(key=lambda row: row["path"])), "prepared transport verifies"),
    (lambda r: r["files"].insert(0, dict(r["files"][-1])), "each file once, sorted"),
    (lambda r: r["files"][0].update(sha256=r["files"][0]["inherited_sha256"]), "differing from the parent"),
    (lambda r: r["files"][0].update(package="sparkcache"), "Python file of the vllm or b12x package"),
    (lambda r: r.update(site_packages="/usr/lib/python3/dist-packages/"), "site-packages directory"),
])
def test_a_manifest_outside_its_owners_is_refused(tmp_path, edit, message):
    record = manifest_record()
    edit(record)
    path = tmp_path / "sources.json"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match=message):
        csf.manifest(path)


def test_the_context_writes_the_merged_files_and_records_them(tmp_path):
    lock, read, receipt = parent()
    record = manifest_record()
    layer = csf.layer(csf.directory_reader(merged_directory(tmp_path)), record)
    result = derived_layer.prepare_layer(layer, lock, read, tmp_path / "context")
    context = Path(result["context"])
    plan = json.loads((context / "plan.json").read_text())
    assert plan["added"] == [SITE + "vllm/model_executor/layers/l2_prefetch.py"]
    assert plan["replaced"] == sorted(SITE + path for path in PARENT_FILES if path in MERGED)
    derived = json.loads((context / "files" / derived_layer.BASE_RECEIPT.lstrip("/")).read_text())
    for path, data in MERGED.items():
        assert (context / "files" / (SITE + path).lstrip("/")).read_bytes() == data
        assert derived["files"][SITE + path] == sha(data)
    # The parent's other files, transport fields and status are kept.
    assert derived["files"][SITE + "vllm/envs.py"] == receipt["files"][SITE + "vllm/envs.py"]
    assert derived["capabilities"] == receipt["capabilities"]
    provenance = json.loads((context / "files" / csf.PROVENANCE.lstrip("/")).read_text())
    assert set(provenance["files"]) == {SITE + path for path in MERGED}
    assert "nvfp4_csf" in provenance["purpose"]
    locked = derived_layer.derived_lock(plan, {"Id": "sha256:" + "b" * 64, "Size": 2000}, "dev-csf")
    assert locked["profiles"] == lock["profiles"] and locked["transport_manifest_sha256"] == "3" * 64


def test_a_source_directory_with_other_bytes_is_refused(tmp_path):
    lock, read, _ = parent()
    changed = dict(MERGED, **{"b12x/_lib/quant/nvfp4_csf.py": b"OTHER = 3\n"})
    layer = csf.layer(csf.directory_reader(merged_directory(tmp_path, changed)), manifest_record())
    with pytest.raises(ValueError, match="differs from the merge's file"):
        derived_layer.prepare_layer(layer, lock, read, tmp_path / "context")


def test_a_parent_whose_file_differs_from_the_manifest_is_refused(tmp_path):
    lock, read, _ = parent()
    record = manifest_record()
    record["files"][0]["inherited_sha256"] = "9" * 64
    layer = csf.layer(csf.directory_reader(merged_directory(tmp_path)), record)
    with pytest.raises(ValueError, match="pinned inherited or resulting"):
        derived_layer.prepare_layer(layer, lock, read, tmp_path / "context")


def test_the_payload_archive_is_read_only_when_it_is_the_pinned_one(tmp_path):
    record = manifest_record()
    archive = tmp_path / "csf_payload.tar"
    archive.write_bytes(payload_bytes(MERGED))
    read = csf.payload_reader(archive, record)
    assert all(read(path) == data for path, data in MERGED.items())
    archive.write_bytes(payload_bytes({**MERGED, "vllm/extra.py": b"x\n"}))
    with pytest.raises(ValueError, match="is not the payload"):
        csf.payload_reader(archive, record)
    other = copy.deepcopy(record)
    other["payload"]["sha256"] = sha(archive.read_bytes())
    with pytest.raises(ValueError, match="differ from the manifest"):
        csf.payload_reader(archive, other)


def test_prepare_refuses_another_parent_release(tmp_path, capsys):
    other = tmp_path / "lock.json"
    other.write_text(json.dumps(installer_image.default_lock() | {"name": "dev-other"}))
    with pytest.raises(SystemExit):
        csf.main(["prepare", "--parent-lock", str(other), "--sources", str(tmp_path), "--output",
                  str(tmp_path / "context")])
    assert f"derives from {PARENT_RELEASE}" in capsys.readouterr().err


def test_the_release_selects_its_builders_and_pins_its_inputs():
    release = json.loads((ROOT / "runtime/releases" / RELEASE / "release.json").read_text())
    assert release["id"] == RELEASE and release["selection"] == "source-build"
    for item in release["inputs"]:
        assert hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest() == item["sha256"], item["path"]
    assert {item["path"] for item in release["inputs"]} >= {
        f"runtime/releases/{PARENT_RELEASE}/installer-image.json",
        "runtime/images/compositions/kraken-csf-sources-20261007/sources.json"}
    builders = {row["id"]: row for row in json.loads((ROOT / "runtime/images/builders.json").read_text())["builders"]}
    assert RELEASE in builders["installer-sircl-layer"]["releases"]
    assert builders["installer-kraken-csf-sources"]["path"] == "runtime/images/derive_kraken_csf_sources.py"
    assert release["image"]["builder"] == builders["installer-sircl-layer"]["path"]
