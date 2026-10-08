"""The SIRCL image layer: its reproducible wheel, its build context and its v3 lock; offline.

The parent image is represented by its two receipts; the native libraries by
files standing in for the ones ``natives`` builds in the parent image.
"""
import base64
import csv
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from runtime.common import image_lock, installer_image, transport
from runtime.images import derived_layer, sircl_layer

SITE = "/usr/local/lib/python3.12/dist-packages/"


def test_the_wheel_is_the_same_bytes_every_time_and_records_every_file(tmp_path):
    first = sircl_layer.write_wheel(tmp_path / "a")
    second = sircl_layer.write_wheel(tmp_path / "b")
    assert first["sha256"] == second["sha256"] and first["name"] == "sparkring_sircl-0.2.0-py3-none-any.whl"
    with zipfile.ZipFile(first["wheel"]) as archive:
        names = archive.namelist()
        info = "sparkring_sircl-0.2.0.dist-info/"
        assert names == sorted(names) and "spark_roce_gid.py" in names and "sparkring_sircl/__init__.py" in names
        assert "sparkring_sircl/oneshot/_roce_proxy.c" in names and "sparkring_sircl/p2p/_p2p_proxy.c" in names
        assert not any("__pycache__" in name or name.startswith("tests/") for name in names)
        assert archive.read(info + "top_level.txt") == b"spark_roce_gid\nsparkring_sircl\n"
        rows = list(csv.reader(io.StringIO(archive.read(info + "RECORD").decode())))
        assert {row[0] for row in rows} == set(names)
        for name, digest, size in rows:
            if name == info + "RECORD":
                assert (digest, size) == ("", "")
                continue
            data = archive.read(name)
            assert digest == "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            assert int(size) == len(data) and b"\r\n" not in data
        assert all(entry.date_time == sircl_layer.ZIP_TIME for entry in archive.infolist())


def test_the_installed_wheel_registers_both_plugins_and_imports_the_resolver(tmp_path):
    written = sircl_layer.write_wheel(tmp_path)
    root = tmp_path / "site"
    with zipfile.ZipFile(written["wheel"]) as archive:
        archive.extractall(root)
    found = {(point.group, point.name): point.value
             for distribution in importlib.metadata.distributions(path=[str(root)])
             for point in distribution.entry_points}
    from spark_transport.sircl.sparkring_sircl.vllm.serve import staging
    for group, points in staging.ENTRY_POINTS.items():
        for name, value in points.items():
            assert found[(group, name)] == value
    assert found[("console_scripts", "sircl-prepare")] == "sparkring_sircl.build:main"
    # Outside the repository the package finds SparkRing's GID resolver as the top-level module.
    probe = ("import sys; sys.path.insert(0, sys.argv[1]); import sparkring_sircl.roce_gid as gid, spark_roce_gid; "
             "print(gid.source_file().parent == __import__('pathlib').Path(sys.argv[1]).resolve())")
    done = subprocess.run([sys.executable, "-I", "-c", probe, str(root)], capture_output=True, text=True, check=True,
                          cwd=tmp_path)
    assert done.stdout.strip() == "True"


def test_a_wheel_that_is_not_this_checkouts_is_refused(tmp_path):
    written = sircl_layer.write_wheel(tmp_path)
    path = Path(written["wheel"])
    path.write_bytes(path.read_bytes() + b"\0")
    with pytest.raises(ValueError, match="not the wheel this checkout writes"):
        sircl_layer.read_wheel(path)


def parent(tmp_path):
    """The parent lock and receipts of a kraken-line image with vLLM in site-packages."""
    base = {"schema": "sparkring-external-installed/v1", "files": {SITE + "vllm/__init__.py": "a" * 64},
            "capabilities": {"runtime_status": {"version": "0.3.4"}}}
    base_raw = derived_layer.canonical_json(base)
    toolchain_raw = derived_layer.canonical_json({"variant": "combined",
                                                  "parent_receipt_sha256": hashlib.sha256(base_raw).hexdigest()},
                                                 sort_keys=False)
    lock = dict(installer_image.default_lock(), parent_receipt_sha256=hashlib.sha256(base_raw).hexdigest(),
                toolchain_receipt_sha256=hashlib.sha256(toolchain_raw).hexdigest())
    files = {derived_layer.BASE_RECEIPT: base_raw, derived_layer.TOOLCHAIN_RECEIPT: toolchain_raw}
    return lock, files.__getitem__, base


def natives(tmp_path, lock, wheel):
    """A natives directory as ``build_natives`` writes it, with stand-in library bytes."""
    from spark_transport.sircl.sparkring_sircl import build
    from spark_transport.sircl.sparkring_sircl.p2p import build as p2p
    directory = tmp_path / "natives"
    directory.mkdir()
    record = {"schema": "sparkring-sircl-natives/v1", "parent_image_id": lock["image_id"],
              "wheel": Path(wheel["wheel"]).name, "wheel_sha256": wheel["sha256"], "compiler": "gcc 13.3.0"}
    for kind, module in (("native", build), ("p2p", p2p)):
        name = module.library_path(directory=Path("/")).name
        data = f"{kind} library".encode()
        (directory / name).write_bytes(data)
        record[kind] = {"path": f"{image_lock.LIBRARY_DIRECTORY}/{name}", "sha256": hashlib.sha256(data).hexdigest(),
                        "source_digest": module.source_digest()}
    (directory / "natives.json").write_text(json.dumps(record))
    return directory


def prepared(tmp_path):
    lock, read, base = parent(tmp_path)
    wheel = sircl_layer.write_wheel(tmp_path / "wheel")
    directory = natives(tmp_path, lock, wheel)
    result = sircl_layer.prepare(lock, read, wheel["wheel"], directory, tmp_path / "context")
    return lock, base, wheel, result


def test_the_context_installs_the_wheel_and_libraries_and_records_them_in_the_parent_receipt(tmp_path):
    lock, base, wheel, result = prepared(tmp_path)
    context = Path(result["context"])
    plan = json.loads((context / "plan.json").read_text())
    derived = json.loads((context / "files" / derived_layer.BASE_RECEIPT.lstrip("/")).read_text())
    layer = json.loads((context / "files" / image_lock.LAYER_RECEIPT.lstrip("/")).read_text())
    assert layer["site_packages"] == SITE and layer["wheel"] == {"name": wheel["name"], "sha256": wheel["sha256"]}
    assert SITE + "spark_roce_gid.py" in layer["files"] and SITE + "sparkring_sircl/build.py" in layer["files"]
    for kind in ("native", "p2p"):
        assert layer["files"][layer[kind]["path"]] == layer[kind]["sha256"]
    # Every added file, the layer receipt included, is in the receipt the image's verify checks.
    assert set(plan["added"]) == set(layer["files"]) | {image_lock.LAYER_RECEIPT}
    for path in plan["added"]:
        data = (context / "files" / path.lstrip("/")).read_bytes()
        assert derived["files"][path] == hashlib.sha256(data).hexdigest()
    assert derived["files"][SITE + "vllm/__init__.py"] == base["files"][SITE + "vllm/__init__.py"]
    assert derived["capabilities"]["sircl"]["receipt_sha256"] == plan["layer_sha256"]
    toolchain = json.loads((context / "files" / derived_layer.TOOLCHAIN_RECEIPT.lstrip("/")).read_text())
    assert toolchain["parent_receipt_sha256"] == plan["receipts"][derived_layer.BASE_RECEIPT]
    assert (context / "Dockerfile").read_text() == derived_layer.dockerfile()
    assert plan["tuning_defaults_sha256"] == transport.tuning_digest(transport.load_tuning())


def test_the_v3_lock_binds_the_built_image_and_the_layer(tmp_path):
    lock, _, wheel, result = prepared(tmp_path)
    plan = json.loads((Path(result["context"]) / "plan.json").read_text())
    image = {"Id": "sha256:" + "7" * 64, "Size": 33_000_000_000}
    probe = {"vllm": {"matches": ["lil-image-aba309e4610c"]}}
    value = sircl_layer.v3_lock(plan, image, "dev-20261009-kraken-sircl-cuda1342-nccl2323-status034", probe)
    for profile in image_lock.profiles_of(value):
        image_lock.validate(value, profile)
    assert value["transports"] == ["prepared", "sircl"] and value["line"] == "kraken" and not value["archived"]
    assert value["sircl"]["wheel"]["sha256"] == wheel["sha256"] and value["sircl"]["vllm_pins"] == probe["vllm"]["matches"]
    assert value["parent_receipt_sha256"] == plan["receipts"][derived_layer.BASE_RECEIPT]
    assert value["download_bytes"] == lock["download_bytes"] + plan["payload_bytes"]
    assert value["transport_profile"] == lock["transport_profile"] and value["image_reference"] == image["Id"]


def test_a_parent_that_already_holds_a_sircl_file_or_lacks_vllm_is_refused(tmp_path):
    lock, read, base = parent(tmp_path)
    wheel = sircl_layer.write_wheel(tmp_path / "wheel")
    directory = natives(tmp_path, lock, wheel)
    held = dict(base, files={**base["files"], SITE + "spark_roce_gid.py": "b" * 64})
    held_raw = derived_layer.canonical_json(held)
    toolchain_raw = derived_layer.canonical_json({"variant": "combined",
                                                  "parent_receipt_sha256": hashlib.sha256(held_raw).hexdigest()},
                                                 sort_keys=False)
    held_lock = dict(lock, parent_receipt_sha256=hashlib.sha256(held_raw).hexdigest(),
                     toolchain_receipt_sha256=hashlib.sha256(toolchain_raw).hexdigest())
    files = {derived_layer.BASE_RECEIPT: held_raw, derived_layer.TOOLCHAIN_RECEIPT: toolchain_raw}
    with pytest.raises(ValueError, match="already records .*spark_roce_gid.py"):
        sircl_layer.prepare(held_lock, files.__getitem__, wheel["wheel"], directory, tmp_path / "context")
    with pytest.raises(ValueError, match="in 0 package directories"):
        sircl_layer.site_packages({"files": {}})


def test_the_site_packages_directory_is_not_a_vllm_subpackage_of_another_package():
    # Kraken-line receipts also record B12X's vLLM integration subpackage.
    files = {SITE + "vllm/__init__.py": "a" * 64, SITE + "b12x/integration/vllm/__init__.py": "b" * 64}
    assert sircl_layer.site_packages({"files": files}) == SITE
    site = "/usr/lib/python3/site-packages/"
    assert sircl_layer.site_packages({"files": {site + "vllm/__init__.py": "a" * 64}}) == site
    with pytest.raises(ValueError, match="in 0 package directories"):
        sircl_layer.site_packages({"files": {SITE + "b12x/integration/vllm/__init__.py": "b" * 64}})


def test_the_v3_lock_may_list_profiles_that_run_only_on_sircl_ring_sessions(tmp_path):
    _, _, _, result = prepared(tmp_path)
    plan = json.loads((Path(result["context"]) / "plan.json").read_text())
    image = {"Id": "sha256:" + "7" * 64, "Size": 33_000_000_000}
    probe = {"vllm": {"matches": ["sparkring-kraken-beta-20261007-bc9ea774"]}}
    research = [name for name in installer_image.SIRCL_ONLY if name not in installer_image.QWEN4_EXP]
    assert research
    listed = [*plan["parent_lock"]["profiles"], *research]
    value = sircl_layer.v3_lock(plan, image, "dev-20261009-kraken-sircl-cuda1342-nccl2323-status034", probe,
                                profiles=reversed(listed))
    assert value["profiles"] == sorted(listed)
    for profile in research:
        assert image_lock.validate(value, profile) is value
    # A v2 lock cannot list them.
    with pytest.raises(ValueError, match="run only on SIRCL ring sessions"):
        image_lock.validate(dict(plan["parent_lock"], profiles=sorted(listed)), research[0])


def test_record_admits_and_writes_the_profiles_it_is_given(tmp_path, monkeypatch):
    _, _, _, result = prepared(tmp_path)
    context = Path(result["context"])
    layer = json.loads((context / "plan.json").read_text())["layer"]
    probe_record = {"package": SITE + "sparkring_sircl/__init__.py",
                    "entry_points": {"vllm.general_plugins": [["sircl", "sparkring_sircl.vllm.plugin:register", "x"]],
                                     "vllm.platform_plugins": [["sircl", "sparkring_sircl.vllm.platform:activate",
                                                                "x"]]},
                    "vllm": {"root": SITE + "vllm", "matches": ["sparkring-kraken-beta-20261007-bc9ea774"]},
                    "library": {"path": layer["native"]["path"], "exists": True},
                    "p2p_library": {"path": layer["p2p"]["path"], "exists": True}}
    image_id = "sha256:" + "8" * 64

    def run(argv, text=True):
        if argv[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(argv, 0, json.dumps([{"Id": image_id, "Size": 1}]), "")
        if sircl_layer.PROBE in argv:
            return subprocess.CompletedProcess(argv, 0, "SIRCL-SERVE-PROBE " + json.dumps(probe_record) + "\n", "")
        return subprocess.CompletedProcess(argv, 0, "", "")

    admitted = []
    monkeypatch.setattr(installer_image, "admit", lambda view, *, run, profile: admitted.append((view["schema"],
                                                                                                profile)))
    research = [name for name in installer_image.SIRCL_ONLY if name not in installer_image.QWEN4_EXP]
    plan = json.loads((context / "plan.json").read_text())
    listed = sorted([*plan["parent_lock"]["profiles"], *research])
    output = tmp_path / "lock.json"
    summary = sircl_layer.record(context, image_id, "dev-20261009-kraken-sircl-cuda1342-nccl2323-status034", output,
                                 run=run, profiles=listed)
    assert summary["vllm_pins"] == ["sparkring-kraken-beta-20261007-bc9ea774"]
    assert json.loads(output.read_text())["profiles"] == listed
    # Admission sees each listed profile through the lock's v2 view.
    assert admitted == [(installer_image.SCHEMA, profile) for profile in listed]


def test_natives_built_from_another_wheel_are_refused(tmp_path):
    lock, read, _ = parent(tmp_path)
    wheel = sircl_layer.write_wheel(tmp_path / "wheel")
    directory = natives(tmp_path, lock, wheel)
    record = json.loads((directory / "natives.json").read_text())
    (directory / "natives.json").write_text(json.dumps(dict(record, wheel_sha256="0" * 64)))
    with pytest.raises(ValueError, match="another parent image or wheel"):
        sircl_layer.prepare(lock, read, wheel["wheel"], directory, tmp_path / "context")


def test_the_libraries_build_in_the_parent_image_with_sircls_own_build_code(tmp_path):
    lock, _, _ = parent(tmp_path)
    wheel = sircl_layer.write_wheel(tmp_path / "wheel")
    from spark_transport.sircl.sparkring_sircl import build
    from spark_transport.sircl.sparkring_sircl.p2p import build as p2p
    calls = []

    def run(argv, text=True):
        calls.append(argv)
        if argv[-1] == "--version":
            return subprocess.CompletedProcess(argv, 0, "gcc (Ubuntu 13.3.0) 13.3.0\nCopyright\n", "")
        output = Path(next(item.split("src=")[1].split(",")[0] for item in argv if "dst=/sircl-out" in item))
        record = {}
        for key, module in (("library", build), ("p2p_library", p2p)):
            name = module.library_path(directory=Path("/")).name
            (output / name).write_bytes(key.encode())
            record[key] = {"path": f"/sircl-out/{name}", "exists": True}
        return subprocess.CompletedProcess(argv, 0, "SIRCL-SERVE-PROBE " + json.dumps(record) + "\n", "")
    result = sircl_layer.build_natives(lock, wheel["wheel"], tmp_path / "natives", run=run)
    assert result["compiler"] == "gcc (Ubuntu 13.3.0) 13.3.0"
    assert result["native"]["source_digest"] == build.source_digest() and result["p2p"]["source_digest"] == p2p.source_digest()
    build_call = calls[0]
    assert build_call[:4] == ["docker", "run", "--rm", "--pull"] and "--network" in build_call
    assert build_call[build_call.index("--network") + 1] == "none" and build_call[-3:] == ["-m", sircl_layer.PROBE,
                                                                                            "--build"]
    assert lock["image_id"] in build_call
