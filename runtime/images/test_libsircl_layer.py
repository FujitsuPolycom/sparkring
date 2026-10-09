"""The libsircl image layer: its build, build context, probe and v3 lock; offline.

A fixture snapshot, synced like the vendored one, stands in for libsircl's
tree; the parent image is represented by its two receipts and the built
library by stand-in bytes.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import pytest

from runtime.common import image_lock, installer_image, libsircl
from runtime.common.test_image_lock import sircl_lock
from runtime.images import derived_layer, libsircl_layer
from scripts import sync_libsircl
from scripts.test_sync_libsircl import snapshot

SITE = "/usr/local/lib/python3.12/dist-packages/"
MAKEFILE = (b"BUILD ?= build\nNVCCFLAGS = -O3 -std=c++17 -gencode arch=compute_120,code=sm_120 "
            b"-gencode arch=compute_121,code=sm_121\n")


def fixture_tree(tmp_path):
    """A vendored tree of one fixture snapshot, as scripts/sync_libsircl.py writes it."""
    files = {"VERSION": b"0.6.0\n", "Makefile": MAKEFILE, "README.md": b"# libsircl\n",
             "verification/hardware/run.log": b"evidence\n"}
    for name in libsircl.NOTICES:
        files[name] = f"notice {name}\n".encode()
    for pack in libsircl_layer.PACKS:
        data = pack.encode() * 64
        files[f"kernels/prebuilt/{pack}.fatbin"] = data
        files[f"kernels/prebuilt/{pack}.fatbin.sha256"] = (hashlib.sha256(data).hexdigest() + "\n").encode()
    directory, digest = snapshot(tmp_path / "snapshot", files)
    tree = tmp_path / "tree"
    sync_libsircl.sync(directory, digest, tree)
    return tree, digest


def parent(**changes):
    """The parent v3 lock and receipts of the SIRCL image, with vLLM in site-packages."""
    base = {"schema": "sparkring-external-installed/v1", "files": {SITE + "vllm/__init__.py": "a" * 64},
            "capabilities": {"runtime_status": {"version": "0.3.4"}, "sircl": {"version": "0.2.0"}}}
    base.update(changes)
    base_raw = derived_layer.canonical_json(base)
    toolchain_raw = derived_layer.canonical_json({"variant": "combined",
                                                  "parent_receipt_sha256": hashlib.sha256(base_raw).hexdigest()},
                                                 sort_keys=False)
    lock = sircl_lock(image_id="sha256:" + "5" * 64, image_reference="sha256:" + "5" * 64,
                      parent_receipt_sha256=hashlib.sha256(base_raw).hexdigest(),
                      toolchain_receipt_sha256=hashlib.sha256(toolchain_raw).hexdigest())
    files = {derived_layer.BASE_RECEIPT: base_raw, derived_layer.TOOLCHAIN_RECEIPT: toolchain_raw}
    return lock, files.__getitem__, base


def natives(tmp_path, lock, tree, *, library=b"\x7fELF libsircl 0.6.0"):
    """A natives directory as ``build_natives`` writes it, with stand-in library bytes."""
    directory = tmp_path / "natives"

    def run(argv, text=True):
        output = Path(next(item.split("src=")[1].split(",")[0] for item in argv if f"dst={libsircl_layer.OUTPUT_MOUNT}"
                           in item))
        (output / "libsircl.so.0.6.0").write_bytes(library)
        (output / "check.log").write_text("9 suites OK\n")
        (output / "build.log").write_text("cc ...\n")
        (output / "compiler.txt").write_text("gcc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0\n")
        run.calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "", "")
    run.calls = []
    result = libsircl_layer.build_natives(lock["image_id"], directory, tree=tree, run=run)
    return directory, result, run.calls


def prepared(tmp_path):
    tree, digest = fixture_tree(tmp_path)
    lock, read, base = parent()
    directory, _, _ = natives(tmp_path, lock, tree)
    result = libsircl_layer.prepare(lock, read, directory, tmp_path / "context", tree=tree)
    return tree, digest, lock, base, result


def test_the_library_builds_in_a_network_less_container_of_the_parent_with_libsircls_own_makefile(tmp_path):
    tree, digest = fixture_tree(tmp_path)
    lock, _, _ = parent()
    directory, result, calls = natives(tmp_path, lock, tree)
    (command,) = calls
    assert command[:4] == ["docker", "run", "--rm", "--pull"] and command[command.index("--network") + 1] == "none"
    assert lock["image_id"] in command and any(item.endswith(f"dst={libsircl_layer.SOURCE_MOUNT},readonly")
                                               for item in command)
    script = command[-1]
    assert f"cd {libsircl_layer.BUILD_PATH}" in script and "unset LD_PRELOAD" in script
    assert "make -j BUILD=build" in script and "make check BUILD=build" in script
    assert result["snapshot"] == digest and result["version"] == "0.6.0"
    assert result["library"] == {"name": "libsircl.so.0.6.0",
                                 "sha256": hashlib.sha256(b"\x7fELF libsircl 0.6.0").hexdigest()}
    assert result["architectures"] == ["sm_120", "sm_121"] and set(result["kernel_packs"]) == set(libsircl_layer.PACKS)
    assert json.loads((directory / "natives.json").read_text()) == result


def test_a_tree_that_differs_from_its_snapshot_is_not_built(tmp_path):
    tree, _ = fixture_tree(tmp_path)
    (tree / "Makefile").write_bytes(MAKEFILE + b"# edited\n")
    with pytest.raises(sync_libsircl.SnapshotError, match="differ from the snapshot"):
        libsircl_layer.build_natives("sha256:" + "5" * 64, tmp_path / "natives", tree=tree,
                                     run=lambda *a, **k: None)


def test_the_context_installs_the_library_notices_plugin_and_receipt_and_records_them(tmp_path):
    tree, digest, lock, base, result = prepared(tmp_path)
    context = Path(result["context"])
    plan = json.loads((context / "plan.json").read_text())
    derived = json.loads((context / "files" / derived_layer.BASE_RECEIPT.lstrip("/")).read_text())
    layer = json.loads((context / "files" / image_lock.LIBSIRCL_RECEIPT.lstrip("/")).read_text())
    library = "/opt/sparkring/libsircl/lib/libsircl.so.0.6.0"
    assert layer["library"] == {"path": library, "sha256": hashlib.sha256(b"\x7fELF libsircl 0.6.0").hexdigest(),
                                "soname": "libnccl.so.2"}
    assert layer["snapshot"] == digest and layer["nccl_api_version"] == 22705 and layer["site_packages"] == SITE
    for name in libsircl.NOTICES:
        assert (context / "files" / f"opt/sparkring/libsircl/{name}").read_bytes() == (tree / name).read_bytes()
    assert layer["plugin"]["path"] == SITE + "sparkring_libsircl.py"
    assert SITE + "sparkring_libsircl-0.6.0.dist-info/entry_points.txt" in layer["files"]
    # Every added file, the layer receipt included, is in the receipt the image's verify checks.
    assert set(plan["added"]) == set(layer["files"]) | {image_lock.LIBSIRCL_RECEIPT}
    for path in plan["added"]:
        assert derived["files"][path] == hashlib.sha256((context / "files" / path.lstrip("/")).read_bytes()).hexdigest()
    assert derived["files"][SITE + "vllm/__init__.py"] == base["files"][SITE + "vllm/__init__.py"]
    assert derived["capabilities"]["libsircl"] == {"version": "0.6.0", "snapshot": digest,
                                                   "receipt": image_lock.LIBSIRCL_RECEIPT,
                                                   "receipt_sha256": plan["layer_sha256"]}
    # The parent's SIRCL capability stays.
    assert derived["capabilities"]["sircl"] == base["capabilities"]["sircl"]
    toolchain = json.loads((context / "files" / derived_layer.TOOLCHAIN_RECEIPT.lstrip("/")).read_text())
    assert toolchain["parent_receipt_sha256"] == plan["receipts"][derived_layer.BASE_RECEIPT]
    assert (context / "Dockerfile").read_text() == derived_layer.dockerfile()


def test_the_installed_plugin_registers_its_entry_point_and_selects_the_library(tmp_path, monkeypatch):
    files = libsircl_layer.plugin_files("/site/", "0.6.0")
    root = tmp_path / "site"
    for path, data in files.items():
        target = root / path.removeprefix("/site/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    points = [(point.group, point.name, point.value) for distribution in importlib.metadata.distributions(path=[str(root)])
              for point in distribution.entry_points]
    assert points == [libsircl.PLUGIN_ENTRY_POINT]
    library = tmp_path / "libsircl.so.0.6.0"
    library.write_bytes(b"library")
    probe = ("import os, sys; sys.path.insert(0, sys.argv[1]); import sparkring_libsircl as p; p.register(); "
             "print(os.environ['VLLM_NCCL_SO_PATH'])")
    environment = {"SPARKRING_LIBSIRCL_LIBRARY": str(library),
                   "SPARKRING_LIBSIRCL_SHA256": hashlib.sha256(b"library").hexdigest(), "SYSTEMROOT": "C:\\Windows"}
    done = subprocess.run([sys.executable, "-I", "-c", probe, str(root)], capture_output=True, text=True, check=True,
                          env=environment, cwd=tmp_path)
    assert done.stdout.strip() == str(library)
    record = files["/site/sparkring_libsircl-0.6.0.dist-info/RECORD"].decode().splitlines()
    assert {row.split(",")[0] for row in record} == {path.removeprefix("/site/") for path in files}


def test_the_v3_lock_adds_the_libsircl_transport_and_layer_to_the_parents(tmp_path):
    _, digest, lock, _, result = prepared(tmp_path)
    plan = json.loads((Path(result["context"]) / "plan.json").read_text())
    image = {"Id": "sha256:" + "7" * 64, "Size": 33_000_000_000}
    value = libsircl_layer.v3_lock(plan, image, "dev-20261010-kraken-sircl-libsircl-cuda1342-nccl2323-status034")
    for profile in image_lock.profiles_of(value):
        assert image_lock.validate(value, profile) is value
    assert value["transports"] == ["libsircl", "prepared", "sircl"] and value["sircl"] == lock["sircl"]
    block = image_lock.libsircl(value)
    assert block["snapshot"] == digest and block["library"]["path"] == "/opt/sparkring/libsircl/lib/libsircl.so.0.6.0"
    assert block["receipt"] == {"path": image_lock.LIBSIRCL_RECEIPT, "sha256": plan["layer_sha256"]}
    assert value["download_bytes"] == lock["download_bytes"] + plan["payload_bytes"]
    assert value["parent_receipt_sha256"] == plan["receipts"][derived_layer.BASE_RECEIPT]
    assert value["image_reference"] == image["Id"] and not value["archived"]


@pytest.mark.parametrize("change, message", [
    ("v2", "derives from a v3 image lock"),
    ("libsircl", "without a libsircl layer"),
    ("recorded", "already records"),
    ("natives", "another libsircl snapshot"),
])
def test_a_parent_or_natives_that_do_not_fit_are_refused(tmp_path, change, message):
    tree, _ = fixture_tree(tmp_path)
    lock, read, base = parent()
    directory, _, _ = natives(tmp_path, lock, tree)
    if change == "v2":
        lock = installer_image.default_lock()
    elif change == "libsircl":
        from runtime.common.test_image_lock import libsircl_lock
        lock = libsircl_lock()
    elif change == "recorded":
        lock, read, _ = parent(files={SITE + "vllm/__init__.py": "a" * 64,
                                      "/opt/sparkring/libsircl/lib/libsircl.so.0.6.0": "b" * 64})
    else:
        record = json.loads((directory / "natives.json").read_text())
        (directory / "natives.json").write_text(json.dumps(dict(record, snapshot="0" * 64)))
    with pytest.raises(ValueError, match=message):
        libsircl_layer.prepare(lock, read, directory, tmp_path / "context", tree=tree)


class BuiltImage:
    """``docker`` as ``record`` calls it against a built image whose files are the context's."""

    def __init__(self, context, image_id, probe):
        self.context, self.image_id, self.probe, self.calls = Path(context), image_id, probe, []

    def __call__(self, argv, text=True):
        self.calls.append(argv)
        if argv[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(argv, 0, json.dumps([{"Id": self.image_id, "Size": 1}]), "")
        if libsircl_layer.PROBE in argv:
            return subprocess.CompletedProcess(argv, 0, libsircl_layer.PROBE_MARK + json.dumps(self.probe) + "\n", "")
        if "/bin/cat" in argv:
            return subprocess.CompletedProcess(argv, 0, (self.context / "files" / argv[-1].lstrip("/")).read_bytes(), "")
        return subprocess.CompletedProcess(argv, 0, "", "")


def probe_record(plan):
    layer = plan["layer"]
    return {"entry_points": [["libsircl", "sparkring_libsircl:register"]], "nccl_get_version": [0, 22705],
            "library": "libsircl", "version": layer["version"], "selected": layer["library"]["path"],
            "plugin_file": layer["plugin"]["path"]}


def test_record_probes_checks_the_layer_admits_and_writes_the_lock(tmp_path, monkeypatch):
    _, _, _, _, result = prepared(tmp_path)
    context = Path(result["context"])
    plan = json.loads((context / "plan.json").read_text())
    image_id = "sha256:" + "8" * 64
    run = BuiltImage(context, image_id, probe_record(plan))
    admitted = []
    monkeypatch.setattr(installer_image, "admit", lambda view, *, run, profile: admitted.append(profile))
    output = tmp_path / "lock.json"
    summary = libsircl_layer.record(context, image_id, "dev-20261010-kraken-sircl-libsircl-cuda1342-nccl2323-status034",
                                    output, run=run)
    lock = json.loads(output.read_text())
    assert summary["nccl_api_version"] == 22705 and summary["transports"] == ["libsircl", "prepared", "sircl"]
    assert lock["image_id"] == image_id and admitted == list(image_lock.profiles_of(lock))
    probe = next(argv for argv in run.calls if libsircl_layer.PROBE in argv)
    assert probe[probe.index("--network") + 1] == "none" and "--read-only" in probe
    assert probe[-2:] == [lock["libsircl"]["library"]["path"], lock["libsircl"]["library"]["sha256"]]


@pytest.mark.parametrize("field, value", [("selected", "/opt/sparkring/toolchain/nccl/lib/libnccl.so.2"),
                                          ("nccl_get_version", [0, 23203]), ("entry_points", [])])
def test_a_built_image_whose_probe_disagrees_is_not_recorded(tmp_path, monkeypatch, field, value):
    _, _, _, _, result = prepared(tmp_path)
    context = Path(result["context"])
    plan = json.loads((context / "plan.json").read_text())
    monkeypatch.setattr(installer_image, "admit", lambda *a, **k: None)
    run = BuiltImage(context, "sha256:" + "8" * 64, dict(probe_record(plan), **{field: value}))
    with pytest.raises(ValueError, match=field):
        libsircl_layer.record(context, "sha256:" + "8" * 64, "x", tmp_path / "lock.json", run=run)
    assert not (tmp_path / "lock.json").exists()


def test_a_layer_receipt_the_images_verification_does_not_cover_is_refused(tmp_path):
    _, _, _, _, result = prepared(tmp_path)
    context = Path(result["context"])
    plan = json.loads((context / "plan.json").read_text())
    block = libsircl_layer.libsircl_block(plan)
    run = BuiltImage(context, "sha256:" + "8" * 64, {})
    receipt = plan["receipts"][derived_layer.BASE_RECEIPT]
    assert libsircl.check_layer("sha256:" + "8" * 64, receipt, block, run=run)["files_verified"] == len(plan["added"]) - 1
    for edit, message in ((lambda b: b["library"].update(sha256="0" * 64), "does not cover its libsircl library"),
                          (lambda b: b["receipt"].update(sha256="0" * 64), "layer receipt differs")):
        changed = json.loads(json.dumps(block))
        edit(changed)
        with pytest.raises(libsircl.LibsirclError, match=message):
            libsircl.check_layer("sha256:" + "8" * 64, receipt, changed, run=run)
    with pytest.raises(libsircl.LibsirclError, match="external-base receipt differs"):
        libsircl.check_layer("sha256:" + "8" * 64, "0" * 64, block, run=run)


def test_the_host_library_names_its_content_addressed_host_path(tmp_path):
    tree, _ = fixture_tree(tmp_path)

    def run(argv, text=True):
        output = Path(next(item.split("src=")[1].split(",")[0] for item in argv
                           if f"dst={libsircl_layer.OUTPUT_MOUNT}" in item))
        for name, data in (("libsircl.so.0.6.0", b"lib"), ("check.log", b""), ("compiler.txt", b"gcc 13\n")):
            (output / name).write_bytes(data)
        return subprocess.CompletedProcess(argv, 0, "", "")
    result = libsircl_layer.host_library("sha256:" + "4" * 64, tmp_path / "host", tree=tree, run=run)
    assert result["builder_image_id"] == "sha256:" + "4" * 64
    assert result["install"]["path"] == f"/var/lib/sparkring/libsircl/{hashlib.sha256(b'lib').hexdigest()}/libsircl.so.0.6.0"


def test_the_vendored_tree_names_the_kernel_packs_and_architectures_the_layer_records():
    summary, files = libsircl_layer.tree_facts()
    assert summary["version"] == "0.6.0"
    assert libsircl_layer.architectures(files) == ["sm_120", "sm_121"]
    packs = libsircl_layer.kernel_packs(files)
    assert packs["sircl_kernels"] == "c2e6e5a1f3c2d6bf8af3bcdb62fece9684ab062980f2bfdedd233d82be4eaa25"
    assert packs["sircl_links"] == "dc9dd167b44c5c6ff32fa2cc07eddceb0a0f45aa72b287fe336d67a9900f6410"
