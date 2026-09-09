"""CPU checks for exact native inputs, source identity, and install boundaries."""
import copy
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from archive_utils import make_archive, read_archive, sha
import native_files
import prepare_context
import install_sources
import verify_sources
from receipt_contract import file_map_hash, validate_receipt
from test_source_image import receipt_fixture


def elf(label):
    return b"\x7fELF\x02\x01" + b"\0" * 10 + (3).to_bytes(2, "little") + (183).to_bytes(2, "little") + b"\0" * 44 + label


class Fixture:
    def __init__(self):
        self.packages = {
            "vllm": {"vllm/__init__.py": (b"version = 1\n", 0o644)},
            "b12x": {"b12x/__init__.py": (b"version = 1\n", 0o644)},
            "sparkcache": {"sparkcache/__init__.py": (b"version = 1\n", 0o644),
                           "sparkcache/native/include/api.h": (b"void snapshot();\n", 0o644),
                           "sparkcache/native/tool.py": (b"print('native')\n", 0o755)},
        }
        self.nccl = {"Makefile": (b"all:\n", 0o644)}
        self.archives = {name: make_archive(files, 0) for name, files in self.packages.items()}
        self.archives["nccl"] = make_archive(self.nccl, 0)
        records = {name: {"revision": "1" * 40, "files": {p: sha(v[0]) for p, v in files.items()},
                          "archive_sha256": sha(self.archives[name]), "distribution_version": "1",
                          "entry_points": {}, "console_scripts": {}}
                   for name, files in self.packages.items()}
        nccl = {"result_tree": "2" * 40, "base_revision": "3" * 40, "patch_sha256": "4" * 64}
        self.lock = {"nccl_build": nccl, "runtime": {}, "sources": {}, "profiles": {"test": {}}}
        self.manifest = {"native_mode": "pinned", "sources": records,
                         "nccl_build": {"tree": nccl["result_tree"], "archive_sha256": sha(self.archives["nccl"]),
                                        "files": {p: sha(v[0]) for p, v in self.nccl.items()}}}
        self.files = {name: (elf(name.encode()) if name in native_files.LIBRARIES else b"license\n", 0o644)
                      for name in native_files.MEMBERS if name != "provenance.json"}
        for name, (component, destination) in native_files.LIBRARIES.items():
            self.lock["runtime"].update({component + "_path": destination, component + "_sha256": sha(self.files[name][0])})
        prefix = "sparkcache/native/"
        native = {p: v for p, v in self.packages["sparkcache"].items() if p.startswith(prefix)}
        tree = native_files.git_tree({p[len(prefix):]: v for p, v in native.items()})
        mapping = file_map_hash({p: sha(v[0]) for p, v in native.items()})
        self.provenance = {"schema": "sparkring-pinned-native-inputs/v1",
                           "archive_headers": {"compression": "none", "format": "ustar", "uid": 0,
                                               "gid": 0, "mode": "0644", "mtime": 0, "ordering": "member path, ascending"},
                           "sources": {"nccl": {"patched_tree": nccl["result_tree"],
                                                "base_revision": nccl["base_revision"], "patch_sha256": nccl["patch_sha256"]},
                                       "sparkcache": {"qualified_native_tree": tree, "selected_native_tree": tree,
                                                      "native_source_directory": "sparkcache/native",
                                                      "native_file_count": len(native), "native_file_map_sha256": mapping}}}
        self.lock["native_files"] = {"schema": "sparkring-pinned-native-archive/v1", "nccl_tree": nccl["result_tree"],
                                     "nccl_archive_sha256": sha(self.archives["nccl"]),
                                     "sparkcache_native_tree": tree, "sparkcache_native_file_map_sha256": mapping}
        self.repack()

    def repack(self):
        self.provenance["files"] = {p: {"sha256": sha(v[0]), "bytes": len(v[0])}
                                    for p, v in self.files.items() if p != "provenance.json"}
        self.files["provenance.json"] = (json.dumps(self.provenance).encode(), 0o644)
        self.data = make_archive(self.files, 0)
        self.lock["native_files"].update(sha256=sha(self.data), bytes=len(self.data),
                                        members={p: {"sha256": sha(v[0]), "bytes": len(v[0])} for p, v in self.files.items()})

    def describe(self):
        return native_files.describe(self.data, self.lock, self.manifest, self.archives)

    def stage_inputs(self, root):
        root.mkdir(parents=True)
        lock = json.dumps(self.lock).encode()
        (root / "source-lock.json").write_bytes(lock)
        (root / native_files.ARCHIVE).write_bytes(self.data)
        for name, data in self.archives.items():
            (root / (name + ".tar")).write_bytes(data)
        self.manifest.update(source_lock_sha256=sha(lock), native_files=self.describe())


class NativeFilesTests(unittest.TestCase):
    def test_source_export_matches_git_native_tree_with_windows_eol_settings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            def git(*args):
                return subprocess.check_output(["git", "-C", str(root), *args])
            git("init", "--quiet")
            git("config", "core.autocrlf", "false")
            git("config", "core.eol", "crlf")
            (root / ".gitattributes").write_bytes(b"*.txt text=auto\n")
            (root / "sparkcache/native").mkdir(parents=True)
            (root / "sparkcache/native/CMakeLists.txt").write_bytes(b"one\ntwo\n")
            (root / "pyproject.toml").write_bytes(
                b'[project]\nname="sparkcache"\nversion="1"\n[build-system]\nbuild-backend="setuptools.build_meta"\n')
            git("add", ".")
            git("-c", "user.name=Source test", "-c", "user.email=test@example.invalid", "commit", "--quiet", "-m", "Define native source")
            head = git("rev-parse", "HEAD").decode().strip()
            data, _ = prepare_context.source_archive("sparkcache", {
                "checkout": str(root), "revision": head, "base_revision": head,
                "result_tree": git("write-tree").decode().strip()}, 0)
            files = read_archive(data)
            self.assertEqual(files["sparkcache/native/CMakeLists.txt"][0], b"one\ntwo\n")
            tree = native_files.git_tree({"CMakeLists.txt": files["sparkcache/native/CMakeLists.txt"]})
            self.assertEqual(tree, git("rev-parse", "HEAD:sparkcache/native").decode().strip())

    def test_complete_prepare_install_and_verify_pinned_mode(self):
        fixture = Fixture()
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            rootfs = temporary / "rootfs"
            base = {
                "/opt/spark-sircl/libspark_transport_capi.so": b"transport",
                "/opt/spark-sircl/sparkring-overlay-manifest.json": b"bundle",
                "/opt/sparkcache-src/sparkcache/native/build-cuda/libspark_cache_placement.so": b"placement",
                "/opt/sparkring/nccl/libnccl.so.2.30.7": b"parent nccl",
                "/opt/sparkring/src/mtp3-mesh/mlx5_rdma_tx_rewrite_probe.c": b"marker source",
                "/opt/sparkring/bin/mlx5-rdma-tx-marker": b"marker binary",
                "/opt/sparkring/bin/serve-with-warmup.py": b"serve",
                "/opt/sparkring/bin/warmup_dflash.py": b"warmup",
                prepare_context.SITE + "/vllm/_C.so": b"inherited extension",
            }
            for name, data in base.items():
                path = rootfs / name[1:]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            base_files = {p: sha(v) for p, v in base.items()}
            critical = {p: base_files[p] for p in prepare_context.CRITICAL}
            inherited_nccl = {p: base_files[p] for p in prepare_context.NCCL}
            retained = prepare_context.retained_files(base_files)
            runtime = fixture.lock["runtime"]
            for field, path in {
                "bundle_manifest_sha256": "/opt/spark-sircl/sparkring-overlay-manifest.json",
                "transport_sha256": "/opt/spark-sircl/libspark_transport_capi.so",
                "marker_source_sha256": "/opt/sparkring/src/mtp3-mesh/mlx5_rdma_tx_rewrite_probe.c",
                "marker_binary_sha256": "/opt/sparkring/bin/mlx5-rdma-tx-marker",
                "placement_sha256": "/opt/sparkcache-src/sparkcache/native/build-cuda/libspark_cache_placement.so",
            }.items():
                runtime[field] = base_files[path]
                if field == "placement_sha256":
                    runtime["placement_path"] = path
            runtime.update(expected_distributions={"torch": "2.13"},
                           readiness_warmup={"helper_sha256": sha(b"warmup")},
                           retained_vllm_native_sha256=file_map_hash(retained))
            for name, record in fixture.manifest["sources"].items():
                expected = {p[len(name) + 1:]: h for p, h in record["files"].items() if p.startswith(name + "/")}
                expected.update({p[len(name) + 1:]: h for p, h in retained.items() if p.startswith(name + "/")})
                fixture.lock["sources"][name] = {"revision": record["revision"],
                                                "installed_file_map_sha256": file_map_hash(expected),
                                                "installed_file_count": len(expected)}
            lock_path = temporary / "lock.json"
            lock_path.write_text(json.dumps(fixture.lock))
            base_path = temporary / "base.json"
            base_path.write_text(json.dumps({"files": base_files}))
            nccl_path, native_path = temporary / "nccl.tar", temporary / "native.tar"
            nccl_path.write_bytes(fixture.archives["nccl"])
            native_path.write_bytes(fixture.data)
            spec = {"base_image": prepare_context.PUBLIC_BASE, "base_receipt": str(base_path),
                    "base_receipt_sha256": sha(base_path.read_bytes()), "source_date_epoch": 0,
                    "source_lock": str(lock_path), "sources": fixture.lock["sources"],
                    "runtime": {"description": "CPU fixture", "expected_distributions": {"torch": "2.13"}},
                    "nccl_archive": str(nccl_path), "nccl_build": fixture.manifest["nccl_build"],
                    "native_files_archive": str(native_path),
                    "native_snapshot": {"cuda_architectures": "121", "compiler": "/opt/cuda-13.3/bin/nvcc", "build": "direct-cxx-cuda/v1"}}
            def source_archive(name, *_):
                return fixture.archives[name], copy.deepcopy(fixture.manifest["sources"][name])
            with patch.dict(prepare_context.CRITICAL, critical, clear=True), \
                    patch.dict(prepare_context.NCCL, inherited_nccl, clear=True), \
                    patch("prepare_context.source_archive", side_effect=source_archive):
                prepare_context.prepare(spec, temporary / "context")
                compile_spec = dict(spec)
                del compile_spec["native_files_archive"]
                prepare_context.prepare(compile_spec, temporary / "compile-context")
            docker = (temporary / "context/Dockerfile").read_text()
            self.assertNotIn("/build_nccl.py &&", docker)
            self.assertNotIn("/build_snapshot.py &&", docker)
            compile_docker = (temporary / "compile-context/Dockerfile").read_text()
            self.assertIn("/build_nccl.py &&", compile_docker)
            self.assertIn("/build_snapshot.py &&", compile_docker)
            payload = read_archive((temporary / "context/payload.tar").read_bytes())
            for name, (data, _) in payload.items():
                path = rootfs / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            root = rootfs / prepare_context.ROOT[1:]
            distributions = {name: {"version": "2.13" if name == "torch" else "1", "entry_points": {}}
                             for name in ("torch", "vllm", "b12x", "sparkcache")}
            with patch("install_sources.distribution_records", return_value=distributions), \
                    patch("verify_sources.distribution_records", return_value=distributions), \
                    patch.object(verify_sources, "sys", SimpleNamespace(modules={})):
                install_sources.stage(root, rootfs)
                install_sources.finalize(root, rootfs)
                result = verify_sources.verify(root, rootfs)
                self.assertEqual(result["native_mode"], "pinned")
                self.assertEqual(result["native_files"], native_files.expected_record(fixture.lock))
                self.assertFalse(result["gpu_qualified"])
                self.assertTrue((root / "nccl.tar").exists())
                self.assertTrue((root / "sparkcache.tar").exists())
                self.assertFalse((root / "snapshot-build-receipt.json").exists())
                (root / native_files.ARCHIVE).write_bytes(b"tampered")
                with self.assertRaisesRegex(ValueError, "archive hash/size"):
                    verify_sources.verify(root, rootfs)

    def test_archive_source_and_install_roundtrip(self):
        fixture = Fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root, rootfs = Path(temporary) / "inputs", Path(temporary) / "rootfs"
            fixture.stage_inputs(root)
            record = native_files.install(root, fixture.manifest, rootfs)
            state = {"native_mode": "pinned", "native_files": record}
            self.assertEqual(native_files.verify(root, fixture.manifest, state, rootfs), record)
            self.assertEqual(len(record["installed_files"]), 6)
            for path, digest in record["installed_files"].items():
                self.assertEqual(sha((rootfs / path[1:]).read_bytes()), digest)

    def test_wrong_archive_or_member_identity_rejects(self):
        fixture = Fixture()
        with self.assertRaisesRegex(ValueError, "archive hash/size"):
            native_files.describe(fixture.data + b"changed", fixture.lock, fixture.manifest, fixture.archives)
        fixture.lock["native_files"]["members"]["provenance.json"]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            fixture.describe()

    def test_extra_duplicate_directory_and_link_members_reject_even_when_archive_pin_changes(self):
        for kind in ("extra", "duplicate", "directory", "link"):
            fixture = Fixture()
            output = io.BytesIO()
            with tarfile.open(fileobj=output, mode="w") as archive:
                for name, (data, _) in fixture.files.items():
                    entry = tarfile.TarInfo(name)
                    entry.size = len(data)
                    archive.addfile(entry, io.BytesIO(data))
                entry = tarfile.TarInfo("provenance.json" if kind == "duplicate" else "extra")
                if kind == "directory":
                    entry.type = tarfile.DIRTYPE
                elif kind == "link":
                    entry.type, entry.linkname = tarfile.SYMTYPE, "/tmp/target"
                archive.addfile(entry)
            fixture.data = output.getvalue()
            fixture.lock["native_files"].update(sha256=sha(fixture.data), bytes=len(fixture.data))
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "six regular"):
                fixture.describe()

    def test_wrong_runtime_path_or_hash_and_non_elf_reject(self):
        for field, value in (("nccl_path", "/tmp/library.so"), ("snapshot_sha256", "0" * 64)):
            fixture = Fixture()
            fixture.lock["runtime"][field] = value
            with self.assertRaisesRegex(ValueError, "runtime path/hash"):
                fixture.describe()
        fixture = Fixture()
        name = "native/libnccl.so.2.30.7"
        fixture.files[name] = (b"not an ELF" * 8, 0o644)
        fixture.repack()
        fixture.lock["runtime"]["nccl_sha256"] = sha(fixture.files[name][0])
        with self.assertRaisesRegex(ValueError, "AArch64 shared ELF"):
            fixture.describe()

    def test_native_source_content_and_modes_are_bound_but_python_changes_are_allowed(self):
        for change in ("bytes", "mode", "python"):
            fixture = Fixture()
            files = fixture.packages["sparkcache"]
            path = "sparkcache/__init__.py" if change == "python" else "sparkcache/native/include/api.h"
            data, mode = files[path]
            files[path] = (data + b"changed" if change != "mode" else data, 0o755 if change == "mode" else mode)
            fixture.archives["sparkcache"] = make_archive(files, 0)
            fixture.manifest["sources"]["sparkcache"].update(archive_sha256=sha(fixture.archives["sparkcache"]),
                                                           files={p: sha(v[0]) for p, v in files.items()})
            if change == "python":
                fixture.describe()
            else:
                with self.subTest(change=change), self.assertRaisesRegex(ValueError, "native source bytes/modes"):
                    fixture.describe()

    def test_native_provenance_and_retained_nccl_archive_cannot_forge_source_identity(self):
        fixture = Fixture()
        fixture.provenance["sources"]["sparkcache"]["qualified_native_tree"] = "0" * 40
        fixture.repack()
        with self.assertRaisesRegex(ValueError, "native source bytes/modes"):
            fixture.describe()
        fixture = Fixture()
        fixture.archives["nccl"] = make_archive({"Makefile": (b"changed", 0o644)}, 0)
        fixture.manifest["nccl_build"].update(archive_sha256=sha(fixture.archives["nccl"]), files={"Makefile": sha(b"changed")})
        with self.assertRaisesRegex(ValueError, "pinned source identity"):
            fixture.describe()

    def test_manifest_state_modes_and_installed_files_cannot_drift(self):
        for change in ("mode", "record", "binary", "extra"):
            fixture = Fixture()
            with tempfile.TemporaryDirectory() as temporary:
                root, rootfs = Path(temporary) / "inputs", Path(temporary) / "rootfs"
                fixture.stage_inputs(root)
                record = native_files.install(root, fixture.manifest, rootfs)
                state = {"native_mode": "pinned", "native_files": record}
                if change == "mode":
                    state["native_mode"] = "compile"
                elif change == "record":
                    fixture.manifest["native_files"]["installed_files"]["/tmp/escaped"] = "0" * 64
                elif change == "binary":
                    (rootfs / fixture.lock["runtime"]["nccl_path"][1:]).write_bytes(b"changed")
                else:
                    (rootfs / native_files.METADATA_ROOT[1:] / "extra").write_bytes(b"extra")
                with self.subTest(change=change), self.assertRaises(ValueError):
                    native_files.verify(root, fixture.manifest, state, rootfs)
        with self.assertRaises(ValueError):
            native_files.mode({"native_mode": "compile", "native_files": {}})
        with self.assertRaises(ValueError):
            native_files.mode({"native_mode": "pinned"})

    def test_all_install_targets_checked_before_any_write(self):
        fixture = Fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root, rootfs = Path(temporary) / "inputs", Path(temporary) / "rootfs"
            fixture.stage_inputs(root)
            conflict = rootfs / native_files.METADATA_ROOT[1:] / "provenance.json"
            conflict.parent.mkdir(parents=True)
            conflict.write_bytes(b"occupied")
            with self.assertRaisesRegex(ValueError, "already exists"):
                native_files.install(root, fixture.manifest, rootfs)
            self.assertFalse((rootfs / fixture.lock["runtime"]["nccl_path"][1:]).exists())
            self.assertEqual(conflict.read_bytes(), b"occupied")

    def test_receipt_mode_and_archive_identity_are_required(self):
        fixture = Fixture()
        lock, document = receipt_fixture()
        lock["native_files"] = fixture.lock["native_files"]
        lock["nccl_build"] = fixture.lock["nccl_build"]
        lock["runtime"].update(fixture.lock["runtime"])
        document["inside_image"]["nccl_sha256"] = lock["runtime"]["nccl_sha256"]
        document["native_mode"] = document["inside_image"]["native_mode"] = "pinned"
        document["inside_image"]["native_files"] = fixture.describe()
        validate_receipt(document, lock)
        document["native_mode"] = "compile"
        with self.assertRaisesRegex(ValueError, "modes disagree"):
            validate_receipt(document, lock)


if __name__ == "__main__":
    unittest.main()
