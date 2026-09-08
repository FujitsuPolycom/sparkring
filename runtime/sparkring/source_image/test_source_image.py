"""CPU-only contract tests for source preparation and image verification."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from archive_utils import make_archive, sha, transform_warmup
from build_snapshot import LIBRARY, verify_library
from prepare_image import materialize
from receipt_contract import file_map_hash, validate_receipt
from verify_image import IMAGE_ROOT, TOOLS, trusted_closure, verify, verify_embedded_closure

HERE = Path(__file__).resolve().parent


def receipt_fixture():
    lock = json.loads((HERE / "glm53-tp4-lock.json").read_bytes())
    inside = {k: lock["runtime"][k] for k in (
        "bundle_manifest_sha256", "transport_sha256", "marker_source_sha256",
        "marker_binary_sha256", "nccl_sha256", "retained_vllm_native_sha256",
        "readiness_warmup")}
    inside.update(checks_passed=True, cuda_initialized=False, model_loaded=False,
                  source_lock_sha256=sha((HERE / "glm53-tp4-lock.json").read_bytes()),
                  inherited_runtime=copy.deepcopy(lock["runtime"]["expected_distributions"]),
                  packages={k: {"revision": v["revision"],
                                "files": v["installed_file_count"],
                                "file_map_sha256": v["installed_file_map_sha256"]}
                            for k, v in lock["sources"].items()})
    image = "sha256:" + "a" * 64
    return lock, {"schema": "sparkring-source-image-receipt/v1",
                  "image_id": image, "image_reference": image, "platform": "linux/arm64",
                  "checks_passed": True, "profile": next(iter(lock["profiles"])),
                  "source_lock_sha256": inside["source_lock_sha256"], "inside_image": inside}


class SourceImageTests(unittest.TestCase):
    def test_cache_profile_requires_the_connectors_exact_native_libraries(self):
        lock, document = receipt_fixture()
        document["profile"] = "tp4-dcp1-mtp3-sparkcache"
        for name in ("snapshot", "placement"):
            document["inside_image"][name + "_sha256"] = lock["runtime"][name + "_sha256"]
        validate_receipt(document, lock)
        for name in ("snapshot", "placement"):
            altered = copy.deepcopy(document)
            altered["inside_image"][name + "_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "native library witness"):
                validate_receipt(altered, lock)

    def test_snapshot_build_rejects_unbound_output_before_installation(self):
        data = b"\x7fELF\x02\x01" + b"\0" * 12 + (183).to_bytes(2, "little") + b"snapshot"
        lock = {"runtime": {"snapshot_path": LIBRARY, "snapshot_sha256": sha(data)}}
        verify_library(data, lock)
        with self.assertRaisesRegex(RuntimeError, "locked library"):
            verify_library(data + b"changed", lock)

    def test_complete_receipt(self):
        lock, document = receipt_fixture()
        self.assertIs(validate_receipt(document, lock), document)

    def test_incomplete_or_changed_witness_rejected(self):
        changes = [
            ("checks_passed", False), ("model_loaded", True), ("cuda_initialized", True),
            ("source_lock_sha256", "b" * 64), ("nccl_sha256", "b" * 64),
            ("retained_vllm_native_sha256", "b" * 64), ("packages", None),
            ("inherited_runtime", None), ("readiness_warmup", None),
        ]
        for key, value in changes:
            with self.subTest(key=key):
                lock, document = receipt_fixture()
                document["inside_image"][key] = value
                with self.assertRaises(ValueError):
                    validate_receipt(document, lock)

    def test_each_package_requires_count_hash_and_revision(self):
        for name in ("vllm", "b12x", "sparkcache"):
            for field in ("files", "file_map_sha256", "revision"):
                with self.subTest(name=name, field=field):
                    lock, document = receipt_fixture()
                    del document["inside_image"]["packages"][name][field]
                    with self.assertRaises(ValueError):
                        validate_receipt(document, lock)

    def test_dependency_version_rejected(self):
        lock, document = receipt_fixture()
        document["inside_image"]["inherited_runtime"]["torch"] = "0.0"
        with self.assertRaises(ValueError):
            validate_receipt(document, lock)

    def test_local_id_is_not_publication(self):
        lock, document = receipt_fixture()
        document["image_reference"] = "registry.invalid/image@" + document["image_id"]
        with self.assertRaises(ValueError):
            validate_receipt(document, lock)

    def test_patch_hash_checked_before_destination_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "source.patch").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "patch differs"):
                materialize("vllm", {"patch": "source.patch", "patch_sha256": "0" * 64},
                            root / "cache", root)
            self.assertFalse((root / "cache").exists())

    def test_cached_checkout_rejects_unstaged_and_untracked_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = root / "cache/vllm"
            repository.mkdir(parents=True)
            def git(*args):
                return subprocess.check_output(["git", "-C", str(repository), *args])
            git("init", "--quiet")
            git("config", "core.autocrlf", "false")
            (repository / "source.py").write_bytes(b"value = 1\n")
            git("add", "source.py")
            git("-c", "user.name=Source test", "-c", "user.email=test@example.invalid",
                "commit", "--quiet", "-m", "Define source fixture")
            (root / "source.patch").write_bytes(b"")
            record = {"patch": "source.patch", "patch_sha256": sha(b""),
                      "base_revision": git("rev-parse", "HEAD").decode().strip(),
                      "result_tree": git("write-tree").decode().strip()}
            self.assertEqual(materialize("vllm", record, root / "cache", root, True)[0], repository)
            (repository / "extra.py").write_bytes(b"value = 2\n")
            with self.assertRaisesRegex(ValueError, "Cached source differs"):
                materialize("vllm", record, root / "cache", root, True)
            (repository / "extra.py").unlink()
            (repository / "source.py").write_bytes(b"value = 3\n")
            with self.assertRaisesRegex(ValueError, "Cached source differs"):
                materialize("vllm", record, root / "cache", root, True)

    def test_locked_startup_transform(self):
        lock, _ = receipt_fixture()
        raw = (HERE / "startup/warmup_dflash.py").read_bytes()
        installed = transform_warmup(raw, lock["startup"]["warmup_transform"])
        self.assertEqual(sha(installed), lock["runtime"]["readiness_warmup"]["helper_sha256"])
        with self.assertRaises(ValueError):
            transform_warmup(installed, lock["startup"]["warmup_transform"])

    def test_locked_patch_inventory(self):
        lock, _ = receipt_fixture()
        for source in [*lock["sources"].values(), lock["nccl_build"]]:
            self.assertEqual(sha((HERE / source["patch"]).read_bytes()), source["patch_sha256"])
        self.assertEqual(file_map_hash(lock["runtime"]["retained_vllm_native_files"]),
                         lock["runtime"]["retained_vllm_native_sha256"])

    def test_verifier_no_network_devices_or_framework_import(self):
        lock, document = receipt_fixture()
        calls = []
        def output(argv):
            calls.append(argv)
            if argv[:3] == ["docker", "image", "inspect"]:
                return json.dumps([{"Id": document["image_id"], "Os": "linux", "Architecture": "arm64",
                                    "RootFS": {"Layers": ["sha256:parent"]}}]).encode()
            self.assertIn("--read-only", argv)
            self.assertEqual(argv[argv.index("--network") + 1], "none")
            self.assertNotIn("--gpus", argv)
            self.assertNotIn("--device", argv)
            self.assertEqual(argv[-5:-1], ["-I", "-S", "-B", "-c"])
            self.assertIn("--mount", argv)
            return json.dumps(document["inside_image"]).encode()
        with tempfile.TemporaryDirectory() as temporary, patch("verify_image.subprocess.check_output", output), \
                patch("verify_image.trusted_closure", return_value={}), \
                patch("verify_image.verify_embedded_closure"):
            result = verify(document["image_id"], document["profile"], HERE / "glm53-tp4-lock.json",
                            Path(temporary) / "receipt.json", Path(temporary))
            self.assertTrue(result["checks_passed"])
        self.assertEqual(len(calls), 3)

    def test_forged_embedded_verifier_lock_and_manifest_rejected_before_execution(self):
        for changed in ("verify_sources.py", "source-lock.json", "manifest.json"):
            with self.subTest(changed=changed):
                closure = {name: name.encode() for name in TOOLS | {"source-lock.json", "manifest.json"}}
                calls = []
                def output(argv):
                    calls.append(argv)
                    if argv[:2] == ["docker", "create"]:
                        return ("c" * 64).encode()
                    self.assertEqual(argv[:2], ["docker", "cp"])
                    name = argv[2].rsplit("/", 1)[1]
                    data = b"forged" if name == changed else closure[name]
                    return make_archive({name: (data, 0o644)}, 0)
                with patch("verify_image.subprocess.check_output", output), \
                        patch("verify_image.subprocess.run") as cleanup:
                    with self.assertRaisesRegex(ValueError, "Embedded verifier input differs"):
                        verify_embedded_closure("sha256:" + "a" * 64, closure)
                    cleanup.assert_called_once()
                self.assertFalse(any(c[:2] == ["docker", "run"] for c in calls))

    def test_context_manifest_and_tool_bytes_are_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock = (HERE / "glm53-tp4-lock.json").read_bytes()
            tools = {name: (HERE / name).read_bytes() for name in TOOLS}
            manifest = json.dumps({"source_lock_sha256": sha(lock),
                                   "tool_hashes": {name: sha(data) for name, data in tools.items()}}).encode()
            closure = {**tools, "manifest.json": manifest, "source-lock.json": lock}
            payload = make_archive({IMAGE_ROOT.lstrip("/") + "/" + name: (data, 0o644)
                                    for name, data in closure.items()}, 0)
            context = make_archive({"Dockerfile": (b"FROM parent\n", 0o644), "payload.tar": (payload, 0o644)}, 0)
            (root / "image-context.tar").write_bytes(context)
            (root / "manifest.json").write_bytes(manifest)
            (root / "context-receipt.json").write_text(json.dumps({"context_sha256": sha(context),
                "payload_sha256": sha(payload), "manifest_sha256": sha(manifest)}))
            self.assertEqual(trusted_closure(root, lock), closure)
            (root / "manifest.json").write_bytes(b"forged")
            with self.assertRaisesRegex(ValueError, "manifest or source lock differs"):
                trusted_closure(root, lock)


if __name__ == "__main__":
    unittest.main()
