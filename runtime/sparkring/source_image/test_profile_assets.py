"""Exercise profile assets across preparation, installation and runtime checks."""
import json
from pathlib import Path
import tempfile
import unittest

from archive_utils import sha
from profile_assets import install_assets, prepare_assets, verify_assets
from verify_sources import serving_argv


class ProfileAssetTests(unittest.TestCase):
    def fixture(self, root):
        repository = root / "repository"
        repository.mkdir()
        payloads = {"code.py": b"value = 1\n", "profile.json": b'{"name":"pair"}\n'}
        files = {"roce/__init__.py": sha(payloads["code.py"])}
        payloads["manifest.json"] = json.dumps({"files": files}).encode()
        destinations = {"code.py": "/opt/sparkring/transports/pair/roce/__init__.py",
                        "manifest.json": "/opt/sparkring/transports/pair/manifest.json",
                        "profile.json": "/opt/sparkring/profiles/pair/profile.json"}
        for name, data in payloads.items():
            (repository / name).write_bytes(data)
        lock = {"profile_assets": {name: {"destination": destinations[name], "sha256": sha(data)}
                                   for name, data in payloads.items()},
                "profiles": {"pair": {"transport_profile": "pair",
                  "transport_manifest_sha256": sha(payloads["manifest.json"]),
                  "installed_profile_path": destinations["profile.json"],
                  "profile_sha256": sha(payloads["profile.json"])}}}
        return repository, lock

    def test_install_verify_and_reject_corrupt_or_extra_transport_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, lock = self.fixture(root)
            data, record = prepare_assets(repository, lock, 0)
            image = root / "image"
            install_assets(data, record, lock, image)
            self.assertEqual(verify_assets(lock, image)["pair"]["files"], 1)
            code = image / "opt/sparkring/transports/pair/roce/__init__.py"
            code.write_bytes(b"value = 2\n")
            with self.assertRaisesRegex(ValueError, "asset differs"):
                verify_assets(lock, image)
            code.write_bytes(b"value = 1\n")
            (code.parent / "injected.py").write_bytes(b"value = 3\n")
            with self.assertRaisesRegex(ValueError, "inventory differs"):
                verify_assets(lock, image)

    def test_conflicting_parent_file_prevents_all_installation_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, lock = self.fixture(root)
            data, record = prepare_assets(repository, lock, 0)
            image = root / "image"
            conflict = image / "opt/sparkring/profiles/pair/profile.json"
            conflict.parent.mkdir(parents=True)
            conflict.write_bytes(b"different")
            with self.assertRaisesRegex(ValueError, "conflicts"):
                install_assets(data, record, lock, image)
            self.assertFalse((image / "opt/sparkring/transports").exists())
            self.assertEqual(conflict.read_bytes(), b"different")

    def test_profile_dispatch_retains_common_verification_boundary(self):
        profile = "glm53-flash-spark-tp2-mtp3"
        manifest = {"runtime_profiles": {profile: {}, "tp4-dcp1-mtp3-prefill": {}},
                    "warmup_argv": ["/opt/sparkring/bin/serve-with-warmup.py"]}
        self.assertEqual(serving_argv(manifest, profile, ["/models/target"], "/usr/bin/python3"),
                         ["/usr/bin/python3", "/opt/sparkring/transports/entrypoint.py", "serve", "/models/target"])
        self.assertEqual(serving_argv(manifest, "tp4-dcp1-mtp3-prefill", ["serve"]),
                         manifest["warmup_argv"] + ["serve"])
        with self.assertRaisesRegex(ValueError, "absent"):
            serving_argv(manifest, "unlisted-profile", [])
        with self.assertRaisesRegex(ValueError, "absent"):
            serving_argv(manifest, "glm53-flash-nvfp4-tp2-mtp3", [])

    def test_canonical_spark_tp2_assets_are_bound_to_the_five_profile_lock(self):
        source = Path(__file__).resolve().parent
        lock = json.loads((source / "glm53-tp4-lock.json").read_bytes())
        name = "glm53-flash-spark-tp2-mtp3"
        self.assertEqual(len(lock["profiles"]), 5)
        self.assertNotIn("glm53-flash-nvfp4-tp2-mtp3", lock["profiles"])
        entry = lock["profiles"][name]
        self.assertEqual(entry["installed_profile_path"],
                         "/opt/sparkring/profiles/glm53-flash-spark-tp2/profile.json")
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory)
            data, record = prepare_assets(source.parents[2], lock, lock["source_date_epoch"])
            install_assets(data, record, lock, image)
            witness = verify_assets(lock, image)[entry["transport_profile"]]
            self.assertEqual(witness["manifest_sha256"], entry["transport_manifest_sha256"])
            profile_path = image / entry["installed_profile_path"].lstrip("/")
            profile = json.loads(profile_path.read_bytes())
            self.assertEqual(profile["name"], name)
            self.assertTrue(profile["model"]["repository"].endswith("-NVFP4-Spark"))
            profile["vllm_args"][profile["vllm_args"].index("--kv-cache-memory-bytes") + 1] = "1"
            profile_path.write_text(json.dumps(profile))
            with self.assertRaisesRegex(ValueError, "asset differs"):
                verify_assets(lock, image)


if __name__ == "__main__":
    unittest.main()
