"""Pinned-source verification rejects modified or incomplete preinstalled models."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.deploy_existing_assets import (
    REMOTE_VERIFY, pinned_model_manifest, validate_existing_assets, verify_existing_models,
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def blob(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + bytes([0]) + data).hexdigest()


class Response(io.BytesIO):
    def __init__(self, url, entries, link=None):
        super().__init__(json.dumps(entries).encode())
        self.url = url
        self.headers = {"Link": link} if link else {}

    def geturl(self):
        return self.url


DATA = {"config.json": b'{"model":"fixture"}\n',
        "model.safetensors.index.json": b'{"weight_map":{"weight":"weights.bin"}}\n',
        "weights.bin": bytes(range(256))}


def metadata():
    return [{"type": "file", "path": name, "size": len(data), "oid": blob(data),
             **({"lfs": {"size": len(data), "oid": sha(data)}} if name == "weights.bin" else {})}
            for name, data in DATA.items()]


def opener(url, timeout):
    return Response(url, metadata())


def remote_probe(root, expected):
    # Windows has no O_NOFOLLOW; directory/link rejection is still exercised.
    prefix = "import os;os.O_NOFOLLOW=getattr(os,'O_NOFOLLOW',0);\n" if os.name == "nt" else ""
    return subprocess.run([sys.executable, "-I", "-c", prefix + REMOTE_VERIFY, str(root)],
                          input=json.dumps(expected).encode(), capture_output=True)


class ExistingAssetsTests(unittest.TestCase):
    def test_source_metadata_retains_raw_pages_and_lfs_git_identities(self):
        result = pinned_model_manifest("owner/model", "a" * 40, opener=opener)
        self.assertEqual(result["files"]["weights.bin"]["sha256"], sha(DATA["weights.bin"]))
        self.assertEqual(result["files"]["config.json"]["git_blob_sha1"], blob(DATA["config.json"]))
        self.assertEqual(json.loads(result["raw_pages"][0]["body_utf8"]), metadata())

    def test_rejects_unpinned_revision_and_unsafe_roots(self):
        with self.assertRaises(ValueError):
            pinned_model_manifest("owner/model", "main", opener=opener)
        for root in ("/", "/model/../other", "/model//weights", "relative", "/model\nunsafe"):
            with self.subTest(root=root), self.assertRaises(ValueError):
                validate_existing_assets({"schema": "sparkring-existing-assets/v1",
                                          "image": "all-ranks-preinstalled", "model_roots": [root] * 4})

    def test_rejects_duplicate_and_incomplete_metadata(self):
        for entries in (metadata() + metadata(), [{**metadata()[0], "lfs": {"oid": "bad"}}],
                        [{**metadata()[0], "path": "../config.json"}]):
            with self.subTest(entries=entries), self.assertRaises(ValueError):
                pinned_model_manifest("owner/model", "a" * 40,
                                      opener=lambda url, timeout: Response(url, entries))

    def test_remote_checks_all_bytes_and_allows_only_hf_auxiliary_cache(self):
        source = pinned_model_manifest("owner/model", "a" * 40, opener=opener)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, data in DATA.items():
                (root / name).write_bytes(data)
            cache = root / ".cache/huggingface/download"
            cache.mkdir(parents=True)
            (cache / "weights.metadata").write_bytes(b"auxiliary")
            result = remote_probe(root, source["files"])
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["model_files"], {k: sha(v) for k, v in DATA.items()})
            (root / "weights.bin").write_bytes(bytes(reversed(range(256))))
            self.assertIn(b"LFS content differs", remote_probe(root, source["files"]).stderr)
            (root / "weights.bin").write_bytes(DATA["weights.bin"])
            (root / "config.json").write_bytes(b"X" * len(DATA["config.json"]))
            self.assertIn(b"Git-blob content differs", remote_probe(root, source["files"]).stderr)
            (root / "config.json").write_bytes(DATA["config.json"])
            (root / "extra.bin").write_bytes(b"unexpected")
            self.assertIn(b"file set differs", remote_probe(root, source["files"]).stderr)

    def test_remote_rejects_symlinks_even_in_auxiliary_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, data in DATA.items():
                (root / name).write_bytes(data)
            cache = root / ".cache/huggingface"
            cache.mkdir(parents=True)
            try:
                (cache / "link").symlink_to(root / "config.json")
            except OSError:
                self.skipTest("Host cannot create symlinks")
            source = pinned_model_manifest("owner/model", "a" * 40, opener=opener)
            self.assertIn(b"contains a symlink", remote_probe(root, source["files"]).stderr)

    def test_every_rank_proves_pinned_metadata_and_same_payload(self):
        target = {"repository": "owner/model", "revision": "a" * 40,
                  "config_sha256": sha(DATA["config.json"]),
                  "index_sha256": sha(DATA["model.safetensors.index.json"])}
        hosts = [{"rank": r, "host": f"node{r}"} for r in range(4)]
        class Runner:
            def remote(self, host, argv, input, timeout):
                assert argv[:3] == ["python3", "-I", "-c"]
                assert set(json.loads(input)) == set(DATA)
                return json.dumps({"model_files": {k: sha(v) for k, v in DATA.items()},
                                   "files_verified": len(DATA), "bytes_verified": sum(map(len, DATA.values()))})
        result = verify_existing_models(Runner(), hosts, ["/models/existing"] * 4, target, opener=opener)
        self.assertEqual(len(result["ranks"]), 4)
        target["config_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "metadata differs"):
            verify_existing_models(Runner(), hosts, ["/models/existing"] * 4, target, opener=opener)


if __name__ == "__main__":
    unittest.main()
