"""GPU-free tests for the runtime-input pinning checker."""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import check_runtime_inputs as checker

COMMIT = "a" * 40
OTHER_COMMIT = "b" * 40
DIGEST = "sha256:" + "c" * 64


def _minimal_lock() -> dict:
    """A lock in which every asserted property holds."""

    return {
        "schema": "sparkring-runtime-lock/v1",
        "base_image": {
            "builder": {"repository": "nvcr.io/nvidia/cuda", "digest": DIGEST},
            "runtime": {"repository": "nvcr.io/nvidia/cuda", "digest": DIGEST},
        },
        "vllm": {"repository": "https://example.invalid/vllm", "commit": COMMIT},
        "sparkinfer": {
            "repository": "https://example.invalid/sparkinfer",
            "commit": COMMIT,
        },
        "flashinfer": {
            "repository": "https://example.invalid/flashinfer",
            "commit": COMMIT,
        },
        "nccl": {"repository": "https://example.invalid/nccl", "commit": COMMIT},
        "deep_gemm": {
            "repository": "https://example.invalid/deepgemm",
            "commit_full": COMMIT,
        },
        "model": {"repository": "org/model", "revision": OTHER_COMMIT},
    }


class ImmutableIdentityTest(unittest.TestCase):
    def test_a_fully_pinned_lock_reports_nothing(self) -> None:
        self.assertEqual(checker.check_immutable_identities(_minimal_lock()), [])

    def test_branch_name_instead_of_commit_is_reported(self) -> None:
        lock = _minimal_lock()
        lock["vllm"]["commit"] = "main"

        subjects = [f.subject for f in checker.check_immutable_identities(lock)]

        self.assertIn("vllm.commit", subjects)

    def test_abbreviated_commit_is_reported(self) -> None:
        lock = _minimal_lock()
        lock["nccl"]["commit"] = COMMIT[:12]

        findings = checker.check_immutable_identities(lock)

        self.assertEqual([f.subject for f in findings], ["nccl.commit"])
        self.assertIn("abbreviated", findings[0].detail)

    def test_absent_commit_is_reported(self) -> None:
        lock = _minimal_lock()
        del lock["sparkinfer"]["commit"]

        subjects = [f.subject for f in checker.check_immutable_identities(lock)]

        self.assertIn("sparkinfer.commit", subjects)

    def test_image_named_by_tag_rather_than_digest_is_reported(self) -> None:
        lock = _minimal_lock()
        lock["base_image"]["builder"] = {
            "repository": "nvcr.io/nvidia/cuda",
            "tag": "13.2.1-devel",
        }

        subjects = [f.subject for f in checker.check_immutable_identities(lock)]

        self.assertIn("base_image.builder.digest", subjects)
        self.assertIn("base_image.builder", subjects)

    def test_model_revision_must_be_a_commit(self) -> None:
        lock = _minimal_lock()
        lock["model"]["revision"] = "main"

        findings = checker.check_immutable_identities(lock)

        self.assertEqual([f.subject for f in findings], ["model.revision"])


class TrackedArtifactTest(unittest.TestCase):
    def test_matching_bytes_report_nothing(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "runtime").mkdir()
            payload = b"overlay contents\n"
            (root / "runtime" / "overlay.py").write_bytes(payload)
            lock = {
                "overlays": [
                    {
                        "path": "runtime/overlay.py",
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ]
            }

            self.assertEqual(checker.check_tracked_artifacts(lock, root), [])

    def test_changed_bytes_are_reported(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "runtime").mkdir()
            (root / "runtime" / "overlay.py").write_bytes(b"changed\n")
            lock = {
                "overlays": [
                    {
                        "path": "runtime/overlay.py",
                        "sha256": hashlib.sha256(b"original\n").hexdigest(),
                    }
                ]
            }

            findings = checker.check_tracked_artifacts(lock, root)

            self.assertEqual(len(findings), 1)
            self.assertIn("tree has", findings[0].detail)

    def test_absent_artifact_is_reported(self) -> None:
        with TemporaryDirectory() as raw:
            lock = {
                "nccl": {
                    "patches": [
                        {"path": "gone.patch", "sha256": "d" * 64},
                    ]
                }
            }

            findings = checker.check_tracked_artifacts(lock, Path(raw))

            self.assertEqual(len(findings), 1)
            self.assertIn("absent from the tree", findings[0].detail)

    def test_malformed_recorded_hash_is_reported(self) -> None:
        with TemporaryDirectory() as raw:
            lock = {"public_runtime_inputs": [{"path": "x", "sha256": "nope"}]}

            findings = checker.check_tracked_artifacts(lock, Path(raw))

            self.assertEqual(len(findings), 1)
            self.assertIn("malformed", findings[0].detail)


class RepositoryLockTest(unittest.TestCase):
    """The lock this repository ships must satisfy every asserted property."""

    def test_tracked_lock_passes_offline_checks(self) -> None:
        self.assertEqual(checker.run(check_remote=False), [])

    def test_main_reports_pass_offline(self) -> None:
        self.assertEqual(checker.main([]), 0)

    def test_every_pinned_source_names_its_repository(self) -> None:
        lock = checker.load_lock()
        for section, _field in checker.COMMIT_FIELDS:
            with self.subTest(section=section):
                self.assertIn(
                    "repository",
                    lock.get(section, {}),
                    f"{section} pins a commit without naming where to fetch it",
                )


if __name__ == "__main__":
    unittest.main()
