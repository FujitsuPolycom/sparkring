"""GPU-free tests for the pinned-image retriever."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pull_pinned_images as puller

DIGEST_A = "sha256:" + "1" * 64
DIGEST_B = "sha256:" + "2" * 64


def _write(root: Path, relative: str, document: dict) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


class CollectPinnedImagesTest(unittest.TestCase):
    def test_reads_both_lock_shapes(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            _write(
                root,
                "runtime/runtime-lock.json",
                {
                    "base_image": {
                        "builder": {"repository": "reg/base", "digest": DIGEST_A},
                        "runtime": {"repository": "reg/base", "digest": DIGEST_B},
                    }
                },
            )
            _write(
                root,
                "runtime/faststart-lock.json",
                {
                    "base_image": {
                        "repository": "reg/faststart",
                        "manifest_digest": DIGEST_A,
                    }
                },
            )

            images = puller.collect_pinned_images(root)

            self.assertEqual(len(images), 3)
            self.assertEqual(
                {image.reference for image in images},
                {
                    f"reg/base@{DIGEST_A}",
                    f"reg/base@{DIGEST_B}",
                    f"reg/faststart@{DIGEST_A}",
                },
            )

    def test_an_image_named_by_tag_is_not_collected(self) -> None:
        """A tag can be repointed, so it is not an identity this pulls."""

        with TemporaryDirectory() as raw:
            root = Path(raw)
            _write(
                root,
                "runtime/runtime-lock.json",
                {"base_image": {"builder": {"repository": "reg/base", "tag": "13.2"}}},
            )

            self.assertEqual(puller.collect_pinned_images(root), [])

    def test_a_malformed_digest_is_not_collected(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            _write(
                root,
                "runtime/faststart-lock.json",
                {"base_image": {"repository": "reg/x", "manifest_digest": "sha256:no"}},
            )

            self.assertEqual(puller.collect_pinned_images(root), [])

    def test_an_absent_lock_is_skipped(self) -> None:
        with TemporaryDirectory() as raw:
            self.assertEqual(puller.collect_pinned_images(Path(raw)), [])


class PlanTest(unittest.TestCase):
    def test_plan_contacts_nothing_and_succeeds(self) -> None:
        self.assertEqual(puller.main(["--plan"]), 0)


class RepositoryLockTest(unittest.TestCase):
    def test_every_tracked_image_is_digest_pinned(self) -> None:
        images = puller.collect_pinned_images()

        self.assertTrue(images, "the tracked locks name no digest-pinned image")
        for image in images:
            with self.subTest(subject=image.subject):
                self.assertRegex(image.digest, r"^sha256:[0-9a-f]{64}$")
                self.assertIn("@sha256:", image.reference)


if __name__ == "__main__":
    unittest.main()
