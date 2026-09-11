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
    def test_reads_complete_faststart_inventory(self):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            _write(
                root,
                "runtime/faststart-lock.json",
                {
                    name: {"repository": "reg/" + name, "manifest_digest": DIGEST_A}
                    for name in (
                        "base_image",
                        "serving_image",
                        "deepseek_v4_flash_0731_hardened_serving_image",
                    )
                },
            )
            images = puller.collect_pinned_images(root)
            self.assertEqual(len(images), 3)
            self.assertEqual({i.digest for i in images}, {DIGEST_A})

    def test_absent_lock_fails(self):
        with TemporaryDirectory() as raw:
            with self.assertRaisesRegex(ValueError, "missing"):
                puller.collect_pinned_images(Path(raw))

    def test_missing_or_invalid_image_slot_fails(self):
        for damage in ("absent", "tag-only", "bad-digest"):
            with self.subTest(damage=damage), TemporaryDirectory() as raw:
                root = Path(raw)
                document = {
                    name: {"repository": "reg/" + name, "manifest_digest": DIGEST_A}
                    for name in (
                        "base_image",
                        "serving_image",
                        "deepseek_v4_flash_0731_hardened_serving_image",
                    )
                }
                if damage == "absent":
                    document.pop("serving_image")
                elif damage == "tag-only":
                    document["serving_image"] = {"repository": "reg/x", "tag": "moving"}
                else:
                    document["serving_image"]["manifest_digest"] = "sha256:no"
                _write(root, "runtime/faststart-lock.json", document)
                with self.assertRaises(ValueError):
                    puller.collect_pinned_images(root)


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
