"""Default build and launch inputs must still describe the published image."""

import gzip
import json
from pathlib import Path

from archive_utils import sha
from prepare_context import startup_archive
from profile_assets import prepare_assets


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]


def published_inputs():
    publication = json.loads((HERE / "publication.json").read_bytes())
    name = "published-" + publication["source_revision"][:8] + "-manifest.json.gz"
    raw = gzip.decompress((HERE / "fixtures" / name).read_bytes())
    assert sha(raw) == publication["manifest_sha256"]
    return publication, json.loads(raw)


def test_default_lock_matches_published_image_and_all_profile_receipts():
    publication, manifest = published_inputs()
    raw = (HERE / "glm53-tp4-lock.json").read_bytes()
    assert sha(raw) == publication["source_lock_sha256"] == manifest["source_lock_sha256"]
    lock = json.loads(raw)
    for name, receipt in publication["profiles"].items():
        assert receipt["source_lock_sha256"] == sha(raw)
        profile = lock["profiles"][name]
        for field in ("profile_sha256", "transport_manifest_sha256", "topology", "collective_backend"):
            if field in receipt:
                assert receipt[field] == profile[field]


def test_default_startup_and_profile_archives_reproduce_published_manifest():
    _, manifest = published_inputs()
    lock = json.loads((HERE / "glm53-tp4-lock.json").read_bytes())
    _, startup = startup_archive(
        {**lock["startup"], "checkout": str(HERE / "startup")}, lock["source_date_epoch"]
    )
    assert startup == manifest["startup_override"]
    _, profiles = prepare_assets(REPOSITORY, lock, lock["source_date_epoch"])
    assert profiles == manifest["profile_assets"]


def test_default_embedded_verifier_tools_match_published_context():
    _, manifest = published_inputs()
    for name, expected in manifest["tool_hashes"].items():
        assert sha((HERE / name).read_bytes()) == expected, name
