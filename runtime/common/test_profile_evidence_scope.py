"""A profile's evidence scope describes measurements; its release field selects the image it runs."""
import re
from pathlib import PurePosixPath

from runtime.common import profiles

# "runs image X", "runs installer image X", "selects image X", "uses image X":
# a claim about the image the profile itself runs, which its release field
# owns. Measurements name their image as "On installer image X, ...".
SELECTION = re.compile(r"\b(?:runs|selects|uses)(?: on)?(?: the)?(?: installer)? image ([A-Za-z0-9._-]+)")


def release_name(profile):
    """The release directory a profile's release field names: runtime/releases/NAME/release.json."""
    return PurePosixPath(profile["release"]).parent.name


def claimed_images(scope):
    return [name.rstrip(".") for name in SELECTION.findall(scope)]


def test_evidence_scope_names_no_image_but_the_release_as_the_one_the_profile_runs():
    for profile_id in profiles.catalog():
        profile, _ = profiles.load(profile_id)
        for name in claimed_images(profile["evidence_scope"]):
            assert name == release_name(profile), (profile_id, name)


def test_a_claim_that_the_profile_runs_another_image_is_found():
    scope = ("On installer image dev-20260927-b12xcache-cuda1342-nccl2323-status032, an installation passed. "
             "This profile runs image dev-20260928-plainstatus-cuda1342-nccl2323-status033, which is another image.")
    assert claimed_images(scope) == ["dev-20260928-plainstatus-cuda1342-nccl2323-status033"]
    assert claimed_images("The profile runs installer image dev-20261001-kraken-cuda1342-nccl2323-status034.") == [
        "dev-20261001-kraken-cuda1342-nccl2323-status034"]


def test_shared_serving_versions_lists_the_profiles_that_select_the_shared_release():
    text = (profiles.ROOT / "runtime/releases/README.md").read_text(encoding="utf-8")
    section = text.split("## Shared serving versions", 1)[1].split("\n## ", 1)[0]
    catalog = profiles.catalog()
    named = {token for token in re.findall(r"`([^`]+)`", section) if token in catalog}
    selecting = {profile_id for profile_id in catalog
                 if release_name(profiles.load(profile_id)[0]) == "shared-2026.09.3"}
    assert named == selecting
