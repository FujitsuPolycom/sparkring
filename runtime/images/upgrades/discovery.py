"""Read publication metadata without treating publication as qualification."""

import datetime
import json
import re
import urllib.request

from .contracts import require
from .io import checked

REGISTRY = "ghcr.io/randomvariable/vllm-b12x-multi"
FEED = "https://randomvariable.github.io/vllm-multiarch-oci/latest-image.json"


def validate_publication(value, *, now=None, max_age_hours=36):
    require(
        isinstance(value, dict)
        and set(value) == {"repository", "tag", "reference", "resolved_at"},
        "Unexpected image-publication record",
    )
    require(
        value["repository"] == REGISTRY,
        "Publication belongs to another registry repository",
    )
    require(
        re.fullmatch(re.escape(REGISTRY) + r"@sha256:[0-9a-f]{64}", value["reference"]),
        "Publication reference is not immutable",
    )
    require(
        re.fullmatch(
            r"vllmb12x-[a-z0-9][a-z0-9-]*-[0-9a-f]{12}-[0-9a-f]{12}-[0-9]{8}-n[1-9][0-9]*",
            value["tag"],
        ),
        "Unexpected immutable publication tag",
    )
    timestamp = datetime.datetime.fromisoformat(
        value["resolved_at"].replace("Z", "+00:00")
    )
    require(timestamp.tzinfo is not None, "Publication time must have a timezone")
    now = now or datetime.datetime.now(datetime.timezone.utc)
    age = (now - timestamp).total_seconds()
    require(
        -300 <= age <= max_age_hours * 3600,
        "Publication record is stale or future-dated",
    )
    return dict(value, publication_is_serving_qualification=False)


def discover_arm64(*, url=FEED, max_age_hours=36):
    require(
        url == FEED,
        "Use the publisher's known metadata feed, not an arbitrary credential-bearing URL",
    )
    with urllib.request.urlopen(url, timeout=20) as response:
        raw = response.read(1024 * 1024 + 1)
    require(len(raw) <= 1024 * 1024, "Publication metadata exceeds size limit")
    result = validate_publication(json.loads(raw), max_age_hours=max_age_hours)
    # Registry inspection fetches manifests/configuration, not image layers.
    raw = checked(
        [
            "docker",
            "buildx",
            "imagetools",
            "inspect",
            "--format",
            "{{json .Image}}",
            result["reference"],
        ],
        seconds=30,
    )
    config = json.loads(raw)
    require(
        config.get("os") == "linux" and config.get("architecture") == "arm64",
        "Resolved image is not Linux ARM64",
    )
    return {
        **result,
        "platform": "linux/arm64",
        "labels": config.get("config", {}).get("Labels", {}),
        "decision": "Foundation candidate only; no automatic replacement of the policy's approved foundation.",
    }
