"""Offline contracts for the public faststart lane."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "runtime"
LOCK = json.loads((RUNTIME / "faststart-lock.json").read_text(encoding="utf-8"))


def test_faststart_lock_pins_arm64_base_and_model():
    assert LOCK["schema"] == "sparkring-faststart-lock/v1"
    base = LOCK["base_image"]
    assert base["platform"] == "linux/arm64"
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", base["manifest_digest"])
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", base["config_digest"])
    model = LOCK["model"]
    assert re.fullmatch(r"[0-9a-f]{40}", model["revision"])
    assert re.fullmatch(r"[0-9a-f]{64}", model["config_sha256"])


def test_faststart_model_pin_matches_full_runtime_lock():
    full = json.loads((RUNTIME / "runtime-lock.json").read_text(encoding="utf-8"))
    assert LOCK["model"]["repository"] == full["model"]["repository"]
    assert LOCK["model"]["revision"] == full["model"]["revision"]
    assert LOCK["model"]["config_sha256"] == full["model"]["config_sha256"]


def test_faststart_container_applies_overlay_before_native_build():
    text = (RUNTIME / "Containerfile.faststart").read_text(encoding="utf-8")
    assert "FROM ${BASE_IMAGE} AS overlay-base" in text
    assert "--fail-closed" in text
    assert ".faststart-overlay-passed" in text
    assert (
        "COPY --from=overlay-base /opt/sparkring/.faststart-overlay-passed"
        in text
    )
    assert "FROM ${BASE_DEVEL_IMAGE} AS native-build" in text
    assert "runtime/patches" in text
    assert "runtime/patches-faststart" in text
    assert "--compatible-base" in text
    assert "--supplemental-root /build/patches-faststart" in text
    assert "runtime-manifest.json" in text
    assert 'ENTRYPOINT ["/opt/sparkring/public-entrypoint.sh"]' in text


def test_faststart_builder_uses_digest_not_human_tag():
    text = (RUNTIME / "build-faststart.sh").read_text(encoding="utf-8")
    assert 'f\'{base["repository"]}@{base["manifest_digest"]}\'' in text
    assert "tag_for_humans" not in text.split("values = {", 1)[1]
    assert '"BASE_CONFIG_DIGEST": base["config_digest"]' in text
    assert 'observed_base_id="$("${ENGINE}" image inspect' in text
    assert "ALLOW_DIRTY_BUILD" in text
    assert "--platform linux/arm64" in text


def test_download_script_uses_same_pinned_base_and_model():
    text = (ROOT / "scripts/download-glm52.sh").read_text(encoding="utf-8")
    base = LOCK["base_image"]
    expected_base = f'{base["repository"]}@{base["manifest_digest"]}'
    assert f'BASE_IMAGE="{expected_base}"' in text
    assert f'MODEL_REPO="{LOCK["model"]["repository"]}"' in text
    assert f'MODEL_REVISION="{LOCK["model"]["revision"]}"' in text
    assert f'CONFIG_SHA256="{LOCK["model"]["config_sha256"]}"' in text


def test_quickstart_names_every_major_gate():
    text = (ROOT / "docs/QUICKSTART.md").read_text(encoding="utf-8")
    for required in (
        "build-faststart.sh",
        "download-glm52.sh",
        "docker save",
        "docker load",
        "rsync",
        "sparkring_site.py",
        "preflight.py",
        "sparkring_launcher.py",
        "--enforce-eager",
        "/v1/models",
    ):
        assert required in text
