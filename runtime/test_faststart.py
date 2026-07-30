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


def test_download_script_uses_pinned_base_and_nf3_recipe():
    text = (ROOT / "scripts/download-glm52.sh").read_text(encoding="utf-8")
    base = LOCK["base_image"]
    expected_base = f'{base["repository"]}@{base["manifest_digest"]}'
    recipe = json.loads(
        (ROOT / "recipes/glm52-nf3-hybrid.json").read_text(encoding="utf-8")
    )
    model = recipe["model"]
    draft = model["mtp_draft"]
    assert f'BASE_IMAGE="{expected_base}"' in text
    assert f'MODEL_REPO="{model["repository"]}"' in text
    assert f'MODEL_REVISION="{model["revision"]}"' in text
    assert f'MODEL_CONFIG_SHA256="{model["config_sha256"]}"' in text
    assert f'MODEL_INDEX_SHA256="{model["index_sha256"]}"' in text
    assert f'DRAFT_REPO="{draft["repository"]}"' in text
    assert f'DRAFT_REVISION="{draft["revision"]}"' in text
    assert (
        f'DRAFT_INPUTSCALES_SHA256="{draft["inputscales_sha256"]}"'
        in text
    )
    assert 'allow_patterns=["mtp-draft/*"]' in text


def test_download_script_can_adopt_verified_existing_payload():
    text = (ROOT / "scripts/download-glm52.sh").read_text(encoding="utf-8")
    assert "verify_glm52_download.py" in text
    assert '"${verify_command[@]}" --adopt' in text
    assert "verification_status" in text


def test_download_container_overrides_inherited_offline_and_root_cache_settings():
    text = (ROOT / "scripts/download-glm52.sh").read_text(encoding="utf-8")
    assert '--env "HOME=/tmp"' in text
    assert '--env "HF_HOME=/tmp/sparkring-huggingface"' in text
    assert '--env "HF_HUB_OFFLINE=0"' in text
    assert text.index('--env "HF_HOME=/tmp/sparkring-huggingface"') < text.index(
        '"${ENGINE}" "${common_run[@]}"'
    )


def test_quickstart_names_every_major_gate():
    text = (ROOT / "docs/QUICKSTART.md").read_text(encoding="utf-8")
    for required in (
        "bootstrap_nf3.py",
        "builds the small NF3 adapter layer",
        "verify_ssh_mesh.py",
        "--scope all-adjacent",
        "--fix",
        "sparkring_site.py",
        "acceptance_gate.py",
        "CUDA graphs through Q40",
        "operator or bot",
        "Management: SSH, downloads, launch, API",
        "--profile nvfp4-rope8",
        "/v1/models",
    ):
        assert required in text


def test_readme_and_quickstart_link_exhaustive_prerequisites():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    quickstart = (ROOT / "docs/QUICKSTART.md").read_text(encoding="utf-8")
    prerequisites = (ROOT / "docs/PREREQUISITES.md").read_text(
        encoding="utf-8"
    )
    assert "docs/PREREQUISITES.md" in readme
    assert "PREREQUISITES.md" in quickstart
    for required in (
        "What the operator must provide",
        "What the bot can discover",
        "Required validation sequence",
        "Ready-to-bootstrap checklist",
        "NVFP4/FP8-RoPE",
    ):
        assert required in prerequisites


def test_nvfp4_rope8_layer_preserves_nf3_and_restores_only_mla():
    containerfile = (
        RUNTIME / "Containerfile.nf3-nvfp4-rope8"
    ).read_text(encoding="utf-8")
    assert "FROM ${MLA_IMAGE} AS mla_source" in containerfile
    assert "FROM ${NF3_IMAGE}" in containerfile
    assert 'rm -rf "${SITE_PACKAGES}/b12x/attention/mla"' in containerfile
    assert "b12x/integration/mla.py" in containerfile
    assert "b12x/integration/sparse_mla_scratch.py" in containerfile
    assert "rm -rf \"${SITE_PACKAGES}/b12x\"" not in containerfile
    assert "verify-nf3-nvfp4-rope8.py" in containerfile
    assert 'org.sparkring.kv_profile="nvfp4-rope8"' in containerfile


def test_nvfp4_rope8_builder_reuses_pinned_nf3_and_faststart_layers():
    text = (
        ROOT / "scripts/build-nf3-nvfp4-rope8-image.sh"
    ).read_text(encoding="utf-8")
    assert 'bash "${ROOT}/scripts/build-nf3-image.sh"' in text
    assert "NF3_IMAGE_ID=" in text
    assert "MLA_IMAGE_ID=" in text
    assert "verify-nf3-nvfp4-rope8.py" in text
    assert "--platform linux/arm64" in text


def test_entrypoint_refuses_image_and_launch_kv_profile_mismatch():
    entrypoint = (RUNTIME / "public-entrypoint.sh").read_text(
        encoding="utf-8"
    )
    assert (
        '[ "${SPARKRING_KV_PROFILE}" = "${VLLM_SPARK_KV_PROFILE}" ]'
        in entrypoint
    )
    assert "verify-nf3-nvfp4-rope8.py" in entrypoint
    nf3 = (RUNTIME / "Containerfile.nf3-bootstrap").read_text(
        encoding="utf-8"
    )
    assert "SPARKRING_KV_PROFILE=fp8" in nf3
