from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASE = "ghcr.io/fujitsupolycom/gb10-vllm-serving@sha256:6fc26fdad81a18f0fff67ce0a05f6d90165625ea2e1cac8a6f39bfb462017028"
PUBLISHED = "sha256:1574ba87fe4a0ad38c25a30087929ad549d823730be83b33e91fe4745b7a6571"


def test_public_lock_pins_published_image() -> None:
    lock = json.loads((HERE.parent / "faststart-lock.json").read_text(encoding="utf-8"))
    image = lock["deepseek_v4_flash_0731_hardened_serving_image"]
    assert image["manifest_digest"] == PUBLISHED
    assert image["base_manifest_digest"] == BASE.rsplit("@", 1)[1]


def test_containerfile_has_thin_and_default_native_targets() -> None:
    text = (HERE / "Containerfile").read_text(encoding="utf-8")
    assert f"FROM {BASE} AS runtime-overlay\n" in text
    assert "COPY LICENSE THIRD_PARTY_NOTICES.md" in text
    assert 'org.opencontainers.image.revision="${SPARKRING_SOURCE_REVISION}"' in text
    assert "FROM runtime-overlay AS thin" in text
    assert text.rstrip().endswith(
        'org.sparkring.native-pr431-reference-sha256="fe8b061337c2932031e20370dce3521a968ee5dc3f14e65ccdadd05ed1f19f8a"'
    )
    assert text.index("\nFROM runtime-overlay AS thin\n") < text.index(
        "\nFROM runtime-overlay AS native\n"
    )


def test_containerfile_patches_installed_and_retained_source() -> None:
    text = (HERE / "Containerfile").read_text(encoding="utf-8")
    assert text.count("apply_runtime_overlay.py") >= 3
    assert "--site-root /opt/r7-src/vllm" in text
    assert "--patch-only" in text
    assert "--component _C_stable_libtorch" in text
    assert "--target _C_stable_libtorch" in text
    assert (
        "COPY --from=native-builder /out/stage/vllm/_C_stable_libtorch.abi3.so "
        "/opt/venv/lib/python3.12/site-packages/vllm/_C_stable_libtorch.abi3.so"
    ) in text
    assert "native_artifact_receipt.py" in text
    assert "verify_image.py" in text


def test_build_script_runs_explicit_launch_environment_verifier() -> None:
    text = (HERE / "build-image.sh").read_text(encoding="utf-8")
    assert "SPARKRING_DEEPSEEK_GB10_TARGET:-native" in text
    assert "--require-launch-env" in text
    assert (
        "LD_PRELOAD=/usr/local/cuda/compat/libcuda.so.1:/opt/sparkring/nccl/libnccl.so.2"
        in text
    )
    assert "--expect-native" in text
    assert 'SPARKRING_SOURCE_REVISION=${source_revision}' in text
    assert '"${repo_root}"' in text
