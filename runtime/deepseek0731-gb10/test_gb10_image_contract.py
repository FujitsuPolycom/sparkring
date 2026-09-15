from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
BASE = "ghcr.io/fujitsupolycom/gb10-vllm-serving@sha256:6fc26fdad81a18f0fff67ce0a05f6d90165625ea2e1cac8a6f39bfb462017028"
PUBLISHED = "sha256:827a8e8c5749b78529cc0015dd174e1b19a0accc116bc142282f8b75428f98bd"


@pytest.mark.skipif(shutil.which("bash") is None, reason="Bash unavailable")
@pytest.mark.parametrize("builder,selector", [
    ("glm53-flash", "LICENSE"), ("deepseek0731-gb10", "LICENSE"),
    ("deepseek0731-gb10", "THIRD_PARTY_NOTICES.md"),
    ("deepseek0731-gb10", "runtime/deepseek0731-gb10"),
])
@pytest.mark.parametrize("damage", ["dirty", "untracked"])
def test_build_input_drift_precedes_external_commands(tmp_path, builder, selector, damage):
    def shell_path(path):
        value = str(path.resolve()).replace("\\", "/")
        return "/mnt/" + value[0].lower() + value[2:] if os.name == "nt" else value

    script = tmp_path / "build-image.sh"
    script.write_text((HERE.parent / builder / "build-image.sh").read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    marker = tmp_path / "operation"
    programs = {"git": '''#!/bin/bash
shift 2
if [[ "$1" == rev-parse ]]; then printf '%s\\n' "$FIXTURE_ROOT"; exit 0; fi
for item in "$@"; do
  if [[ "$item" == "$SELECTOR" ]]; then
    if [[ "$1" == diff && "$DAMAGE" == dirty ]]; then exit 1; fi
    if [[ "$1" == ls-files && "$DAMAGE" == untracked ]]; then echo untracked; fi
  fi
done
''', "docker": '#!/bin/sh\ntouch "$MARKER"\nexit 97\n',
        "python3": '#!/bin/sh\ntouch "$MARKER"\nexit 97\n'}
    for name, body in programs.items():
        path = fake_bin / name
        path.write_text(body, encoding="utf-8", newline="\n")
        path.chmod(0o755)
    environment = {"FIXTURE_ROOT": shell_path(tmp_path), "SELECTOR": selector, "DAMAGE": damage,
        "MARKER": shell_path(marker), "BUILD_RECEIPT": shell_path(tmp_path / "receipt.json")}
    command = 'export PATH=' + shlex.quote(shell_path(fake_bin)) + ':"$PATH"\n'
    command += "\n".join("export " + key + "=" + shlex.quote(value) for key, value in environment.items())
    command += "\nbash " + shlex.quote(shell_path(script))
    result = subprocess.run(["bash", "-c", command], text=True, capture_output=True, timeout=10)
    assert result.returncode == (78 if builder == "glm53-flash" else 2), result.stderr
    assert "builder inputs" in result.stderr and not marker.exists()


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
