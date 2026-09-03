from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LAUNCHER = HERE / "launch-rank.sh"
ENVIRONMENT = HERE / "runtime.env.example"
IMAGE_ID = "sha256:c3f85b2350609b6ff1201b8c5998f881ff4cef8b671d6783b543f841040915c0"


def _defaults() -> dict[str, str]:
    values: dict[str, str] = {}
    for line in ENVIRONMENT.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([A-Z0-9_]+)=(?:'([^']*)'|([^#\s]+))", line)
        if match:
            values[match.group(1)] = match.group(2) or match.group(3)
    return values


def _bash_path(path: Path) -> str:
    value = str(path)
    if os.name != "nt":
        return value
    drive, tail = os.path.splitdrive(value)
    assert drive, value
    normalized = tail.lstrip("\\/").replace("\\", "/")
    return f"/mnt/{drive[0].lower()}/{normalized}"


def test_environment_exposes_reproducible_operator_defaults() -> None:
    values = _defaults()
    assert values["IMAGE_ID"] == IMAGE_ID
    assert values["IMAGE_REF"] == (
        "ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@"
        "sha256:4ce98659c30d9e9c313b1018a2675e5f135a0404e7cc00951b4ade161c0a711f"
    )
    assert values["MAX_MODEL_LEN"] == "1048576"
    assert values["SERVED_MODEL_NAME"] == "glm-5.3-flash"
    assert values["DECODE_CONTEXT_PARALLEL_SIZE"] == "4"
    assert values["MAX_NUM_BATCHED_TOKENS"] == "8192"
    assert values["PREFILL_SCHEDULE_INTERVAL"] == "2"
    assert values["MAX_IMAGES_PER_PROMPT"] == "4"
    assert values["MAX_VIDEOS_PER_PROMPT"] == "1"
    assert values["KV_CACHE_MEMORY_BYTES"] == "auto"
    assert values["SPARKCACHE_ENABLED"] == "1"
    assert values["ENABLE_PROMPT_TOKENS_DETAILS"] == "1"
    assert values["SPARKCACHE_ACCESS_MODE"] == "read-write"
    assert values["SPARKCACHE_SHARED_PREFIX_LEASE_TTL_SECONDS"] == "300"
    assert values["SPARKCACHE_MAX_SPAN_TOKENS"] == "1048576"
    assert values["CP_KV_CACHE_INTERLEAVE_SIZE"] == "auto"
    assert values["B12X_MLA_CKV_GATHER"] == "auto"
    assert values["JIT_CACHE_NAMESPACE"] == (
        "glm53-flash-sm121-vllm-22ffe140-b12x-6255090a"
    )


def test_launcher_resolves_dcp_profiles_and_prompt_token_details(
    tmp_path: Path,
) -> None:
    subprocess.run(["bash", "-n", _bash_path(LAUNCHER)], check=True, cwd=ROOT)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture = tmp_path / "docker-arguments.txt"
    docker = fake_bin / "docker"
    docker.write_text(
        """#!/bin/sh
if [ "$1" = image ] && [ "$2" = inspect ]; then
  printf '%s\n' "$EXPECTED_IMAGE_ID"
elif [ "$1" = container ] && [ "$2" = inspect ]; then
  exit 1
elif [ "$1" = run ]; then
  printf '%s\n' "$@" > "$CAPTURE_PATH"
else
  exit 97
fi
        """,
        encoding="utf-8",
        newline="\n",
    )
    os.chmod(docker, 0o755)
    sha256sum = fake_bin / "sha256sum"
    sha256sum.write_text(
        """#!/bin/sh
case "$2" in
  */target/config.json) hash=676382abd1e90a6c85f0c8f33d45441ecd45fd514fd7b63ce5610e732d8e4996 ;;
  */target/model.safetensors.index.json) hash=0d1d9e6b226e76520e182de10d4e7194cc885c5cb1bf885bb90de1916ce312cb ;;
  */draft/config.json) hash=c4aeac0101196a6e26705b34c45230bcd0c7c68ee2d2d1efdb242087f3712573 ;;
  */draft/model.safetensors) hash=b33c03475ba7322cf398828f2d8d1be376df30dc05c6b40c28c8ea8da23e410b ;;
  *) exit 95 ;;
esac
printf '%s  %s\n' "$hash" "$2"
        """,
        encoding="utf-8",
        newline="\n",
    )
    os.chmod(sha256sum, 0o755)

    directories = {name: tmp_path / name for name in ("target", "draft", "cache")}
    for directory in directories.values():
        directory.mkdir()
    for path in (
        directories["target"] / "config.json",
        directories["target"] / "model.safetensors.index.json",
        directories["draft"] / "config.json",
        directories["draft"] / "model.safetensors",
    ):
        path.write_text("fixture", encoding="utf-8")

    for dcp, interleave, gather, kv_bytes in (
        (1, "1", "0", "27917287424"),
        (2, "4", "1", "32212254720"),
        (4, "4", "1", "25769803776"),
    ):
        config = tmp_path / f"dcp{dcp}.env"
        config.write_text(
            "\n".join(
                (
                    "HOST_IP=rank0.example.net",
                    "MASTER_ADDR=rank0.example.net",
                    f"TARGET_MODEL_HOST_PATH={_bash_path(directories['target'])}",
                    f"DFLASH_MODEL_HOST_PATH={_bash_path(directories['draft'])}",
                    f"CACHE_HOST_ROOT={_bash_path(directories['cache'])}",
                    f"PATH={_bash_path(fake_bin)}:$PATH",
                    f"export CAPTURE_PATH={_bash_path(capture)}",
                    f"export EXPECTED_IMAGE_ID={IMAGE_ID}",
                    "IMAGE_REF=test-image:r8",
                    f"IMAGE_ID={IMAGE_ID}",
                    f"DECODE_CONTEXT_PARALLEL_SIZE={dcp}",
                )
            ),
            encoding="utf-8",
            newline="\n",
        )
        result = subprocess.run(
            ["bash", _bash_path(LAUNCHER), "0", _bash_path(config)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        arguments = capture.read_text(encoding="utf-8").splitlines()
        assert "test-image:r8" in arguments
        dcp_index = arguments.index("--decode-context-parallel-size")
        assert arguments[dcp_index + 1] == str(dcp)
        interleave_index = arguments.index("--cp-kv-cache-interleave-size")
        assert arguments[interleave_index + 1] == interleave
        assert f"VLLM_B12X_MLA_CKV_GATHER={gather}" in arguments
        kv_index = arguments.index("--kv-cache-memory-bytes")
        assert arguments[kv_index + 1] == kv_bytes
        assert "--enable-prompt-tokens-details" in arguments
        assert "--language-model-only" not in arguments
        mm_index = arguments.index("--limit-mm-per-prompt")
        assert json.loads(arguments[mm_index + 1]) == {"image": 4, "video": 1}
        assert "--kv-transfer-config" in arguments
        jit_namespace = "glm53-flash-sm121-vllm-22ffe140-b12x-6255090a"
        assert f"VLLM_CACHE_ROOT=/cache/jit/vllm/{jit_namespace}" in arguments
        assert (
            f"B12X_CUTE_COMPILE_CACHE_DIR=/cache/jit/b12x/{jit_namespace}"
            in arguments
        )
        assert f"TRITON_CACHE_DIR=/cache/jit/triton/{jit_namespace}" in arguments
        assert (
            f"TORCHINDUCTOR_CACHE_DIR=/cache/jit/torchinductor/{jit_namespace}"
            in arguments
        )
        connector = json.loads(arguments[arguments.index("--kv-transfer-config") + 1])
        extra = connector["kv_connector_extra_config"]
        assert extra["spark_cache_publication_schema"] == "tail-cow-v2"
        assert extra["spark_cache_model_profile"] == "glm53-flash-hybrid"
        assert extra["spark_cache_access_mode"] == "read-write"
        assert extra["spark_cache_shared_prefix_lease_ttl_seconds"] == 300
        assert "spark_cache_store" not in extra
        assert "spark_cache_restore" not in extra

    config = tmp_path / "dcp1-vllm-prefix-only.env"
    config.write_text(
        "\n".join(
            (
                "HOST_IP=rank0.example.net",
                "MASTER_ADDR=rank0.example.net",
                f"TARGET_MODEL_HOST_PATH={_bash_path(directories['target'])}",
                f"DFLASH_MODEL_HOST_PATH={_bash_path(directories['draft'])}",
                f"CACHE_HOST_ROOT={_bash_path(directories['cache'])}",
                f"PATH={_bash_path(fake_bin)}:$PATH",
                f"export CAPTURE_PATH={_bash_path(capture)}",
                f"export EXPECTED_IMAGE_ID={IMAGE_ID}",
                "IMAGE_REF=test-image:r8",
                f"IMAGE_ID={IMAGE_ID}",
                "DECODE_CONTEXT_PARALLEL_SIZE=1",
                "SPARKCACHE_ENABLED=0",
            )
        ),
        encoding="utf-8",
        newline="\n",
    )
    result = subprocess.run(
        ["bash", _bash_path(LAUNCHER), "0", _bash_path(config)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    arguments = capture.read_text(encoding="utf-8").splitlines()
    assert "--enable-prefix-caching" in arguments
    assert "--kv-transfer-config" not in arguments
    assert "org.sparkring.sparkcache.enabled=0" in arguments

    disabled = tmp_path / "prompt-token-details-disabled.env"
    disabled.write_text(
        config.read_text(encoding="utf-8")
        + "\nENABLE_PROMPT_TOKENS_DETAILS=0\n",
        encoding="utf-8",
        newline="\n",
    )
    result = subprocess.run(
        ["bash", _bash_path(LAUNCHER), "0", _bash_path(disabled)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    arguments = capture.read_text(encoding="utf-8").splitlines()
    assert "--enable-prompt-tokens-details" not in arguments


def test_launcher_rejects_invalid_prompt_tokens_details_setting(tmp_path: Path) -> None:
    config = tmp_path / "prompt-token-details-invalid.env"
    config.write_text(
        f"source '{_bash_path(ENVIRONMENT)}'\nENABLE_PROMPT_TOKENS_DETAILS=invalid\n",
        encoding="utf-8",
        newline="\n",
    )

    result = subprocess.run(
        ["bash", _bash_path(LAUNCHER), "0", _bash_path(config)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 78
    assert "ENABLE_PROMPT_TOKENS_DETAILS must be 0 or 1" in result.stderr


def test_launcher_rejects_shared_prefix_retention_above_five_minutes(
    tmp_path: Path,
) -> None:
    config = tmp_path / "shared-prefix-retention-invalid.env"
    config.write_text(
        f"source '{_bash_path(ENVIRONMENT)}'\n"
        "SPARKCACHE_SHARED_PREFIX_LEASE_TTL_SECONDS=301\n",
        encoding="utf-8",
        newline="\n",
    )

    result = subprocess.run(
        ["bash", _bash_path(LAUNCHER), "0", _bash_path(config)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 78
    assert "must be between 1 and 300" in result.stderr


@pytest.mark.skipif(
    os.name == "nt",
    reason="WSL DrvFS reports Windows temporary files as mode 0777",
)
def test_launcher_renders_optional_multi_key_authentication(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture = tmp_path / "docker-arguments.txt"
    docker = fake_bin / "docker"
    docker.write_text(
        """#!/bin/sh
if [ "$1" = image ] && [ "$2" = inspect ]; then
  printf '%s\n' "$EXPECTED_IMAGE_ID"
elif [ "$1" = container ] && [ "$2" = inspect ]; then
  exit 1
elif [ "$1" = run ]; then
  printf '%s\n' "$@" > "$CAPTURE_PATH"
else
  exit 97
fi
        """,
        encoding="utf-8",
        newline="\n",
    )
    os.chmod(docker, 0o755)
    sha256sum = fake_bin / "sha256sum"
    sha256sum.write_text(
        """#!/bin/sh
case "$2" in
  */target/config.json) hash=676382abd1e90a6c85f0c8f33d45441ecd45fd514fd7b63ce5610e732d8e4996 ;;
  */target/model.safetensors.index.json) hash=0d1d9e6b226e76520e182de10d4e7194cc885c5cb1bf885bb90de1916ce312cb ;;
  */draft/config.json) hash=c4aeac0101196a6e26705b34c45230bcd0c7c68ee2d2d1efdb242087f3712573 ;;
  */draft/model.safetensors) hash=b33c03475ba7322cf398828f2d8d1be376df30dc05c6b40c28c8ea8da23e410b ;;
  *) exit 95 ;;
esac
printf '%s  %s\n' "$hash" "$2"
        """,
        encoding="utf-8",
        newline="\n",
    )
    os.chmod(sha256sum, 0o755)

    directories = {name: tmp_path / name for name in ("target", "draft", "cache")}
    for directory in directories.values():
        directory.mkdir()
    for path in (
        directories["target"] / "config.json",
        directories["target"] / "model.safetensors.index.json",
        directories["draft"] / "config.json",
        directories["draft"] / "model.safetensors",
    ):
        path.write_text("fixture", encoding="utf-8")

    def launch(name: str, *overrides: str) -> subprocess.CompletedProcess[str]:
        config = tmp_path / f"{name}.env"
        config.write_text(
            "\n".join(
                (
                    "HOST_IP=rank0.example.net",
                    "MASTER_ADDR=rank0.example.net",
                    f"TARGET_MODEL_HOST_PATH={_bash_path(directories['target'])}",
                    f"DFLASH_MODEL_HOST_PATH={_bash_path(directories['draft'])}",
                    f"CACHE_HOST_ROOT={_bash_path(directories['cache'])}",
                    f"PATH={_bash_path(fake_bin)}:$PATH",
                    f"export CAPTURE_PATH={_bash_path(capture)}",
                    f"export EXPECTED_IMAGE_ID={IMAGE_ID}",
                    "IMAGE_REF=test-image:r8",
                    f"IMAGE_ID={IMAGE_ID}",
                )
                + overrides
            ),
            encoding="utf-8",
            newline="\n",
        )
        return subprocess.run(
            ["bash", _bash_path(LAUNCHER), "0", _bash_path(config)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    # Unset is the default and must render exactly the keyless command.
    result = launch("auth-unset")
    assert result.returncode == 0, result.stderr
    keyless = capture.read_text(encoding="utf-8").splitlines()
    assert "--api-key" not in keyless

    # A two-key file renders one --api-key option carrying both keys.
    keys = tmp_path / "api-keys"
    keys.write_text("k1\n\nk2\n", encoding="utf-8", newline="\n")
    os.chmod(keys, 0o600)
    result = launch("auth-set", f"API_KEYS_FILE={_bash_path(keys)}")
    assert result.returncode == 0, result.stderr
    keyed = capture.read_text(encoding="utf-8").splitlines()
    assert keyed.count("--api-key") == 1
    index = keyed.index("--api-key")
    assert keyed[index + 1 : index + 3] == ["k1", "k2"]
    assert keyed[index + 3] == "--host"
    warmup_credential = "SPARKRING_WARMUP_API_KEY=k1"
    warmup_index = keyed.index(warmup_credential)
    assert keyed[warmup_index - 1] == "-e"
    command_without_warmup_credential = (
        keyed[: warmup_index - 1] + keyed[warmup_index + 1 :]
    )
    assert [
        argument
        for argument in command_without_warmup_credential
        if argument not in ("--api-key", "k1", "k2")
    ] == keyless

    # A file readable beyond its owner is refused before the container starts.
    loose = tmp_path / "api-keys-loose"
    loose.write_text("k1\n", encoding="utf-8", newline="\n")
    os.chmod(loose, 0o644)
    result = launch("auth-loose", f"API_KEYS_FILE={_bash_path(loose)}")
    assert result.returncode == 78, result.stdout + result.stderr
    assert "API_KEYS_FILE must be mode 0600" in result.stderr


def test_launcher_fails_closed_on_unusable_api_key_files() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    assert ': "${API_KEYS_FILE:=}"' in launcher
    assert "API_KEYS_FILE is not a readable regular file" in launcher
    assert "API_KEYS_FILE contains no non-empty keys" in launcher
    assert "API_KEYS_FILE contains whitespace in a key" in launcher
    assert "API_KEYS_FILE must be mode 0600" in launcher


def test_launcher_can_use_vllm_prefix_cache_without_sparkcache() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    assert "SPARKCACHE_ENABLED must be 0 or 1" in launcher
    assert "kv_transfer_args=()" in launcher
    assert "--enable-prefix-caching" in launcher


def test_launcher_exposes_independent_sparkcache_access_mode() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    assert "SPARKCACHE_ACCESS_MODE must be read-write" in launcher
    assert '"spark_cache_access_mode": os.environ["SPARKCACHE_ACCESS_MODE"]' in launcher
    assert '"spark_cache_store": True' not in launcher
    assert '"spark_cache_restore": True' not in launcher


def test_launcher_bounds_shared_prefix_retention() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    assert "SPARKCACHE_SHARED_PREFIX_LEASE_TTL_SECONDS:=300" in launcher
    assert "SPARKCACHE_SHARED_PREFIX_LEASE_TTL_SECONDS must be between 1 and 300" in launcher
    assert (
        '"spark_cache_shared_prefix_lease_ttl_seconds": integer('
        in launcher
    )


def test_launcher_rejects_unsupported_dcp_geometry() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    assert "DECODE_CONTEXT_PARALLEL_SIZE must be 1, 2, or 4" in launcher
    assert "GLM-5.3 DCP2/DCP4 requires" in launcher
    assert "IMAGE_ID must be an immutable local image ID" in launcher


def test_public_operator_documents_use_portable_examples_and_resolving_links() -> None:
    documents = (
        HERE / "README.md",
        HERE / "LIVE_VALIDATION.md",
        ROOT / "docs" / "GLM53_JJ_R8_GB10_SPARKCACHE_TP4_QUICKSTART.md",
    )
    private_address = re.compile(
        r"\b(?:10\.|192\.168\.|172\.(?:1[6-9]|2[0-9]|3[01])\.)"
    )
    link = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    for document in documents:
        text = document.read_text(encoding="utf-8")
        assert private_address.search(text) is None
        for target in link.findall(text):
            if target.startswith(("http://", "https://")):
                continue
            relative = target.split("#", maxsplit=1)[0]
            assert (document.parent / relative).resolve().exists(), (document, target)

    quickstart = documents[-1].read_text(encoding="utf-8")
    assert "sha256:4ce98659c30d9e9c313b1018a2675e5f135a0404e7cc00951b4ade161c0a711f" in quickstart
    assert "sha256:3c377f1e4136285ebf66c32c36c3d01fd929f8aba0836cd0a16ed63cfd7e1762" in quickstart
    assert "DECODE_CONTEXT_PARALLEL_SIZE=4  # change to 1 or 2" in quickstart
    assert "fanout_image_archive.py" in quickstart


def _launcher_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, Path]]:
    """Return (fake_bin, capture, directories) for a launcher dry run."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(parents=True)
    capture = tmp_path / "docker-arguments.txt"
    docker = fake_bin / "docker"
    docker.write_text(
        """#!/bin/sh
if [ "$1" = image ] && [ "$2" = inspect ]; then
  printf '%s\n' "$EXPECTED_IMAGE_ID"
elif [ "$1" = container ] && [ "$2" = inspect ]; then
  exit 1
elif [ "$1" = run ]; then
  printf '%s\n' "$@" > "$CAPTURE_PATH"
else
  exit 97
fi
        """,
        encoding="utf-8",
        newline="\n",
    )
    os.chmod(docker, 0o755)
    sha256sum = fake_bin / "sha256sum"
    sha256sum.write_text(
        """#!/bin/sh
case "$2" in
  */target/config.json) hash=676382abd1e90a6c85f0c8f33d45441ecd45fd514fd7b63ce5610e732d8e4996 ;;
  */target/model.safetensors.index.json) hash=0d1d9e6b226e76520e182de10d4e7194cc885c5cb1bf885bb90de1916ce312cb ;;
  */draft/config.json) hash=c4aeac0101196a6e26705b34c45230bcd0c7c68ee2d2d1efdb242087f3712573 ;;
  */draft/model.safetensors) hash=b33c03475ba7322cf398828f2d8d1be376df30dc05c6b40c28c8ea8da23e410b ;;
  *) exit 95 ;;
esac
printf '%s  %s\n' "$hash" "$2"
        """,
        encoding="utf-8",
        newline="\n",
    )
    os.chmod(sha256sum, 0o755)
    directories = {name: tmp_path / name for name in ("target", "draft", "cache")}
    for directory in directories.values():
        directory.mkdir()
    for path in (
        directories["target"] / "config.json",
        directories["target"] / "model.safetensors.index.json",
        directories["draft"] / "config.json",
        directories["draft"] / "model.safetensors",
    ):
        path.write_text("fixture", encoding="utf-8")
    return fake_bin, capture, directories


def _run_launcher(
    tmp_path: Path, name: str, *extra_lines: str
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    fake_bin, capture, directories = _launcher_fixture(tmp_path / name)
    config = tmp_path / f"{name}.env"
    config.write_text(
        "\n".join(
            (
                "HOST_IP=rank0.example.net",
                "MASTER_ADDR=rank0.example.net",
                f"TARGET_MODEL_HOST_PATH={_bash_path(directories['target'])}",
                f"DFLASH_MODEL_HOST_PATH={_bash_path(directories['draft'])}",
                f"CACHE_HOST_ROOT={_bash_path(directories['cache'])}",
                f"PATH={_bash_path(fake_bin)}:$PATH",
                f"export CAPTURE_PATH={_bash_path(capture)}",
                f"export EXPECTED_IMAGE_ID={IMAGE_ID}",
                "IMAGE_REF=test-image:r8",
                f"IMAGE_ID={IMAGE_ID}",
                *extra_lines,
            )
        ),
        encoding="utf-8",
        newline="\n",
    )
    result = subprocess.run(
        ["bash", _bash_path(LAUNCHER), "0", _bash_path(config)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    arguments = (
        capture.read_text(encoding="utf-8").splitlines() if capture.exists() else []
    )
    return result, arguments


def test_launcher_exposes_multimodal_and_text_only_modes(
    tmp_path: Path,
) -> None:
    values = _defaults()
    assert values["MULTIMODAL_INPUTS"] == "1"
    assert values["MAX_IMAGES_PER_PROMPT"] == "4"
    assert values["MAX_VIDEOS_PER_PROMPT"] == "1"

    result, arguments = _run_launcher(tmp_path, "multimodal-default")
    assert result.returncode == 0, result.stderr
    assert "--language-model-only" not in arguments
    limit = json.loads(arguments[arguments.index("--limit-mm-per-prompt") + 1])
    assert limit == {"image": 4, "video": 1}
    assert "org.sparkring.multimodal-inputs=1" in arguments
    assert "--kv-transfer-config" in arguments

    result, arguments = _run_launcher(
        tmp_path,
        "text-only",
        "MULTIMODAL_INPUTS=0",
    )
    assert result.returncode == 0, result.stderr
    assert "--language-model-only" in arguments
    assert "--limit-mm-per-prompt" not in arguments
    assert "org.sparkring.multimodal-inputs=0" in arguments
    assert "--kv-transfer-config" in arguments
    assert arguments[arguments.index("--load-format") + 1] == "fastsafetensors"

    result, arguments = _run_launcher(
        tmp_path,
        "custom-multimodal-limits",
        "MULTIMODAL_INPUTS=1",
        "MAX_IMAGES_PER_PROMPT=2",
        "MAX_VIDEOS_PER_PROMPT=0",
    )
    assert result.returncode == 0, result.stderr
    limit = json.loads(arguments[arguments.index("--limit-mm-per-prompt") + 1])
    assert limit == {"image": 2, "video": 0}
    assert "--kv-transfer-config" in arguments


def test_launcher_rejects_invalid_multimodal_mode(
    tmp_path: Path,
) -> None:
    result, arguments = _run_launcher(
        tmp_path, "bad-multimodal-value", "MULTIMODAL_INPUTS=yes"
    )
    assert result.returncode != 0
    assert "MULTIMODAL_INPUTS must be 0" in result.stderr
    assert arguments == []
