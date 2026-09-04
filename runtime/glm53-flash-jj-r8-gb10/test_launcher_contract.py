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
SIRCL_ENVIRONMENT = HERE / "sircl-fused.env.example"
IMAGE_ID = "sha256:058b17b49ee3b5ffd805fa4a17e4d9efcb885f92349b98a8c8623bd7f0f96dd4"


def _defaults(path: Path = ENVIRONMENT) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
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
        "sha256:e34aa58fda32c2cc63bc70de680b50c5f2bb69c1e0ad3c5bce0782c6501f7d34"
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
    assert values["SPARKCACHE_CACHE_NAMESPACE"] == (
        "glm53-flash-vllm-e02b1746-b12x-9ae41c5c-"
        "dcp4-page-tail-cow-v2"
    )
    assert values["SPARKCACHE_MAX_SPAN_TOKENS"] == "1048576"
    assert values["CP_KV_CACHE_INTERLEAVE_SIZE"] == "auto"
    assert values["B12X_MLA_CKV_GATHER"] == "auto"
    assert values["VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL"] == "0"
    assert values["VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL_RAIL_MODE"] == "single"
    assert values["VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL_EXPOSURE"] == "sync"
    assert values["JIT_CACHE_NAMESPACE"] == (
        "glm53-flash-sm121-vllm-e02b1746-b12x-9ae41c5c"
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
  */libspark_transport_capi.so) hash=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ;;
  */sparkring-overlay-manifest.json) hash=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb ;;
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
        jit_namespace = "glm53-flash-sm121-vllm-e02b1746-b12x-9ae41c5c"
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
    runtime_readme = documents[0].read_text(encoding="utf-8")
    assert "sha256:e34aa58fda32c2cc63bc70de680b50c5f2bb69c1e0ad3c5bce0782c6501f7d34" in quickstart
    assert "sha256:3c377f1e4136285ebf66c32c36c3d01fd929f8aba0836cd0a16ed63cfd7e1762" in quickstart
    assert "DECODE_CONTEXT_PARALLEL_SIZE=4  # change to 1 or 2" in quickstart
    assert "fanout_image_archive.py" in quickstart
    assert "sircl-fused.env.example" in quickstart
    assert "SIRCL_ENABLED=1" in quickstart
    assert "Q=8/16/32/64/128" in runtime_readme
    assert "Q128 through Q8192" in runtime_readme
    assert re.search(r"primary\s+ports 19006/19007", runtime_readme)
    assert "67,109,888-byte mapped arena" in runtime_readme
    assert re.search(
        r"independent from SparkCache's two 3-GiB\s+asynchronous", runtime_readme
    )


def _launcher_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, Path]]:
    """Return (fake_bin, capture, directories) for a launcher dry run."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(parents=True)
    capture = tmp_path / "docker-arguments.txt"
    docker = fake_bin / "docker"
    docker.write_text(
        """#!/bin/sh
if [ "$1" = image ] && [ "$2" = inspect ]; then
  case "$4" in
    *sircl.native-sha256*) printf '%s\n' "$EXPECTED_SIRCL_NATIVE_SHA256" ;;
    *sircl.manifest-sha256*) printf '%s\n' "$EXPECTED_SIRCL_MANIFEST_SHA256" ;;
    *) printf '%s\n' "$EXPECTED_IMAGE_ID" ;;
  esac
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
  */libspark_transport_capi.so) hash=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ;;
  */sparkring-overlay-manifest.json) hash=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb ;;
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
                (
                    "export EXPECTED_SIRCL_NATIVE_SHA256="
                    "61aa0ec56a1b438439bed8611dab0353d2c72c10af02bbd917fb77c87b33e5fc"
                ),
                (
                    "export EXPECTED_SIRCL_MANIFEST_SHA256="
                    "85a231e6d2a290f7d6cccbc2cc6b1ccad7a6adbefc7ce4dde05b158f249aadd4"
                ),
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


def test_launcher_keeps_sircl_disabled_by_default(tmp_path: Path) -> None:
    assert _defaults()["SIRCL_ENABLED"] == "0"
    result, arguments = _run_launcher(tmp_path, "sircl-disabled")
    assert result.returncode == 0, result.stderr
    assert "org.sparkring.sircl.enabled=0" in arguments
    assert "PYTHONPATH=/opt/spark-sircl" not in arguments
    assert not any("SPARK_TP4_LIBRARY=" in argument for argument in arguments)


def test_fused_sircl_overlay_is_complete_and_sanitized() -> None:
    values = _defaults(SIRCL_ENVIRONMENT)
    assert values == {
        "SIRCL_ENABLED": "1",
        "SIRCL_BUNDLE_HOST_ROOT": None,
        "SPARK_TP4_PEER0": "REPLACE_WITH_PRIMARY_PEER_0_ADDRESS",
        "SPARK_TP4_PEER1": "REPLACE_WITH_PRIMARY_PEER_1_ADDRESS",
        "SPARK_TP4_DEVICE0": "rocep1s0f0",
        "SPARK_TP4_DEVICE1": "rocep1s0f1",
        "SPARK_TP4_GID0": "3",
        "SPARK_TP4_GID1": "3",
        "SPARK_TP4_GRAPH_CONTROL_PORT0": "9970",
        "SPARK_TP4_GRAPH_CONTROL_PORT1": "9971",
        "SPARK_TP4_GRAPH_SUBMIT_CPU": "10",
        "SPARK_TP4_GRAPH_PROGRESS_CPU": "11",
        "SPARK_TP4_MAX_INFLIGHT": "64",
        "SPARK_TP4_CONTROL_CONNECT_TIMEOUT_SECONDS": "10",
        "SPARK_TP4_GRAPH_DIRECT_DOORBELL": "1",
        "VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL": "1",
        "VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL_RAIL_MODE": "dual",
        "VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL_EXPOSURE": "fused",
        "SPARK_TP4_BIDIRECTIONAL_PREFILL_CONTROL_PORT0": "19000",
        "SPARK_TP4_BIDIRECTIONAL_PREFILL_CONTROL_PORT1": "19001",
        "SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_PEER0": (
            "REPLACE_WITH_SECONDARY_PEER_0_ADDRESS"
        ),
        "SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_PEER1": (
            "REPLACE_WITH_SECONDARY_PEER_1_ADDRESS"
        ),
        "SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_DEVICE0": (
            "REPLACE_WITH_SECONDARY_DEVICE_0"
        ),
        "SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_DEVICE1": (
            "REPLACE_WITH_SECONDARY_DEVICE_1"
        ),
        "SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_GID0": "3",
        "SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_GID1": "3",
        "SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_CONTROL_PORT0": "19100",
        "SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_CONTROL_PORT1": "19101",
        "SPARK_TP4_BIDIRECTIONAL_PREFILL_TIMEOUT_SECONDS": "120",
    }
    raw = SIRCL_ENVIRONMENT.read_text(encoding="utf-8").lower()
    for forbidden in ("192.168.", "10.0.", "172.16.", "@"):
        assert forbidden not in raw


def test_launcher_uses_the_image_embedded_sircl_bundle_by_default(
    tmp_path: Path,
) -> None:
    result, arguments = _run_launcher(
        tmp_path,
        "sircl-embedded",
        f"source '{_bash_path(SIRCL_ENVIRONMENT)}'",
        "SPARK_TP4_PEER0=192.0.2.11",
        "SPARK_TP4_PEER1=192.0.2.13",
        "SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_PEER0=192.0.2.12",
        "SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_PEER1=192.0.2.14",
        "SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_DEVICE0=rocep2s0f0",
        "SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_DEVICE1=rocep2s0f1",
    )

    assert result.returncode == 0, result.stderr
    assert not any(argument.endswith(":/opt/spark-sircl:ro") for argument in arguments)
    assert "PYTHONPATH=/opt/spark-sircl" in arguments
    assert "SPARK_TP4_LIBRARY=/opt/spark-sircl/libspark_transport_capi.so" in arguments
    assert (
        "org.sparkring.sircl.native-sha256="
        "61aa0ec56a1b438439bed8611dab0353d2c72c10af02bbd917fb77c87b33e5fc"
    ) in arguments
    assert (
        "org.sparkring.sircl.manifest-sha256="
        "85a231e6d2a290f7d6cccbc2cc6b1ccad7a6adbefc7ce4dde05b158f249aadd4"
    ) in arguments


def test_launcher_rejects_an_image_without_embedded_sircl_labels(
    tmp_path: Path,
) -> None:
    result, arguments = _run_launcher(
        tmp_path,
        "sircl-not-embedded",
        "SIRCL_ENABLED=1",
        "SPARK_TP4_PEER0=192.0.2.11",
        "SPARK_TP4_PEER1=192.0.2.13",
        "export EXPECTED_SIRCL_NATIVE_SHA256=",
        "export EXPECTED_SIRCL_MANIFEST_SHA256=",
    )

    assert result.returncode == 78
    assert "image has no receipt-bound embedded SIRCL bundle" in result.stderr
    assert arguments == []


def _sircl_bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "sircl-bundle"
    bundle.mkdir()
    for name in (
        "sitecustomize.py",
        "spark_collective_audit.py",
        "spark_cudagraph_replay_timing.py",
        "spark_graph_status_reporter.py",
        "spark_persistent_output_ring.py",
        "spark_tp4_backend.py",
        "spark_tp4_capability.py",
        "spark_tp4_health_gate.py",
        "spark_tp4_port_namespace.py",
        "spark_tp4_query_contract.py",
        "spark_tp4_query_row_provider.py",
        "sparkring-overlay-manifest.json",
        "libspark_transport_capi.so",
    ):
        (bundle / name).write_text("fixture", encoding="utf-8")
    return bundle


def test_launcher_accepts_read_only_external_sircl_override(tmp_path: Path) -> None:
    bundle = _sircl_bundle(tmp_path)

    result, arguments = _run_launcher(
        tmp_path,
        "sircl-enabled",
        "SIRCL_ENABLED=1",
        f"SIRCL_BUNDLE_HOST_ROOT={_bash_path(bundle)}",
        "SPARK_TP4_PEER0=192.0.2.11",
        "SPARK_TP4_PEER1=192.0.2.13",
        "SPARK_TP4_DEVICE0=rocep1s0f0",
        "SPARK_TP4_DEVICE1=rocep1s0f1",
    )
    assert result.returncode == 0, result.stderr
    assert (
        f"{_bash_path(bundle)}:/opt/spark-sircl:ro" in arguments
    )
    for setting in (
        "PYTHONPATH=/opt/spark-sircl",
        "SPARK_TP4_LIBRARY=/opt/spark-sircl/libspark_transport_capi.so",
        "VLLM_SPARK_TP4_MODE=custom",
        "VLLM_SPARK_TP4_GRAPH_WIDTH4096_RESEARCH=1",
        "VLLM_SPARK_SHARED_CAPTURE_STREAM=1",
        "SPARK_TP4_PEER0=192.0.2.11",
        "SPARK_TP4_PEER1=192.0.2.13",
        "SPARK_TP4_GRAPH_SUBMIT_CPU=10",
        "SPARK_TP4_GRAPH_PROGRESS_CPU=11",
        "SPARK_TP4_GRAPH_DIRECT_DOORBELL=0",
        "VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL=0",
        "VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL_RAIL_MODE=single",
        "VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL_EXPOSURE=sync",
        "SPARK_TP4_GRAPH_STATUS_PATH=/cache/jit/sircl-graph-rank0.json",
        "org.sparkring.sircl.enabled=1",
        "org.sparkring.sircl.direct-doorbell=0",
        "org.sparkring.sircl.prefill-exposure=sync",
        "org.sparkring.sircl.prefill-rail-mode=single",
        "org.sparkring.sircl.native-sha256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "org.sparkring.sircl.manifest-sha256=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    ):
        assert setting in arguments
    assert "SPARK_CUDAGRAPH_REPLAY_TIMING=1" not in arguments


def test_launcher_configures_fused_dual_rail_prefill(tmp_path: Path) -> None:
    bundle = _sircl_bundle(tmp_path)
    result, arguments = _run_launcher(
        tmp_path,
        "sircl-fused-prefill",
        f"source '{_bash_path(SIRCL_ENVIRONMENT)}'",
        f"SIRCL_BUNDLE_HOST_ROOT={_bash_path(bundle)}",
        "SPARK_TP4_PEER0=192.0.2.11",
        "SPARK_TP4_PEER1=192.0.2.13",
        "SPARK_TP4_DEVICE0=rocep1s0f0",
        "SPARK_TP4_DEVICE1=rocep1s0f1",
        "SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_PEER0=192.0.2.12",
        "SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_PEER1=192.0.2.14",
        "SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_DEVICE0=rocep2s0f0",
        "SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_DEVICE1=rocep2s0f1",
    )

    assert result.returncode == 0, result.stderr
    for setting in (
        "VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL=1",
        "VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL_RAIL_MODE=dual",
        "VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL_EXPOSURE=fused",
        "SPARK_TP4_GRAPH_DIRECT_DOORBELL=1",
        "SPARK_TP4_GRAPH_CONTROL_PORT0=9970",
        "SPARK_TP4_GRAPH_CONTROL_PORT1=9971",
        "SPARK_TP4_GRAPH_SUBMIT_CPU=10",
        "SPARK_TP4_GRAPH_PROGRESS_CPU=11",
        "SPARK_TP4_MAX_INFLIGHT=64",
        "SPARK_TP4_CONTROL_CONNECT_TIMEOUT_SECONDS=10",
        "SPARK_TP4_BIDIRECTIONAL_PREFILL_CONTROL_PORT0=19000",
        "SPARK_TP4_BIDIRECTIONAL_PREFILL_CONTROL_PORT1=19001",
        "SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_PEER0=192.0.2.12",
        "SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_PEER1=192.0.2.14",
        "SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_DEVICE0=rocep2s0f0",
        "SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_DEVICE1=rocep2s0f1",
        "SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_GID0=3",
        "SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_GID1=3",
        "SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_CONTROL_PORT0=19100",
        "SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_CONTROL_PORT1=19101",
        "SPARK_TP4_BIDIRECTIONAL_PREFILL_TIMEOUT_SECONDS=120",
        "org.sparkring.sircl.prefill-exposure=fused",
        "org.sparkring.sircl.prefill-rail-mode=dual",
        "org.sparkring.sircl.direct-doorbell=1",
    ):
        assert setting in arguments


@pytest.mark.parametrize(
    ("name", "lines", "message"),
    (
        (
            "prefill-without-sircl",
            ("VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL=1",),
            "bidirectional prefill requires SIRCL_ENABLED=1",
        ),
        (
            "dual-without-prefill",
            ("VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL_RAIL_MODE=dual",),
            "dual-rail bidirectional prefill requires",
        ),
        (
            "fused-with-single-rail",
            (
                "SIRCL_ENABLED=1",
                "VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL=1",
                "VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL_EXPOSURE=fused",
            ),
            "fused prefill exposure requires dual rail mode",
        ),
        (
            "invalid-exposure",
            ("VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL_EXPOSURE=async",),
            "VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL_EXPOSURE must be",
        ),
    ),
)
def test_launcher_rejects_incoherent_prefill_configuration(
    tmp_path: Path, name: str, lines: tuple[str, ...], message: str
) -> None:
    result, arguments = _run_launcher(tmp_path, name, *lines)
    assert result.returncode == 78
    assert message in result.stderr
    assert arguments == []


def test_launcher_replay_timing_is_explicitly_opt_in(
    tmp_path: Path,
) -> None:
    bundle = _sircl_bundle(tmp_path)

    result, arguments = _run_launcher(
        tmp_path,
        "sircl-timing",
        "SIRCL_ENABLED=1",
        f"SIRCL_BUNDLE_HOST_ROOT={_bash_path(bundle)}",
        "SPARK_TP4_PEER0=192.0.2.11",
        "SPARK_TP4_PEER1=192.0.2.13",
        "SPARK_TP4_DEVICE0=rocep1s0f0",
        "SPARK_TP4_DEVICE1=rocep1s0f1",
        "SPARK_CUDAGRAPH_REPLAY_TIMING=1",
        "SPARK_CUDAGRAPH_REPLAY_TIMING_SAMPLES=257",
    )

    assert result.returncode == 0, result.stderr
    assert "SPARK_CUDAGRAPH_REPLAY_TIMING=1" in arguments
    assert "SPARK_CUDAGRAPH_REPLAY_TIMING_SAMPLES=257" in arguments
    assert (
        "SPARK_CUDAGRAPH_REPLAY_TIMING_ARM_PATH="
        "/cache/jit/sircl-replay-timing.arm"
    ) in arguments


def test_launcher_can_time_stock_nccl_without_enabling_sircl(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "timing-bundle"
    bundle.mkdir()
    for name in (
        "sitecustomize.py",
        "spark_cudagraph_replay_timing.py",
        "spark_graph_status_reporter.py",
    ):
        (bundle / name).write_text("fixture", encoding="utf-8")

    result, arguments = _run_launcher(
        tmp_path,
        "nccl-timing",
        "SPARK_CUDAGRAPH_REPLAY_TIMING=1",
        "SPARK_CUDAGRAPH_REPLAY_TIMING_SAMPLES=513",
        (
            "SPARK_CUDAGRAPH_REPLAY_TIMING_BUNDLE_HOST_ROOT="
            f"{_bash_path(bundle)}"
        ),
    )

    assert result.returncode == 0, result.stderr
    assert (
        f"{_bash_path(bundle)}:/opt/spark-replay-timing:ro"
        in arguments
    )
    assert "PYTHONPATH=/opt/spark-replay-timing" in arguments
    assert "SPARK_CUDAGRAPH_REPLAY_TIMING=1" in arguments
    assert "SPARK_CUDAGRAPH_REPLAY_TIMING_SAMPLES=513" in arguments
    assert (
        "SPARK_CUDAGRAPH_REPLAY_TIMING_STATUS_PATH="
        "/cache/jit/cudagraph-replay-rank0.json"
    ) in arguments
    assert "VLLM_SPARK_TP4_MODE=custom" not in arguments
    assert "org.sparkring.sircl.enabled=0" in arguments


def test_launcher_rejects_incomplete_sircl_configuration(tmp_path: Path) -> None:
    result, arguments = _run_launcher(
        tmp_path,
        "sircl-incomplete",
        "SIRCL_ENABLED=1",
    )
    assert result.returncode != 0
    assert "SIRCL requires" in result.stderr
    assert arguments == []
