"""Offline contracts for the DeepSeek two-Spark env-driven launcher."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "deepseek_v4_pair_serve.sh"
TEMPLATE = ROOT / "scripts" / "config" / "deepseek-v4-flash-0731-pair.env.example"
RECIPE = ROOT / "recipes" / "deepseek-v4-flash-0731-pair.json"
CYCLE_LAUNCHER = ROOT / "scripts" / "deepseek_v4_cycle_serve.sh"
CYCLE_TEMPLATE = ROOT / "scripts" / "config" / "deepseek-v4-flash-0731.env.example"
CYCLE_RECIPE = ROOT / "recipes" / "deepseek-v4-flash-0731.json"
RUNTIME_LOCK = ROOT / "runtime" / "faststart-lock.json"


def _bash_path(path: str | Path) -> str:
    value = str(path)
    if os.name != "nt":
        return value
    drive, tail = os.path.splitdrive(value)
    assert drive, value
    return f"/mnt/{drive[0].lower()}/{tail.lstrip('\\/').replace('\\', '/')}"


def _run_launcher(
    env_file: Path,
    mode: str | None = "--check",
    launcher: Path = LAUNCHER,
    path_prefix: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    arguments = [str(env_file)] if mode is None else [mode, str(env_file)]
    if os.name != "nt":
        command = ["bash", str(launcher), *arguments]
        environment = os.environ.copy()
        if path_prefix is not None:
            environment["PATH"] = f"{path_prefix}{os.pathsep}{environment['PATH']}"
    else:
        rendered = " ".join(
            shlex.quote(value)
            for value in (
                "bash",
                _bash_path(launcher),
                *([_bash_path(env_file)] if mode is None else [mode, _bash_path(env_file)]),
            )
        )
        path_assignment = (
            f"export PATH={shlex.quote(_bash_path(path_prefix))}:$PATH; "
            if path_prefix is not None
            else ""
        )
        command = ["bash", "-lc", f"{path_assignment}exec {rendered}"]
        environment = os.environ.copy()
    return subprocess.run(
        command,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def _env_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def _launcher_image(path: Path) -> str:
    match = re.search(r"^image=(\S+)$", path.read_text(encoding="utf-8"), re.MULTILINE)
    assert match is not None
    return match.group(1)


@pytest.fixture
def pair_env(tmp_path: Path) -> Path:
    model = tmp_path / "model"
    cache = tmp_path / "cache"
    model.mkdir()
    cache.mkdir()
    content = TEMPLATE.read_text(encoding="utf-8")
    replacements = {
        "<NODE_RANK_0_OR_1>": "0",
        "<RANK0_FABRIC_IP>": "10.42.0.1",
        "<ABSOLUTE_MODEL_DIRECTORY>": _bash_path(model),
        "<ABSOLUTE_CACHE_DIRECTORY>": _bash_path(cache),
        "<FABRIC_IFNAME>": "fabric0",
        "<RANK_FABRIC_IP>": "10.42.0.1",
        "<GID_INDEX>": "3",
    }
    for placeholder, value in replacements.items():
        content = content.replace(placeholder, value)
    env_file = tmp_path / "rank-0.env"
    env_file.write_text(content, encoding="utf-8", newline="\n")
    return env_file


@pytest.fixture
def cycle_env(tmp_path: Path) -> Path:
    model = tmp_path / "model"
    cache = tmp_path / "cache"
    model.mkdir()
    cache.mkdir()
    content = CYCLE_TEMPLATE.read_text(encoding="utf-8")
    replacements = {
        "<NODE_RANK_0_TO_3>": "0",
        "<RANK0_FABRIC_IP>": "10.43.0.1",
        "<ABSOLUTE_MODEL_DIRECTORY>": _bash_path(model),
        "<ABSOLUTE_CACHE_DIRECTORY>": _bash_path(cache),
        "<NCCL_SOCKET_IFNAME>": "fabric0",
        "<RANK_FABRIC_IP>": "10.43.0.1",
    }
    for placeholder, value in replacements.items():
        content = content.replace(placeholder, value)
    env_file = tmp_path / "rank-0.env"
    env_file.write_text(content, encoding="utf-8", newline="\n")
    return env_file


def test_pair_env_defaults_match_recipe() -> None:
    values = _env_values(TEMPLATE)
    recipe = json.loads(RECIPE.read_text(encoding="utf-8"))
    assert recipe["runtime"]["launcher"] == "scripts/deepseek_v4_pair_serve.sh"
    serving = recipe["serving"]
    assert int(values["API_PORT"]) == 8000
    assert int(values["MASTER_PORT"]) == 29500
    assert int(values["NUM_SPECULATIVE_TOKENS"]) == serving["speculation"][
        "num_speculative_tokens"
    ]
    assert int(values["MAX_MODEL_LEN"]) == serving["max_model_len"]
    assert int(values["MAX_NUM_SEQS"]) == serving["max_num_seqs"]
    assert int(values["MAX_NUM_BATCHED_TOKENS"]) == serving[
        "max_num_batched_tokens"
    ]


def test_pair_env_explains_host_cache_mapping() -> None:
    source = TEMPLATE.read_text(encoding="utf-8")
    assert "CACHE_HOST_PATH=/cache/test" in source
    assert "/cache/test/jit/triton" in source
    assert "/cache/test/jit/tilelang" in source
    assert "/cache/test/jit/b12x-cute" in source
    assert "/cache/test/nccl-fr" in source
    assert "Leave the container-internal cache and recorder paths" in source
    assert "configure the host location only through" in source


@pytest.mark.parametrize("template", [TEMPLATE, CYCLE_TEMPLATE])
def test_deepseek_templates_persist_native_jit_and_flight_recorder_data(
    template: Path,
) -> None:
    values = _env_values(template)

    assert values["TILELANG_CACHE_DIR"] == "/cache/jit/tilelang"
    assert values["B12X_CUTE_COMPILE_CACHE_DIR"] == "/cache/jit/b12x-cute"
    assert values["TORCH_NCCL_TRACE_BUFFER_SIZE"] == "2000"
    assert values["TORCH_NCCL_DUMP_ON_TIMEOUT"] == "1"
    assert values["TORCH_NCCL_ENABLE_MONITORING"] == "1"
    assert values["TORCH_FR_DUMP_TEMP_FILE"] == (
        "/cache/nccl-fr/comm_lib_trace_rank_"
    )
    assert values["TORCH_NCCL_DEBUG_INFO_PIPE_FILE"] == "/tmp/fr_dump_pipe_"


@pytest.mark.parametrize("launcher", [LAUNCHER, CYCLE_LAUNCHER])
def test_deepseek_launchers_prepare_persistent_cache_directories(
    launcher: Path,
) -> None:
    source = launcher.read_text(encoding="utf-8")

    assert '"$CACHE_HOST_PATH/jit/tilelang"' in source
    assert '"$CACHE_HOST_PATH/jit/b12x-cute"' in source
    assert '"$CACHE_HOST_PATH/nccl-fr"' in source


@pytest.mark.parametrize(
    ("launcher", "environment_fixture"),
    [(LAUNCHER, "pair_env"), (CYCLE_LAUNCHER, "cycle_env")],
)
def test_deepseek_launchers_create_persistent_cache_directories_before_run(
    launcher: Path,
    environment_fixture: str,
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> None:
    env_file = request.getfixturevalue(environment_fixture)
    cache = tmp_path / "cache"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker = fake_bin / "docker"
    cache_for_bash = shlex.quote(_bash_path(cache))
    docker.write_text(
        f"""#!/bin/sh
case "$1 $2" in
  "image inspect") exit 0 ;;
  "container inspect") exit 1 ;;
  "run -d")
    cache={cache_for_bash}
    test -d "$cache/jit/tilelang" || exit 91
    test -d "$cache/jit/b12x-cute" || exit 92
    test -d "$cache/nccl-fr" || exit 93
    printf '%s\n' fake-container-id
    ;;
  *) exit 94 ;;
esac
""",
        encoding="utf-8",
        newline="\n",
    )
    docker.chmod(0o755)

    result = _run_launcher(
        env_file,
        mode="--run",
        launcher=launcher,
        path_prefix=fake_bin,
    )

    assert result.returncode == 0, result.stderr
    assert "fake-container-id" in result.stdout
    assert (cache / "jit" / "tilelang").is_dir()
    assert (cache / "jit" / "b12x-cute").is_dir()
    assert (cache / "nccl-fr").is_dir()


def test_cycle_env_defaults_match_recipe() -> None:
    values = _env_values(CYCLE_TEMPLATE)
    recipe = json.loads(CYCLE_RECIPE.read_text(encoding="utf-8"))
    assert recipe["runtime"]["launcher"] == "scripts/deepseek_v4_cycle_serve.sh"
    serving = recipe["serving"]
    assert int(values["NUM_SPECULATIVE_TOKENS"]) == serving["speculation"][
        "num_speculative_tokens"
    ]
    assert int(values["MAX_MODEL_LEN"]) == serving["max_model_len"]
    assert int(values["MAX_NUM_SEQS"]) == serving["max_num_seqs"]
    assert int(values["MAX_NUM_BATCHED_TOKENS"]) == serving[
        "max_num_batched_tokens"
    ]


def test_launchers_pin_the_hardened_image_from_the_runtime_lock() -> None:
    lock = json.loads(RUNTIME_LOCK.read_text(encoding="utf-8"))
    image = lock["deepseek_v4_flash_0731_hardened_serving_image"]
    expected = f"{image['repository']}@{image['manifest_digest']}"

    assert _launcher_image(LAUNCHER) == expected
    assert _launcher_image(CYCLE_LAUNCHER) == expected


def test_cycle_check_renders_operator_settings(cycle_env: Path) -> None:
    result = _run_launcher(cycle_env, launcher=CYCLE_LAUNCHER)

    assert result.returncode == 0, result.stderr
    assert "--tensor-parallel-size 4" in result.stdout
    assert "--nnodes 4" in result.stdout
    assert "--node-rank 0" in result.stdout
    assert "MAX_MODEL_LEN: 1048576" in result.stdout
    assert "NUM_SPECULATIVE_TOKENS: 5" in result.stdout


def test_cycle_rank_three_is_headless(cycle_env: Path) -> None:
    content = cycle_env.read_text(encoding="utf-8")
    content = content.replace("NODE_RANK=0", "NODE_RANK=3")
    content = content.replace("VLLM_HOST_IP=10.43.0.1", "VLLM_HOST_IP=10.43.0.4")
    cycle_env.write_text(content, encoding="utf-8", newline="\n")

    result = _run_launcher(cycle_env, launcher=CYCLE_LAUNCHER)

    assert result.returncode == 0, result.stderr
    assert "--node-rank 3" in result.stdout
    assert "--headless" in result.stdout


def test_check_renders_every_operator_setting(pair_env: Path) -> None:
    content = pair_env.read_text(encoding="utf-8")
    content = content.replace("API_PORT=8000", "API_PORT=8123")
    content = content.replace(
        "NUM_SPECULATIVE_TOKENS=5", "NUM_SPECULATIVE_TOKENS=7"
    )
    content = content.replace("MAX_MODEL_LEN=1048576", "MAX_MODEL_LEN=262144")
    content = content.replace("MAX_NUM_SEQS=32", "MAX_NUM_SEQS=12")
    content = content.replace(
        "MAX_NUM_BATCHED_TOKENS=4096", "MAX_NUM_BATCHED_TOKENS=8192"
    )
    pair_env.write_text(content, encoding="utf-8", newline="\n")

    result = _run_launcher(pair_env)

    assert result.returncode == 0, result.stderr
    assert "MAX_MODEL_LEN: 262144" in result.stdout
    assert "MAX_NUM_SEQS: 12" in result.stdout
    assert "MAX_NUM_BATCHED_TOKENS: 8192" in result.stdout
    assert "NUM_SPECULATIVE_TOKENS: 7" in result.stdout
    assert "--max-model-len 262144" in result.stdout
    assert "--max-num-seqs 12" in result.stdout
    assert "--max-num-batched-tokens 8192" in result.stdout
    assert "num_speculative_tokens" in result.stdout
    assert "--port 8123" in result.stdout
    assert f"{_bash_path(pair_env.parent / 'model')}" in result.stdout
    assert f"{_bash_path(pair_env.parent / 'cache')}" in result.stdout


def test_env_file_alone_defaults_to_check(pair_env: Path) -> None:
    result = _run_launcher(pair_env, mode=None)

    assert result.returncode == 0, result.stderr
    assert "Local rank input checks passed." in result.stdout


def test_rank_one_is_headless(pair_env: Path) -> None:
    content = pair_env.read_text(encoding="utf-8").replace("NODE_RANK=0", "NODE_RANK=1")
    content = content.replace("VLLM_HOST_IP=10.42.0.1", "VLLM_HOST_IP=10.42.0.2")
    pair_env.write_text(content, encoding="utf-8", newline="\n")

    result = _run_launcher(pair_env)

    assert result.returncode == 0, result.stderr
    assert "--node-rank 1" in result.stdout
    assert "--headless" in result.stdout


def test_unresolved_template_fails_before_launch(tmp_path: Path) -> None:
    env_file = tmp_path / "rank.env"
    env_file.write_text(TEMPLATE.read_text(encoding="utf-8"), encoding="utf-8")

    result = _run_launcher(env_file)

    assert result.returncode != 0
    assert "unresolved placeholders" in result.stderr


def test_invalid_serving_integer_fails(pair_env: Path) -> None:
    content = pair_env.read_text(encoding="utf-8").replace(
        "MAX_NUM_SEQS=32", "MAX_NUM_SEQS=0"
    )
    pair_env.write_text(content, encoding="utf-8", newline="\n")

    result = _run_launcher(pair_env)

    assert result.returncode != 0
    assert "MAX_NUM_SEQS must be greater than zero" in result.stderr


def test_rank_zero_requires_its_own_fabric_address(pair_env: Path) -> None:
    content = pair_env.read_text(encoding="utf-8").replace(
        "MASTER_ADDR=10.42.0.1", "MASTER_ADDR=10.42.0.9"
    )
    pair_env.write_text(content, encoding="utf-8", newline="\n")

    result = _run_launcher(pair_env)

    assert result.returncode != 0
    assert "rank-0 MASTER_ADDR must equal rank-0 VLLM_HOST_IP" in result.stderr


def test_launcher_uses_host_ipc_and_16g_shm_declaration() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "--ipc host" in source
    assert "--shm-size 16g" in source
