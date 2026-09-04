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
) -> subprocess.CompletedProcess[str]:
    arguments = [str(env_file)] if mode is None else [mode, str(env_file)]
    environment = os.environ.copy()
    sysfs_root = env_file.parent / "infiniband"
    netdev_ipv4_root = env_file.parent / "netdev-ipv4"
    fixture_assignments: list[str] = []
    if sysfs_root.exists():
        environment["SPARKRING_INFINIBAND_SYSFS_ROOT"] = _bash_path(sysfs_root)
        fixture_assignments.append(
            f"SPARKRING_INFINIBAND_SYSFS_ROOT={_bash_path(sysfs_root)}"
        )
    if netdev_ipv4_root.exists():
        environment["SPARKRING_NETDEV_IPV4_ROOT"] = _bash_path(netdev_ipv4_root)
        fixture_assignments.append(
            f"SPARKRING_NETDEV_IPV4_ROOT={_bash_path(netdev_ipv4_root)}"
        )
    if os.name != "nt":
        command = ["bash", str(launcher), *arguments]
    else:
        rendered = " ".join(
            shlex.quote(value)
            for value in (
                "env",
                *fixture_assignments,
                "bash",
                _bash_path(launcher),
                *([_bash_path(env_file)] if mode is None else [mode, _bash_path(env_file)]),
            )
        )
        command = ["bash", "-lc", f"exec {rendered}"]
    return subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
        env=environment,
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


def _ipv4_gid(address: str) -> str:
    octets = [int(value) for value in address.split(".")]
    assert len(octets) == 4
    return "0000:0000:0000:0000:0000:ffff:" + "".join(
        f"{value:02x}" for value in octets[:2]
    ) + ":" + "".join(f"{value:02x}" for value in octets[2:])


def _write_gid(
    root: Path,
    *,
    hca: str,
    port: int,
    index: int,
    address: str,
    netdev: str,
    gid_type: str = "RoCE v2",
    state: str = "4: ACTIVE",
    link_layer: str = "Ethernet",
) -> None:
    port_root = root / hca / "ports" / str(port)
    (port_root / "gids").mkdir(parents=True, exist_ok=True)
    (port_root / "gid_attrs" / "types").mkdir(parents=True, exist_ok=True)
    (port_root / "gid_attrs" / "ndevs").mkdir(parents=True, exist_ok=True)
    (port_root / "state").write_text(
        state + "\n", encoding="utf-8", newline="\n"
    )
    (port_root / "link_layer").write_text(
        link_layer + "\n", encoding="utf-8", newline="\n"
    )
    (port_root / "gids" / str(index)).write_text(
        _ipv4_gid(address) + "\n", encoding="utf-8", newline="\n"
    )
    (port_root / "gid_attrs" / "types" / str(index)).write_text(
        gid_type + "\n", encoding="utf-8", newline="\n"
    )
    (port_root / "gid_attrs" / "ndevs" / str(index)).write_text(
        netdev + "\n", encoding="utf-8", newline="\n"
    )
    netdev_root = root.parent / "netdev-ipv4"
    netdev_root.mkdir(exist_ok=True)
    (netdev_root / netdev).write_text(
        address + "\n", encoding="utf-8", newline="\n"
    )


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
    _write_gid(
        tmp_path / "infiniband",
        hca="rocep1s0f0",
        port=1,
        index=3,
        address="10.42.0.1",
        netdev="fabric0",
    )
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
    _write_gid(
        tmp_path / "infiniband",
        hca="rocep1s0f0",
        port=1,
        index=3,
        address="10.43.0.1",
        netdev="fabric0",
    )
    _write_gid(
        tmp_path / "infiniband",
        hca="rocep1s0f1",
        port=1,
        index=5,
        address="10.44.0.1",
        netdev="fabric1",
    )
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
    assert values["NCCL_IB_GID_AUTO"] == "1"
    assert values["NCCL_IB_GID_INDEX"] == ""


def test_pair_env_explains_host_cache_mapping() -> None:
    source = TEMPLATE.read_text(encoding="utf-8")
    assert "CACHE_HOST_PATH=/cache/test" in source
    assert "/cache/test/jit/triton" in source
    assert "Leave XDG_CACHE_HOME, TRITON_CACHE_DIR, and VLLM_CACHE_ROOT unchanged" in source
    assert "configure the host location only through" in source


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
    assert values["NCCL_IB_GID_AUTO"] == "1"
    assert values["NCCL_IB_GID_INDEX"] == ""


def test_launchers_pin_the_hardened_image_from_the_runtime_lock() -> None:
    lock = json.loads(RUNTIME_LOCK.read_text(encoding="utf-8"))
    image = lock["deepseek_v4_flash_0731_hardened_serving_image"]
    expected = f"{image['repository']}@{image['manifest_digest']}"

    assert _launcher_image(LAUNCHER) == expected
    assert _launcher_image(CYCLE_LAUNCHER) == expected


def test_cycle_check_renders_operator_settings(cycle_env: Path) -> None:
    result = _run_launcher(cycle_env, launcher=CYCLE_LAUNCHER)

    assert result.returncode == 0, result.stderr
    assert "rocep1s0f0:1 usable RoCEv2/IPv4 indexes: 3" in result.stdout
    assert "rocep1s0f1:1 usable RoCEv2/IPv4 indexes: 5" in result.stdout
    assert "NCCL_IB_GID_INDEX will be unset in the container" in result.stdout
    assert "--env NCCL_IB_GID_INDEX" in result.stdout
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


def test_auto_gid_mode_ignores_a_stale_pin(pair_env: Path) -> None:
    content = pair_env.read_text(encoding="utf-8").replace(
        "NCCL_IB_GID_INDEX=", "NCCL_IB_GID_INDEX=17"
    )
    pair_env.write_text(content, encoding="utf-8", newline="\n")

    result = _run_launcher(pair_env)

    assert result.returncode == 0, result.stderr
    assert "ignoring configured NCCL_IB_GID_INDEX=17" in result.stdout
    assert "--env NCCL_IB_GID_INDEX" in result.stdout


def test_auto_gid_mode_is_the_default_when_switch_is_omitted(pair_env: Path) -> None:
    content = pair_env.read_text(encoding="utf-8").replace(
        "NCCL_IB_GID_AUTO=1\n", ""
    )
    pair_env.write_text(content, encoding="utf-8", newline="\n")

    result = _run_launcher(pair_env)

    assert result.returncode == 0, result.stderr
    assert "automatic validation passed" in result.stdout


def test_pinned_gid_escape_hatch_preserves_the_configured_index(pair_env: Path) -> None:
    content = pair_env.read_text(encoding="utf-8")
    content = content.replace("NCCL_IB_GID_AUTO=1", "NCCL_IB_GID_AUTO=0")
    content = content.replace("NCCL_IB_GID_INDEX=", "NCCL_IB_GID_INDEX=007")
    pair_env.write_text(content, encoding="utf-8", newline="\n")

    result = _run_launcher(pair_env)

    assert result.returncode == 0, result.stderr
    assert "pinned NCCL_IB_GID_INDEX=007" in result.stdout
    assert "--env NCCL_IB_GID_INDEX" not in result.stdout


def test_pinned_gid_escape_hatch_requires_a_decimal_index(pair_env: Path) -> None:
    content = pair_env.read_text(encoding="utf-8")
    content = content.replace("NCCL_IB_GID_AUTO=1", "NCCL_IB_GID_AUTO=0")
    pair_env.write_text(content, encoding="utf-8", newline="\n")

    result = _run_launcher(pair_env)

    assert result.returncode != 0
    assert "requires a decimal NCCL_IB_GID_INDEX" in result.stderr


def test_auto_gid_mode_rejects_a_selected_member_without_rocev2(cycle_env: Path) -> None:
    gid_type = (
        cycle_env.parent
        / "infiniband"
        / "rocep1s0f1"
        / "ports"
        / "1"
        / "gid_attrs"
        / "types"
        / "5"
    )
    gid_type.write_text("RoCE v1\n", encoding="utf-8", newline="\n")

    result = _run_launcher(cycle_env, launcher=CYCLE_LAUNCHER)

    assert result.returncode != 0
    assert "rocep1s0f0:1 usable RoCEv2/IPv4 indexes: 3" in result.stdout
    assert "rocep1s0f1:1 usable RoCEv2/IPv4 indexes: none" in result.stderr
    assert "automatic RoCEv2 GID validation failed" in result.stderr


def test_auto_gid_mode_rejects_a_rocev2_gid_without_a_current_ipv4_match(
    pair_env: Path,
) -> None:
    (pair_env.parent / "netdev-ipv4" / "fabric0").write_text(
        "10.42.0.9\n", encoding="utf-8", newline="\n"
    )
    content = pair_env.read_text(encoding="utf-8").replace(
        "VLLM_HOST_IP=10.42.0.1", "VLLM_HOST_IP=10.42.0.8"
    )
    content = content.replace("NODE_RANK=0", "NODE_RANK=1")
    pair_env.write_text(content, encoding="utf-8", newline="\n")

    result = _run_launcher(pair_env)

    assert result.returncode != 0
    assert "RoCE v2 indexes seen: 3" in result.stderr


def test_run_fails_gid_validation_before_docker_is_consulted(pair_env: Path) -> None:
    gid_type = (
        pair_env.parent
        / "infiniband"
        / "rocep1s0f0"
        / "ports"
        / "1"
        / "gid_attrs"
        / "types"
        / "3"
    )
    gid_type.write_text("RoCE v1\n", encoding="utf-8", newline="\n")

    result = _run_launcher(pair_env, mode="--run")

    assert result.returncode != 0
    assert "automatic RoCEv2 GID validation failed" in result.stderr
    assert "docker is unavailable" not in result.stderr


def test_selector_supports_exact_port_rail_and_plane_fields(cycle_env: Path) -> None:
    _write_gid(
        cycle_env.parent / "infiniband",
        hca="rocep1s0f00",
        port=1,
        index=9,
        address="10.46.0.1",
        netdev="fabric9",
    )
    content = cycle_env.read_text(encoding="utf-8").replace(
        "NCCL_IB_HCA=rocep1s0f0,rocep1s0f1",
        "NCCL_IB_HCA==rocep1s0f0:1:0:0,rocep1s0f1::1:1",
    )
    cycle_env.write_text(content, encoding="utf-8", newline="\n")

    result = _run_launcher(cycle_env, launcher=CYCLE_LAUNCHER)

    assert result.returncode == 0, result.stderr
    assert "rocep1s0f0:1 usable RoCEv2/IPv4 indexes: 3" in result.stdout
    assert "rocep1s0f1:1 usable RoCEv2/IPv4 indexes: 5" in result.stdout


def test_selector_supports_exact_exclusion(cycle_env: Path) -> None:
    _write_gid(
        cycle_env.parent / "infiniband",
        hca="rocep1s0f2",
        port=1,
        index=7,
        address="10.45.0.1",
        netdev="fabric2",
    )
    content = cycle_env.read_text(encoding="utf-8").replace(
        "NCCL_IB_HCA=rocep1s0f0,rocep1s0f1", "NCCL_IB_HCA=^=rocep1s0f2"
    )
    cycle_env.write_text(content, encoding="utf-8", newline="\n")

    result = _run_launcher(cycle_env, launcher=CYCLE_LAUNCHER)

    assert result.returncode == 0, result.stderr
    assert "rocep1s0f0:1 usable RoCEv2/IPv4 indexes: 3" in result.stdout
    assert "rocep1s0f1:1 usable RoCEv2/IPv4 indexes: 5" in result.stdout
    assert "rocep1s0f2:1" not in result.stdout


def test_selector_prefix_matching_cannot_silently_widen_pair(pair_env: Path) -> None:
    _write_gid(
        pair_env.parent / "infiniband",
        hca="rocep1s0f00",
        port=1,
        index=9,
        address="10.46.0.1",
        netdev="fabric9",
    )

    result = _run_launcher(pair_env)

    assert result.returncode != 0
    assert "must resolve to exactly one active HCA/port; resolved 2" in result.stderr


def test_selector_ignores_entries_after_nccls_32_entry_cap(pair_env: Path) -> None:
    ignored_names = ",".join(f"missing{index}" for index in range(32))
    selector = ignored_names + ",rocep1s0f0"
    content = pair_env.read_text(encoding="utf-8").replace(
        "NCCL_IB_HCA=rocep1s0f0", f"NCCL_IB_HCA={selector}"
    )
    pair_env.write_text(content, encoding="utf-8", newline="\n")

    result = _run_launcher(pair_env)

    assert result.returncode != 0
    assert "first 32 non-empty NCCL_IB_HCA entries" in result.stderr
    assert "matched no active RDMA HCA/port" in result.stderr


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
