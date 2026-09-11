"""Offline contracts for the DeepSeek-V4.1-Flash four-Spark cycle launcher."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "deepseek_v41_cycle_serve.sh"
TEMPLATE = ROOT / "scripts" / "config" / "deepseek-v41-flash-cycle.env.example"
RECIPE = ROOT / "recipes" / "deepseek-v41-flash-cycle.json"
PATCHES = ROOT / "runtime" / "deepseek-v41-gb10" / "patches"
RECEIPT = ROOT / "runtime" / "deepseek-v41-gb10" / "image-receipt.json"

requires_bash = pytest.mark.skipif(os.name == "nt", reason="bash launcher contract")


def _run(env_file: Path, mode: str = "--check") -> subprocess.CompletedProcess[str]:
    return subprocess.run(["bash", str(LAUNCHER), mode, str(env_file)], text=True, capture_output=True, check=False)


def _env_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def _resolved_env(tmp_path: Path, rank: int = 0, **overrides: str) -> Path:
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    (model / "config.json").write_text(json.dumps({"architectures": ["DeepseekV41ForCausalLM"]}), encoding="utf-8")
    (model / "model-00048-of-00048.safetensors").write_bytes(b"")
    cache = tmp_path / "cache"
    cache.mkdir(exist_ok=True)
    nccl = tmp_path / "libnccl.so.2"
    nccl.write_bytes(b"\x7fELF")
    values = _env_values(TEMPLATE)
    values.update(
        {
            "NODE_RANK": str(rank),
            "MASTER_ADDR": "203.0.113.10",
            "VLLM_HOST_IP": "203.0.113.10" if rank == 0 else f"203.0.113.{10 + rank}",
            "MODEL_HOST_PATH": str(model),
            "CACHE_HOST_PATH": str(cache),
            "PATCH_DIR": str(PATCHES),
            "NCCL_SO_HOST_PATH": str(nccl),
            "IMAGE_ID": "sha256:" + "0" * 64,
            "NCCL_SOCKET_IFNAME": "eth0",
            "GLOO_SOCKET_IFNAME": "eth0",
            "NCCL_IB_GID_INDEX": "3",
        }
    )
    values.update(overrides)
    env_file = tmp_path / f"rank-{rank}.env"
    env_file.write_text("".join(f"{k}={v}\n" for k, v in values.items()), encoding="utf-8")
    return env_file


@requires_bash
def test_template_placeholders_are_rejected() -> None:
    result = _run(TEMPLATE)
    assert result.returncode == 20
    assert "unresolved placeholders" in result.stderr


def test_patch_manifest_matches_md5sums() -> None:
    listed = {line.split()[1]: line.split()[0] for line in (PATCHES / "MD5SUMS").read_text().splitlines() if line.strip()}
    mounted = [line.split()[0] for line in (PATCHES / "mounts.txt").read_text().splitlines() if line.strip()]
    assert len(mounted) == 7
    for name in mounted:
        digest = hashlib.md5((PATCHES / name).read_bytes()).hexdigest()  # noqa: S324 - integrity pin, not security
        assert listed[name] == digest, name


@requires_bash
def test_check_renders_the_recipe_contract(tmp_path: Path) -> None:
    recipe = json.loads(RECIPE.read_text(encoding="utf-8"))
    serving = recipe["serving"]
    result = _run(_resolved_env(tmp_path))
    assert result.returncode == 0, result.stderr
    rendered = result.stdout.splitlines()[-1]
    argv = shlex.split(rendered)
    joined = " ".join(argv)
    assert "--tensor-parallel-size 4" in joined and "--nnodes 4" in joined
    assert f"--max-model-len {serving['max_model_len']}" in joined
    assert f"--max-num-seqs {serving['max_num_seqs']}" in joined
    assert f"--max-num-batched-tokens {serving['max_num_batched_tokens']}" in joined
    assert f"--gpu-memory-utilization {serving['gpu_memory_utilization']:.2f}" in joined
    assert f"--block-size {serving['block_size']}" in joined
    assert f"--load-format {serving['load_format']}" in joined
    assert "--served-model-name deepseek-v4.1-flash" in joined
    assert '{"cpu_offload": false}' in argv[argv.index("--engram-config") + 1]
    spec = json.loads(argv[argv.index("--speculative-config") + 1])
    assert spec == {
        "method": "dspark",
        "num_speculative_tokens": serving["speculation"]["num_speculative_tokens"],
        "draft_sample_method": serving["speculation"]["draft_sample_method"],
        "rejection_sample_method": "block",
        "enable_adaptive_verification": False,
    }
    graphs = json.loads(argv[argv.index("--compilation-config") + 1])
    assert graphs["cudagraph_mode"] == "FULL_AND_PIECEWISE"
    assert graphs["cudagraph_capture_sizes"] == sorted({*range(5, 41, 5), *range(6, 49, 6)})
    assert "-e LD_PRELOAD=/opt/sparkring/nccl/libnccl.so.2" in joined
    assert "-e VLLM_NCCL_SO_PATH=/opt/sparkring/nccl/libnccl.so.2" in joined
    assert "-e DSV41_ENGRAM_DISK=1" in joined
    assert "-e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0" in joined
    assert "-e DSV41_ENGRAM_BALANCED=1" in joined and "-e DSV41_ENGRAM_PACKED_DIR=/cache/engram-packed" in joined
    assert "-e VLLM_USE_BREAKABLE_CUDAGRAPH=1" in joined
    assert "--tool-call-parser deepseek_v41" in joined and "--reasoning-parser deepseek_v41" in joined
    assert '--limit-mm-per-prompt {"image":4}' in joined
    assert joined.count(":ro") >= 9  # model, nccl, seven patches
    assert "--headless" not in joined


@requires_bash
def test_api_key_file_passes_every_key(tmp_path: Path) -> None:
    keys = tmp_path / "keys"
    keys.write_text("k-one\n\nk-two\n", encoding="utf-8")
    result = _run(_resolved_env(tmp_path, API_KEY_FILE=str(keys)))
    assert result.returncode == 0, result.stderr
    assert "--api-key k-one k-two" in result.stdout
    bare = _run(_resolved_env(tmp_path))
    assert bare.returncode == 0, bare.stderr
    assert "--api-key" not in bare.stdout
    empty = tmp_path / "empty"
    empty.write_text("\n", encoding="utf-8")
    result = _run(_resolved_env(tmp_path, API_KEY_FILE=str(empty)))
    assert result.returncode != 0
    assert "has no keys" in result.stderr


@requires_bash
def test_worker_ranks_are_headless_and_eager_drops_graphs(tmp_path: Path) -> None:
    result = _run(_resolved_env(tmp_path, rank=2, ENFORCE_EAGER="1", TEXT_ONLY="1"))
    assert result.returncode == 0, result.stderr
    joined = result.stdout.splitlines()[-1]
    assert "--headless" in joined
    assert "--enforce-eager" in joined and "--compilation-config" not in joined
    assert "--language-model-only" in joined and "--limit-mm-per-prompt" not in joined


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"NUM_SPECULATIVE_TOKENS": "4"}, "dspark_block_size"),
        ({"NCCL_SWITCHLESS_RING_ONLY": "0"}, "NCCL_SWITCHLESS_RING_ONLY"),
        ({"NCCL_IB_HCA": "rocep1s0f0"}, "exactly two RoCE devices"),
        ({"GLOO_SOCKET_IFNAME": "eth1"}, "must match"),
        ({"NODE_RANK": "0", "VLLM_HOST_IP": "203.0.113.11"}, "rank-0 MASTER_ADDR"),
    ],
)
@requires_bash
def test_contract_violations_fail_closed(tmp_path: Path, override: dict[str, str], message: str) -> None:
    result = _run(_resolved_env(tmp_path, **override))
    assert result.returncode == 20
    assert message in result.stderr


def test_recipe_and_receipt_agree_on_identities() -> None:
    recipe = json.loads(RECIPE.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert recipe["model"]["revision"] == "dba1be0a40aa45a94ad051997016db3960a90277"
    assert receipt["vllm_commit"] in recipe["runtime"]["image_note"]
    assert receipt["flashinfer_commit"] in recipe["runtime"]["image_note"]
    assert receipt["base_image"] in recipe["runtime"]["image_note"]
    listed = {line.split()[1]: line.split()[0] for line in (PATCHES / "MD5SUMS").read_text().splitlines() if line.strip()}
    assert receipt["patch_md5"] == listed
