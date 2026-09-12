"""Offline checks for the pinned Vision-Exp cycle; never start containers."""

import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "runtime/deepseek-vision-exp"
PROFILE = json.loads((HERE / "profile.json").read_text())
RECIPE = json.loads((ROOT / "recipes/deepseek-v4-flash-vision-exp-tp4.json").read_text())
SPEC = importlib.util.spec_from_file_location("vision_compose_check", HERE / "check_compose.py")
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


def template():
    values = {}
    for line in (HERE / "cycle.env.example").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        assert key not in values
        assert not re.search(r"\s+[A-Z_][A-Z_0-9]*=", value), line
        values[key] = value
    return values


def test_artifacts_and_fixture_are_pinned():
    for ref in (PROFILE["image"]["reference"], PROFILE["transport"]["donor_image"]):
        assert re.fullmatch(r"ghcr.io/[\w/-]+@sha256:[0-9a-f]{64}", ref)
    assert PROFILE["platform"] == "linux/arm64"
    assert re.fullmatch(r"[0-9a-f]{40}", PROFILE["upstream_recipe"]["revision"])
    raw = gzip.decompress((HERE / "upstream/docker-compose.dspark.yml.gz").read_bytes())
    assert hashlib.sha256(raw).hexdigest() == PROFILE["upstream_recipe"]["compose_sha256"]
    patch = (ROOT / PROFILE["transport"]["patch"]).read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(patch).hexdigest() == PROFILE["transport"]["patch_sha256"]
    assert "MIT License" in (HERE / "upstream/LICENSE").read_text()


def test_template_matches_recipe_and_required_upstream_unlock():
    values = template()
    serving = RECIPE["serving"]
    assert values["DSPARK_VLLM_IMAGE"] == PROFILE["image"]["reference"]
    assert values["DSPARK_MODEL"] == RECIPE["model"]["repository"] == PROFILE["model"]["repository"]
    assert values["DSPARK_REVISION"] == RECIPE["model"]["revision"] == PROFILE["model"]["revision"]
    assert values["SERVED_MODEL_NAME"] == serving["served_model_name"]
    for key, field in (("NNODES", "node_count"), ("TP_SIZE", "tensor_parallel_size"),
                       ("MAX_NUM_SEQS", "max_num_seqs"), ("MAX_NUM_BATCHED_TOKENS", "max_num_batched_tokens"),
                       ("DSPARK_MAX_INFLIGHT_PREFILLS", "in_flight_prefills")):
        assert int(values[key]) == serving[field]
    assert values["DSPARK_ENABLE_DSPARK_BLOCK_K"] == "1"
    assert int(values["MTP_NUM_TOKENS"]) == serving["speculation"]["num_speculative_tokens"] == 5
    assert values["DSPARK_ENABLE_SP_INDEXER"] == "1"
    assert values["DEFAULT_THINKING"] == serving["default_thinking"]
    assert PROFILE["overlay"]["flashinfer_package_overlay"] is False
    assert PROFILE["status"] == RECIPE["status"] == "research-only"


@pytest.mark.parametrize("rank", range(4))
def test_compose_resolves_each_rank_without_launching(tmp_path, rank):
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("Docker Compose CLI is unavailable; no daemon or hardware is required")
    version = subprocess.run([docker, "compose", "version"], capture_output=True, timeout=20)
    if version.returncode:
        pytest.skip("Docker Compose plugin is unavailable")
    values = template()
    values.update({
        "NODE_RANK": str(rank), "HEADLESS": "" if rank == 0 else "1",
        "MASTER_ADDR": "192.0.2.10", "VLLM_HOST_IP": f"192.0.2.{10 + rank}",
        "NCCL_SOCKET_IFNAME": "mgmt0", "TP_SOCKET_IFNAME": "mgmt0", "GLOO_SOCKET_IFNAME": "mgmt0",
        "NCCL_IB_HCA": "fabric0,fabric1", "NCCL_IB_GID_INDEX": "3",
        "HF_CACHE": str(tmp_path / "hf"), "DSPARK_TMP_HOST": str(tmp_path / "compiler"),
        "SPARKRING_NCCL_LIBRARY": str(tmp_path / "libnccl.so.2.30.7"),
        "VLLM_API_KEY": "test-only",
    })
    assert not any("REPLACE_" in value for value in values.values())
    library_path = Path(values["SPARKRING_NCCL_LIBRARY"])
    library_path.write_bytes(b"Offline path fixture; not an executable library")
    env_file = tmp_path / "rank.env"
    env_file.write_text("".join(f"{key}={value}\n" for key, value in values.items()))
    compose = tmp_path / "docker-compose.dspark.yml"
    compose.write_bytes(gzip.decompress((HERE / "upstream/docker-compose.dspark.yml.gz").read_bytes()))
    # Shell variables override --env-file. Exclude test-controlled settings so
    # the developer's ambient model configuration cannot affect this check.
    environment = {key: value for key, value in os.environ.items() if key not in values}
    result = subprocess.run([
        docker, "compose", "--env-file", str(env_file), "-f", str(compose),
        "-f", str(HERE / "compose.override.yml"), "config", "--format", "json",
    ], capture_output=True, text=True, env=environment, timeout=30)
    assert result.returncode == 0, result.stderr
    service = json.loads(result.stdout)["services"]["vllm-dspark"]
    actual = service["environment"]
    assert service["image"] == PROFILE["image"]["reference"]
    assert service["restart"] == "no"
    for key in ("NODE_RANK", "HEADLESS", "NNODES", "TP_SIZE", "DSPARK_REVISION",
                "DSPARK_ENABLE_DSPARK_BLOCK_K", "DSPARK_MAX_INFLIGHT_PREFILLS", "DSPARK_ENABLE_SP_INDEXER"):
        assert actual[key] == values[key]
    assert actual["NCCL_SWITCHLESS_RING_ONLY"] == "1"
    assert actual["LD_PRELOAD"] == actual["VLLM_NCCL_SO_PATH"] == PROFILE["transport"]["container_path"]
    mounts = [row for row in service["volumes"] if row["target"] == PROFILE["transport"]["container_path"]]
    assert len(mounts) == 1 and mounts[0]["read_only"]
    assert mounts[0].get("bind", {}).get("create_host_path", False) is False
    command = "\n".join(service["command"])
    assert "--served-model-name deepseek-v4-flash-vision-exp" in command
    assert "--tensor-parallel-size 4" in command and "--nnodes 4" in command
    assert "--max-num-seqs 48" in command and "--max-num-batched-tokens 12288" in command
    assert ("--headless" in command) == (rank != 0)
    document = json.loads(result.stdout)
    assert CHECKER.validate(document, rank, values["SPARKRING_NCCL_LIBRARY"])["configuration_validated"]
    library_mount = next(row for row in document["services"]["vllm-dspark"]["volumes"]
                         if row["target"] == PROFILE["transport"]["container_path"])
    # Legacy Compose omits false; the explicit and omitted forms must agree.
    library_mount.setdefault("bind", {}).pop("create_host_path", None)
    assert CHECKER.validate(document, rank, library_path)["configuration_validated"]
    library_mount["bind"]["create_host_path"] = True
    with pytest.raises(ValueError, match="NCCL mount differs"):
        CHECKER.validate(document, rank, library_path)
    library_mount["bind"].pop("create_host_path")
    library_path.unlink()
    with pytest.raises(ValueError, match="existing file"):
        CHECKER.validate(document, rank, library_path)
    library_path.write_bytes(b"Offline path fixture; not an executable library")
    # A shell export overrides --env-file before rendering. Validate the
    # resolved result rather than trusting the file alone.
    for key in ("DSPARK_ENABLE_DSPARK_BLOCK_K", "DSPARK_MAX_INFLIGHT_PREFILLS", "DSPARK_ASYNC_SCHEDULING", "NODE_RANK"):
        original = document["services"]["vllm-dspark"]["environment"][key]
        document["services"]["vllm-dspark"]["environment"][key] = "999"
        with pytest.raises(ValueError, match="Resolved setting differs"):
            CHECKER.validate(document, rank, values["SPARKRING_NCCL_LIBRARY"])
        document["services"]["vllm-dspark"]["environment"][key] = original
    service = document["services"]["vllm-dspark"]
    original_command = service["command"][:]
    for flag, value in (("--kv-cache-dtype", "fp8"), ("--block-size", "128"),
                        ("--moe-backend", "other"), ("--tokenizer-mode", "auto"),
                        ("--tool-call-parser", "other"), ("--distributed-executor-backend", "ray")):
        service["command"] = [re.sub(r"(" + re.escape(flag) + r"\s+)\S+", r"\g<1>" + value, text)
                              for text in original_command]
        with pytest.raises(ValueError, match="Resolved serving argument differs"):
            CHECKER.validate(document, rank, library_path)
        service["command"] = original_command[:]
    for duplicate in ("--tensor-parallel-size=2", "--tensor_parallel_size=2", "-tp=2", "-tp 2", "-pp 2"):
        service["command"] = original_command[:]
        service["command"][-1] += " " + duplicate
        with pytest.raises(ValueError, match="Resolved serving argument differs"):
            CHECKER.validate(document, rank, library_path)
    service["command"] = [re.sub(r"--max-model-len\s+\S+", "", text)
                          for text in original_command]
    service["command"][-1] += " --max-model-len"
    with pytest.raises(ValueError, match="--max-model-len has no value"):
        CHECKER.validate(document, rank, library_path)
    service["command"] = original_command
    document["services"]["vllm-dspark"]["image"] = "unreviewed:tag"
    with pytest.raises(ValueError, match="Resolved image"):
        CHECKER.validate(document, rank, values["SPARKRING_NCCL_LIBRARY"])
