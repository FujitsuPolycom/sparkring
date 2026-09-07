"""Validate resolved Compose settings without contacting Docker or model hosts."""

import argparse
import json
from pathlib import Path
import shlex
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def validate(document, rank, nccl_library):
    profile = json.loads((HERE / "profile.json").read_text())
    recipe = json.loads((ROOT / "recipes/deepseek-v4-flash-vision-exp-tp4.json").read_text())
    service = document["services"]["vllm-dspark"]
    env = service["environment"]
    expected = {
        "NODE_RANK": str(rank), "HEADLESS": "" if rank == 0 else "1",
        "NNODES": "4", "TP_SIZE": "4", "MTP_NUM_TOKENS": "5",
        "DSPARK_MODEL": profile["model"]["repository"],
        "DSPARK_REVISION": profile["model"]["revision"],
        "DSPARK_ENABLE_DSPARK_BLOCK_K": "1", "DSPARK_MAX_INFLIGHT_PREFILLS": "2",
        "DSPARK_ENABLE_SP_INDEXER": "1", "DEFAULT_THINKING": "off",
        "LD_PRELOAD": profile["transport"]["container_path"],
        "VLLM_NCCL_SO_PATH": profile["transport"]["container_path"],
    }
    expected.update({key: value for key, value in recipe["serving"]["nccl"].items()
                     if not value.startswith("<")})
    for key, value in expected.items():
        if str(env.get(key, "")) != value:
            raise ValueError(f"Resolved setting differs from the profile: {key}")
    if service["image"] != profile["image"]["reference"] or service["restart"] != "no":
        raise ValueError("Resolved image or restart policy differs from the profile")
    if any("REPLACE_" in str(value) for value in env.values()):
        raise ValueError("Resolved configuration contains an unreplaced template value")
    if not env.get("VLLM_API_KEY"):
        raise ValueError("Configure VLLM_API_KEY before exposing the API")
    for key in ("MASTER_ADDR", "VLLM_HOST_IP", "NCCL_SOCKET_IFNAME", "TP_SOCKET_IFNAME",
                "GLOO_SOCKET_IFNAME", "NCCL_IB_HCA", "NCCL_IB_GID_INDEX"):
        if not env.get(key):
            raise ValueError(f"Configure the rank-specific setting: {key}")
    command = "\n".join(service["command"])
    marker = "exec /usr/local/bin/vllm serve "
    if command.count(marker) != 1:
        raise ValueError("Resolved serving command differs from the pinned recipe")
    args = shlex.split(command.split(marker, 1)[1])
    if args[0] != profile["model"]["repository"]:
        raise ValueError("Resolved model differs from the profile")
    for flag, value in {
        "--served-model-name": recipe["serving"]["served_model_name"],
        "--tensor-parallel-size": "4", "--nnodes": "4", "--node-rank": str(rank),
        "--max-model-len": "1048576", "--max-num-seqs": "48",
        "--max-num-batched-tokens": "12288", "--gpu-memory-utilization": "0.80",
        "--long-prefill-token-threshold": "1024",
    }.items():
        if args.count(flag) != 1:
            raise ValueError(f"Resolved serving argument differs: {flag}")
        actual = args[args.index(flag) + 1]
        equal = float(actual) == float(value) if flag == "--gpu-memory-utilization" else actual == value
        if not equal:
            raise ValueError(f"Resolved serving argument differs: {flag}")
    if ("--headless" in args) != (rank != 0):
        raise ValueError("Headless mode differs from the selected rank")
    mounts = [row for row in service["volumes"]
              if row["target"] == profile["transport"]["container_path"]]
    if (len(mounts) != 1 or not mounts[0].get("read_only")
            or mounts[0].get("bind", {}).get("create_host_path") is not False
            or Path(mounts[0]["source"]).resolve() != Path(nccl_library).resolve()):
        raise ValueError("NCCL mount differs from the verified library path or read-only contract")
    return {"configuration_validated": True, "rank": rank, "image": service["image"],
            "model": recipe["serving"]["served_model_name"], "hardware_validated": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, choices=range(4), required=True)
    parser.add_argument("--nccl-library", type=Path, required=True)
    options = parser.parse_args()
    try:
        result = validate(json.load(sys.stdin), options.rank, options.nccl_library)
    except (KeyError, ValueError, IndexError, TypeError) as error:
        parser.exit(1, f"Configuration rejected: {error}\n")
    print(json.dumps(result))
