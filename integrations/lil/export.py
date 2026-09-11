"""Render a lil image bundle using the SparkRing operator launcher's argument builder."""

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile

from plan import ROOT, keys, read_json, render, require
import mesh_check

LAUNCHER = ROOT / "runtime/glm53-flash-jj-r8-gb10/launch-rank.sh"
FABRIC_KEYS = {
    "HOST_IP",
    "MASTER_ADDR",
    "SOCKET_IFNAME",
    "NCCL_IB_HCA",
    "SPARK_TP4_PEER0",
    "SPARK_TP4_PEER1",
    "SPARK_TP4_DEVICE0",
    "SPARK_TP4_DEVICE1",
    "SPARK_TP4_GID0",
    "SPARK_TP4_GID1",
    "SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_PEER0",
    "SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_PEER1",
    "SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_DEVICE0",
    "SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_DEVICE1",
}
OPTIONAL_FABRIC_KEYS = {
    "NCCL_IB_GID_INDEX",
    "SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_GID0",
    "SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_GID1",
}


def export(descriptor, site, fabric, bundle_id):
    require(
        isinstance(bundle_id, str)
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,79}", bundle_id),
        "bundle id must start with an ASCII letter or digit and contain only ASCII letters, digits or hyphens (maximum 80 characters)",
    )
    require(os.name == "posix", "run the exporter in Linux or WSL")
    plan = render(descriptor, site)
    keys(fabric, ("ranks",))
    require(
        len(fabric["ranks"]) == len(plan["ranks"]),
        "fabric requires one record per rank",
    )
    runtime = plan["resolved_runtime"]
    ranks = []
    for rank, network in zip(plan["ranks"], fabric["ranks"], strict=True):
        keys(network, FABRIC_KEYS, OPTIONAL_FABRIC_KEYS)
        require(
            all(
                isinstance(v, str) and v and not any(c in v for c in "\r\n\x00")
                for v in network.values()
            ),
            "fabric values must be nonempty strings",
        )
        values = dict(network)
        s = rank["settings"]
        mesh_preflight = (
            mesh_check.command(
                rank["rank"], rank["host"],
                runtime["operator_image"]["image_id"], network,
            )
            if s["speculator"] == "mtp" else None
        )
        values.update(
            {
                "IMAGE_REF": rank["image"],
                "IMAGE_ID": runtime["operator_image"]["image_id"],
                "TARGET_MODEL_HOST_PATH": site["storage"]["target"],
                "DFLASH_MODEL_HOST_PATH": site["storage"].get("draft", ""),
                "SPECULATION_METHOD": s["speculator"],
                "TARGET_MODEL_VARIANT": "nvfp4-spark"
                if s["speculator"] == "mtp"
                else "nvfp4",
                "MAX_CUDAGRAPH_CAPTURE_SIZE": str(
                    s["max_num_seqs"] * (s["speculative_tokens"] + 1)
                ),
                "SERVED_MODEL_NAME": "glm-5.3-flash-spark"
                if s["speculator"] == "mtp"
                else "glm-5.3-flash",
                "CACHE_HOST_ROOT": site["storage"]["jit"],
                "CONTAINER_PREFIX": bundle_id,
                "SIRCL_ENABLED": "1",
                "VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL": "1",
                "VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL_RAIL_MODE": "dual",
                "VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL_EXPOSURE": "fused",
                "SPARK_TP4_GRAPH_DIRECT_DOORBELL": "1",
                "SPARKRING_DECLARED_SIRCL_NATIVE_SHA256": runtime["sircl"][
                    "native_sha256"
                ],
                "SPARKRING_DECLARED_SIRCL_MANIFEST_SHA256": runtime["sircl"][
                    "overlay_manifest_sha256"
                ],
                "SPARKRING_PRINT_CONTAINER_SPEC": "1",
                "SPARKRING_OFFLINE_SPEC": "1",
                "SPARKCACHE_ENABLED": "1" if rank["cache"] else "0",
                "DFLASH_WARMUP": "1",
                "SPARKCACHE_ASYNC_PAGE_CAPTURE": "1"
                if rank["cache"] and rank["cache"]["access_mode"] != "restore-only"
                else "0",
                "SPARKCACHE_PUBLICATION_SCHEMA": "tail-cow-v2",
                "SPARKCACHE_CLEAR_ONCE": "",
                "JIT_CACHE_NAMESPACE": bundle_id,
            }
        )
        for key, source in {
            "PORT": "port",
            "TENSOR_PARALLEL_SIZE": "tp",
            "DECODE_CONTEXT_PARALLEL_SIZE": "dcp",
            "MAX_MODEL_LEN": "max_model_len",
            "MAX_NUM_SEQS": "max_num_seqs",
            "MAX_NUM_BATCHED_TOKENS": "max_num_batched_tokens",
            "KV_CACHE_MEMORY_BYTES": "kv_cache_memory_bytes",
            "NUM_SPECULATIVE_TOKENS": "speculative_tokens",
        }.items():
            values[key] = str(s[source])
        if rank["cache"]:
            values["SPARKCACHE_CACHE_NAMESPACE"] = rank["cache"]["namespace"]
            values["SPARKCACHE_ACCESS_MODE"] = rank["cache"]["access_mode"]
        with tempfile.TemporaryDirectory(prefix="sparkring-render-") as tmp:
            config = Path(tmp) / "site.env"
            config.write_text(
                "\n".join(f"{k}={shlex.quote(v)}" for k, v in values.items()) + "\n",
                encoding="utf-8",
            )
            env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "LC_ALL": "C.UTF-8"}
            process = subprocess.run(
                ["bash", str(LAUNCHER), str(rank["rank"]), str(config)],
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )
            require(process.returncode == 0, process.stderr.strip())
            argv = json.loads(process.stdout)["argv"]
        checks = [
            {
                "argv": [
                    "docker",
                    "image",
                    "inspect",
                    "--format",
                    "{{.Id}}",
                    rank["image"],
                ],
                "expected": runtime["operator_image"]["image_id"],
            }
        ]
        for mount in rank["mounts"]:
            checks.append({"argv": ["test", "-d", mount["source"]], "expected": ""})
        if rank["cache"]:
            cache_mount = rank["mounts"][-1]
            argv[2:2] = ["-v", cache_mount["source"] + ":/cache/jit/sparkcache-context"]
        for role, checkpoint in plan["checkpoints"].items():
            for file, digest in (
                ("config.json", checkpoint["config_sha256"]),
                (
                    "model.safetensors.index.json"
                    if role == "target"
                    else "model.safetensors",
                    checkpoint.get(
                        "weight_index_sha256", checkpoint.get("weights_sha256")
                    ),
                ),
            ):
                filename = site["storage"][role] + "/" + file
                checks.append(
                    {
                        "argv": ["sha256sum", "--", filename],
                        "expected": digest + "  " + filename,
                    }
                )
        libraries = {
            "nccl": "/opt/sparkring/nccl/libnccl.so.2",
            "sircl": "/opt/spark-sircl/libspark_transport_capi.so",
            "cache_placement": "/opt/sparkcache-src/sparkcache/native/build-cuda/libspark_cache_placement.so",
            "cache_capture": "/opt/sparkcache-src/sparkcache/native/build-cuda/libspark_cache_snapshot.so",
        }
        for role, filename in libraries.items():
            checks.append(
                {
                    "argv": [
                        "docker",
                        "run",
                        "--rm",
                        "--entrypoint",
                        "sha256sum",
                        rank["image"],
                        filename,
                    ],
                    "expected": plan["native_identities"][role] + "  " + filename,
                }
            )
        if mesh_preflight:
            checks.append(mesh_preflight)
        ranks.append(
            {
                "rank": rank["rank"],
                "host": rank["host"],
                "name": bundle_id + "-r" + str(rank["rank"]),
                "argv": argv,
                "checks": checks,
            }
        )
    return {"schema": "lil-image-bundle/v1", "id": bundle_id, "ranks": ranks}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--descriptor",
        default=str(Path(__file__).with_name("glm53-mtp3.json")),
        help="image profile; defaults to the published NVFP4-Spark MTP3 mesh",
    )
    p.add_argument("--site", required=True)
    p.add_argument("--fabric", required=True)
    p.add_argument("--id", required=True)
    a = p.parse_args()
    try:
        print(
            json.dumps(
                export(
                    read_json(a.descriptor),
                    read_json(a.site),
                    read_json(a.fabric),
                    a.id,
                ),
                indent=2,
            )
        )
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
