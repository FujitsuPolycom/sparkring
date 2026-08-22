#!/usr/bin/env python3
"""Emit an offline NCCL-versus-SIRCL plan for the four-Spark DeepSeek profile.

The canonical DeepSeek recipe owns every model, scheduler, speculation, and
memory value.  This module refuses recipe drift and limits the candidate arm to
the environment overlay that activates the research width-4096 graph session.
It has no execution mode and never contacts a configured host.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
DEFAULT_RECIPE = REPO / "recipes" / "deepseek-v4-flash-0731.json"
DEFAULT_LOCK = REPO / "runtime" / "faststart-lock.json"
BASE_ENV_TEMPLATE = "scripts/config/deepseek-v4-flash-0731.env.example"
SIRCL_ENV_TEMPLATE = (
    "scripts/config/deepseek-v4-flash-0731-sircl-research.env.example"
)

EXPECTED_SERVING = {
    "served_model_name": "deepseek-v4-flash-0731",
    "tensor_parallel_size": 4,
    "node_count": 4,
    "distributed_executor_backend": "mp",
    "dtype": "bfloat16",
    "max_model_len": 1048576,
    "max_num_seqs": 32,
    "max_num_batched_tokens": 4096,
    "gpu_memory_utilization": 0.7,
    "kv_cache_memory_bytes_per_rank": 17179869184,
    "kv_cache_dtype": "fp8_ds_mla",
    "tokenizer_mode": "deepseek_v4",
    "speculation": {
        "method": "dspark",
        "num_speculative_tokens": 5,
        "moe_backend": "b12x",
    },
    "tool_call_parser": "deepseek_v4",
}
SOURCE_RECIPE_BATCH_TOKENS = 8192

SIRCL_ENVIRONMENT = {
    "PYTHONPATH": "/opt/spark-vllm",
    "SPARK_TP4_LIBRARY": (
        "/opt/sparkring/spark_transport/libspark_transport_capi.so"
    ),
    "VLLM_SPARK_TP4_MODE": "custom",
    "VLLM_SPARK_TP4_GRAPH_WIDTH4096_RESEARCH": "1",
    "VLLM_SPARK_SHARED_CAPTURE_STREAM": "1",
    "VLLM_SPARK_TP4_GRAPH_Q1": "0",
    "VLLM_SPARK_TP4_GRAPH_DUAL_PORT_Q40": "0",
    "SPARK_TP4_GRAPH_CONTROL_PORT0": "9970",
    "SPARK_TP4_GRAPH_CONTROL_PORT1": "9971",
    "SPARK_TP4_GRAPH_SUBMIT_CPU": "10",
    "SPARK_TP4_GRAPH_PROGRESS_CPU": "11",
    "SPARK_TP4_MAX_INFLIGHT": "64",
    "SPARK_TP4_CONTROL_CONNECT_TIMEOUT_SECONDS": "10",
}

SIRCL_RUNTIME_MOUNTS = (
    {
        "name": "parallel_state",
        "source": "/path/to/sircl-runtime/parallel_state.py",
        "destination": (
            "/opt/venv/lib/python3.12/site-packages/vllm/distributed/parallel_state.py"
        ),
    },
    {
        "name": "cudagraph_utils",
        "source": "/path/to/sircl-runtime/cudagraph_utils.py",
        "destination": (
            "/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/"
            "cudagraph_utils.py"
        ),
    },
    {
        "name": "native_library",
        "source": "/path/to/sircl-runtime/libspark_transport_capi.so",
        "destination": (
            "/opt/sparkring/spark_transport/libspark_transport_capi.so"
        ),
    },
    {
        "name": "backend",
        "source": "/path/to/sircl-runtime/spark_tp4_backend.py",
        "destination": "/opt/spark-vllm/spark_tp4_backend.py",
    },
    {
        "name": "port_namespace",
        "source": "/path/to/sircl-runtime/spark_tp4_port_namespace.py",
        "destination": "/opt/spark-vllm/spark_tp4_port_namespace.py",
    },
    {
        "name": "query_row_provider",
        "source": "/path/to/sircl-runtime/spark_tp4_query_row_provider.py",
        "destination": "/opt/spark-vllm/spark_tp4_query_row_provider.py",
    },
)

BENCHMARK_CONCURRENCY_LEVELS = (1, 2, 4, 8, 16, 32)
BENCHMARK_CONTEXTS = (2048, 8192)


class PlanError(ValueError):
    """The canonical launch contract cannot produce a safe A/B plan."""


def _load(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PlanError(f"cannot read JSON input {path}: {error}") from error
    if not isinstance(document, dict):
        raise PlanError(f"JSON input must be an object: {path}")
    return document


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _benchmark_identity(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = path.resolve()
    if not resolved.is_file():
        raise PlanError(f"benchmark harness is not a file: {resolved}")
    text = resolved.read_text(encoding="utf-8")
    version_match = re.search(r'^VERSION\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
    identity: dict[str, Any] = {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "version": None if version_match is None else version_match.group(1),
        "git_revision": None,
        "git_dirty": None,
    }
    root = subprocess.run(
        ["git", "-C", str(resolved.parent), "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
        text=True,
    )
    if root.returncode != 0:
        return identity
    repository = Path(root.stdout.strip()).resolve()
    revision = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    relative = resolved.relative_to(repository)
    status = subprocess.run(
        ["git", "-C", str(repository), "status", "--short", "--", str(relative)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    identity["git_revision"] = revision
    identity["git_dirty"] = bool(status.strip())
    return identity


def _require_exact_serving(recipe: dict[str, Any]) -> dict[str, Any]:
    if recipe.get("schema") != "sparkring-recipe/v1":
        raise PlanError("DeepSeek recipe schema must be sparkring-recipe/v1")
    if recipe.get("recipe_id") != "deepseek-v4-flash-0731":
        raise PlanError("A/B plan requires the four-Spark DeepSeek recipe")
    hardware = recipe.get("hardware")
    if not isinstance(hardware, dict) or hardware.get("ranks") != 4:
        raise PlanError("A/B plan requires exactly four Spark ranks")
    if hardware.get("topology") != "direct-cycle-4":
        raise PlanError("A/B plan requires the direct four-Spark cycle")
    serving = recipe.get("serving")
    if not isinstance(serving, dict):
        raise PlanError("DeepSeek recipe serving contract is missing")
    source_expected = {**EXPECTED_SERVING, "max_num_batched_tokens": SOURCE_RECIPE_BATCH_TOKENS}
    observed = {key: serving.get(key) for key in source_expected}
    if observed != source_expected:
        differences = {
            key: {"expected": expected, "observed": observed.get(key)}
            for key, expected in source_expected.items()
            if observed.get(key) != expected
        }
        raise PlanError(
            "four-Spark DeepSeek quickstart contract drifted: "
            + json.dumps(differences, sort_keys=True, separators=(",", ":"))
        )
    return {**serving, "max_num_batched_tokens": EXPECTED_SERVING["max_num_batched_tokens"]}


def _image(lock: dict[str, Any]) -> str:
    serving_image = lock.get("serving_image")
    if not isinstance(serving_image, dict):
        raise PlanError("runtime lock serving_image is missing")
    repository = serving_image.get("repository")
    digest = serving_image.get("manifest_digest")
    if not isinstance(repository, str) or not repository:
        raise PlanError("runtime lock serving-image repository is missing")
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        raise PlanError("runtime lock serving-image digest is invalid")
    return f"{repository}@{digest}"


def vllm_arguments(serving: dict[str, Any]) -> list[str]:
    speculation = json.dumps(
        {
            "method": serving["speculation"]["method"],
            "num_speculative_tokens": serving["speculation"][
                "num_speculative_tokens"
            ],
            "moe_backend": serving["speculation"]["moe_backend"],
        },
        separators=(",", ":"),
    )
    return [
        "serve",
        "/models/deepseek-v4-flash-0731",
        "--tensor-parallel-size",
        str(serving["tensor_parallel_size"]),
        "--nnodes",
        str(serving["node_count"]),
        "--node-rank",
        "{rank}",
        "--master-addr",
        "{rank0_fabric_addr}",
        "--master-port",
        "29500",
        "--distributed-executor-backend",
        serving["distributed_executor_backend"],
        "--dtype",
        serving["dtype"],
        "--max-model-len",
        str(serving["max_model_len"]),
        "--max-num-seqs",
        str(serving["max_num_seqs"]),
        "--max-num-batched-tokens",
        str(serving["max_num_batched_tokens"]),
        "--async-scheduling",
        "--scheduler-reserve-full-isl",
        "--no-enable-prefix-caching",
        "--gpu-memory-utilization",
        f"{serving['gpu_memory_utilization']:.2f}",
        "--kv-cache-memory-bytes",
        str(serving["kv_cache_memory_bytes_per_rank"]),
        "--kv-cache-dtype",
        serving["kv_cache_dtype"],
        "--block-size",
        "256",
        "--tokenizer-mode",
        serving["tokenizer_mode"],
        "--kernel-config",
        '{"enable_cutedsl_warmup":false}',
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        serving["tool_call_parser"],
        "--speculative-config",
        speculation,
        "--served-model-name",
        serving["served_model_name"],
        "{api_or_headless}",
    ]


def docker_arguments(
    *, image: str, vllm_args: list[str], candidate: bool
) -> list[str]:
    arguments = [
        "docker",
        "run",
        "-d",
        "--name",
        "deepseek-v4-flash-r{rank}",
        "--network",
        "host",
        "--ipc",
        "host",
        "--shm-size",
        "16g",
        "--gpus",
        "all",
        "--ulimit",
        "memlock=-1:-1",
        "--device",
        "/dev/infiniband",
        "-v",
        "/path/to/deepseek-v4-flash-0731:/models/deepseek-v4-flash-0731:ro",
        "-v",
        "/path/to/jit-cache:/cache",
        "--env-file",
        "/path/to/rank-{rank}.env",
    ]
    if candidate:
        arguments.extend(
            [
                "--env-file",
                "/path/to/rank-{rank}-sircl.env",
            ]
        )
    for mount in SIRCL_RUNTIME_MOUNTS:
        arguments.extend(
            [
                "-v",
                f"{mount['source']}:{mount['destination']}:ro",
            ]
        )
    arguments.extend(["--entrypoint", "/opt/venv/bin/vllm", image])
    arguments.extend(vllm_args)
    return arguments


def benchmark_arguments(concurrency: int) -> list[str]:
    """Return one sustained-decode invocation matching the accepted method.

    Each concurrency level runs in a separate process so context calibration,
    hidden warmup, and a failed cell cannot contaminate later cells.
    """

    if concurrency not in BENCHMARK_CONCURRENCY_LEVELS:
        raise PlanError(
            "benchmark concurrency must be one of "
            f"{BENCHMARK_CONCURRENCY_LEVELS}: {concurrency}"
        )
    return [
        "{python}",
        "{llm_decode_bench}",
        "--host",
        "{rank0_api_host}",
        "--port",
        "8000",
        "--model",
        EXPECTED_SERVING["served_model_name"],
        "--concurrency",
        str(concurrency),
        "--contexts",
        "2k,8k",
        "--duration",
        "90",
        "--max-tokens",
        "8192",
        "--temperature",
        "0",
        "--decode-warmup-seconds",
        "10",
        "--cell-warmup-timeout-seconds",
        "180",
        "--unique-context-percent",
        "100",
        "--token-targeting",
        "exact",
        "--max-total-tokens",
        "2198756",
        "--skip-prefill",
        "--hw-ssh-hosts",
        "{rank_labeled_ssh_hosts}",
        "--output",
        f"{{output_dir}}/2k8k-c{concurrency}.json",
    ]


def aligned_c32_arguments() -> list[str]:
    """Return the TP2-baseline-aligned 16K/C32 decode invocation."""

    return [
        "{python}",
        "{llm_decode_bench}",
        "--host",
        "{rank0_api_host}",
        "--port",
        "8000",
        "--model",
        EXPECTED_SERVING["served_model_name"],
        "--temperature",
        "1.0",
        "--display-mode",
        "plain",
        "--hw-ssh-hosts",
        "{rank_labeled_ssh_hosts}",
        "--kv-budget",
        "2198756",
        "--contexts",
        "16k",
        "--concurrency",
        "32",
        "--duration",
        "240",
        "--max-tokens",
        "32768",
        "--unique-context-percent",
        "100",
        "--token-targeting",
        "exact",
        "--skip-prefill",
        "--isolated-server",
        "--cell-warmup-timeout-seconds",
        "600",
        "--decode-warmup-seconds",
        "10",
        "--output",
        "{output_dir}/temp1-16k-c32.json",
    ]


def build_plan(
    recipe_path: Path,
    lock_path: Path,
    benchmark_script: Path | None = None,
) -> dict[str, Any]:
    recipe = _load(recipe_path)
    lock = _load(lock_path)
    serving = _require_exact_serving(recipe)
    image = _image(lock)
    common_vllm = vllm_arguments(serving)
    return {
        "schema": "sparkring-deepseek-sircl-ab-plan/v1",
        "safety": "OFFLINE",
        "dry_run": True,
        "execution_supported": False,
        "scope": {
            "lane": "public-functional",
            "maturity": "research-only",
            "hardware": "four directly cabled NVIDIA DGX Sparks",
            "topology": "TP4/DCP1 direct-cycle-4",
        },
        "launch_readiness": {
            "status": "waiting-for-explicit-authorization",
            "remote_actions_performed": False,
            "model_actions_performed": False,
            "required_before_launch": [
                "The running DGX4 benchmark has completed and its receipt is immutable.",
                "The control receipt, harness file, image, and model identities are hashed.",
                "Four rank environments resolve both direct SIRCL peers and contain no placeholders.",
                "The seven SIRCL runtime mount inputs are built, immutable, and bound by SHA-256.",
                "The user explicitly authorizes stopping the control and starting the candidate.",
            ],
        },
        "inputs": {
            "recipe": str(recipe_path),
            "recipe_sha256": _sha256(recipe_path),
            "runtime_lock": str(lock_path),
            "runtime_lock_sha256": _sha256(lock_path),
            "image": image,
        },
        "fixed_serving_contract": EXPECTED_SERVING,
        "derived_graph_contract": {
            "draft_plus_target_rows_per_sequence": 6,
            "maximum_verification_query_rows": 192,
            "width_elements": 4096,
            "sircl_session_capacity_query_rows": 512,
        },
        "arms": {
            "patched_nccl_control": {
                "environment_templates": [BASE_ENV_TEMPLATE],
                "environment_delta": {},
                "runtime_mounts": list(SIRCL_RUNTIME_MOUNTS),
                "runtime_mount_sha256_required": True,
                "docker_argv_template": docker_arguments(
                    image=image, vllm_args=common_vllm, candidate=False
                ),
            },
            "sircl_width4096_candidate": {
                "environment_templates": [
                    BASE_ENV_TEMPLATE,
                    SIRCL_ENV_TEMPLATE,
                ],
                "environment_delta": SIRCL_ENVIRONMENT,
                "runtime_mounts": list(SIRCL_RUNTIME_MOUNTS),
                "runtime_mount_sha256_required": True,
                "site_specific_environment": [
                    "SPARK_TP4_PEER0",
                    "SPARK_TP4_PEER1",
                    "SPARK_TP4_DEVICE0",
                    "SPARK_TP4_DEVICE1",
                    "SPARK_TP4_GID0",
                    "SPARK_TP4_GID1",
                    "SPARK_TP4_GRAPH_STATUS_PATH",
                ],
                "docker_argv_template": docker_arguments(
                    image=image, vllm_args=common_vllm, candidate=True
                ),
            },
        },
        "candidate_gates": {
            "all_ranks": [0, 1, 2, 3],
            "captured_nodes_minimum": 1,
            "published_sequence_must_advance": True,
            "published_consumed_completed_must_match": True,
            "overflow_sequence": 0,
            "bounded_output_comparison_required": True,
            "draft_acceptance_metrics_required": True,
            "post_run_rank_health_required": True,
            "required_runtime_paths": [
                "/opt/spark-vllm/sitecustomize.py",
                *[mount["destination"] for mount in SIRCL_RUNTIME_MOUNTS],
            ],
        },
        "benchmark_contract": {
            "harness": "llm_decode_bench.py",
            "resolved_harness_identity": _benchmark_identity(benchmark_script),
            "harness_identity_required": [
                "version",
                "git_revision",
                "sha256",
            ],
            "mode": "duration",
            "duration_seconds_per_cell": 90,
            "contexts_total_chat_tokens": list(BENCHMARK_CONTEXTS),
            "concurrency_levels": list(BENCHMARK_CONCURRENCY_LEVELS),
            "one_invocation_per_concurrency": True,
            "maximum_output_tokens_per_request": 8192,
            "ignore_eos": True,
            "temperature": 0.0,
            "decode_warmup": {
                "concurrency": 1,
                "largest_requested_context": 8192,
                "seconds": 10,
            },
            "context_targeting": "exact",
            "unique_context_percent": 100.0,
            "context_sharing_mode": "fully_unique_per_stream",
            "prefill_reporting": "skipped",
            "headline_selection": (
                "Use completed OpenAI usage or stream timestamps only when "
                "their measurement-window coverage passes the harness gate; "
                "otherwise use the exact Prometheus generation-token delta "
                "over the 90-second post-warmup wall window."
            ),
            "paired_source_must_match": True,
            "invocations": [
                benchmark_arguments(concurrency)
                for concurrency in BENCHMARK_CONCURRENCY_LEVELS
            ],
            "aligned_tp2_c32_comparison": {
                "purpose": (
                    "Match the established TP2 base workload while changing "
                    "only the server, topology, and hardware-monitor targets."
                ),
                "context_tokens": 16384,
                "concurrency": 32,
                "duration_seconds": 240,
                "maximum_output_tokens": 32768,
                "temperature": 1.0,
                "kv_budget_tokens": 2198756,
                "readiness_timeout_seconds": 600,
                "all_requests_must_be_running_before_measurement": True,
                "isolated_server": True,
                "invocation": aligned_c32_arguments(),
            },
            "acceptance_gates_per_cell": {
                "num_errors": 0,
                "warmup_timed_out": False,
                "capacity_limited": False,
                "underfilled": False,
                "max_queue_reqs": 0,
                "effective_concurrency_must_equal_requested": True,
                "server_output_tokens_minimum": 1,
                "measurement_wall_seconds_minimum": 89.5,
                "speculative_acceptance_recorded": True,
                "hardware_samples_recorded": True,
            },
        },
        "limitations": [
            "The plan does not start, stop, inspect, or contact a DGX Spark.",
            "The SIRCL arm remains research-only until a matched live A/B passes.",
            "Width-256 drafter and large eager-prefill collectives remain on NCCL.",
        ],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument("--runtime-lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--benchmark-script", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plan = build_plan(
        args.recipe.resolve(),
        args.runtime_lock.resolve(),
        None if args.benchmark_script is None else args.benchmark_script.resolve(),
    )
    payload = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
