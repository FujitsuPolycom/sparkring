import copy
import importlib.util
import json
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
MODULE_PATH = HERE / "deepseek_sircl_ab_plan.py"
SPEC = importlib.util.spec_from_file_location("deepseek_sircl_ab_plan", MODULE_PATH)
assert SPEC and SPEC.loader
plan_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plan_module)


def _recipe() -> dict:
    return json.loads(
        (REPO / "recipes" / "deepseek-v4-flash-0731.json").read_text(
            encoding="utf-8"
        )
    )


def test_plan_preserves_exact_four_spark_quickstart_contract() -> None:
    plan = plan_module.build_plan(
        REPO / "recipes" / "deepseek-v4-flash-0731.json",
        REPO / "runtime" / "faststart-lock.json",
    )

    assert plan["dry_run"] is True
    assert plan["execution_supported"] is False
    assert plan["launch_readiness"]["status"] == "waiting-for-explicit-authorization"
    assert plan["launch_readiness"]["remote_actions_performed"] is False
    assert plan["launch_readiness"]["model_actions_performed"] is False
    assert plan["fixed_serving_contract"] == plan_module.EXPECTED_SERVING
    assert plan["fixed_serving_contract"]["speculation"] == {
        "method": "dspark",
        "num_speculative_tokens": 5,
        "moe_backend": "b12x",
    }
    assert plan["derived_graph_contract"] == {
        "draft_plus_target_rows_per_sequence": 6,
        "maximum_verification_query_rows": 192,
        "width_elements": 4096,
        "sircl_session_capacity_query_rows": 512,
    }


def test_candidate_changes_only_environment_and_runtime_mounts() -> None:
    plan = plan_module.build_plan(
        REPO / "recipes" / "deepseek-v4-flash-0731.json",
        REPO / "runtime" / "faststart-lock.json",
    )
    control = plan["arms"]["patched_nccl_control"]
    candidate = plan["arms"]["sircl_width4096_candidate"]

    control_argv = control["docker_argv_template"]
    candidate_argv = candidate["docker_argv_template"]
    stripped = list(candidate_argv)
    overlay = "/path/to/rank-{rank}-sircl.env"
    start = stripped.index(overlay) - 1
    del stripped[start : start + 2]
    assert stripped == control_argv
    for mount in plan_module.SIRCL_RUNTIME_MOUNTS:
        mount_arg = f"{mount['source']}:{mount['destination']}:ro"
        control_start = control_argv.index(mount_arg) - 1
        candidate_start = candidate_argv.index(mount_arg) - 1
        assert control_argv[control_start] == "-v"
        assert candidate_argv[candidate_start] == "-v"
    assert candidate["environment_delta"] == plan_module.SIRCL_ENVIRONMENT
    assert candidate["runtime_mounts"] == list(
        plan_module.SIRCL_RUNTIME_MOUNTS
    )
    assert candidate["runtime_mount_sha256_required"] is True
    assert control["runtime_mounts"] == candidate["runtime_mounts"]
    assert control["runtime_mount_sha256_required"] is True


def test_vllm_arguments_match_quickstart_values() -> None:
    args = plan_module.vllm_arguments(plan_module.EXPECTED_SERVING)

    expected_pairs = {
        "--tensor-parallel-size": "4",
        "--nnodes": "4",
        "--max-model-len": "1048576",
        "--max-num-seqs": "32",
        "--max-num-batched-tokens": "4096",
        "--gpu-memory-utilization": "0.70",
        "--kv-cache-memory-bytes": "17179869184",
        "--kv-cache-dtype": "fp8_ds_mla",
    }
    for option, expected in expected_pairs.items():
        assert args[args.index(option) + 1] == expected
    assert "--async-scheduling" in args
    assert "--scheduler-reserve-full-isl" in args
    assert "--no-enable-prefix-caching" in args
    assert args[args.index("--block-size") + 1] == "256"
    speculative = json.loads(args[args.index("--speculative-config") + 1])
    assert speculative == {
        "method": "dspark",
        "num_speculative_tokens": 5,
        "moe_backend": "b12x",
    }


def test_benchmark_contract_matches_sustained_decode_method() -> None:
    plan = plan_module.build_plan(
        REPO / "recipes" / "deepseek-v4-flash-0731.json",
        REPO / "runtime" / "faststart-lock.json",
    )
    benchmark = plan["benchmark_contract"]

    assert benchmark["mode"] == "duration"
    assert benchmark["duration_seconds_per_cell"] == 90
    assert benchmark["contexts_total_chat_tokens"] == [2048, 8192]
    assert benchmark["concurrency_levels"] == [1, 2, 4, 8, 16, 32]
    assert benchmark["ignore_eos"] is True
    assert benchmark["maximum_output_tokens_per_request"] == 8192
    assert benchmark["context_targeting"] == "exact"
    assert benchmark["unique_context_percent"] == 100.0
    assert benchmark["paired_source_must_match"] is True

    for concurrency, args in zip(
        benchmark["concurrency_levels"], benchmark["invocations"], strict=True
    ):
        assert args[args.index("--concurrency") + 1] == str(concurrency)
        assert args[args.index("--contexts") + 1] == "2k,8k"
        assert args[args.index("--duration") + 1] == "90"
        assert args[args.index("--decode-warmup-seconds") + 1] == "10"
        assert args[args.index("--max-total-tokens") + 1] == "2198756"
        assert "--respect-eos" not in args
        assert "--skip-prefill" in args


def test_candidate_gates_cover_every_runtime_mount() -> None:
    plan = plan_module.build_plan(
        REPO / "recipes" / "deepseek-v4-flash-0731.json",
        REPO / "runtime" / "faststart-lock.json",
    )

    required = set(plan["candidate_gates"]["required_runtime_paths"])
    assert "/opt/spark-vllm/sitecustomize.py" in required
    assert {
        mount["destination"] for mount in plan_module.SIRCL_RUNTIME_MOUNTS
    } <= required


def test_aligned_c32_invocation_matches_tp2_workload() -> None:
    plan = plan_module.build_plan(
        REPO / "recipes" / "deepseek-v4-flash-0731.json",
        REPO / "runtime" / "faststart-lock.json",
    )
    comparison = plan["benchmark_contract"]["aligned_tp2_c32_comparison"]
    args = comparison["invocation"]

    expected_pairs = {
        "--temperature": "1.0",
        "--kv-budget": "2198756",
        "--contexts": "16k",
        "--concurrency": "32",
        "--duration": "240",
        "--max-tokens": "32768",
        "--unique-context-percent": "100",
        "--token-targeting": "exact",
        "--cell-warmup-timeout-seconds": "600",
        "--decode-warmup-seconds": "10",
    }
    for option, expected in expected_pairs.items():
        assert args[args.index(option) + 1] == expected
    assert "--skip-prefill" in args
    assert comparison["all_requests_must_be_running_before_measurement"] is True


def test_benchmark_identity_binds_modified_file_bytes(tmp_path: Path) -> None:
    harness = tmp_path / "llm_decode_bench.py"
    harness.write_text('VERSION = "test-version"\n', encoding="utf-8")

    identity = plan_module._benchmark_identity(harness)

    assert identity == {
        "path": str(harness.resolve()),
        "sha256": plan_module._sha256(harness),
        "version": "test-version",
        "git_revision": None,
        "git_dirty": None,
    }


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("serving", "max_num_seqs"), 16),
        (("serving", "max_num_batched_tokens"), 2048),
        (("serving", "speculation", "num_speculative_tokens"), 7),
        (("serving", "kv_cache_memory_bytes_per_rank"), 20000000000),
    ),
)
def test_plan_rejects_quickstart_drift(
    tmp_path: Path, path: tuple[str, ...], value: object
) -> None:
    recipe = copy.deepcopy(_recipe())
    target = recipe
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(recipe), encoding="utf-8")

    with pytest.raises(plan_module.PlanError, match="quickstart contract drifted"):
        plan_module.build_plan(
            recipe_path,
            REPO / "runtime" / "faststart-lock.json",
        )


def test_sircl_overlay_has_no_site_addresses() -> None:
    text = (
        REPO
        / "scripts"
        / "config"
        / "deepseek-v4-flash-0731-sircl-research.env.example"
    ).read_text(encoding="utf-8")

    assert "<DIRECT_PEER_ADDRESS_ON_DEVICE0>" in text
    assert "<DIRECT_PEER_ADDRESS_ON_DEVICE1>" in text
    assert "VLLM_SPARK_TP4_GRAPH_WIDTH4096_RESEARCH=1" in text
    assert "VLLM_SPARK_SHARED_CAPTURE_STREAM=1" in text

    values = {}
    for line in text.splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        values[name] = value
    for name, expected in plan_module.SIRCL_ENVIRONMENT.items():
        assert values[name] == expected
