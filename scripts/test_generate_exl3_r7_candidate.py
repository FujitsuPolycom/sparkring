from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import generate_exl3_r7_candidate as candidate  # noqa: E402
import sparkring_runtime as runtime  # noqa: E402


def inputs() -> tuple[dict, dict, dict]:
    return (
        json.loads(candidate.TEMPLATE_PATH.read_text(encoding="utf-8")),
        json.loads(candidate.PINS_PATH.read_text(encoding="utf-8")),
        json.loads(candidate.RECIPE_PATH.read_text(encoding="utf-8")),
    )


def test_generated_profile_uses_generic_launcher_contract(tmp_path: Path) -> None:
    template, pins, recipe = inputs()
    document = candidate.generate(template, pins, recipe)
    path = tmp_path / "launch.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    profile = runtime.load_runtime_profile(path)

    assert profile.source_schema == "sparkring-runtime-profile/v1"
    assert profile.model_family == "exl3-r7"
    assert "--speculative-config" not in profile.extra_vllm_args
    assert profile.extra_vllm_args[profile.extra_vllm_args.index("--kv-cache-dtype") + 1] == "auto"
    assert profile.extra_vllm_args[profile.extra_vllm_args.index("--load-format") + 1] == "instanttensor"
    assert json.loads(profile.extra_vllm_args[
        profile.extra_vllm_args.index("--quantization-config") + 1
    ]) == candidate.ONLINE_QUANTIZATION_CONFIG
    assert json.loads(profile.extra_vllm_args[
        profile.extra_vllm_args.index("--hf-overrides") + 1
    ]) == candidate.HF_OVERRIDES
    assert len(candidate.GLM52_INDEX_TOPK_PATTERN) == 78
    assert "--enable-auto-tool-choice" in profile.extra_vllm_args
    assert json.loads(profile.extra_vllm_args[
        profile.extra_vllm_args.index("--default-chat-template-kwargs") + 1
    ]) == candidate.DEFAULT_CHAT_TEMPLATE_KWARGS
    assert "--enforce-eager" not in profile.extra_vllm_args
    assert profile.extra_vllm_args[
        profile.extra_vllm_args.index("--max-cudagraph-capture-size") + 1
    ] == "32"
    compilation = json.loads(profile.extra_vllm_args[
        profile.extra_vllm_args.index("--compilation-config") + 1
    ])
    assert compilation["cudagraph_mode"] == "FULL_AND_PIECEWISE"
    assert compilation["cudagraph_capture_sizes"] == list(range(1, 33))
    assert not any("lmcache" in value.lower() for value in profile.extra_vllm_args)
    assert profile.environment["SPARK_CONTEXT_CACHE_ENABLE"] == "0"
    assert profile.environment["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert profile.environment["CUDA_DEVICE_MAX_CONNECTIONS"] == "32"
    assert profile.environment["CUTE_DSL_ARCH"] == "sm_121a"
    assert profile.environment["LD_PRELOAD"] == (
        f"{candidate.CUDA_COMPAT_LIBRARY}:{candidate.PATCHED_NCCL_LIBRARY}"
    )
    assert profile.environment["ONLINE_QUANT"] == "exl3-b6"
    assert profile.environment["INSTANTTENSOR_BACKEND"] == "BUFFERED"
    assert profile.environment["INSTANTTENSOR_BUFFER_SIZE"] == "536870912"
    assert profile.environment["INSTANTTENSOR_COPY"] == "0"
    assert profile.environment["INSTANTTENSOR_MAX_FREE_MEM_USAGE"] == "0.05"
    assert profile.environment["B12X_MOE_FORCE_A16"] == "1"
    assert profile.environment["B12X_DENSE_SPLITK_TURBO"] == "1"
    assert profile.environment["B12X_MLA_SM120_UNIFIED"] == "1"
    assert profile.environment["B12X_W4A16_TC_DECODE"] == "1"
    assert profile.environment["B12X_W4A8_TINY_DECODE"] == "1"
    assert profile.environment["OMP_NUM_THREADS"] == "16"
    assert profile.environment["SAFETENSORS_FAST_GPU"] == "1"
    assert profile.environment["PYTHONPATH"] == "/opt/sparkring-r7-tvm-ffi:/opt/spark-vllm"
    assert profile.environment["VLLM_USE_AOT_COMPILE"] == "1"
    assert profile.environment["VLLM_USE_BREAKABLE_CUDAGRAPH"] == "0"
    assert profile.environment["VLLM_USE_MEGA_AOT_ARTIFACT"] == "1"
    assert profile.environment["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] == "1"
    assert profile.environment["VLLM_SPARK_MAX_QUERY_ROWS"] == "8"
    assert profile.environment["VLLM_SPARK_SHARED_CAPTURE_STREAM"] == "1"
    assert profile.environment["VLLM_SPARK_TP4_MODE"] == "custom"
    assert profile.environment["VLLM_SPARK_TP4_ALLGATHER_MODE"] == "custom"
    assert profile.environment["VLLM_SPARK_TP4_VOCAB_MODE"] == "custom"
    assert profile.environment["VLLM_SPARK_TP4_GRAPH_Q1"] == "1"
    assert profile.environment["VLLM_SPARK_TP4_INDEXER_GRAPH_CUSTOM"] == "0"
    assert profile.environment["SPARK_TP4_GRAPH_SUBMIT_CPU"] == "10"
    assert profile.environment["SPARK_TP4_GRAPH_PROGRESS_CPU"] == "11"
    assert profile.environment["SPARK_TP4_GRAPH_VOCAB_PROGRESS_CPU"] == "12"
    assert profile.environment["SPARK_TP4_GRAPH_INDEXER_PROGRESS_CPU"] == "14"
    assert profile.environment["VLLM_NCCL_SO_PATH"] == candidate.PATCHED_NCCL_LIBRARY
    assert candidate.PATCHED_NCCL_LIBRARY in profile.environment["LD_PRELOAD"].split(":")
    assert profile.environment["VLLM_PREFIX_CACHE_RETENTION_INTERVAL"] is None
    assert profile.environment["VLLM_B12X_ABSORB_BMM"] == "0"
    assert profile.environment["VLLM_DCP_GLOBAL_TOPK"] == "1"
    assert profile.environment["VLLM_DCP_SHARD_DRAFT"] == "1"
    assert profile.environment["VLLM_USE_B12X_DCP_A2A"] == "1"
    assert profile.environment["VLLM_USE_B12X_FP8_GEMM"] == "1"
    assert profile.environment["VLLM_USE_B12X_MHC"] == "1"
    assert profile.environment["VLLM_USE_B12X_MOE"] == "1"
    assert profile.environment["VLLM_USE_B12X_SPARSE_INDEXER"] == "1"
    assert profile.environment["VLLM_USE_B12X_WO_PROJECTION"] == "1"
    assert profile.environment["VLLM_USE_FLASHINFER_SAMPLER"] == "1"
    assert profile.environment["VLLM_EXL3_ONLINE_CACHE_MODE"] == "readwrite"
    assert profile.environment["VLLM_EXL3_ONLINE_TRELLIS_BITS"] == "6"
    assert "VLLM_EXL3_R7_FUSED" not in profile.environment
    assert "VLLM_SPARK_R7_NONFINITE_TRACE" not in profile.environment
    assert profile.environment["SPARK_TP4_DCP_COLLECTIVE_AUDIT"] == "1"
    assert profile.environment["SPARK_TP4_GRAPH_STATUS_PATH"] == (
        candidate.DCP_GRAPH_STATUS_PATH
    )
    assert "SPARKRING_R7_NONFINITE_TRACE" not in profile.environment
    assert "VLLM_SPARK_FINITE_TRACE" not in profile.environment
    assert profile.environment["NCCL_ALGO"] == "Ring"
    assert profile.environment["NCCL_CUMEM_ENABLE"] == "0"
    assert profile.environment["NCCL_IB_DISABLE"] == "0"
    assert profile.environment["NCCL_IGNORE_CPU_AFFINITY"] == "1"
    assert profile.environment["NCCL_MAX_NCHANNELS"] == "4"
    assert profile.environment["NCCL_MIN_NCHANNELS"] == "4"
    assert profile.environment["NCCL_NET"] == "IB"
    assert profile.environment["NCCL_SKIP_TREE_CONNECT"] == "1"
    assert profile.environment["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert profile.environment["TORCHINDUCTOR_COMPILE_THREADS"] == "1"
    assert profile.environment["TORCH_USE_RTLD_GLOBAL"] == "1"
    assert "/proc/self/maps" in profile.attestation_hook[2]
    assert ("/var/tmp/sparkring-r7-online", "/cache/exl3-online", "rw") in profile.extra_volumes
    assert not any(
        mode == "ro" and host.startswith("/var/tmp/sparkring-r7-")
        for host, _container, mode in profile.extra_volumes
    )
    assert profile.attestation_hook[:2] == ("/bin/sh", "-c")
    assert template["model_config_sha256"] in profile.attestation_hook[2]
    assert template["model_index_sha256"] in profile.attestation_hook[2]
    assert "sha256sum --check --strict" in profile.attestation_hook[2]
    assert f"test -r {candidate.CUDA_COMPAT_LIBRARY}" in profile.attestation_hook[2]
    assert f"test -r {candidate.PATCHED_NCCL_LIBRARY}" in profile.attestation_hook[2]
    assert (
        f"{candidate.ENTRYPOINT_SHA256}  {candidate.ENTRYPOINT_CONTAINER_PATH}"
        in profile.attestation_hook[2]
    )
    assert (
        f"{candidate.WEIGHT_UTILS_SHA256}  {candidate.WEIGHT_UTILS_CONTAINER_PATH}"
        in profile.attestation_hook[2]
    )
    assert (
        f"{candidate.EXL3_SCRATCH_SHA256}  {candidate.EXL3_SCRATCH_CONTAINER_PATH}"
        in profile.attestation_hook[2]
    )
    assert candidate.EXL3_SCRATCH_SHA256 == (
        "8e0051faf9b8bac9eefd6f38a5f0133a30bca4c0b5ab41962537e2f13cf968f4"
    )
    assert (
        f"{candidate.CUDAGRAPH_UTILS_SHA256}  "
        f"{candidate.CUDAGRAPH_UTILS_CONTAINER_PATH}"
        in profile.attestation_hook[2]
    )
    assert candidate.CUDAGRAPH_UTILS_SHA256 == (
        "ef03d64297ed2d1a5161847b48a435bf8ae5feda7a5b81b668d00ae9a1d65a2a"
    )
    assert (
        f"{candidate.QUACK_LAYOUT_UTILS_SHA256}  "
        f"{candidate.QUACK_LAYOUT_UTILS_CONTAINER_PATH}"
        in profile.attestation_hook[2]
    )
    assert (
        f"{candidate.QUACK_COPY_UTILS_SHA256}  "
        f"{candidate.QUACK_COPY_UTILS_CONTAINER_PATH}"
        in profile.attestation_hook[2]
    )
    assert (
        f"{candidate.DCP_AUDIT_SHA256}  {candidate.DCP_AUDIT_CONTAINER_PATH}"
        in profile.attestation_hook[2]
    )
    assert f'assert md.version("apache-tvm-ffi") == "{candidate.TVM_FFI_VERSION}"' in profile.attestation_hook[2]
    assert f'assert tvm_ffi.__version__ == "{candidate.TVM_FFI_VERSION}"' in profile.attestation_hook[2]
    assert candidate.TVM_FFI_CONTAINER_PATH in profile.attestation_hook[2]
    assert candidate.TVM_FFI_WHEEL_SHA256 == "3829216a8500c2f61062e48c627f6db6c3fa49416b3ffa85bc04243ae5d759f7"
    assert profile.health_check == (
        "/opt/venv/bin/python",
        "/opt/sparkring-r7/verify_runtime.py",
    )


def test_generator_rejects_unpinned_image() -> None:
    template, pins, recipe = inputs()
    template["image_id"] = "latest"
    with pytest.raises(candidate.CandidateError, match="exact local ARM64 image ID"):
        candidate.generate(template, pins, recipe)


def test_generated_bf16_baseline_disables_online_k6() -> None:
    template, pins, recipe = inputs()
    template["online_quantization"] = "none"
    document = candidate.generate(template, pins, recipe)

    assert "--quantization-config" not in document["extra_vllm_args"]
    assert document["environment"]["ONLINE_QUANT"] == "none"
    for name in (
        "VLLM_EXL3_ENCODER_SOURCE",
        "VLLM_EXL3_ONLINE_CACHE_DIR",
        "VLLM_EXL3_ONLINE_CACHE_MODE",
        "VLLM_EXL3_ONLINE_TRELLIS_BITS",
    ):
        assert document["environment"][name] is None


def test_generated_fp8_ds_mla_discriminator_changes_only_kv_format() -> None:
    template, pins, recipe = inputs()
    baseline = candidate.generate(template, pins, recipe)
    template["kv_cache_dtype"] = "fp8_ds_mla"
    document = candidate.generate(template, pins, recipe)

    kv_index = document["extra_vllm_args"].index("--kv-cache-dtype") + 1
    assert document["extra_vllm_args"][kv_index] == "fp8_ds_mla"
    baseline_args = list(baseline["extra_vllm_args"])
    baseline_args[baseline_args.index("--kv-cache-dtype") + 1] = "fp8_ds_mla"
    assert document["extra_vllm_args"] == baseline_args
    assert document["environment"] == baseline["environment"]
    assert document["extra_volumes"] == baseline["extra_volumes"]


def test_generator_rejects_unknown_kv_cache_dtype() -> None:
    template, pins, recipe = inputs()
    template["kv_cache_dtype"] = "fp8"
    with pytest.raises(candidate.CandidateError, match="auto or fp8_ds_mla"):
        candidate.generate(template, pins, recipe)


def test_generated_clean_profile_removes_only_nonfinite_trace_contract() -> None:
    template, pins, recipe = inputs()
    template["nonfinite_trace"] = True
    traced = candidate.generate(template, pins, recipe)
    template["nonfinite_trace"] = False
    clean = candidate.generate(template, pins, recipe)

    expected_environment = dict(traced["environment"])
    del expected_environment["VLLM_SPARK_R7_NONFINITE_TRACE"]
    assert clean["environment"] == expected_environment

    trace_volume = {
        "host": candidate.NONFINITE_TRACE_HOST_PATH,
        "container": candidate.NONFINITE_TRACE_CONTAINER_PATH,
        "mode": "ro",
    }
    assert clean["extra_volumes"] == [
        volume for volume in traced["extra_volumes"] if volume != trace_volume
    ]
    trace_attestation = (
        f"{candidate.NONFINITE_TRACE_SHA256}  "
        f"{candidate.NONFINITE_TRACE_CONTAINER_PATH}"
    )
    assert trace_attestation in traced["attestation_hook"][2]
    assert trace_attestation not in clean["attestation_hook"][2]

    traced_without_contract = dict(traced)
    traced_without_contract["environment"] = expected_environment
    traced_without_contract["extra_volumes"] = clean["extra_volumes"]
    traced_without_contract["attestation_hook"] = clean["attestation_hook"]
    assert clean == traced_without_contract


def test_generator_rejects_non_boolean_trace_selector() -> None:
    template, pins, recipe = inputs()
    template["nonfinite_trace"] = "false"
    with pytest.raises(candidate.CandidateError, match="true or false"):
        candidate.generate(template, pins, recipe)


def test_generated_stock_transport_disables_sircl_and_uses_nccl_ib() -> None:
    template, pins, recipe = inputs()
    template["transport"] = "stock-nccl-ib"
    document = candidate.generate(template, pins, recipe)
    environment = document["environment"]

    assert environment["NCCL_NET"] == "IB"
    assert environment["NCCL_IB_DISABLE"] == "0"
    assert environment["NCCL_CROSS_NIC"] == "1"
    assert environment["NCCL_MAX_NCHANNELS"] == "4"
    assert environment["NCCL_MIN_NCHANNELS"] == "4"
    assert environment["TORCH_USE_RTLD_GLOBAL"] == "1"
    assert environment["VLLM_NCCL_SO_PATH"] == candidate.PATCHED_NCCL_LIBRARY
    assert environment["LD_PRELOAD"] == (
        f"{candidate.CUDA_COMPAT_LIBRARY}:{candidate.PATCHED_NCCL_LIBRARY}"
    )
    assert candidate.PATCHED_NCCL_LIBRARY in document["attestation_hook"][2]
    assert "import torch; import instanttensor; import vllm" in document["attestation_hook"][2]
    assert "/proc/self/maps" in document["attestation_hook"][2]
    assert "assert len(mapped) == 1" in document["attestation_hook"][2]
    for name in (
        "VLLM_SPARK_MAX_QUERY_ROWS",
        "VLLM_SPARK_TP4_ALLGATHER_MODE",
        "VLLM_SPARK_TP4_ALLGATHER_POLICY",
        "VLLM_SPARK_TP4_GRAPH_Q1",
        "VLLM_SPARK_TP4_INDEXER_GRAPH_CUSTOM",
        "VLLM_SPARK_TP4_MODE",
        "VLLM_SPARK_TP4_PREFILL_Q512",
        "VLLM_SPARK_TP4_VOCAB_MODE",
    ):
        assert environment[name] is None


def test_generated_hybrid_transport_changes_only_nccl_fallback_lane() -> None:
    template, pins, recipe = inputs()
    baseline = candidate.generate(template, pins, recipe)
    template["transport"] = "sircl-nccl-ib"
    document = candidate.generate(template, pins, recipe)
    environment = document["environment"]

    expected_environment = dict(baseline["environment"])
    expected_environment.update({
        "LD_PRELOAD": (
            f"{candidate.CUDA_COMPAT_LIBRARY}:{candidate.PATCHED_NCCL_LIBRARY}"
        ),
        "NCCL_CROSS_NIC": "1",
        "NCCL_IB_DISABLE": "0",
        "NCCL_MAX_NCHANNELS": "4",
        "NCCL_MIN_NCHANNELS": "4",
        "NCCL_NET": "IB",
        "TORCH_USE_RTLD_GLOBAL": "1",
        "VLLM_NCCL_SO_PATH": candidate.PATCHED_NCCL_LIBRARY,
    })
    assert environment == expected_environment
    assert environment["VLLM_SPARK_TP4_MODE"] == "custom"
    assert environment["VLLM_SPARK_TP4_ALLGATHER_MODE"] == "custom"
    assert environment["VLLM_SPARK_TP4_VOCAB_MODE"] == "custom"
    assert environment["VLLM_SPARK_TP4_GRAPH_Q1"] == "1"
    assert environment["VLLM_SPARK_TP4_INDEXER_GRAPH_CUSTOM"] == "0"
    assert candidate.PATCHED_NCCL_LIBRARY in document["attestation_hook"][2]
    assert "import torch; import instanttensor; import vllm" in document["attestation_hook"][2]
    assert "/proc/self/maps" in document["attestation_hook"][2]
    assert "assert len(mapped) == 1" in document["attestation_hook"][2]

    normalized = dict(document)
    normalized["environment"] = baseline["environment"]
    normalized["attestation_hook"] = baseline["attestation_hook"]
    assert normalized == baseline


def test_generated_indexer_graph_transport_is_a_separate_fail_closed_selector() -> None:
    template, pins, recipe = inputs()
    template["transport"] = "sircl-nccl-ib"
    baseline = candidate.generate(template, pins, recipe)
    template["indexer_graph_custom"] = True
    document = candidate.generate(template, pins, recipe)

    expected_environment = dict(baseline["environment"])
    expected_environment["VLLM_SPARK_TP4_INDEXER_GRAPH_CUSTOM"] = "1"
    assert document["environment"] == expected_environment
    for symbol in (
        "spark_tp4_indexer_graph_create",
        "spark_tp4_indexer_capture_allgather",
        "spark_tp4_indexer_get_graph_status",
        "spark_tp4_indexer_graph_destroy",
    ):
        assert symbol in document["attestation_hook"][2]
    assert "\\'spark_tp4_indexer" not in document["attestation_hook"][2]

    normalized = dict(document)
    normalized["environment"] = baseline["environment"]
    normalized["attestation_hook"] = baseline["attestation_hook"]
    assert normalized == baseline


def test_generator_rejects_indexer_graph_without_sircl() -> None:
    template, pins, recipe = inputs()
    template["transport"] = "stock-nccl-ib"
    template["indexer_graph_custom"] = True
    with pytest.raises(candidate.CandidateError, match="requires a SIRCL transport"):
        candidate.generate(template, pins, recipe)


def test_generator_rejects_unknown_transport() -> None:
    template, pins, recipe = inputs()
    template["transport"] = "hybrid"
    with pytest.raises(candidate.CandidateError, match="sircl-nccl-ib"):
        candidate.generate(template, pins, recipe)


def test_generator_rejects_non_boolean_indexer_graph_selector() -> None:
    template, pins, recipe = inputs()
    template["indexer_graph_custom"] = "1"
    with pytest.raises(candidate.CandidateError, match="true or false"):
        candidate.generate(template, pins, recipe)


def test_recipe_records_the_operator_accepted_mtp4_serving_contract() -> None:
    _, _, recipe = inputs()
    serving = recipe["serving"]
    assert serving["tensor_parallel_size"] == 4
    assert serving["decode_context_parallel_size"] == 4
    assert serving["mtp_policy"] == "fixed-4"
    assert serving["kv_cache_dtype"] == "nvfp4_ds_mla"
    assert serving["kv_dynamic_per_token_scale"] is True
    assert serving["kv_fp8_rope"] is True
    assert serving["max_model_len"] == 262_144
    assert serving["max_num_batched_tokens"] == 4_096
    assert serving["exact_q40_policy"]["capacity_rows"] == 40
    assert serving["kv_cache_bytes_per_rank"] == 9_250_000_000
    assert serving["gpu_memory_utilization"] == 0.85
    assert serving["execution_mode"] == "FULL_AND_PIECEWISE"


def test_r7_entrypoint_applies_explicit_environment_unsets() -> None:
    entrypoint_path = ROOT / "runtime/exl3-r7/entrypoint.sh"
    if not entrypoint_path.exists():
        pytest.skip(
            "runtime/exl3-r7/entrypoint.sh is on the builder branch; "
            "the public lane cannot verify the entrypoint hash"
        )
    entrypoint = entrypoint_path.read_bytes()
    source = entrypoint.decode("utf-8")

    assert hashlib.sha256(entrypoint).hexdigest() == candidate.ENTRYPOINT_SHA256
    assert 'if [[ -n "${SPARKRING_EXPLICITLY_UNSET:-}" ]]' in source
    assert 'unset "${name}"' in source
    assert "unset SPARKRING_EXPLICITLY_UNSET" in source
