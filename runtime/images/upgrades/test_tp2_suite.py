"""Native cache-argument adaptation retains identity checks and model settings."""

import json

import pytest

from runtime.images.upgrades.contracts import Refused
from runtime.images.upgrades.tp2_suite import native_arguments, option


def metadata():
    return {
        "files": {
            "/opt/sparkring/sparkcache/lib/libspark_cache_placement.so": "a" * 64,
            "/opt/sparkring/sparkcache/lib/libspark_cache_snapshot.so": "b" * 64,
            "/opt/sparkring/contracts/boundary-native.json": "c" * 64,
        },
        "active_contracts": [
            "/opt/sparkring/contracts/vllm-connector-jobs-native.json"
        ],
        "boundary_runtime": {
            "path": "/opt/sparkring/contracts/boundary-native.json",
            "sha256": "c" * 64,
        },
    }


def args():
    config = {
        "kv_connector": "SparkBoundaryCacheConnector",
        "kv_connector_extra_config": {
            "spark_cache_root": "/cache/persistent/model",
            "spark_cache_model_profile": "qwen-hybrid",
            "spark_cache_cuda_placement_library": "/former/placement.so",
            "spark_cache_cuda_placement_library_sha256": "d" * 64,
            "spark_cache_async_page_capture_library": "/former/snapshot.so",
            "spark_cache_async_page_capture_library_sha256": "e" * 64,
        },
    }
    return [
        "serve",
        "/models/target",
        "--max-model-len",
        "262144",
        "--kv-transfer-config",
        json.dumps(config),
    ]


def test_native_cache_bindings_use_attested_libraries_and_runtime_identity():
    original = args()
    result, changes = native_arguments(original, metadata())
    config = json.loads(result[-1])["kv_connector_extra_config"]
    assert config["spark_cache_cuda_placement_library_sha256"] == "a" * 64
    assert config["spark_cache_async_page_capture_library_sha256"] == "b" * 64
    assert config["spark_cache_boundary_contract_sha256"] == "c" * 64
    assert config["spark_cache_root"] == "/cache/persistent/model"
    assert original == args() and result[:4] == original[:4]
    assert "boundary-runtime-identity" in changes


def test_missing_boundary_identity_cannot_be_worked_around():
    value = metadata()
    value["boundary_runtime"] = None
    with pytest.raises(Refused, match="boundary-runtime"):
        native_arguments(args(), value)


def test_multiple_active_source_lease_contracts_require_explicit_selection():
    value = metadata()
    value["active_contracts"].append(
        "/opt/sparkring/contracts/vllm-connector-jobs-another.json"
    )
    with pytest.raises(Refused, match="one explicit"):
        native_arguments(args(), value)


def test_duplicate_serving_options_are_not_silently_replaced():
    with pytest.raises(Refused, match="occur once"):
        option(["--port", "1", "--port", "2"], "--port", 18000)
