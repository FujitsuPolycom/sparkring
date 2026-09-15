"""Native cache-argument adaptation retains identity checks and model settings."""

import json

import pytest

from runtime.images.upgrades.contracts import Refused
from runtime.images.upgrades.tp2_suite import native_arguments, option
from runtime.images.upgrades.tp2_suite import fault_command


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


def test_fault_helper_mounts_only_the_owned_test_root_without_gpu_or_network():
    root = "/var/tmp/sparkring-upgrade-qualification/trial/glm-r0"
    argv = fault_command(
        "trial", {"root": root, "persistent": "cache"}, "corrupt", "sha256:" + "a" * 64
    )
    assert argv.count("--mount") == 1
    assert argv[argv.index("--mount") + 1] == f"type=bind,src={root},dst={root}"
    assert argv[argv.index("--network") + 1] == "none"
    assert "--privileged" not in argv and "--gpus" not in argv
    assert "--read-only" in argv and "DAC_OVERRIDE" in argv
    assert "--workers-stopped" in argv
    with pytest.raises(Refused, match="qualification root"):
        fault_command(
            "trial",
            {"root": "/home/user", "persistent": "cache"},
            "corrupt",
            "sha256:" + "a" * 64,
        )


def test_publication_failure_preserves_cold_response_and_stops_workers(
    tmp_path, monkeypatch
):
    from runtime.common.container_spec import ContainerSpec
    from runtime.images.upgrades import tp2_suite as suite

    site = dict(
        hosts=["u@h0", "u@h1"],
        hostnames=["h0", "h1"],
        api_host="192.0.2.1",
        port=18016,
        model="fixture",
        cache_checks=True,
    )
    specs = [
        ContainerSpec(
            name=f"test-r{rank}",
            image_id="sha256:" + "a" * 64,
            entrypoint=("python",),
            command=("serve",),
            environment={},
            mounts=(),
        )
        for rank in (0, 1)
    ]
    stops = []

    class FakePair:
        def __init__(self, *args, **kwargs):
            pass

        def call(self, rank, argv, **kwargs):
            if argv[:3] == ["docker", "ps", "-q"]:
                return b""
            if argv[-1] == "verify":
                return (
                    b'{"schema":"sparkring-native-verification/v1","files_verified":1}'
                )
            return b"{}"

        def create(self, *args):
            pass

        def start(self, *args):
            return "fixture log command"

        def stop(self, rank, name):
            stops.append(rank)

    monkeypatch.setattr(suite, "Pair", FakePair)
    monkeypatch.setattr(suite, "load_policy", lambda p: {})
    monkeypatch.setattr(suite, "load_site", lambda p: (site, []))
    monkeypatch.setattr(
        suite, "prepare_specs", lambda *a: (specs, [{"persistent": "cache"}] * 2, [])
    )
    monkeypatch.setattr(suite, "remote_cache_roots", lambda *a: None)
    monkeypatch.setattr(suite, "wait_ready", lambda *a, **k: None)
    monkeypatch.setattr(
        suite,
        "needle_fixture",
        lambda *a: {"messages": [], "expected": "SR-CODE", "sha256": "b" * 64},
    )
    replies = iter(
        [{"content": "42"}, {"content": "SR-CODE", "usage": {"prompt_tokens": 5000}}]
    )
    monkeypatch.setattr(suite, "chat", lambda *a, **k: next(replies))

    def no_publication(*a, **k):
        raise ValueError("No persisted chunks")

    monkeypatch.setattr(suite, "wait_publication", no_publication)
    with pytest.raises(ValueError, match="No persisted"):
        suite.run(
            tmp_path / "site.json",
            tmp_path / "policy.json",
            tmp_path / "lease.json",
            "sha256:" + "a" * 64,
            tmp_path / "out",
            run_id="fixture",
            input_sha256="c" * 64,
            gate_id="models",
        )
    result = json.loads((tmp_path / "out/result.json").read_text())
    assert result["outcome"] == "failed"
    assert result["evidence"]["cache"]["cold"]["content"] == "SR-CODE"
    assert result["evidence"]["failure"]["phase"] == "cache-publication"
    assert "restored" not in result["evidence"]["cache"]
    assert stops == [0, 1]
