"""Offline endpoint/collector contracts; no vLLM installation or GPU required."""

import asyncio
from enum import Enum
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib
from types import ModuleType, SimpleNamespace as NS
import zipfile

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from sparkring_runtime_status import collector as c
from sparkring_runtime_status import plugin as p


def config(tp=2):
    return NS(parallel_config=NS(tensor_parallel_size=tp, pipeline_parallel_size=1,
                                 decode_context_parallel_size=1, data_parallel_size=1,
                                 prefill_context_parallel_size=1),
              scheduler_config=NS(max_num_batched_tokens=8192, max_num_seqs=16,
                                  enable_chunked_prefill=True),
              cache_config=NS(block_size=1440, cache_dtype="fp8",
                              enable_prefix_caching=True, mamba_cache_mode="align",
                              recurrent_checkpoint_policy="aligned"),
              speculative_config=NS(method="mtp", num_speculative_tokens=3),
              kv_transfer_config=NS(kv_connector="SparkCacheConnector", kv_role="kv_both",
                                     kv_connector_extra_config={"token": "test-only"}),
              kernel_config=NS(linear_backend="b12x", moe_backend="b12x"),
              compilation_config=NS(max_cudagraph_capture_size=64))


def worker(rank=0):
    return NS(rank=rank, local_rank=0, vllm_config=config(), use_v2_model_runner=True)


def rows():
    return [c.worker_snapshot(worker(rank), environ={}, modules={}) for rank in range(2)]


class Engine:
    def __init__(self, result=None, release=None, error=None):
        self.result = rows() if result is None else result
        self.release = release
        self.error = error
        self.calls = []
        self.cancelled = False

    async def collective_rpc(self, method, timeout):
        self.calls.append((method, timeout))
        try:
            if self.release is not None:
                await self.release.wait()
            if self.error:
                raise self.error
            return self.result
        except asyncio.CancelledError:
            self.cancelled = True
            raise


def service(engine, **kwargs):
    return p.StatusService(config(), NS(block_size=32, api_key="test-only"), engine,
                           receipt={"state": "unknown"}, environ={}, **kwargs)


def test_configuration_distinguishes_arguments_from_effective_and_observed():
    result = asyncio.run(service(Engine()).snapshot())
    assert result["configured"]["arguments"]["block_size"]["value"] == 32
    assert result["effective"]["block_size"]["value"] == 1440
    assert result["effective"]["num_speculative_tokens"]["value"] == 3
    assert result["effective"]["max_cudagraph_capture_size"]["value"] == 64
    assert result["observed"]["kernel_execution"]["state"] == "not_observed"
    assert result["workers"]["state"] == "complete"
    assert result["workers"]["ranks"][1]["identity"]["rank"]["value"] == 1
    assert "test-only" not in json.dumps(result)


def test_no_properties_repr_or_tensor_reads():
    class Trap:
        @property
        def tensor_parallel_size(self):
            raise AssertionError("property evaluated")

        def __getattr__(self, name):
            raise AssertionError("dynamic lookup")

        def __repr__(self):
            raise AssertionError("repr")

        def item(self):
            raise AssertionError("tensor read")

    obj = worker()
    obj.vllm_config.parallel_config = Trap()
    obj.model_runner = Trap()
    obj._b12x_session = Trap()
    result = c.worker_snapshot(obj, environ={}, modules={})
    assert result["effective"]["tensor_parallel_size"]["state"] == "unknown"
    assert result["effective"]["kernel_preparation"]["state"] == "unknown"
    assert "test-only" not in json.dumps(result)


def test_environment_allowlist_and_unknown_unset_flags():
    result = c.environment({"API_KEY": "test-only", "VLLM_API_KEY": "test-only",
                            "SPARKCACHE_ENABLED": "1", "VLLM_QWEN3_8_PREFILL_COALESCE": "0",
                            "VLLM_QWEN3_8_HC_PREFILL_MODE": "shard",
                            "SPARKRING_TRANSPORT_PROFILE": "secret-value",
                            "QWEN_HC_FUSION": "invalid"})
    assert result["SPARKCACHE_ENABLED"]["value"] is True
    assert result["VLLM_QWEN3_8_PREFILL_COALESCE"]["value"] is False
    assert result["VLLM_QWEN3_8_HC_PREFILL_MODE"]["value"] == "shard"
    assert result["QWEN_HC_FUSION"]["state"] == "unknown"
    assert result["B12X_AUTOTUNE"]["state"] == "unknown"
    assert "test-only" not in json.dumps(result)
    assert "secret-value" not in json.dumps(result)


def test_first_use_marker_is_not_real_request_proof():
    obj = worker()
    model = NS(hc_prefill_mode="off", _modules={"hyper_connection_workspace": NS(tp_size=4)})
    obj.model_runner = NS(model=NS(_modules={"model": model}))
    module = NS(announced={"prefill", "test-only"})
    result = c.worker_snapshot(obj, environ={}, modules={"qwen4_hc_fusion": module})
    assert result["effective"]["hc_projection_tp_size"]["value"] == 4
    assert result["effective"]["hc_prefill_row_ownership"]["value"] == "off"
    marker = result["observed"]["hyperconnection"]
    assert marker["categories"] == ["prefill"]
    assert marker["phase"] == "unspecified_includes_warmup"
    assert marker["current_request_execution"] == "not_observed"


def test_qwen_multimodal_language_model_chain_exposes_resident_hc_fields():
    class Module:
        def __init__(self, **children):
            self._modules = children

        def __getattr__(self, name):
            raise AssertionError("module property or dynamic lookup")

    text_model = Module(hyper_connection_workspace=NS(tp_size=4))
    text_model.hc_prefill_mode = "off"
    causal_lm = Module(model=text_model)
    conditional_generation = Module(language_model=causal_lm, visual=NS(secret="test-only"))
    obj = worker()
    obj.model_runner = NS(model=conditional_generation)
    result = c.worker_snapshot(obj, environ={}, modules={})
    assert result["effective"]["hc_projection_tp_size"]["value"] == 4
    assert result["effective"]["hc_prefill_row_ownership"]["value"] == "off"
    assert "test-only" not in json.dumps(result)


@pytest.mark.parametrize("stored_world_size", [False, True])
def test_pcp_worker_count_matches_addressed_executor(stored_world_size):
    cfg = config()
    cfg.parallel_config.prefill_context_parallel_size = 2
    cfg.parallel_config.data_parallel_size = 3
    cfg.parallel_config.distributed_executor_backend = "mp"
    if stored_world_size:
        cfg.parallel_config.world_size = 4
    engine = Engine(result=[c.worker_snapshot(worker(rank), environ={}, modules={}) for rank in range(4)])
    status = p.StatusService(cfg, NS(), engine, receipt={"state": "unknown"}, environ={})
    result = asyncio.run(status.snapshot())
    assert result["workers"]["expected_count"] == 4
    assert result["workers"]["state"] == "complete"
    assert result["workers"]["missing_count"] == 0


def test_external_launcher_reports_only_its_local_executor_worker():
    cfg = config()
    cfg.parallel_config.distributed_executor_backend = "external_launcher"
    cfg.parallel_config.data_parallel_size = 3
    cfg.parallel_config.world_size = 6
    engine = Engine(result=[c.worker_snapshot(worker(4), environ={}, modules={})])
    status = p.StatusService(cfg, NS(), engine, receipt={"state": "unknown"}, environ={})
    result = asyncio.run(status.snapshot())
    assert result["workers"]["expected_count"] == 1
    assert result["workers"]["state"] == "complete"
    assert result["workers"]["ranks"][0]["identity"]["rank"]["value"] == 4


def test_resident_transport_policy_and_loader_options_are_metadata_only():
    obj = worker()
    obj.vllm_config.load_config = NS(load_format="b12x", model_loader_extra_config={
        "read_mode": "bounce", "io_threads": 8, "access_key": "test-only"})
    obj.vllm_config.additional_config = {"gdn_decode_kernel": "b12x", "token": "test-only"}
    policy = NS(MODE="both", LIMIT=20480)
    result = c.worker_snapshot(obj, environ={}, modules={"qwen38_collective_policy": policy})
    assert result["effective"]["loader_read_mode"]["value"] == "bounce"
    assert result["effective"]["loader_io_threads"]["value"] == 8
    assert result["effective"]["gdn_decode_kernel"]["value"] == "b12x"
    assert result["effective"]["qwen_collective_routing_mode"]["value"] == "both"
    assert result["effective"]["qwen_collective_allreduce_cutoff_bytes"]["value"] == 20480
    assert result["observed"]["transport"]["state"] == "not_observed"
    assert "test-only" not in json.dumps(result)


def test_resident_plan_sample_is_bounded_and_cannot_prepare():
    class Plan:
        @property
        def selection(self):
            raise AssertionError("must not invoke plan properties")

        def memory_requirements(self):
            raise AssertionError("must not resolve device or programs")

    plan = Plan()
    plan.query = NS(max_rows=8, in_features=4608, out_features=2560,
                    request="test-only")
    plan._prepared = NS(closed=False,
                        selection=NS(component_id="gemm.blockscaled_precision", source="tuned",
                                     config=NS(backend="cute", split_k_slices=2, tile_n=128,
                                               request_text="test-only")),
                        device=NS(identity=NS(compute_capability=(12, 1), sm_count=48)))
    obj = worker()
    obj._b12x_session = NS(state="FROZEN", _plans=[plan] * 50)
    before = plan._prepared
    result = c.prepared_choices(obj)
    assert result["plan_count"] == 50 and result["inspected_count"] == 32
    assert result["truncated"] is True and len(result["selections"]) == 32
    assert result["device"] == {"compute_capability": [12, 1], "sm_count": 48}
    assert result["phase"] == "preparation" and result["execution"] == "not_observed"
    assert result["selections"][0]["config"]["split_k_slices"] == 2
    assert plan._prepared is before
    assert "test-only" not in json.dumps(result)


@pytest.mark.parametrize("component,configuration,query", [
    ("gemm.blockscaled_precision", {"mode": "fused_mxfp8", "split_k": 4},
     {"num_tokens": 8, "in_features": 4096, "out_features": 512}),
    ("sequence.gdn_prefill", {"algorithm": "sequential", "segment_tokens": 256,
                              "v_split": 64, "k_split": 1, "stages": 3, "window_tiles": 64},
     {"max_tokens": 8192, "key_heads": 4, "value_heads": 8,
      "head_dim": 128, "checkpoint_export": True}),
])
def test_prepared_mxfp8_and_gdn_choices_use_current_metadata_fields(component, configuration, query):
    plan = NS(query=NS(**query), _prepared=NS(closed=False,
              selection=NS(component_id=component, source="cached", config=NS(**configuration))))
    obj = worker()
    obj._b12x_session = NS(state="FROZEN", _plans=[plan])
    result = c.prepared_choices(obj)
    assert result["selections"][0]["config"] == configuration
    assert result["selections"][0]["query"] == query
    assert result["execution"] == "not_observed"


def test_no_speculation_is_known_disabled_missing_config_remains_unknown():
    cfg = config()
    cfg.speculative_config = None
    assert c.configuration(cfg)["speculative_enabled"]["value"] is False
    assert c.configuration(None)["speculative_enabled"]["state"] == "unknown"


def test_enum_serialization():
    class Mode(Enum):
        FULL_AND_PIECEWISE = 3
    cfg = config()
    cfg.compilation_config.cudagraph_mode = Mode.FULL_AND_PIECEWISE
    assert c.configuration(cfg)["cudagraph_mode"]["value"] == "FULL_AND_PIECEWISE"


def test_receipt_is_allowlisted_and_not_a_fresh_file_audit(tmp_path):
    data = {"schema": "sparkring-external-installed/v1",
            "base": {"config_id": "sha256:" + "c" * 64, "reference": "test-only"},
            "composition_sha256": "d" * 64,
            "sources": {"vllm": {"commit": "a" * 40, "archive_sha256": "b" * 64,
                                   "archive": "test-only"}},
            "files": {"test-only": "test-only"}, "tokens": "test-only",
            "capabilities": {"transport_profile": "tp2-rocenante-adaptive-prepared",
                             "transport_manifest_sha256": "e" * 64}}
    raw = json.dumps(data).encode()
    (tmp_path / "external-base-installed.json").write_bytes(raw)
    result = c.provenance(tmp_path)
    assert result["receipt_sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["sources"]["vllm"]["commit"] == "a" * 40
    assert result["runtime_image_id"]["state"] == "unknown"
    assert result["verification"] == "receipt_read_at_plugin_startup_no_fresh_file_audit"
    assert "test-only" not in json.dumps(result)


@pytest.mark.parametrize("payload,reason", [(b"broken", "receipt_unreadable"),
                                           (b"{}", "unsupported_receipt_schema"),
                                           (b"[" * 2048 + b"]" * 2048, "receipt_unreadable"),
                                           (b" " * (c.MAX_RECEIPT_BYTES + 1), "receipt_too_large")],
                         ids=["malformed", "wrong-schema", "nested", "oversized"])
def test_bad_receipt_is_reported_not_fatal(tmp_path, payload, reason):
    (tmp_path / "external-base-installed.json").write_bytes(payload)
    assert c.provenance(tmp_path) == {"state": "unknown", "reason": reason}


def test_absent_receipt(tmp_path):
    assert c.provenance(tmp_path)["reason"] == "installed_receipt_missing"


def test_one_pending_rpc_survives_timeouts_and_client_cancellation():
    async def run():
        release = asyncio.Event()
        engine = Engine(release=release)
        status = service(engine, wait_seconds=0.001, cache_seconds=60)
        results = await asyncio.gather(*(status.snapshot() for _ in range(8)))
        assert all(r["workers"]["state"] == "pending" for r in results)
        assert len(engine.calls) == 1 and engine.calls[0] == (p.METHOD, None)
        request = asyncio.create_task(status.snapshot())
        await asyncio.sleep(0)
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        assert engine.cancelled is False and status.pending is not None
        release.set()
        # A slow collection is cached from completion, not from its start.
        status.last_attempt -= 120
        await status.pending
        result = await status.snapshot()
        assert result["workers"]["state"] == "complete"
        for _ in range(8):
            await status.snapshot()
        assert len(engine.calls) == 1 and not engine.cancelled
    asyncio.run(run())


def test_schema_accepts_complete_partial_missing_and_initializing():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(Path(__file__).with_name("schema-v1.json").read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)
    for engine in (Engine(), Engine(result=[rows()[0]]), None):
        validator.validate(asyncio.run(service(engine).snapshot()))
    validator.validate({"schema": c.SCHEMA, "state": "initializing"})


def test_offline_wheel_contains_both_official_entrypoints(tmp_path):
    root = Path(__file__).parent
    project = tmp_path / "project"
    project.mkdir()
    for name in ("pyproject.toml", "README.md"):
        shutil.copyfile(root / name, project / name)
    shutil.copytree(root / "sparkring_runtime_status", project / "sparkring_runtime_status",
                    ignore=shutil.ignore_patterns("__pycache__"))
    metadata = tomllib.loads((project / "pyproject.toml").read_text())
    assert set(metadata["project"]["entry-points"]) == {"vllm.endpoint_plugins", "vllm.general_plugins"}
    result = subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps",
                             "--no-build-isolation", "--no-index", "--wheel-dir", "wheels", "."],
                            cwd=project, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    wheels = list((project / "wheels").glob("*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as archive:
        name = next(n for n in archive.namelist() if n.endswith("/entry_points.txt"))
        entrypoints = archive.read(name).decode()
        assert "[vllm.endpoint_plugins]" in entrypoints
        assert "sparkring_status = sparkring_runtime_status.plugin:StatusPlugin" in entrypoints
        assert "[vllm.general_plugins]" in entrypoints
        assert "sparkring_status = sparkring_runtime_status.plugin:register_worker_method" in entrypoints
        assert "sparkring_runtime_status/collector.py" in archive.namelist()


def test_failed_rpc_is_rate_limited_and_exception_details_redacted():
    async def run():
        engine = Engine(error=RuntimeError("api key test-only"))
        status = service(engine)
        first = await status.snapshot()
        assert first["workers"]["state"] == "error"
        assert first["workers"]["error"] == "worker_rpc_failed"
        await status.snapshot()
        assert len(engine.calls) == 1
        assert "test-only" not in json.dumps(first)
    asyncio.run(run())


def test_partial_rank_errors_and_missing_rank_count():
    result = asyncio.run(service(Engine(result=[rows()[0]])).snapshot())
    assert result["workers"]["state"] == "partial"
    assert result["workers"]["missing_count"] == 1
    assert result["workers"]["missing_rpc_slots"] == [1]
    failed = {"schema": c.WORKER_SCHEMA, "error": "snapshot_failed",
              "identity": {"rank": c.fact(1, source="worker_instance")}}
    result = asyncio.run(service(Engine(result=[rows()[0], failed])).snapshot())
    assert result["workers"]["state"] == "partial"
    assert result["workers"]["received_count"] == 2


def test_duplicate_rank_identities_never_report_complete():
    result = asyncio.run(service(Engine(result=[rows()[0], rows()[0]])).snapshot())
    assert result["workers"]["state"] == "partial"
    assert result["workers"]["rank_identities_unique"] is False


def test_stale_data_remains_marked_while_refresh_is_pending():
    async def run():
        engine = Engine()
        status = service(engine, wait_seconds=0.001)
        await status.snapshot()
        status.cached_monotonic -= 10
        status.last_attempt -= 10
        engine.release = asyncio.Event()
        result = await status.snapshot()
        assert result["workers"]["state"] == "pending"
        assert result["workers"]["stale"] is True
        assert result["workers"]["cache_age_seconds"] >= 10
        assert result["workers"]["received_count"] == 2
        engine.release.set()
        await status.pending
    asyncio.run(run())


def test_route_initialization_read_only_and_no_engine_degrades():
    app = FastAPI()
    plugin = p.StatusPlugin()
    plugin.attach_router(app)
    with TestClient(app) as client:
        assert client.get("/v1/sparkring/status").status_code == 503
        app.state.sparkring_status_service = service(None)
        response = client.get("/v1/sparkring/status")
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        assert response.json()["workers"]["state"] == "unavailable"
        assert client.post("/v1/sparkring/status").status_code == 405


def test_worker_registration_is_unique_idempotent_and_does_not_wrap_execution(monkeypatch):
    module = ModuleType("vllm.v1.worker.worker_base")
    class WorkerBase:
        def execute_model(self):
            pass
    original = WorkerBase.execute_model
    module.WorkerBase = WorkerBase
    monkeypatch.setitem(sys.modules, module.__name__, module)
    p.register_worker_method()
    p.register_worker_method()
    assert WorkerBase.execute_model is original
    instance = WorkerBase()
    instance.rank = 0
    assert getattr(instance, p.METHOD)()["schema"] == c.WORKER_SCHEMA
    setattr(WorkerBase, p.METHOD, lambda self: None)
    with pytest.raises(RuntimeError, match="already registered"):
        p.register_worker_method()


def test_official_authentication_middleware_guards_status_if_source_available():
    source_root = os.environ.get("SPARKRING_TEST_VLLM_ROOT")
    if not source_root:
        pytest.skip("set SPARKRING_TEST_VLLM_ROOT to test the pinned vLLM middleware")
    source = Path(source_root) / "vllm/entrypoints/serve/middleware/authenticate.py"
    if not source.exists():
        pytest.skip("optional pinned vLLM source is unavailable")
    spec = importlib.util.spec_from_file_location("status_test_vllm_auth", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    app = FastAPI()
    p.StatusPlugin().attach_router(app)
    app.state.sparkring_status_service = service(None)
    app.add_middleware(module.AuthenticationMiddleware, tokens=["test-secret"])
    with TestClient(app) as client:
        assert client.get("/v1/sparkring/status").status_code == 401
        response = client.get("/v1/sparkring/status", headers={"Authorization": "Bearer test-secret"})
        assert response.status_code == 200
        assert "test-secret" not in response.text
