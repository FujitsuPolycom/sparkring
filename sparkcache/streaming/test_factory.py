from __future__ import annotations

import dataclasses
import json
import logging
import types
import threading
import time
from pathlib import Path

import pytest

from sparkcache.persistent_context_cache.cache_manifest import (
    CacheIdentity,
    ContextChunk,
    ManifestStore,
    StateRecord,
)
from sparkcache.spark_context_cache_codec import LayerPlan, pack_positions
from sparkcache.streaming.factory import (
    ProductionStreamingSettings,
    SchedulerStreamingSnapshotAdapter,
    WorkerStreamingSnapshotAdapter,
    build_glm52_source_inventory,
    verify_production_lease_contract,
)
from sparkcache.streaming.block_lease import BlockLeaseRegistry, LeaseCapacity
from sparkcache.streaming.native_ring import NativeRingConfig
from sparkcache.streaming.timing import (
    TIMING_PREFIX,
    StreamingTimingTrace,
)

# The fake connector deliberately resolves these names from its defining
# module, exactly as the production adapter resolves the flat connector's
# canonical storage ABI exports.
CONNECTOR_ABI_EXPORTS = (ContextChunk, StateRecord, pack_positions)


class _ExtraConfig:
    def __init__(self, values: dict[str, object]) -> None:
        self._values = values

    def get_from_extra_config(self, key: str, default: object) -> object:
        return self._values.get(key, default)


def _connector(*, extra: dict[str, object] | None = None):
    return types.SimpleNamespace(
        _kv_transfer_config=_ExtraConfig(extra or {}),
    )


def test_worker_settings_require_attested_absolute_snapshot_library(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(
        "SPARK_CONTEXT_CACHE_STREAMING_NATIVE_LIBRARY",
        raising=False,
    )
    monkeypatch.delenv(
        "SPARK_CONTEXT_CACHE_STREAMING_NATIVE_LIBRARY_SHA256",
        raising=False,
    )

    with pytest.raises(
        RuntimeError,
        match="absolute native library path.*lowercase SHA-256",
    ):
        ProductionStreamingSettings.from_connector(_connector())


def test_worker_settings_enable_timing_only_with_explicit_one() -> None:
    base = {
        "spark_cache_streaming_native_library": (
            "/opt/spark/lib/libspcc_snapshot.so"
        ),
        "spark_cache_streaming_native_library_sha256": "a" * 64,
    }

    assert not ProductionStreamingSettings.from_connector(
        _connector(extra=base)
    ).timing_enabled
    assert ProductionStreamingSettings.from_connector(
        _connector(
            extra={
                **base,
                "spark_cache_streaming_timing": "1",
            }
        )
    ).timing_enabled
    with pytest.raises(RuntimeError, match="timing flag"):
        ProductionStreamingSettings.from_connector(
            _connector(
                extra={
                    **base,
                    "spark_cache_streaming_timing": "true",
                }
            )
        )


def test_worker_adapter_construction_is_cuda_and_native_inert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = ProductionStreamingSettings(
        native_library_path=Path("/opt/spark/lib/libspcc_snapshot.so"),
        native_library_sha256="a" * 64,
    )
    touched: list[object] = []

    adapter = WorkerStreamingSnapshotAdapter(
        types.SimpleNamespace(),
        settings=settings,
        ring_builder=lambda *_args, **_kwargs: touched.append("native"),
    )

    assert adapter.status()["bound"] is False
    assert touched == []


class _FakeRows:
    def __init__(
        self,
        *,
        pointer: int,
        width: int,
        rows: int = 4096,
        contiguous: bool = True,
    ) -> None:
        self._pointer = pointer
        self._width = width
        self.shape = (rows, width)
        self.device = types.SimpleNamespace(type="cuda", index=0)
        self._contiguous = contiguous

    def data_ptr(self) -> int:
        return self._pointer

    def element_size(self) -> int:
        return 1

    def stride(self, dimension: int) -> int:
        return (self._width, 1)[dimension]

    def is_contiguous(self) -> bool:
        return self._contiguous


class _FakeTensor:
    def __init__(self, rows: _FakeRows) -> None:
        self.rows = rows
        self.device = rows.device

    def data_ptr(self) -> int:
        return self.rows.data_ptr()

    def is_contiguous(self) -> bool:
        return self.rows.is_contiguous()


class _FakeConnector:
    pass


def _glm_inventory_connector(*, noncontiguous: str | None = None):
    plans: list[LayerPlan] = []
    tensors = {}
    pointer = 0x100000
    for ordinal in range(79):
        name = f"model.layers.{ordinal:02d}.mla"
        plans.append(LayerPlan(name, "target_ckv", 368))
        rows = _FakeRows(
            pointer=pointer,
            width=368,
            contiguous=name != noncontiguous,
        )
        tensors[name] = _FakeTensor(rows)
        pointer += 0x10000
    for ordinal in range(22):
        name = f"model.layers.{ordinal:02d}.indexer"
        plans.append(LayerPlan(name, "sparse_indexer", 132))
        rows = _FakeRows(
            pointer=pointer,
            width=132,
            contiguous=name != noncontiguous,
        )
        tensors[name] = _FakeTensor(rows)
        pointer += 0x10000
    connector = _FakeConnector()
    connector._plans = tuple(sorted(plans, key=lambda plan: plan.name))
    connector._layer_tensors = tensors
    connector._rows_view = lambda tensor: tensor.rows
    return connector


def test_glm_source_inventory_is_exact_grouped_and_retains_row_views() -> None:
    connector = _glm_inventory_connector()

    inventory = build_glm52_source_inventory(connector)

    assert len(inventory.sources) == 101
    assert [source.record_kind for source in inventory.sources] == [
        *([0] * 79),
        *([1] * 22),
    ]
    assert [source.source_layer_ordinal for source in inventory.sources[:79]] == list(
        range(79)
    )
    assert [source.source_layer_ordinal for source in inventory.sources[79:]] == list(
        range(22)
    )
    assert {source.bytes_per_token for source in inventory.sources[:79]} == {368}
    assert {source.bytes_per_token for source in inventory.sources[79:]} == {132}
    assert len(inventory.retained_row_views) == 101


def test_glm_source_inventory_rejects_copy_requiring_layout() -> None:
    connector = _glm_inventory_connector(
        noncontiguous="model.layers.03.mla"
    )

    with pytest.raises(RuntimeError, match="contiguous aliasing row view"):
        build_glm52_source_inventory(connector)


class _FakeRing:
    def __init__(self, config: NativeRingConfig) -> None:
        self.config = config
        self.configured_sources = None
        self.active_ticket_count = 0
        self.shutdown_count = 0

    def configure_sources(self, sources) -> None:
        self.configured_sources = tuple(sources)

    def shutdown(self) -> None:
        self.shutdown_count += 1


def _worker_connector(tmp_path: Path):
    connector = _glm_inventory_connector()
    connector._block_size = 64
    connector._dcp_degree = 4
    connector._worker_rank = lambda: 2
    connector._store = ManifestStore(tmp_path)
    connector._load_lock = threading.Lock()
    connector._held = set()
    connector.counters = {}
    identity = CacheIdentity(
        target_checkpoint="a" * 64,
        draft_checkpoint="a" * 64,
        quantization_layout="nvfp4_ds_mla-per-token-v1",
        rope_layout="glm52-rope-v1",
        tp_degree=4,
        dcp_degree=4,
        dcp_shard_rank=2,
        chunk_tokens=256,
        boundary_hidden_policy="live_forward",
        draft_kv_policy="colocated_target",
    )
    connector._identity = lambda rank: dataclasses.replace(
        identity,
        dcp_shard_rank=rank,
    )
    connector._identity_base = {
        "draft_kv_policy": "colocated_target",
    }
    return connector


def test_worker_bind_builds_only_proven_profile_once(
    tmp_path: Path,
) -> None:
    connector = _worker_connector(tmp_path)
    built: list[tuple[NativeRingConfig, Path, str]] = []

    def build_ring(
        config: NativeRingConfig,
        *,
        library_path: Path,
        expected_sha256: str,
    ) -> _FakeRing:
        built.append((config, library_path, expected_sha256))
        return _FakeRing(config)

    adapter = WorkerStreamingSnapshotAdapter(
        connector,
        settings=ProductionStreamingSettings(
            native_library_path=Path("/opt/spark/lib/libspcc_snapshot.so"),
            native_library_sha256="b" * 64,
        ),
        ring_builder=build_ring,
        progress_thread_initializer=lambda _device: None,
    )

    adapter.bind_kv_caches()
    adapter.bind_kv_caches()

    assert len(built) == 1
    config, path, digest = built[0]
    assert config == NativeRingConfig(
        arena_mode=1,
        slot_bytes=64 * 1024 * 1024,
        slot_count=2,
        max_sources=101,
        max_rows=1024,
        device_ordinal=0,
    )
    assert str(path).replace("\\", "/") == "/opt/spark/lib/libspcc_snapshot.so"
    assert digest == "b" * 64
    assert adapter.status()["bound"] is True
    assert adapter.status()["background_progress_alive"] is True
    time.sleep(0.02)
    assert (
        adapter.status()["counters"].get("background_progress_polls", 0) == 0
    )
    assert len(adapter._retained_row_views) == 101
    adapter.shutdown()
    assert adapter.status()["closed"] is True
    assert adapter.status()["background_progress_alive"] is False


def test_worker_finishes_manifest_without_another_foreground_callback(
    tmp_path: Path,
) -> None:
    connector = _worker_connector(tmp_path)
    published = threading.Event()
    initialized_devices: list[int] = []

    class Runtime:
        counters: dict[str, int] = {}

        def __init__(self) -> None:
            self.active_contexts = 0
            self.needs_progress = False
            self.poll_calls = 0
            self._committed: set[str] = set()

        def begin_context(self, **_kwargs) -> bool:
            self.active_contexts = 1
            return True

        def accept_completed_prefill(self, **_kwargs):
            self.needs_progress = True
            return types.SimpleNamespace(
                submitted_batches=1,
                active=True,
                aborted=False,
                reason=None,
            )

        def poll(self) -> int:
            self.poll_calls += 1
            self.active_contexts = 0
            self.needs_progress = True
            self._committed.add("9" * 64)
            return 1

        def take_aborted(self) -> dict[str, str]:
            return {}

        def take_committed(self) -> set[str]:
            committed, self._committed = self._committed, set()
            if committed:
                self.needs_progress = False
                published.set()
            return committed

        def shutdown(self) -> None:
            return None

    adapter = WorkerStreamingSnapshotAdapter(
        connector,
        settings=ProductionStreamingSettings(
            native_library_path=Path("/opt/spark/lib/libspcc_snapshot.so"),
            native_library_sha256="b" * 64,
        ),
        ring_builder=lambda config, **_kwargs: _FakeRing(config),
        progress_poll_seconds=0.001,
        progress_thread_initializer=initialized_devices.append,
    )
    adapter.bind_kv_caches()
    runtime = Runtime()
    adapter._runtime = runtime

    adapter.offer_completed(
        types.SimpleNamespace(
            request_id="request-idle-tail",
            digest="9" * 64,
            span_tokens=32768,
            completed_tokens=32768,
            block_ids=tuple(range(128)),
        ),
        producer_stream=7,
    )

    assert published.wait(1.0)
    assert initialized_devices == [0]
    assert connector._held == {"9" * 64}
    assert adapter.status()["counters"]["digests_advertised"] == 1
    assert runtime.poll_calls == 1
    adapter.shutdown()


def test_background_writer_failure_aborts_cache_without_failing_serving(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    connector = _worker_connector(tmp_path)
    aborted = threading.Event()

    class Runtime:
        counters: dict[str, int] = {}

        def __init__(self) -> None:
            self.active_contexts = 0
            self.needs_progress = False
            self._aborted: dict[str, str] = {}
            self._polled = False

        def begin_context(self, **_kwargs) -> bool:
            self.active_contexts = 1
            return True

        def accept_completed_prefill(self, **_kwargs):
            self.needs_progress = True
            return types.SimpleNamespace(
                submitted_batches=1,
                active=True,
                aborted=False,
                reason=None,
            )

        def poll(self) -> int:
            if self._polled:
                return 0
            self._polled = True
            self.active_contexts = 0
            self._aborted["request-writer-failure"] = "writer_result_failed"
            return 1

        def take_aborted(self) -> dict[str, str]:
            values, self._aborted = self._aborted, {}
            if values:
                self.needs_progress = False
                aborted.set()
            return values

        def take_committed(self) -> set[str]:
            return set()

        def shutdown(self) -> None:
            return None

    adapter = WorkerStreamingSnapshotAdapter(
        connector,
        settings=ProductionStreamingSettings(
            native_library_path=Path("/opt/spark/lib/libspcc_snapshot.so"),
            native_library_sha256="b" * 64,
        ),
        ring_builder=lambda config, **_kwargs: _FakeRing(config),
        progress_poll_seconds=0.001,
        progress_thread_initializer=lambda _device: None,
    )
    adapter.bind_kv_caches()
    adapter._runtime = Runtime()

    with caplog.at_level(
        logging.INFO,
        logger="vllm.spark_context_cache.streaming",
    ):
        adapter.offer_completed(
            types.SimpleNamespace(
                request_id="request-writer-failure",
                digest="8" * 64,
                span_tokens=32768,
                completed_tokens=32768,
                block_ids=tuple(range(128)),
            ),
            producer_stream=7,
        )
        assert aborted.wait(1.0)
        assert adapter.poll() == 0

    terminal_aborts = [
        record
        for record in caplog.records
        if '"event":"aborted"' in record.getMessage()
        and '"request_id":"request-writer-failure"' in record.getMessage()
    ]
    assert len(terminal_aborts) == 1
    assert connector._held == set()
    status = adapter.status()
    assert status["background_progress_error"] is None
    assert status["counters"]["runtime_aborted"] == 1
    adapter.shutdown()


def test_background_and_foreground_poll_are_serialized(tmp_path: Path) -> None:
    connector = _worker_connector(tmp_path)
    poll_entered = threading.Event()
    release_poll = threading.Event()

    class Runtime:
        counters: dict[str, int] = {}

        def __init__(self) -> None:
            self.active_contexts = 0
            self.needs_progress = False
            self.overlapped = False
            self._inside_poll = threading.Lock()
            self._first = True

        def begin_context(self, **_kwargs) -> bool:
            self.active_contexts = 1
            return True

        def accept_completed_prefill(self, **_kwargs):
            self.needs_progress = True
            return types.SimpleNamespace(
                submitted_batches=1,
                active=True,
                aborted=False,
                reason=None,
            )

        def poll(self) -> int:
            if not self._inside_poll.acquire(blocking=False):
                self.overlapped = True
                return 0
            try:
                if self._first:
                    self._first = False
                    poll_entered.set()
                    assert release_poll.wait(1.0)
                    self.needs_progress = False
                    self.active_contexts = 0
                return 0
            finally:
                self._inside_poll.release()

        def take_aborted(self) -> dict[str, str]:
            return {}

        def take_committed(self) -> set[str]:
            return set()

        def shutdown(self) -> None:
            return None

    adapter = WorkerStreamingSnapshotAdapter(
        connector,
        settings=ProductionStreamingSettings(
            native_library_path=Path("/opt/spark/lib/libspcc_snapshot.so"),
            native_library_sha256="b" * 64,
        ),
        ring_builder=lambda config, **_kwargs: _FakeRing(config),
        progress_poll_seconds=0.001,
        progress_thread_initializer=lambda _device: None,
    )
    adapter.bind_kv_caches()
    runtime = Runtime()
    adapter._runtime = runtime
    adapter.offer_completed(
        types.SimpleNamespace(
            request_id="request-serialized",
            digest="7" * 64,
            span_tokens=32768,
            completed_tokens=32768,
            block_ids=tuple(range(128)),
        ),
        producer_stream=7,
    )
    assert poll_entered.wait(1.0)

    foreground = threading.Thread(target=adapter.poll)
    foreground.start()
    time.sleep(0.02)
    assert foreground.is_alive()
    release_poll.set()
    foreground.join(1.0)

    assert not foreground.is_alive()
    assert runtime.overlapped is False
    adapter.shutdown()


def test_worker_poll_emits_registered_final_timing_once() -> None:
    digest = "9" * 64
    lines: list[str] = []

    class Runtime:
        active_contexts = 0
        counters: dict[str, int] = {}

        def poll(self) -> int:
            return 1

        def take_aborted(self) -> dict[str, str]:
            return {}

        def take_committed(self) -> set[str]:
            return {digest}

    connector = types.SimpleNamespace(
        _load_lock=threading.Lock(),
        _held=set(),
        counters={},
        _store=None,
    )
    adapter = WorkerStreamingSnapshotAdapter(
        connector,
        settings=ProductionStreamingSettings(
            native_library_path=Path("/opt/spark/lib/libspcc_snapshot.so"),
            native_library_sha256="b" * 64,
            timing_enabled=True,
        ),
    )
    timing = StreamingTimingTrace(enabled=True, sink=lines.append)
    timing.register_final("request-timing", 7, 32768)
    timing.mark("request-timing", 7, "final_watermark", at_ns=100)
    adapter._timing_trace = timing
    adapter._runtime = Runtime()
    adapter._leases = types.SimpleNamespace(
        active_leases=0,
        leased_blocks=0,
    )
    adapter._ring = types.SimpleNamespace(active_ticket_count=0)
    adapter._bound = True
    adapter._contexts["request-timing"] = (
        digest,
        32768,
        time.monotonic(),
    )
    adapter._admitted_requests.add("request-timing")
    adapter._terminal_pending_requests.add("request-timing")

    assert adapter.poll() == 1

    assert len(lines) == 1
    assert lines[0].startswith(TIMING_PREFIX)
    payload = json.loads(lines[0].removeprefix(TIMING_PREFIX))
    assert payload["request_id"] == "request-timing"
    assert payload["batch_index"] == 7
    assert "adapter_observed" in payload["stage_delta_ns"]


def test_bound_timing_trace_uses_the_configured_worker_logger(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    adapter = WorkerStreamingSnapshotAdapter(
        _worker_connector(tmp_path),
        settings=ProductionStreamingSettings(
            native_library_path=Path("/opt/spark/lib/libspcc_snapshot.so"),
            native_library_sha256="b" * 64,
            timing_enabled=True,
        ),
        ring_builder=lambda config, **_kwargs: _FakeRing(config),
        progress_thread_initializer=lambda _device: None,
    )
    adapter.bind_kv_caches()
    assert adapter._timing_trace is not None
    adapter._timing_trace.register_final("request-timing", 7, 32768)
    adapter._timing_trace.mark(
        "request-timing",
        7,
        "final_watermark",
        at_ns=100,
    )

    with caplog.at_level(
        logging.INFO,
        logger="vllm.spark_context_cache.streaming",
    ):
        adapter._timing_trace.mark(
            "request-timing",
            7,
            "adapter_observed",
            at_ns=200,
        )

    lines = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith(TIMING_PREFIX)
    ]
    assert len(lines) == 1
    adapter.shutdown()


def test_background_cuda_initialization_failure_is_sticky(
    tmp_path: Path,
) -> None:
    def fail_device_initialization(_device: int) -> None:
        raise RuntimeError("cannot establish CUDA device")

    adapter = WorkerStreamingSnapshotAdapter(
        _worker_connector(tmp_path),
        settings=ProductionStreamingSettings(
            native_library_path=Path("/opt/spark/lib/libspcc_snapshot.so"),
            native_library_sha256="b" * 64,
        ),
        ring_builder=lambda config, **_kwargs: _FakeRing(config),
        progress_thread_initializer=fail_device_initialization,
    )
    adapter.bind_kv_caches()
    deadline = time.monotonic() + 1.0
    while adapter.status()["background_progress_error"] is None:
        assert time.monotonic() < deadline
        time.sleep(0.001)

    with pytest.raises(RuntimeError, match="background progress failed"):
        adapter.poll()

    assert adapter.status()["counters"]["background_progress_failed"] == 1
    adapter.shutdown()


def test_scheduler_delays_only_observed_streaming_requests() -> None:
    adapter = SchedulerStreamingSnapshotAdapter(types.SimpleNamespace())
    metadata = types.SimpleNamespace(
        streaming_snapshot_offers=[
            types.SimpleNamespace(request_id="streaming")
        ],
        preempted_request_ids=(),
    )

    adapter.observe_metadata(metadata)

    assert adapter.request_finished("ordinary", (1, 2)) is False
    assert adapter.request_finished("streaming", (3, 4)) is True
    assert adapter.take_finished({"ordinary"}) == set()
    assert adapter.take_finished({"streaming"}) == {"streaming"}
    assert adapter.status()["eligible_requests"] == 0


def test_scheduler_same_step_preemption_then_resume_keeps_current_offer() -> None:
    adapter = SchedulerStreamingSnapshotAdapter(types.SimpleNamespace())
    adapter.observe_metadata(
        types.SimpleNamespace(
            streaming_snapshot_offers=[
                types.SimpleNamespace(request_id="preempted")
            ],
            preempted_request_ids=("preempted",),
        )
    )

    assert adapter.request_finished("preempted", ()) is True


def test_worker_preemption_retires_old_finished_sending_ownership() -> None:
    class _Runtime:
        active_contexts = 0
        counters: dict[str, int] = {}

        def preempt(self, _request_id: str) -> bool:
            # The optional cache transaction may already have committed or
            # aborted even though the adapter still remembers the old offer.
            return False

        def poll(self) -> int:
            return 0

        def take_aborted(self) -> dict[str, str]:
            return {}

        def take_committed(self) -> set[str]:
            return set()

    adapter = WorkerStreamingSnapshotAdapter(
        types.SimpleNamespace(),
        settings=ProductionStreamingSettings(
            native_library_path=Path("/opt/spark/lib/libspcc_snapshot.so"),
            native_library_sha256="d" * 64,
        ),
    )
    adapter._runtime = _Runtime()
    adapter._bound = True
    adapter._leases = BlockLeaseRegistry(
        LeaseCapacity(max_active_leases=2, max_leased_blocks=16)
    )
    adapter._seen_requests.add("preempted")
    adapter._pending_finished_requests.add("preempted")

    adapter.handle_preemptions(
        types.SimpleNamespace(preempted_request_ids=("preempted",))
    )

    # The scheduler adapter drops eligibility for this old generation. If the
    # client now aborts before resume, scheduler request_finished() returns
    # false and deletes it; the worker must not echo the ID as finished_sending.
    assert "preempted" not in adapter._seen_requests
    assert "preempted" not in adapter._pending_finished_requests
    assert adapter.take_finished({"preempted"}) == set()


def test_lease_contract_is_verified_from_explicit_paths(tmp_path: Path) -> None:
    vllm_root = tmp_path / "site-packages"
    source = vllm_root / "vllm" / "source.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"pinned allocator semantics\n")
    import hashlib
    import json

    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    contract = tmp_path / "lease-contract.json"
    contract.write_text(
        json.dumps(
            {
                "schema": "sparkring-vllm-kv-block-lease-contract/v1",
                "files": [
                    {
                        "path": "vllm/source.py",
                        "sha256": digest,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    settings = ProductionStreamingSettings(
        native_library_path=Path("/opt/spark/lib/libspcc_snapshot.so"),
        native_library_sha256="c" * 64,
        vllm_root=vllm_root,
        lease_contract_path=contract,
    )

    assert verify_production_lease_contract(settings) == (source.resolve(),)

    source.write_bytes(b"changed\n")
    with pytest.raises(RuntimeError, match="lease-contract attestation failed"):
        verify_production_lease_contract(settings)


class _AbortOnceRuntime:
    def __init__(self) -> None:
        self.accept_count = 0
        self.cancelled = []

    def begin_context(self, **_kwargs) -> bool:
        return True

    def accept_completed_prefill(self, **_kwargs):
        self.accept_count += 1
        return types.SimpleNamespace(aborted=True, reason="native_backpressure")

    def cancel(self, request_id: str, *, reason: str) -> bool:
        self.cancelled.append((request_id, reason))
        return True


def test_worker_ignores_later_watermarks_after_fail_open_abort() -> None:
    adapter = WorkerStreamingSnapshotAdapter(
        types.SimpleNamespace(),
        settings=ProductionStreamingSettings(
            native_library_path=Path("/opt/spark/lib/libspcc_snapshot.so"),
            native_library_sha256="d" * 64,
        ),
    )
    runtime = _AbortOnceRuntime()
    adapter._runtime = runtime
    adapter._bound = True
    offer = types.SimpleNamespace(
        request_id="request",
        digest="e" * 64,
        span_tokens=4096,
        completed_tokens=4096,
        block_ids=tuple(range(16)),
    )

    adapter.offer_completed(offer, producer_stream=0x123)
    offer.completed_tokens = 8192
    adapter.offer_completed(offer, producer_stream=0x123)

    assert runtime.accept_count == 1
    assert adapter.status()["counters"]["offers_suppressed_after_abort"] == 1


def test_worker_reports_only_seen_finished_requests_without_active_leases() -> None:
    adapter = WorkerStreamingSnapshotAdapter(
        types.SimpleNamespace(),
        settings=ProductionStreamingSettings(
            native_library_path=Path("/opt/spark/lib/libspcc_snapshot.so"),
            native_library_sha256="d" * 64,
        ),
    )
    adapter._runtime = object()
    adapter._bound = True
    adapter._leases = BlockLeaseRegistry(
        LeaseCapacity(max_active_leases=2, max_leased_blocks=16)
    )
    adapter._seen_requests.add("seen-no-lease")
    adapter.poll = lambda: 0

    assert adapter.take_finished({"seen-no-lease", "ordinary"}) == {
        "seen-no-lease"
    }
    assert adapter._leases.counters["requests_finished"] == 1
    assert "ordinary" not in adapter._seen_requests


def test_worker_retains_seen_finished_request_until_lease_completes() -> None:
    class _Fence:
        def __init__(self) -> None:
            self.done = False

        def query(self) -> bool:
            return self.done

        def synchronize(self) -> None:
            self.done = True

    adapter = WorkerStreamingSnapshotAdapter(
        types.SimpleNamespace(),
        settings=ProductionStreamingSettings(
            native_library_path=Path("/opt/spark/lib/libspcc_snapshot.so"),
            native_library_sha256="d" * 64,
        ),
    )
    adapter._runtime = object()
    adapter._bound = True
    adapter._leases = BlockLeaseRegistry(
        LeaseCapacity(max_active_leases=2, max_leased_blocks=16)
    )
    adapter._seen_requests.add("seen-with-lease")
    adapter.poll = lambda: 0
    fence = _Fence()
    lease = adapter._leases.try_reserve("seen-with-lease", (7,))
    assert lease is not None
    lease.submit(lambda: fence)

    # vLLM supplies a newly finished ID only once. The connector must retain
    # it if the asynchronous send is not ready during that call.
    assert adapter.take_finished({"seen-with-lease"}) == set()

    fence.done = True

    # A later empty call must still release the request after its lease drains.
    assert adapter.take_finished(set()) == {"seen-with-lease"}
    assert "seen-with-lease" not in adapter._seen_requests
    # A repeated or reordered upstream finished-ID set cannot make one worker
    # contribute a second vote to vLLM's occurrence-counting aggregator.
    assert adapter.take_finished({"seen-with-lease"}) == set()


def test_four_rank_finished_sending_requires_every_worker_once() -> None:
    class _Fence:
        def __init__(self, *, done: bool) -> None:
            self.done = done

        def query(self) -> bool:
            return self.done

        def synchronize(self) -> None:
            self.done = True

    class _PinnedVllmQuorum:
        """Characterize KVOutputAggregator's exact four-vote countdown."""

        def __init__(self) -> None:
            self.remaining: dict[str, int] = {}

        def aggregate(self, per_rank: list[set[str]]) -> set[str]:
            finished: set[str] = set()
            for request_ids in per_rank:
                for request_id in request_ids:
                    remaining = self.remaining.get(request_id, 4) - 1
                    if remaining == 0:
                        finished.add(request_id)
                        self.remaining.pop(request_id, None)
                    else:
                        self.remaining[request_id] = remaining
            return finished

    adapters: list[WorkerStreamingSnapshotAdapter] = []
    fences: list[_Fence | None] = [None]
    for rank in range(4):
        adapter = WorkerStreamingSnapshotAdapter(
            types.SimpleNamespace(),
            settings=ProductionStreamingSettings(
                native_library_path=Path("/opt/spark/lib/libspcc_snapshot.so"),
                native_library_sha256="d" * 64,
            ),
        )
        adapter._runtime = object()
        adapter._bound = True
        adapter._leases = BlockLeaseRegistry(
            LeaseCapacity(max_active_leases=2, max_leased_blocks=16)
        )
        adapter._seen_requests.add("request")
        adapter.poll = lambda: 0
        if rank:
            fence = _Fence(done=False)
            lease = adapter._leases.try_reserve("request", (rank,))
            assert lease is not None
            lease.submit(lambda fence=fence: fence)
            fences.append(fence)
        adapters.append(adapter)

    quorum = _PinnedVllmQuorum()

    # Rank 0 is ready, but one worker vote cannot release scheduler blocks.
    first = [adapter.take_finished({"request"}) for adapter in adapters]
    assert first == [{"request"}, set(), set(), set()]
    assert quorum.aggregate(first) == set()

    # The other ranks complete in later, differently grouped get_finished()
    # calls. Each adapter retained the ID from the original one-shot input.
    for index in range(1, 4):
        fence = fences[index]
        assert fence is not None
        fence.done = True
        votes = [adapter.take_finished(set()) for adapter in adapters]
        expected = {"request"} if index == 3 else set()
        assert quorum.aggregate(votes) == expected

    # Repeating the original ID cannot contribute a fifth/duplicate vote.
    duplicate = [adapter.take_finished({"request"}) for adapter in adapters]
    assert duplicate == [set(), set(), set(), set()]
    assert quorum.aggregate(duplicate) == set()


def test_worker_cancels_changed_offer_identity_without_new_submission() -> None:
    adapter = WorkerStreamingSnapshotAdapter(
        types.SimpleNamespace(),
        settings=ProductionStreamingSettings(
            native_library_path=Path("/opt/spark/lib/libspcc_snapshot.so"),
            native_library_sha256="d" * 64,
        ),
    )
    runtime = _AbortOnceRuntime()
    adapter._runtime = runtime
    adapter._bound = True
    adapter._contexts["request"] = ("1" * 64, 4096, time.monotonic())
    adapter._admitted_requests.add("request")
    changed = types.SimpleNamespace(
        request_id="request",
        digest="2" * 64,
        span_tokens=4096,
        completed_tokens=4096,
        block_ids=tuple(range(16)),
    )

    adapter.offer_completed(changed, producer_stream=0x123)

    assert runtime.accept_count == 0
    assert runtime.cancelled == [
        ("request", "offer_identity_changed")
    ]
    assert "request" not in adapter._admitted_requests
    assert "request" in adapter._suppressed_requests


class _TerminalRuntime:
    active_contexts = 0

    def __init__(self) -> None:
        self.counters = {}
        self._committed = {"1" * 64}
        self._aborted = {"aborted-request": "writer_result_failed"}

    def poll(self) -> int:
        return 2

    def take_committed(self) -> set[str]:
        committed, self._committed = self._committed, set()
        return committed

    def take_aborted(self) -> dict[str, str]:
        aborted, self._aborted = self._aborted, {}
        return aborted


class _DrainingAbortRuntime:
    def __init__(self, ring: _FakeRing) -> None:
        self.active_contexts = 1
        self.counters = {}
        self._ring = ring
        self._poll_count = 0
        self._aborted = {"aborted-request": "preempted"}

    def poll(self) -> int:
        self._poll_count += 1
        if self._poll_count == 1:
            return 0
        self.active_contexts = 0
        self._ring.active_ticket_count = 0
        return 1

    def take_committed(self) -> set[str]:
        return set()

    def take_aborted(self) -> dict[str, str]:
        aborted, self._aborted = self._aborted, {}
        return aborted


def test_worker_emits_drained_after_aborted_writer_releases_final_ticket(
    caplog: pytest.LogCaptureFixture,
) -> None:
    connector = types.SimpleNamespace(
        _load_lock=threading.Lock(),
        _held=set(),
        counters={},
    )
    adapter = WorkerStreamingSnapshotAdapter(
        connector,
        settings=ProductionStreamingSettings(
            native_library_path=Path("/opt/spark/lib/libspcc_snapshot.so"),
            native_library_sha256="f" * 64,
        ),
    )
    ring = _FakeRing(
        NativeRingConfig(
            arena_mode=1,
            slot_bytes=64 * 1024 * 1024,
            slot_count=2,
            max_sources=101,
            max_rows=1024,
            device_ordinal=0,
        )
    )
    ring.active_ticket_count = 1
    adapter._runtime = _DrainingAbortRuntime(ring)
    adapter._ring = ring
    adapter._leases = BlockLeaseRegistry(
        LeaseCapacity(max_active_leases=2, max_leased_blocks=2)
    )
    adapter._bound = True
    adapter._contexts = {
        "aborted-request": ("2" * 64, 4096, time.monotonic()),
    }

    with caplog.at_level(
        logging.INFO,
        logger="vllm.spark_context_cache.streaming",
    ):
        adapter.poll()
        first_events = [
            record.getMessage()
            for record in caplog.records
            if record.getMessage().startswith("[sparkcache-streaming-status]")
        ]
        assert not any('"event":"drained"' in event for event in first_events)
        adapter.poll()

    events = [
        json.loads(record.getMessage().split("] ", 1)[1])
        for record in caplog.records
        if record.getMessage().startswith("[sparkcache-streaming-status]")
    ]
    drained = [event for event in events if event["event"] == "drained"]
    assert len(drained) == 1
    assert drained[0]["active_contexts"] == 0
    assert drained[0]["active_leases"] == 0
    assert drained[0]["leased_blocks"] == 0
    assert drained[0]["active_tickets"] == 0


def test_worker_abort_reports_request_leases_separately_from_global_leases(
    caplog: pytest.LogCaptureFixture,
) -> None:
    connector = types.SimpleNamespace(
        _load_lock=threading.Lock(),
        _held=set(),
        counters={},
    )
    adapter = WorkerStreamingSnapshotAdapter(
        connector,
        settings=ProductionStreamingSettings(
            native_library_path=Path("/opt/spark/lib/libspcc_snapshot.so"),
            native_library_sha256="f" * 64,
        ),
    )
    leases = BlockLeaseRegistry(
        LeaseCapacity(max_active_leases=2, max_leased_blocks=2)
    )
    assert leases.try_reserve("other-request", (7,)) is not None
    adapter._runtime = _TerminalRuntime()
    adapter._leases = leases
    adapter._bound = True
    adapter._contexts = {
        "aborted-request": ("2" * 64, 4096, time.monotonic()),
    }

    with caplog.at_level(
        logging.INFO,
        logger="vllm.spark_context_cache.streaming",
    ):
        adapter.poll()

    events = [
        json.loads(record.getMessage().split("] ", 1)[1])
        for record in caplog.records
        if record.getMessage().startswith("[sparkcache-streaming-status]")
    ]
    aborted = next(event for event in events if event["event"] == "aborted")
    assert aborted["leases_after_abort"] == 0
    assert aborted["request_leases_after_abort"] == 0
    assert aborted["global_active_leases_after_abort"] == 1


def test_worker_advertises_only_post_manifest_committed_digests() -> None:
    connector = types.SimpleNamespace(
        _load_lock=threading.Lock(),
        _held=set(),
        counters={},
    )
    adapter = WorkerStreamingSnapshotAdapter(
        connector,
        settings=ProductionStreamingSettings(
            native_library_path=Path("/opt/spark/lib/libspcc_snapshot.so"),
            native_library_sha256="f" * 64,
        ),
    )
    adapter._runtime = _TerminalRuntime()
    adapter._bound = True
    adapter._contexts = {
        "committed-request": ("1" * 64, 4096, time.monotonic()),
        "aborted-request": ("2" * 64, 4096, time.monotonic()),
    }
    adapter._admitted_requests.update(adapter._contexts)
    adapter._terminal_pending_requests.update(adapter._contexts)

    assert connector._held == set()
    assert adapter.poll() == 2

    assert connector._held == {"1" * 64}
    assert connector.counters["streaming_store_committed"] == 1
    assert "committed-request" not in adapter._admitted_requests
    assert "aborted-request" not in adapter._admitted_requests
    assert "aborted-request" in adapter._suppressed_requests
    assert "2" * 64 not in connector._held


def test_finished_scheduler_generation_retains_context_until_late_commit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    digest = "3" * 64
    runtime = _TerminalRuntime()
    runtime._committed = set()
    runtime._aborted = {}
    connector = types.SimpleNamespace(
        _load_lock=threading.Lock(),
        _held=set(),
        counters={},
    )
    adapter = WorkerStreamingSnapshotAdapter(
        connector,
        settings=ProductionStreamingSettings(
            native_library_path=Path("/opt/spark/lib/libspcc_snapshot.so"),
            native_library_sha256="f" * 64,
        ),
    )
    adapter._runtime = runtime
    adapter._leases = BlockLeaseRegistry(
        LeaseCapacity(max_active_leases=2, max_leased_blocks=2)
    )
    adapter._bound = True
    adapter._seen_requests.add("late-commit")
    adapter._admitted_requests.add("late-commit")
    adapter._terminal_pending_requests.add("late-commit")
    adapter._contexts["late-commit"] = (
        digest,
        65536,
        time.monotonic(),
    )
    adapter._last_watermark_at["late-commit"] = time.monotonic()

    # vLLM may release scheduler ownership before the writer's manifest
    # publication becomes visible to the next worker-side poll.
    assert adapter.take_finished({"late-commit"}) == {"late-commit"}
    assert "late-commit" not in adapter._seen_requests
    assert "late-commit" in adapter._contexts
    assert "late-commit" in adapter._terminal_pending_requests

    runtime._committed = {digest}
    with caplog.at_level(
        logging.INFO,
        logger="vllm.spark_context_cache.streaming",
    ):
        adapter.poll()

    events = [
        json.loads(record.getMessage().split("] ", 1)[1])
        for record in caplog.records
        if record.getMessage().startswith("[sparkcache-streaming-status]")
    ]
    committed = next(event for event in events if event["event"] == "committed")
    assert committed["request_id"] == "late-commit"
    assert committed["context_digest"] == digest
    assert committed["span_tokens"] == 65536
    assert committed["final_tail_ms"] is not None
    assert connector._held == {digest}
    assert "late-commit" not in adapter._contexts
    assert "late-commit" not in adapter._terminal_pending_requests
    assert "late-commit" not in adapter._last_watermark_at
