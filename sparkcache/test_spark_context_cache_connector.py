"""GPU-free tests for the SparkRing persistent context-cache connector.

Simulates a four-rank DCP4 store -> pool wipe -> restore cycle with real
byte comparisons on CPU torch tensors, plus the fail-closed sabotage paths.
vLLM is stubbed the same way as the sibling backend suites.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import struct
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

import spark_context_cache_codec as codec


def _install_vllm_stubs() -> None:
    if "vllm" in sys.modules:
        return
    vllm = types.ModuleType("vllm")
    config = types.ModuleType("vllm.config")

    class VllmConfig:  # noqa: D401 - stub
        pass

    config.VllmConfig = VllmConfig
    logger_mod = types.ModuleType("vllm.logger")

    class _Logger:
        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

    logger_mod.init_logger = lambda name: _Logger()
    distributed = types.ModuleType("vllm.distributed")

    class _Group:
        rank_in_group = 0
        world_size = 4

    distributed.get_dcp_group = lambda: _Group
    kv_transfer = types.ModuleType("vllm.distributed.kv_transfer")
    kv_connector = types.ModuleType("vllm.distributed.kv_transfer.kv_connector")
    v1 = types.ModuleType("vllm.distributed.kv_transfer.kv_connector.v1")
    base = types.ModuleType("vllm.distributed.kv_transfer.kv_connector.v1.base")
    metrics = types.ModuleType("vllm.distributed.kv_transfer.kv_connector.v1.metrics")
    import dataclasses as _dc

    @_dc.dataclass
    class KVConnectorStats:
        data: dict = _dc.field(default_factory=dict)

        def reset(self):
            raise NotImplementedError

        def aggregate(self, other):
            raise NotImplementedError

        def reduce(self):
            raise NotImplementedError

        def is_empty(self):
            raise NotImplementedError

    metrics.KVConnectorStats = KVConnectorStats

    import enum

    class KVConnectorRole(enum.Enum):
        SCHEDULER = 0
        WORKER = 1

    class KVConnectorMetadata:
        pass

    class KVConnectorBase_V1:
        def __init__(self, *, vllm_config, role, kv_cache_config):
            self._vllm_config = vllm_config
            self._role = role
            self._kv_transfer_config = vllm_config.kv_transfer_config
            self._metadata = None

        def bind_connector_metadata(self, metadata):
            self._metadata = metadata

        def clear_connector_metadata(self):
            self._metadata = None

        def _get_connector_metadata(self):
            return self._metadata

    base.KVConnectorBase_V1 = KVConnectorBase_V1
    base.KVConnectorMetadata = KVConnectorMetadata
    base.KVConnectorRole = KVConnectorRole
    for name, module in (
        ("vllm", vllm),
        ("vllm.config", config),
        ("vllm.logger", logger_mod),
        ("vllm.distributed", distributed),
        ("vllm.distributed.kv_transfer", kv_transfer),
        ("vllm.distributed.kv_transfer.kv_connector", kv_connector),
        ("vllm.distributed.kv_transfer.kv_connector.v1", v1),
        ("vllm.distributed.kv_transfer.kv_connector.v1.base", base),
        ("vllm.distributed.kv_transfer.kv_connector.v1.metrics", metrics),
    ):
        sys.modules[name] = module


_install_vllm_stubs()

import spark_context_cache_connector as connector_module  # noqa: E402
from spark_context_cache_connector import (  # noqa: E402
    SparkCacheConnectorMetadata,
    SparkContextCacheConnector,
    _ReqPlan,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # noqa: E402
    KVConnectorRole,
)


class CodecTests(unittest.TestCase):
    def test_owned_positions_interleave_one(self) -> None:
        self.assertEqual(codec.owned_positions(8, 4, 0), (0, 4))
        self.assertEqual(codec.owned_positions(8, 4, 3), (3, 7))
        union: set[int] = set()
        for rank in range(4):
            union.update(codec.owned_positions(1024, 4, rank))
        self.assertEqual(union, set(range(1024)))

    def test_local_slots_dense_prefix(self) -> None:
        # ordinal o = p // 4; block table [7, 2]; block size 4
        slots = codec.local_slots_for_positions(
            (1, 5, 9, 13, 17, 21, 25, 29), (7, 2), 4, 4
        )
        self.assertEqual(slots, (28, 29, 30, 31, 8, 9, 10, 11))
        with self.assertRaises(codec.CodecError):
            codec.local_slots_for_positions((33,), (7, 2), 4, 4)

    def test_record_round_trip_and_trailing_rejection(self) -> None:
        plans = codec.build_layer_plans({"a.attn": 8, "b.indexer": 4, "c.draft": 2})
        rows = 3
        payload = codec.pack_record(
            plans, "target_ckv", {"a.attn": bytes(range(24))}, rows
        )
        split = codec.unpack_record(plans, "target_ckv", payload, rows)
        self.assertEqual(split, {"a.attn": bytes(range(24))})
        with self.assertRaises(codec.CodecError):
            codec.unpack_record(plans, "target_ckv", payload + b"x", rows)

    def test_layer_plans_require_all_kinds(self) -> None:
        with self.assertRaises(codec.CodecError):
            codec.build_layer_plans({"a.attn": 8, "b.indexer": 4})

    def test_digest_binds_identity_salt(self) -> None:
        tokens = list(range(64))
        self.assertNotEqual(
            codec.context_digest(tokens, "salt-a"),
            codec.context_digest(tokens, "salt-b"),
        )

    def test_vectorized_integer_codec_matches_v1_wire_bytes(self) -> None:
        tokens = [0, 1, 255, 65535, 2**32 - 1]
        legacy_token_bytes = b"".join(
            token.to_bytes(4, "little", signed=False) for token in tokens
        )
        legacy_digest = hashlib.sha256(b"identity\x00" + legacy_token_bytes).hexdigest()

        self.assertEqual(codec.context_digest(tokens, "identity"), legacy_digest)
        self.assertEqual(codec.pack_positions(tokens), legacy_token_bytes)
        self.assertEqual(codec.unpack_positions(legacy_token_bytes), tuple(tokens))
        self.assertEqual(
            legacy_token_bytes,
            struct.pack("<5I", *tokens),
        )

    def test_vectorized_integer_codec_rejects_out_of_range_values(self) -> None:
        for bad in (-1, 2**32):
            with self.subTest(bad=bad):
                with self.assertRaises(codec.CodecError):
                    codec.pack_positions([bad])
                with self.assertRaises(codec.CodecError):
                    codec.context_digest([bad], "identity")


def _make_connector(
    root: Path,
    rank: int,
    block_size: int = 64,
    extra_config: dict[str, object] | None = None,
    role: KVConnectorRole = KVConnectorRole.WORKER,
    override_worker_rank: bool = True,
) -> SparkContextCacheConnector:
    values = {
        "spark_cache_root": str(root),
        "spark_cache_min_span_tokens": "256",
        "spark_cache_target_id": "test-target",
        "spark_cache_draft_id": "test-draft",
        "spark_cache_draft_policy": "separate",
    }
    values.update(extra_config or {})
    kv_transfer_config = types.SimpleNamespace(
        get_from_extra_config=lambda key, default=None: values.get(key, default)
    )
    vllm_config = types.SimpleNamespace(
        kv_transfer_config=kv_transfer_config,
        cache_config=types.SimpleNamespace(block_size=block_size),
        parallel_config=types.SimpleNamespace(
            tensor_parallel_size=4, decode_context_parallel_size=4
        ),
        model_config=types.SimpleNamespace(model="test-target"),
    )
    connector = SparkContextCacheConnector(
        vllm_config=vllm_config,
        role=role,
        kv_cache_config=None,
    )
    if override_worker_rank:
        connector._worker_rank = lambda: rank  # type: ignore[method-assign]
    return connector


class _FakeCudaTensor:
    def __init__(self, pointer: int, width: int):
        self.shape = (8, 64, width)
        self.device = types.SimpleNamespace(type="cuda", index=0)
        self._pointer = pointer
        self._width = width

    def dim(self):
        return 3

    def __getitem__(self, _index):
        return types.SimpleNamespace(
            numel=lambda: self._width,
            element_size=lambda: 1,
        )

    def is_contiguous(self):
        return True

    def element_size(self):
        return 1

    def stride(self):
        return (64 * self._width, self._width, 1)

    def data_ptr(self):
        return self._pointer


def _fake_cuda_pools():
    return {
        name: _FakeCudaTensor(0x100000 + index * 0x10000, width)
        for index, (name, width) in enumerate(_LAYERS.items())
    }


_LAYERS = {
    "model.layers.0.self_attn.attn": 368,
    "model.layers.0.self_attn.indexer_cache": 128,
    "draft.layers.0.self_attn.attn": 368,
}


class NativeRestoreSelectionTests(unittest.TestCase):
    def test_native_restore_is_disabled_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connector = _make_connector(Path(directory), 0)

        self.assertFalse(connector._native_restore_enabled)
        self.assertIsNone(connector._native_adapter)
        self.assertEqual(connector._load_thread_limit, 1)

    def test_disabled_native_mode_ignores_stale_native_settings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connector = _make_connector(
                Path(directory),
                0,
                extra_config={
                    "spark_cache_native_restore": "0",
                    "spark_cache_native_library": "not-absolute",
                    "spark_cache_native_library_sha256": "UPPERCASE",
                    "spark_cache_native_arena_bytes": "not-an-integer",
                    "spark_cache_native_io_workers": "also-invalid",
                },
            )
            connector.register_kv_caches(_make_pools(8, 64))

        self.assertFalse(connector._native_restore_enabled)
        self.assertIsNone(connector._native_adapter)

    def test_native_restore_requires_all_three_attested_settings(self) -> None:
        cases = (
            {},
            {"spark_cache_native_library": "/tmp/placement.so"},
            {
                "spark_cache_native_library": "/tmp/placement.so",
                "spark_cache_native_library_sha256": "0" * 64,
            },
        )
        for missing in cases:
            with self.subTest(missing=missing):
                with tempfile.TemporaryDirectory() as directory:
                    settings = {
                        "spark_cache_native_restore": "1",
                        **missing,
                    }
                    with self.assertRaisesRegex(
                        RuntimeError, "native restore requires"
                    ):
                        _make_connector(Path(directory), 0, extra_config=settings)

    def test_native_library_hash_failure_stops_registration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "placement.so"
            artifact.write_bytes(b"not-the-pinned-library")
            connector = _make_connector(
                Path(directory),
                0,
                extra_config={
                    "spark_cache_native_restore": "1",
                    "spark_cache_native_library": str(artifact),
                    "spark_cache_native_library_sha256": "0" * 64,
                    "spark_cache_native_arena_bytes": str(64 * 1024 * 1024),
                    "spark_cache_load_threads": "2",
                },
            )

            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                connector.register_kv_caches(_fake_cuda_pools())

            self.assertIsNone(connector._native_adapter)
            self.assertEqual(
                connector._load_thread_limit,
                1,
                "native restores must be serialized regardless of requested"
                " Python load-thread count",
            )

    def test_attested_native_adapter_is_configured_after_cache_registration(
        self,
    ) -> None:
        calls = []
        adapter = types.SimpleNamespace(
            configure=lambda plans, tensors: calls.append(
                ("configure", tuple(plans), dict(tensors))
            ),
            close=lambda: calls.append(("close",)),
        )

        class FakeLibrary:
            @classmethod
            def load(cls, path, *, expected_sha256):
                calls.append(("load", Path(path), expected_sha256))
                return "attested-library"

        class FakeAdapter:
            @classmethod
            def create(cls, library, **kwargs):
                calls.append(("create", library, kwargs))
                return adapter

        components = types.SimpleNamespace(
            NativePlacementLibrary=FakeLibrary,
            NativePlacementAdapter=FakeAdapter,
            ArenaMode=types.SimpleNamespace(MAPPED_HOST=1),
            RecordKind=types.SimpleNamespace(
                TARGET_CKV=0,
                SPARSE_INDEXER=1,
                MTP_DRAFT_KV=2,
            ),
            execute_native_restore=lambda **_kwargs: None,
        )

        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "placement.so"
            artifact.write_bytes(b"mock")
            connector = _make_connector(
                Path(directory),
                0,
                extra_config={
                    "spark_cache_native_restore": "true",
                    "spark_cache_native_library": str(artifact),
                    "spark_cache_native_library_sha256": "a" * 64,
                    "spark_cache_native_arena_bytes": str(128 * 1024 * 1024),
                },
            )
            with mock.patch.object(
                connector_module,
                "_load_native_components",
                return_value=components,
            ):
                connector.register_kv_caches(_fake_cuda_pools())

        self.assertEqual(calls[0][0], "load")
        self.assertEqual(calls[1][0], "create")
        self.assertEqual(calls[2][0], "configure")
        create = calls[1][2]
        self.assertEqual(create["arena_bytes"], 128 * 1024 * 1024)
        self.assertEqual(create["arena_mode"], 1)
        self.assertEqual(create["device_ordinal"], 0)
        self.assertIs(connector._native_adapter, adapter)

    def test_scheduler_role_never_creates_a_native_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "placement.so"
            artifact.write_bytes(b"scheduler-does-not-load-this")
            connector = _make_connector(
                Path(directory),
                0,
                extra_config={
                    "spark_cache_native_restore": "1",
                    "spark_cache_native_library": str(artifact),
                    "spark_cache_native_library_sha256": "b" * 64,
                    "spark_cache_native_arena_bytes": str(64 * 1024 * 1024),
                },
                role=KVConnectorRole.SCHEDULER,
            )
            with mock.patch.object(
                connector_module,
                "_load_native_components",
                side_effect=AssertionError(
                    "scheduler role must not load native placement"
                ),
            ):
                connector.register_kv_caches(_fake_cuda_pools())

        self.assertIsNone(connector._native_adapter)

    def test_enabled_native_load_never_falls_back_to_python_assembly(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connector = _make_connector(Path(directory), 0, 64)
            connector.register_kv_caches(_make_pools(8, 64))
            plan = _ReqPlan(
                "native-restore",
                "9" * 64,
                1024,
                (3, 0, 5, 1),
                False,
            )
            connector.bind_connector_metadata(
                SparkCacheConnectorMetadata(
                    plans=[dataclasses.replace(plan, is_store=True)]
                )
            )
            connector.wait_for_save()
            _drain_store(connector)
            lookup = connector._store.lookup(
                connector._identity(0), plan.digest, verify_chunks=False
            )
            self.assertTrue(lookup.is_hit)
            connector._native_restore_enabled = True
            connector._native_adapter = object()
            connector._native_arena_bytes = 64 * 1024 * 1024
            connector._native_io_workers = 4
            connector._native_required_record_mask = 0b111
            observed = {}

            def execute(**kwargs):
                observed.update(kwargs)
                return types.SimpleNamespace(
                    verified_chunks=4,
                    verified_encoded_bytes=12345,
                    slabs=1,
                    read_and_hash_ms=2.0,
                    parse_and_submit_ms=1.0,
                    finish_ms=0.5,
                )

            connector._native_execute_restore = execute
            connector._store.restore = mock.Mock(
                side_effect=AssertionError(
                    "native selection must not enter Python assembly"
                )
            )

            self.assertTrue(connector._load_one(plan))

        self.assertEqual(observed["request_id"], "native-restore")
        self.assertEqual(observed["lookup"], lookup)
        self.assertEqual(
            observed["slots"],
            tuple(
                codec.local_slots_for_positions(
                    codec.owned_positions(1024, 4, 0),
                    plan.block_ids,
                    64,
                    4,
                )
            ),
        )
        self.assertEqual(connector.counters["native_load_verified"], 1)

    def test_native_failure_invalidates_entry_without_python_fallback(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connector = _make_connector(Path(directory), 0, 64)
            connector.register_kv_caches(_make_pools(8, 64))
            plan = _ReqPlan(
                "native-failure",
                "8" * 64,
                1024,
                (3, 0, 5, 1),
                False,
            )
            connector.bind_connector_metadata(
                SparkCacheConnectorMetadata(
                    plans=[dataclasses.replace(plan, is_store=True)]
                )
            )
            connector.wait_for_save()
            _drain_store(connector)
            self.assertIn(plan.digest, connector._held)
            connector._native_restore_enabled = True
            connector._native_adapter = object()
            connector._native_arena_bytes = 64 * 1024 * 1024
            connector._native_io_workers = 4
            connector._native_required_record_mask = 0b111
            connector._native_execute_restore = mock.Mock(
                side_effect=Exception("authenticated chunk changed")
            )
            connector._store.restore = mock.Mock(
                side_effect=AssertionError(
                    "partial native failure must never fall back"
                )
            )

            self.assertFalse(connector._load_one(plan))

            self.assertNotIn(plan.digest, connector._held)
            self.assertFalse(
                connector._store.lookup(
                    connector._identity(0),
                    plan.digest,
                    verify_chunks=False,
                ).is_hit
            )

    def test_shutdown_never_closes_adapter_under_a_live_loader(self) -> None:
        class HungThread:
            def __init__(self):
                self.join_timeouts = []

            def join(self, timeout=None):
                self.join_timeouts.append(timeout)

            def is_alive(self):
                return True

        with tempfile.TemporaryDirectory() as directory:
            connector = _make_connector(Path(directory), 0)
            adapter = mock.Mock()
            thread = HungThread()
            connector._native_adapter = adapter
            connector._load_threads = [thread]
            connector.wait_for_pending_loads = mock.Mock(return_value=False)

            connector.shutdown()

        adapter.close.assert_not_called()
        self.assertIs(connector._native_adapter, adapter)
        self.assertEqual(connector.counters["native_shutdown_handle_leaked"], 1)
        self.assertEqual(len(thread.join_timeouts), 1)


def _make_pools(blocks: int, block_size: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260728)
    return {
        name: torch.randint(
            0,
            256,
            (blocks, block_size, width),
            dtype=torch.uint8,
            generator=generator,
        )
        for name, width in _LAYERS.items()
    }


def _empty_scheduler_output():
    return types.SimpleNamespace(
        scheduled_new_reqs=[],
        num_scheduled_tokens={},
        scheduled_cached_reqs=types.SimpleNamespace(
            req_ids=[],
            resumed_req_ids=set(),
            num_computed_tokens=[],
            new_block_ids=[],
        ),
    )


def _drain(connector: SparkContextCacheConnector, timeout: float = 30.0):
    assert connector.wait_for_pending_loads(timeout=timeout)
    _, received = connector.get_finished(set())
    return received


def _drain_store(
    connector: SparkContextCacheConnector,
    timeout: float = 30.0,
) -> None:
    assert connector.wait_for_pending_stores(timeout=timeout)


class ConnectorRoundTripTests(unittest.TestCase):
    SPAN = 1024
    BLOCK_SIZE = 64

    def _plan(self) -> _ReqPlan:
        # span 1024, dcp4 -> 256 local ordinals -> 4 blocks of 64
        return _ReqPlan(
            request_id="req-0",
            digest="f" * 64,
            span_tokens=self.SPAN,
            block_ids=(3, 0, 5, 1),
            is_store=True,
        )

    def test_four_rank_store_wipe_restore_is_byte_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pools: list[dict[str, torch.Tensor]] = []
            originals: list[dict[str, torch.Tensor]] = []
            connectors = []
            for rank in range(4):
                root = Path(directory) / f"rank{rank}"
                connector = _make_connector(root, rank, self.BLOCK_SIZE)
                pool = _make_pools(8, self.BLOCK_SIZE)
                connector.register_kv_caches(pool)
                connectors.append(connector)
                pools.append(pool)
                originals.append({k: v.clone() for k, v in pool.items()})
            store_meta = SparkCacheConnectorMetadata(plans=[self._plan()])
            for connector in connectors:
                connector.bind_connector_metadata(store_meta)
                connector.wait_for_save()
                _drain_store(connector)
                self.assertEqual(connector.counters["store_committed"], 1)
                self.assertEqual(connector.counters["store_failed"], 0)

            load_plan = dataclasses.replace(self._plan(), is_store=False)
            load_meta = SparkCacheConnectorMetadata(plans=[load_plan])
            for rank, connector in enumerate(connectors):
                for tensor in pools[rank].values():
                    tensor.zero_()
                connector.bind_connector_metadata(load_meta)
                connector.start_load_kv(None)
                self.assertEqual(_drain(connector), {"req-0"})
                self.assertEqual(connector.get_finished(set()), (None, None))
                self.assertEqual(connector.counters["load_verified"], 1)
                self.assertEqual(connector.get_block_ids_with_load_errors(), set())

            slots = codec.local_slots_for_positions(
                codec.owned_positions(self.SPAN, 4, 0),
                self._plan().block_ids,
                self.BLOCK_SIZE,
                4,
            )
            slot_tensor = torch.tensor(slots, dtype=torch.long)
            for rank in range(4):
                for name in _LAYERS:
                    restored = pools[rank][name].reshape(-1, _LAYERS[name])
                    original = originals[rank][name].reshape(-1, _LAYERS[name])
                    torch.testing.assert_close(
                        restored[slot_tensor],
                        original[slot_tensor],
                        rtol=0,
                        atol=0,
                    )
                    untouched = torch.ones(restored.shape[0], dtype=torch.bool)
                    untouched[slot_tensor] = False
                    self.assertTrue(
                        (restored[untouched] == 0).all(),
                        "load wrote outside the restored slots",
                    )

    def test_bit_flip_on_one_rank_reports_load_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rank2"
            connector = _make_connector(root, 2, self.BLOCK_SIZE)
            pool = _make_pools(8, self.BLOCK_SIZE)
            connector.register_kv_caches(pool)
            connector.bind_connector_metadata(
                SparkCacheConnectorMetadata(plans=[self._plan()])
            )
            connector.wait_for_save()
            _drain_store(connector)
            chunk_files = sorted((root / "chunks").glob("*.spcc"))
            self.assertTrue(chunk_files)
            corrupted = bytearray(chunk_files[0].read_bytes())
            corrupted[len(corrupted) // 2] ^= 0x40
            chunk_files[0].write_bytes(bytes(corrupted))

            load_plan = dataclasses.replace(self._plan(), is_store=False)
            connector.bind_connector_metadata(
                SparkCacheConnectorMetadata(plans=[load_plan])
            )
            connector.start_load_kv(None)
            self.assertEqual(_drain(connector), {"req-0"})
            self.assertEqual(connector.counters["load_failed"], 1)
            self.assertEqual(
                connector.get_block_ids_with_load_errors(),
                set(load_plan.block_ids),
            )
            # errors drain once reported
            self.assertEqual(connector.get_block_ids_with_load_errors(), set())

    def test_truncated_manifest_is_clean_miss(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rank1"
            connector = _make_connector(root, 1, self.BLOCK_SIZE)
            pool = _make_pools(8, self.BLOCK_SIZE)
            connector.register_kv_caches(pool)
            connector.bind_connector_metadata(
                SparkCacheConnectorMetadata(plans=[self._plan()])
            )
            connector.wait_for_save()
            _drain_store(connector)
            manifests = sorted((root / "manifests").rglob("*.json"))
            self.assertTrue(manifests)
            payload = manifests[0].read_bytes()
            manifests[0].write_bytes(payload[: len(payload) // 2])

            load_plan = dataclasses.replace(self._plan(), is_store=False)
            connector.bind_connector_metadata(
                SparkCacheConnectorMetadata(plans=[load_plan])
            )
            connector.start_load_kv(None)
            self.assertEqual(_drain(connector), {"req-0"})
            self.assertEqual(connector.counters["load_failed"], 1)
            self.assertEqual(
                connector.get_block_ids_with_load_errors(),
                set(load_plan.block_ids),
            )


class SchedulerChunkedPrefillTests(unittest.TestCase):
    def test_store_plan_accumulates_full_block_table(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connector = _make_connector(Path(directory), 0, block_size=64)
            token_ids = list(range(1100))  # span aligns to 1024
            step1 = types.SimpleNamespace(
                scheduled_new_reqs=[
                    types.SimpleNamespace(
                        req_id="req-c",
                        prompt_token_ids=token_ids,
                        num_computed_tokens=0,
                        block_ids=([10, 11],),
                    )
                ],
                num_scheduled_tokens={"req-c": 512},
                scheduled_cached_reqs=types.SimpleNamespace(
                    req_ids=[],
                    resumed_req_ids=set(),
                    num_computed_tokens=[],
                    new_block_ids=[],
                ),
            )
            meta1 = connector.build_connector_meta(step1)
            self.assertEqual(meta1.plans, [])
            step2 = types.SimpleNamespace(
                scheduled_new_reqs=[],
                num_scheduled_tokens={"req-c": 512},
                scheduled_cached_reqs=types.SimpleNamespace(
                    req_ids=["req-c"],
                    resumed_req_ids=set(),
                    num_computed_tokens=[512],
                    new_block_ids=[([12, 13],)],
                ),
            )
            meta2 = connector.build_connector_meta(step2)
            self.assertEqual(len(meta2.plans), 1)
            plan = meta2.plans[0]
            self.assertTrue(plan.is_store)
            self.assertEqual(plan.span_tokens, 1024)
            self.assertEqual(plan.block_ids, (10, 11, 12, 13))
            self.assertEqual(connector._store_progress, {})


class StartupDiscoveryTests(unittest.TestCase):
    def test_startup_discovers_manifest_without_restoring_chunk_payloads(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rank1"
            writer = _make_connector(root, 1, 64)
            writer.register_kv_caches(_make_pools(8, 64))
            plan = _ReqPlan("seed", "a" * 64, 1024, (3, 0, 5, 1), True)
            writer._store_one(plan)

            restarted = _make_connector(root, 1, 64)
            restarted._store.restore = mock.Mock(
                side_effect=AssertionError(
                    "startup discovery must not materialize chunk payloads"
                )
            )
            original_read_bytes = Path.read_bytes

            def reject_chunk_reads(path: Path) -> bytes:
                if path.suffix == ".spcc":
                    raise AssertionError("startup discovery must be O(manifest bytes)")
                return original_read_bytes(path)

            with mock.patch.object(Path, "read_bytes", reject_chunk_reads):
                restarted.register_kv_caches(_make_pools(8, 64))

            self.assertIn(plan.digest, restarted._held)
            restarted._store.restore.assert_not_called()

    def test_startup_rejects_and_removes_truncated_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rank1"
            writer = _make_connector(root, 1, 64)
            writer.register_kv_caches(_make_pools(8, 64))
            plan = _ReqPlan("seed", "b" * 64, 1024, (3, 0, 5, 1), True)
            writer._store_one(plan)
            manifest_path = next((root / "manifests").rglob("*.json"))
            manifest = manifest_path.read_bytes()
            manifest_path.write_bytes(manifest[: len(manifest) // 2])

            restarted = _make_connector(root, 1, 64)
            restarted.register_kv_caches(_make_pools(8, 64))

            self.assertNotIn(plan.digest, restarted._held)
            self.assertFalse(manifest_path.exists())
            self.assertEqual(restarted.counters["discovery_rejected"], 1)

    def test_startup_rejects_invalid_chunk_descriptor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rank1"
            writer = _make_connector(root, 1, 64)
            writer.register_kv_caches(_make_pools(8, 64))
            plan = _ReqPlan("seed", "c" * 64, 1024, (3, 0, 5, 1), True)
            writer._store_one(plan)
            chunk_path = next((root / "chunks").glob("*.spcc"))
            manifest_path = next((root / "manifests").rglob("*.json"))
            manifest = json.loads(manifest_path.read_text(encoding="ascii"))
            manifest["chunks"][0]["bytes"] = -1
            manifest_path.write_text(json.dumps(manifest), encoding="ascii")

            restarted = _make_connector(root, 1, 64)
            restarted.register_kv_caches(_make_pools(8, 64))

            self.assertNotIn(plan.digest, restarted._held)
            self.assertFalse(manifest_path.exists())
            self.assertTrue(chunk_path.exists())

    def test_rejected_manifest_cannot_delete_shared_healthy_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rank1"
            writer = _make_connector(root, 1, 64)
            writer.register_kv_caches(_make_pools(8, 64))
            healthy = _ReqPlan("healthy", "1" * 64, 1024, (3, 0, 5, 1), True)
            malicious = dataclasses.replace(
                healthy,
                request_id="malicious",
                digest="2" * 64,
            )
            writer._store_one(healthy)
            writer._store_one(malicious)
            chunk_path = next((root / "chunks").glob("*.spcc"))
            identity = writer._identity(1)
            malicious_manifest = (
                root / "manifests" / identity.storage_key / f"{malicious.digest}.json"
            )
            manifest = json.loads(malicious_manifest.read_text(encoding="ascii"))
            manifest["chunks"][0]["bytes"] = chunk_path.stat().st_size + 1
            malicious_manifest.write_text(json.dumps(manifest), encoding="ascii")

            restarted = _make_connector(root, 1, 64)
            restarted.register_kv_caches(_make_pools(8, 64))

            self.assertNotIn(malicious.digest, restarted._held)
            self.assertFalse(malicious_manifest.exists())
            self.assertTrue(chunk_path.exists())
            self.assertIn(healthy.digest, restarted._held)
            lookup = restarted._store.lookup(identity, healthy.digest)
            self.assertTrue(lookup.is_hit, lookup.reason)
            self.assertIsNotNone(restarted._store.restore(lookup))

    def test_discovery_cannot_erase_concurrent_commit_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rank1"
            writer = _make_connector(root, 1, 64)
            writer.register_kv_caches(_make_pools(8, 64))
            existing = _ReqPlan("existing", "3" * 64, 1024, (3, 0, 5, 1), True)
            writer._store_one(existing)

            restarted = _make_connector(root, 1, 64)
            restarted.register_kv_caches(_make_pools(8, 64))
            committed_digest = "4" * 64
            with restarted._store_cv:
                restarted._store_inflight = 1

            scan_started = threading.Event()
            publication_finished = threading.Event()
            original_lookup = restarted._store.lookup

            def gated_lookup(*args, **kwargs):
                scan_started.set()
                publication_finished.wait(timeout=0.2)
                return original_lookup(*args, **kwargs)

            restarted._store.lookup = gated_lookup
            discovery = threading.Thread(target=restarted.discover_manifests)

            def publish() -> None:
                restarted._finish_store(committed_digest, committed=True)
                publication_finished.set()

            publisher = threading.Thread(target=publish)
            discovery.start()
            self.assertTrue(scan_started.wait(timeout=5))
            publisher.start()
            discovery.join(timeout=5)
            publisher.join(timeout=5)

            self.assertFalse(discovery.is_alive())
            self.assertFalse(publisher.is_alive())
            self.assertIn(existing.digest, restarted._held)
            self.assertIn(committed_digest, restarted._held)

    def test_startup_rejects_manifest_with_missing_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rank1"
            writer = _make_connector(root, 1, 64)
            writer.register_kv_caches(_make_pools(8, 64))
            plan = _ReqPlan("missing", "d" * 64, 1024, (3, 0, 5, 1), True)
            writer._store_one(plan)
            next((root / "chunks").glob("*.spcc")).unlink()

            restarted = _make_connector(root, 1, 64)
            restarted._store.restore = mock.Mock(
                side_effect=AssertionError(
                    "missing chunks must be rejected before restore"
                )
            )
            restarted.register_kv_caches(_make_pools(8, 64))

            self.assertNotIn(plan.digest, restarted._held)
            restarted._store.restore.assert_not_called()
            self.assertFalse(next((root / "manifests").rglob("*.json"), None))

    def test_startup_rejects_manifest_with_wrong_chunk_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rank1"
            writer = _make_connector(root, 1, 64)
            writer.register_kv_caches(_make_pools(8, 64))
            plan = _ReqPlan("short", "f" * 64, 1024, (3, 0, 5, 1), True)
            writer._store_one(plan)
            chunk_path = next((root / "chunks").glob("*.spcc"))
            encoded = chunk_path.read_bytes()
            chunk_path.write_bytes(encoded[:-1])

            restarted = _make_connector(root, 1, 64)
            restarted._store.restore = mock.Mock(
                side_effect=AssertionError(
                    "wrong-size chunks must be rejected before restore"
                )
            )
            restarted.register_kv_caches(_make_pools(8, 64))

            self.assertNotIn(plan.digest, restarted._held)
            restarted._store.restore.assert_not_called()
            self.assertFalse(next((root / "manifests").rglob("*.json"), None))
            self.assertTrue(
                chunk_path.exists(),
                "metadata-only startup cannot prove an unshared chunk is bad",
            )

    def test_corrupt_chunk_is_offered_then_fails_closed_and_revokes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rank1"
            writer = _make_connector(root, 1, 64)
            writer.register_kv_caches(_make_pools(8, 64))
            plan = _ReqPlan("corrupt", "e" * 64, 1024, (3, 0, 5, 1), True)
            writer._store_one(plan)
            chunk_path = next((root / "chunks").glob("*.spcc"))
            encoded = bytearray(chunk_path.read_bytes())
            encoded[len(encoded) // 2] ^= 0x40
            chunk_path.write_bytes(encoded)

            restarted = _make_connector(root, 1, 64)
            pool = _make_pools(8, 64)
            for tensor in pool.values():
                tensor.zero_()
            restarted.register_kv_caches(pool)
            self.assertIn(plan.digest, restarted._held)

            load_plan = dataclasses.replace(plan, is_store=False)
            restarted.bind_connector_metadata(
                SparkCacheConnectorMetadata(plans=[load_plan])
            )
            restarted.start_load_kv(None)

            self.assertEqual(_drain(restarted), {"corrupt"})
            self.assertEqual(
                restarted.get_block_ids_with_load_errors(),
                set(load_plan.block_ids),
            )
            self.assertNotIn(plan.digest, restarted._held)
            self.assertEqual(
                restarted.get_kv_connector_stats().data,
                {"reports": [{"rank": 1, "held": []}]},
            )
            for tensor in pool.values():
                self.assertTrue((tensor == 0).all())


class SweepTests(unittest.TestCase):
    def test_sweep_invalidates_only_damaged_entries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rank1"
            connector = _make_connector(root, 1, 64)
            pool = _make_pools(8, 64)
            connector.register_kv_caches(pool)
            plan = _ReqPlan("req-s", "a" * 64, 1024, (3, 0, 5, 1), True)
            connector.bind_connector_metadata(SparkCacheConnectorMetadata(plans=[plan]))
            connector.wait_for_save()
            _drain_store(connector)
            # healthy sweep keeps the entry
            result = connector.sweep_integrity()
            self.assertEqual(result["checked"], 1)
            self.assertEqual(result["invalidated"], 0)
            # damage it, then sweep removes it
            chunk = sorted((root / "chunks").glob("*.spcc"))[0]
            payload = bytearray(chunk.read_bytes())
            payload[len(payload) // 2] ^= 0x20
            chunk.write_bytes(bytes(payload))
            result = connector.sweep_integrity()
            self.assertEqual(result["checked"], 1)
            self.assertEqual(result["invalidated"], 1)
            self.assertEqual(connector.sweep_integrity()["checked"], 0)


class AsyncStoreTests(unittest.TestCase):
    def test_commit_runs_off_request_path_and_publishes_only_after_success(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connector = _make_connector(Path(directory), 0, 64)
            connector.register_kv_caches(_make_pools(8, 64))
            digest = "6" * 64
            connector.bind_connector_metadata(
                SparkCacheConnectorMetadata(
                    plans=[
                        _ReqPlan(
                            "store-me",
                            digest,
                            1024,
                            (3, 0, 5, 1),
                            True,
                        )
                    ]
                )
            )
            commit_started = threading.Event()
            release_commit = threading.Event()
            wait_returned = threading.Event()
            caller_errors: list[BaseException] = []
            original_commit = connector._store.commit

            def gated_commit(**kwargs):
                commit_started.set()
                if not release_commit.wait(timeout=5):
                    raise TimeoutError("test did not release background commit")
                return original_commit(**kwargs)

            def call_wait_for_save() -> None:
                try:
                    connector.wait_for_save()
                except BaseException as error:  # noqa: BLE001 - test capture
                    caller_errors.append(error)
                finally:
                    wait_returned.set()

            connector._store.commit = gated_commit
            connector.sweep_integrity = mock.Mock(
                side_effect=AssertionError(
                    "request completion must not sweep the whole store"
                )
            )
            caller = threading.Thread(target=call_wait_for_save)
            caller.start()
            self.assertTrue(commit_started.wait(timeout=2))
            try:
                self.assertTrue(
                    wait_returned.wait(timeout=0.5),
                    "wait_for_save blocked on the durable NVMe commit",
                )
                self.assertNotIn(digest, connector._held)
                self.assertEqual(
                    connector.get_kv_connector_stats().data,
                    {"reports": [{"rank": 0, "held": []}]},
                )
            finally:
                release_commit.set()
                caller.join(timeout=5)

            self.assertEqual(caller_errors, [])
            self.assertTrue(connector.wait_for_pending_stores(timeout=5))
            self.assertIn(digest, connector._held)
            self.assertEqual(connector.counters["store_committed"], 1)
            connector.sweep_integrity.assert_not_called()

    def test_snapshot_owns_bytes_before_source_blocks_can_be_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connector = _make_connector(Path(directory), 0, 64)
            pool = _make_pools(8, 64)
            original = {name: tensor.clone() for name, tensor in pool.items()}
            connector.register_kv_caches(pool)
            plan = _ReqPlan(
                "store-me",
                "7" * 64,
                1024,
                (3, 0, 5, 1),
                True,
            )
            connector.bind_connector_metadata(SparkCacheConnectorMetadata(plans=[plan]))
            commit_started = threading.Event()
            release_commit = threading.Event()
            original_commit = connector._store.commit

            def gated_commit(**kwargs):
                commit_started.set()
                self.assertTrue(release_commit.wait(timeout=5))
                return original_commit(**kwargs)

            connector._store.commit = gated_commit
            connector.wait_for_save()
            self.assertTrue(commit_started.wait(timeout=2))
            for tensor in pool.values():
                tensor.zero_()
            release_commit.set()
            self.assertTrue(connector.wait_for_pending_stores(timeout=5))

            connector.bind_connector_metadata(
                SparkCacheConnectorMetadata(
                    plans=[dataclasses.replace(plan, is_store=False)]
                )
            )
            connector.start_load_kv(None)
            self.assertEqual(_drain(connector), {"store-me"})

            slots = codec.local_slots_for_positions(
                codec.owned_positions(1024, 4, 0),
                plan.block_ids,
                64,
                4,
            )
            slot_tensor = torch.tensor(slots, dtype=torch.long)
            for name in _LAYERS:
                actual = pool[name].reshape(-1, _LAYERS[name])
                expected = original[name].reshape(-1, _LAYERS[name])
                torch.testing.assert_close(
                    actual[slot_tensor],
                    expected[slot_tensor],
                    rtol=0,
                    atol=0,
                )

    def test_busy_saver_rejects_before_taking_another_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connector = _make_connector(Path(directory), 0, 64)
            connector.register_kv_caches(_make_pools(8, 64))
            first = _ReqPlan(
                "store-a",
                "8" * 64,
                1024,
                (3, 0, 5, 1),
                True,
            )
            second = dataclasses.replace(
                first,
                request_id="store-b",
                digest="9" * 64,
            )
            commit_started = threading.Event()
            release_commit = threading.Event()
            original_commit = connector._store.commit

            def gated_commit(**kwargs):
                commit_started.set()
                self.assertTrue(release_commit.wait(timeout=5))
                return original_commit(**kwargs)

            connector._store.commit = gated_commit
            with mock.patch.object(
                connector,
                "_snapshot_store",
                wraps=connector._snapshot_store,
            ) as snapshot:
                connector.bind_connector_metadata(
                    SparkCacheConnectorMetadata(plans=[first])
                )
                connector.wait_for_save()
                self.assertTrue(commit_started.wait(timeout=2))
                connector.bind_connector_metadata(
                    SparkCacheConnectorMetadata(plans=[second])
                )
                connector.wait_for_save()

                self.assertEqual(snapshot.call_count, 1)
                self.assertEqual(connector.counters["store_skipped_busy"], 1)
                self.assertNotIn(second.digest, connector._held)

            release_commit.set()
            self.assertTrue(connector.wait_for_pending_stores(timeout=5))
            self.assertIn(first.digest, connector._held)
            self.assertNotIn(second.digest, connector._held)

    def test_busy_rejection_preserves_a_preexisting_committed_digest(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connector = _make_connector(Path(directory), 0, 64)
            connector.register_kv_caches(_make_pools(8, 64))
            existing = _ReqPlan(
                "existing",
                "a" * 64,
                1024,
                (3, 0, 5, 1),
                True,
            )
            connector.bind_connector_metadata(
                SparkCacheConnectorMetadata(plans=[existing])
            )
            connector.wait_for_save()
            _drain_store(connector)
            self.assertIn(existing.digest, connector._held)
            self.assertTrue(
                connector._store.lookup(
                    connector._identity(0),
                    existing.digest,
                ).is_hit
            )

            active = dataclasses.replace(
                existing,
                request_id="active",
                digest="b" * 64,
            )
            commit_started = threading.Event()
            release_commit = threading.Event()
            original_commit = connector._store.commit

            def gated_commit(**kwargs):
                commit_started.set()
                self.assertTrue(release_commit.wait(timeout=5))
                return original_commit(**kwargs)

            connector._store.commit = gated_commit
            connector.bind_connector_metadata(
                SparkCacheConnectorMetadata(plans=[active])
            )
            connector.wait_for_save()
            self.assertTrue(commit_started.wait(timeout=2))
            connector.bind_connector_metadata(
                SparkCacheConnectorMetadata(plans=[existing])
            )
            connector.wait_for_save()

            self.assertIn(existing.digest, connector._held)
            self.assertTrue(
                connector._store.lookup(
                    connector._identity(0),
                    existing.digest,
                ).is_hit
            )
            release_commit.set()
            self.assertTrue(connector.wait_for_pending_stores(timeout=5))

    def test_commit_failure_revokes_and_releases_capacity_for_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connector = _make_connector(Path(directory), 0, 64)
            connector.register_kv_caches(_make_pools(8, 64))
            failed = _ReqPlan(
                "store-fails",
                "a" * 64,
                1024,
                (3, 0, 5, 1),
                True,
            )
            retry = dataclasses.replace(
                failed,
                request_id="store-retry",
                digest="b" * 64,
            )
            original_commit = connector._store.commit
            connector._store.commit = mock.Mock(
                side_effect=OSError("simulated fsync failure")
            )
            connector.bind_connector_metadata(
                SparkCacheConnectorMetadata(plans=[failed])
            )

            connector.wait_for_save()
            self.assertTrue(connector.wait_for_pending_stores(timeout=5))

            self.assertEqual(connector.counters["store_failed"], 1)
            self.assertNotIn(failed.digest, connector._held)
            self.assertEqual(
                connector.get_kv_connector_stats().data,
                {"reports": [{"rank": 0, "held": []}]},
            )

            connector._store.commit = original_commit
            connector.bind_connector_metadata(
                SparkCacheConnectorMetadata(plans=[retry])
            )
            connector.wait_for_save()
            self.assertTrue(connector.wait_for_pending_stores(timeout=5))

            self.assertEqual(connector.counters["store_committed"], 1)
            self.assertIn(retry.digest, connector._held)

    def test_new_commit_does_no_work_per_preexisting_entry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connector = _make_connector(Path(directory), 0, 64)
            connector.register_kv_caches(_make_pools(8, 64))
            for index in range(5):
                connector.bind_connector_metadata(
                    SparkCacheConnectorMetadata(
                        plans=[
                            _ReqPlan(
                                f"seed-{index}",
                                f"{index + 1:064x}",
                                1024,
                                (3, 0, 5, 1),
                                True,
                            )
                        ]
                    )
                )
                connector.wait_for_save()
                self.assertTrue(connector.wait_for_pending_stores(timeout=5))

            original_commit = connector._store.commit
            with (
                mock.patch.object(
                    connector._store,
                    "commit",
                    wraps=original_commit,
                ) as commit,
                mock.patch.object(
                    connector._store,
                    "restore",
                    side_effect=AssertionError(
                        "store completion must not read prior entries"
                    ),
                ) as restore,
                mock.patch.object(
                    connector,
                    "sweep_integrity",
                    side_effect=AssertionError(
                        "store completion must not sweep prior entries"
                    ),
                ) as sweep,
            ):
                connector.bind_connector_metadata(
                    SparkCacheConnectorMetadata(
                        plans=[
                            _ReqPlan(
                                "new-store",
                                "f" * 64,
                                1024,
                                (3, 0, 5, 1),
                                True,
                            )
                        ]
                    )
                )
                connector.wait_for_save()
                self.assertTrue(connector.wait_for_pending_stores(timeout=5))

            self.assertEqual(commit.call_count, 1)
            restore.assert_not_called()
            sweep.assert_not_called()
            self.assertIn("f" * 64, connector._held)

    def test_unexpected_saver_exception_releases_reserved_capacity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connector = _make_connector(Path(directory), 0, 64)
            connector.register_kv_caches(_make_pools(8, 64))
            connector._store.commit = mock.Mock(
                return_value=types.SimpleNamespace(committed_tokens=1024)
            )
            digest = "c" * 64
            connector.bind_connector_metadata(
                SparkCacheConnectorMetadata(
                    plans=[
                        _ReqPlan(
                            "broken-receipt",
                            digest,
                            1024,
                            (3, 0, 5, 1),
                            True,
                        )
                    ]
                )
            )

            connector.wait_for_save()

            self.assertTrue(connector.wait_for_pending_stores(timeout=1))
            self.assertEqual(connector.counters["store_failed"], 1)
            self.assertNotIn(digest, connector._held)

    def test_shutdown_joins_saver_and_rejects_post_shutdown_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connector = _make_connector(Path(directory), 0, 64)
            connector.register_kv_caches(_make_pools(8, 64))
            initial = _ReqPlan(
                "initial",
                "d" * 64,
                1024,
                (3, 0, 5, 1),
                True,
            )
            connector.bind_connector_metadata(
                SparkCacheConnectorMetadata(plans=[initial])
            )
            connector.wait_for_save()
            self.assertTrue(connector.wait_for_pending_stores(timeout=5))
            saver = connector._store_thread
            self.assertIsNotNone(saver)

            connector.shutdown()

            self.assertFalse(saver.is_alive())
            self.assertIsNone(connector._store_thread)
            after_shutdown = dataclasses.replace(
                initial,
                request_id="too-late",
                digest="e" * 64,
            )
            connector.bind_connector_metadata(
                SparkCacheConnectorMetadata(plans=[after_shutdown])
            )
            with mock.patch.object(
                connector,
                "_snapshot_store",
                wraps=connector._snapshot_store,
            ) as snapshot:
                connector.wait_for_save()

            snapshot.assert_not_called()
            self.assertEqual(connector.counters["store_skipped_busy"], 1)
            self.assertNotIn(after_shutdown.digest, connector._held)


class SchedulerRetirementTests(unittest.TestCase):
    def test_load_errors_retire_the_admitted_entry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rank0"
            connector = _make_connector(root, 0, 64)
            pool = _make_pools(8, 64)
            connector.register_kv_caches(pool)
            digest = "b" * 64
            plan = _ReqPlan("req-r", digest, 1024, (3, 0, 5, 1), True)
            connector.bind_connector_metadata(SparkCacheConnectorMetadata(plans=[plan]))
            connector.wait_for_save()
            _drain_store(connector)
            identity = connector._identity(0)
            self.assertTrue(connector._store.lookup(identity, digest).is_hit)

            connector._admitted["req-r"] = digest
            # the runtime finishes the failed request first; the digest must
            # still be retired when the callback arrives afterwards
            connector.update_connector_output(
                types.SimpleNamespace(invalid_block_ids={3, 0})
            )
            # entry retired so the next request is a clean miss
            self.assertFalse(connector._store.lookup(identity, digest).is_hit)
            self.assertEqual(connector._admitted, {})
            self.assertEqual(connector.counters["scheduler_retired"], 1)

    def test_clean_output_does_not_retire(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rank0"
            connector = _make_connector(root, 0, 64)
            pool = _make_pools(8, 64)
            connector.register_kv_caches(pool)
            digest = "c" * 64
            connector.bind_connector_metadata(
                SparkCacheConnectorMetadata(
                    plans=[_ReqPlan("req-k", digest, 1024, (3, 0, 5, 1), True)]
                )
            )
            connector.wait_for_save()
            _drain_store(connector)
            connector._admitted["req-k"] = digest
            connector.update_connector_output(
                types.SimpleNamespace(invalid_block_ids=set())
            )
            self.assertTrue(
                connector._store.lookup(connector._identity(0), digest).is_hit
            )


class QuorumAdmissionTests(unittest.TestCase):
    """A damaged entry must simply stop being offered, so the request is an
    ordinary miss that re-prefills - never a failed request."""

    def _request(self, tokens: int = 1100):
        return types.SimpleNamespace(
            request_id="req-q", prompt_token_ids=list(range(tokens))
        )

    def test_no_offer_until_every_rank_confirms(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            c = _make_connector(Path(directory) / "r0", 0, 64)
            c.register_kv_caches(_make_pools(8, 64))
            req = self._request()
            digest = c._digest(list(req.prompt_token_ids), 1024)
            # store exists locally but nobody has confirmed yet
            c.bind_connector_metadata(
                SparkCacheConnectorMetadata(
                    plans=[_ReqPlan("s", digest, 1024, (3, 0, 5, 1), True)]
                )
            )
            c.wait_for_save()
            _drain_store(c)
            self.assertEqual(c.get_num_new_matched_tokens(req, 0), (0, False))

            # three of four ranks confirm -> still not offered
            for rank in (0, 1, 2):
                c.update_connector_output(
                    types.SimpleNamespace(
                        invalid_block_ids=set(),
                        kv_connector_stats=types.SimpleNamespace(
                            data={
                                "spark_context_cache": {"rank": rank, "held": [digest]}
                            }
                        ),
                    )
                )
            self.assertEqual(c.get_num_new_matched_tokens(req, 0), (0, False))

            # fourth rank confirms -> now offered
            c.update_connector_output(
                types.SimpleNamespace(
                    invalid_block_ids=set(),
                    kv_connector_stats=types.SimpleNamespace(
                        data={"spark_context_cache": {"rank": 3, "held": [digest]}}
                    ),
                )
            )
            matched, _ = c.get_num_new_matched_tokens(req, 0)
            self.assertEqual(matched, 1024)

    def test_rank_withdrawing_stops_the_offer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            c = _make_connector(Path(directory) / "r0", 0, 64)
            c.register_kv_caches(_make_pools(8, 64))
            req = self._request()
            digest = c._digest(list(req.prompt_token_ids), 1024)
            c.bind_connector_metadata(
                SparkCacheConnectorMetadata(
                    plans=[_ReqPlan("s", digest, 1024, (3, 0, 5, 1), True)]
                )
            )
            c.wait_for_save()
            _drain_store(c)
            for rank in range(4):
                c.update_connector_output(
                    types.SimpleNamespace(
                        invalid_block_ids=set(),
                        kv_connector_stats=types.SimpleNamespace(
                            data={
                                "spark_context_cache": {"rank": rank, "held": [digest]}
                            }
                        ),
                    )
                )
            self.assertEqual(c.get_num_new_matched_tokens(req, 0)[0], 1024)

            # rank 2's sweep finds damage and it stops holding the digest
            c.update_connector_output(
                types.SimpleNamespace(
                    invalid_block_ids=set(),
                    kv_connector_stats=types.SimpleNamespace(
                        data={"spark_context_cache": {"rank": 2, "held": []}}
                    ),
                )
            )
            # entry is silently withdrawn: plain miss, request re-prefills
            self.assertEqual(c.get_num_new_matched_tokens(req, 0), (0, False))
            self.assertGreaterEqual(c.counters["quorum_incomplete"], 1)

    def test_scheduler_stats_do_not_require_an_initialized_dcp_group(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connector = _make_connector(
                Path(directory),
                0,
                64,
                role=KVConnectorRole.SCHEDULER,
                override_worker_rank=False,
            )
            distributed = sys.modules["vllm.distributed"]
            with mock.patch.object(
                distributed,
                "get_dcp_group",
                side_effect=RuntimeError("DCP group is not initialized"),
            ):
                self.assertIsNone(connector.get_kv_connector_stats())

    def test_worker_reports_only_verified_digests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "r1"
            c = _make_connector(root, 1, 64)
            c.register_kv_caches(_make_pools(8, 64))
            digest = "a" * 64
            c.bind_connector_metadata(
                SparkCacheConnectorMetadata(
                    plans=[_ReqPlan("s", digest, 1024, (3, 0, 5, 1), True)]
                )
            )
            c.wait_for_save()
            _drain_store(c)
            self.assertIn(digest, c._held)

            chunk = sorted((root / "chunks").glob("*.spcc"))[0]
            payload = bytearray(chunk.read_bytes())
            payload[len(payload) // 2] ^= 0x10
            chunk.write_bytes(bytes(payload))
            c.sweep_integrity()
            self.assertNotIn(digest, c._held)

            # An empty report is semantically meaningful: it revokes this
            # rank's prior confirmations at the scheduler. Returning None
            # here would leave a stale full quorum indefinitely.
            stats = c.get_kv_connector_stats()
            self.assertIsNotNone(stats)
            self.assertEqual(
                stats.data,
                {"reports": [{"rank": 1, "held": []}]},
            )
            c._quorum[digest] = {0, 1, 2, 3}
            c._absorb_quorum(types.SimpleNamespace(kv_connector_stats=stats))
            self.assertEqual(c._quorum[digest], {0, 2, 3})


class AsyncRestoreTests(unittest.TestCase):
    SPAN = 1024
    BLOCKS = (3, 0, 5, 1)

    def _store_entry(self, connector, digest, block_ids=None):
        connector.bind_connector_metadata(
            SparkCacheConnectorMetadata(
                plans=[
                    _ReqPlan(
                        "seed",
                        digest,
                        self.SPAN,
                        tuple(block_ids or self.BLOCKS),
                        True,
                    )
                ]
            )
        )
        connector.wait_for_save()
        _drain_store(connector)
        connector.clear_connector_metadata()

    def _confirm_quorum(self, connector, digest):
        for rank in range(4):
            connector.update_connector_output(
                types.SimpleNamespace(
                    invalid_block_ids=set(),
                    kv_connector_stats=types.SimpleNamespace(
                        data={
                            "spark_context_cache": {
                                "rank": rank,
                                "held": [digest],
                            }
                        }
                    ),
                )
            )

    def _blocks_stub(self, block_ids=None):
        table = list(block_ids or self.BLOCKS)
        return types.SimpleNamespace(get_block_ids=lambda: (table,))

    def test_full_quorum_parks_only_the_restoring_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connector = _make_connector(Path(directory), 0, 64)
            connector.register_kv_caches(_make_pools(8, 64))
            tokens = list(range(1100))
            digest = connector._digest(tokens, self.SPAN)
            connector.bind_connector_metadata(
                SparkCacheConnectorMetadata(
                    plans=[_ReqPlan("seed", digest, self.SPAN, self.BLOCKS, True)]
                )
            )
            connector.wait_for_save()
            _drain_store(connector)
            for rank in range(4):
                connector.update_connector_output(
                    types.SimpleNamespace(
                        invalid_block_ids=set(),
                        kv_connector_stats=types.SimpleNamespace(
                            data={
                                "spark_context_cache": {
                                    "rank": rank,
                                    "held": [digest],
                                }
                            }
                        ),
                    )
                )

            request = types.SimpleNamespace(
                request_id="restore-me", prompt_token_ids=tokens
            )
            self.assertEqual(
                connector.get_num_new_matched_tokens(request, 0),
                (self.SPAN, True),
            )
            blocks = types.SimpleNamespace(get_block_ids=lambda: (list(self.BLOCKS),))
            connector.update_state_after_alloc(request, blocks, self.SPAN)
            metadata = connector.build_connector_meta(_empty_scheduler_output())

            self.assertEqual(len(metadata.plans), 1)
            self.assertEqual(metadata.plans[0].request_id, "restore-me")
            self.assertFalse(metadata.plans[0].is_store)

    def test_restore_reports_finished_only_after_background_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connector = _make_connector(Path(directory), 0, 64)
            pool = _make_pools(8, 64)
            connector.register_kv_caches(pool)
            digest = "d" * 64
            connector.bind_connector_metadata(
                SparkCacheConnectorMetadata(
                    plans=[_ReqPlan("seed", digest, self.SPAN, self.BLOCKS, True)]
                )
            )
            connector.wait_for_save()
            _drain_store(connector)

            started = threading.Event()
            release = threading.Event()
            original_restore = connector._store.restore

            def gated_restore(lookup):
                started.set()
                self.assertTrue(release.wait(timeout=30))
                return original_restore(lookup)

            connector._store.restore = gated_restore
            try:
                connector.bind_connector_metadata(
                    SparkCacheConnectorMetadata(
                        plans=[
                            _ReqPlan(
                                "restore-me",
                                digest,
                                self.SPAN,
                                self.BLOCKS,
                                False,
                            )
                        ]
                    )
                )
                enqueue_started = time.perf_counter()
                connector.start_load_kv(None)
                enqueue_ms = 1e3 * (time.perf_counter() - enqueue_started)
                self.assertTrue(started.wait(timeout=30))
                poll_started = time.perf_counter()
                for _ in range(10_000):
                    self.assertEqual(connector.get_finished(set()), (None, None))
                poll_us = 1e6 * (time.perf_counter() - poll_started) / 10_000
                self.assertLess(
                    enqueue_ms,
                    1.0,
                    "restore enqueue exceeded the sub-ms control-path budget",
                )
                self.assertLess(
                    poll_us,
                    10.0,
                    "empty completion polling exceeded 10 us/call",
                )
            finally:
                release.set()

            self.assertEqual(_drain(connector), {"restore-me"})
            self.assertEqual(connector.get_finished(set()), (None, None))
            self.assertEqual(connector.counters["load_verified"], 1)

    def test_corrupt_restore_finishes_for_clean_recompute(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rank2"
            connector = _make_connector(root, 2, 64)
            pool = _make_pools(8, 64)
            connector.register_kv_caches(pool)
            digest = "e" * 64
            self._store_entry(connector, digest)
            chunk = sorted((root / "chunks").glob("*.spcc"))[0]
            payload = bytearray(chunk.read_bytes())
            payload[len(payload) // 2] ^= 0x08
            chunk.write_bytes(bytes(payload))
            for tensor in pool.values():
                tensor.zero_()

            connector.bind_connector_metadata(
                SparkCacheConnectorMetadata(
                    plans=[
                        _ReqPlan(
                            "bad-restore",
                            digest,
                            self.SPAN,
                            self.BLOCKS,
                            False,
                        )
                    ]
                )
            )
            connector.start_load_kv(None)

            self.assertEqual(_drain(connector), {"bad-restore"})
            self.assertEqual(connector.get_finished(set()), (None, None))
            self.assertEqual(
                connector.get_block_ids_with_load_errors(), set(self.BLOCKS)
            )
            self.assertEqual(connector.counters["load_failed"], 1)
            self.assertNotIn(digest, connector._held)
            for tensor in pool.values():
                self.assertTrue((tensor == 0).all())

    def test_two_restoring_requests_complete_independently(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connector = _make_connector(Path(directory), 0, 64)
            pool = _make_pools(8, 64)
            connector.register_kv_caches(pool)
            digest_a, blocks_a = "1" * 64, (3, 0, 5, 1)
            digest_b, blocks_b = "2" * 64, (2, 4, 6, 7)
            for request_id, digest, blocks in (
                ("seed-a", digest_a, blocks_a),
                ("seed-b", digest_b, blocks_b),
            ):
                connector.bind_connector_metadata(
                    SparkCacheConnectorMetadata(
                        plans=[
                            _ReqPlan(
                                request_id,
                                digest,
                                self.SPAN,
                                blocks,
                                True,
                            )
                        ]
                    )
                )
                connector.wait_for_save()
                _drain_store(connector)
            for tensor in pool.values():
                tensor.zero_()
            connector.bind_connector_metadata(
                SparkCacheConnectorMetadata(
                    plans=[
                        _ReqPlan("restore-a", digest_a, self.SPAN, blocks_a, False),
                        _ReqPlan("restore-b", digest_b, self.SPAN, blocks_b, False),
                    ]
                )
            )
            connector.start_load_kv(None)

            self.assertTrue(connector.wait_for_pending_loads(timeout=30))
            self.assertEqual(
                connector.get_finished(set())[1], {"restore-a", "restore-b"}
            )
            self.assertEqual(connector.counters["load_verified"], 2)

    def test_verified_restore_does_not_rewrite_existing_entry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connector = _make_connector(Path(directory), 0, 64)
            connector.register_kv_caches(_make_pools(8, 64))
            tokens = list(range(1100))
            digest = connector._digest(tokens, self.SPAN)
            self._store_entry(connector, digest)
            self._confirm_quorum(connector, digest)
            request = types.SimpleNamespace(
                request_id="restore-me", prompt_token_ids=tokens
            )
            self.assertEqual(
                connector.get_num_new_matched_tokens(request, 0),
                (self.SPAN, True),
            )
            connector.update_state_after_alloc(request, self._blocks_stub(), self.SPAN)
            connector.build_connector_meta(_empty_scheduler_output())

            resumed = types.SimpleNamespace(
                scheduled_new_reqs=[
                    types.SimpleNamespace(
                        req_id="restore-me",
                        prompt_token_ids=tokens,
                        num_computed_tokens=self.SPAN,
                        block_ids=(list(self.BLOCKS),),
                    )
                ],
                num_scheduled_tokens={"restore-me": len(tokens) - self.SPAN},
                scheduled_cached_reqs=types.SimpleNamespace(
                    req_ids=[],
                    resumed_req_ids=set(),
                    num_computed_tokens=[],
                    new_block_ids=[],
                ),
            )
            self.assertEqual(connector.build_connector_meta(resumed).plans, [])
            self.assertNotIn("restore-me", connector._admitted)

    def test_failed_restore_recompute_can_republish(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connector = _make_connector(Path(directory), 0, 64)
            connector.register_kv_caches(_make_pools(8, 64))
            tokens = list(range(1100))
            digest = connector._digest(tokens, self.SPAN)
            self._store_entry(connector, digest)
            self._confirm_quorum(connector, digest)
            request = types.SimpleNamespace(
                request_id="restore-me", prompt_token_ids=tokens
            )
            connector.get_num_new_matched_tokens(request, 0)
            connector.update_state_after_alloc(request, self._blocks_stub(), self.SPAN)
            connector.build_connector_meta(_empty_scheduler_output())
            connector.update_connector_output(
                types.SimpleNamespace(
                    invalid_block_ids={self.BLOCKS[0]},
                    kv_connector_stats=None,
                )
            )

            recompute = types.SimpleNamespace(
                scheduled_new_reqs=[
                    types.SimpleNamespace(
                        req_id="restore-me",
                        prompt_token_ids=tokens,
                        num_computed_tokens=0,
                        block_ids=(list(self.BLOCKS),),
                    )
                ],
                num_scheduled_tokens={"restore-me": len(tokens)},
                scheduled_cached_reqs=types.SimpleNamespace(
                    req_ids=[],
                    resumed_req_ids=set(),
                    num_computed_tokens=[],
                    new_block_ids=[],
                ),
            )
            plans = connector.build_connector_meta(recompute).plans
            self.assertEqual(len(plans), 1)
            self.assertTrue(plans[0].is_store)
            self.assertEqual(plans[0].digest, digest)

    def test_oversize_restore_and_store_are_declined(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connector = _make_connector(Path(directory), 0, 64)
            connector._max_span = 1024
            tokens = list(range(2200))
            request = types.SimpleNamespace(
                request_id="too-big", prompt_token_ids=tokens
            )
            self.assertEqual(
                connector.get_num_new_matched_tokens(request, 0), (0, False)
            )

            step = types.SimpleNamespace(
                scheduled_new_reqs=[
                    types.SimpleNamespace(
                        req_id="too-big",
                        prompt_token_ids=tokens,
                        num_computed_tokens=0,
                        block_ids=([10, 11],),
                    )
                ],
                num_scheduled_tokens={"too-big": len(tokens)},
                scheduled_cached_reqs=types.SimpleNamespace(
                    req_ids=[],
                    resumed_req_ids=set(),
                    num_computed_tokens=[],
                    new_block_ids=[],
                ),
            )
            self.assertEqual(connector.build_connector_meta(step).plans, [])


class QuorumStatsAggregationTests(unittest.TestCase):
    """The executor merges every worker's stats object before the scheduler
    sees it, so aggregate() must be concrete and must union the reports."""

    def test_four_rank_reports_merge_without_loss(self) -> None:
        cls = connector_module.SparkCacheStats
        acc = cls(data={"reports": [{"rank": 0, "held": ["a" * 64]}]})
        for rank in (1, 2, 3):
            acc = acc.aggregate(
                cls(data={"reports": [{"rank": rank, "held": ["a" * 64]}]})
            )
        ranks = {r["rank"] for r in acc.data["reports"]}
        self.assertEqual(ranks, {0, 1, 2, 3})
        self.assertFalse(acc.is_empty())
        self.assertEqual(acc.reduce()["spark_cache_ranks_reporting"], 4)

    def test_later_report_replaces_same_rank(self) -> None:
        cls = connector_module.SparkCacheStats
        acc = cls(data={"reports": [{"rank": 2, "held": ["b" * 64]}]})
        acc = acc.aggregate(cls(data={"reports": [{"rank": 2, "held": []}]}))
        held = {r["rank"]: r["held"] for r in acc.data["reports"]}
        self.assertEqual(held[2], [])

    def test_scheduler_absorbs_merged_reports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            c = _make_connector(Path(directory) / "r0", 0, 64)
            digest = "c" * 64
            merged = types.SimpleNamespace(
                invalid_block_ids=set(),
                kv_connector_stats=types.SimpleNamespace(
                    data={"reports": [{"rank": r, "held": [digest]} for r in range(4)]}
                ),
            )
            c.update_connector_output(merged)
            self.assertEqual(c._quorum[digest], {0, 1, 2, 3})


class StatsPicklabilityTests(unittest.TestCase):
    """Stats objects cross the worker->engine shared-memory queue, which
    pickles them. A class defined inside a function is NOT picklable and
    silently kills the async output thread, hanging the request."""

    def test_stats_object_survives_pickle_round_trip(self) -> None:
        import pickle

        original = connector_module.SparkCacheStats(
            data={"reports": [{"rank": 2, "held": ["d" * 64]}]}
        )
        restored = pickle.loads(pickle.dumps(original))
        self.assertEqual(restored.data, original.data)
        self.assertFalse(restored.is_empty())

    def test_stats_class_is_module_level(self) -> None:
        self.assertNotIn("<locals>", connector_module.SparkCacheStats.__qualname__)


if __name__ == "__main__":
    unittest.main()
