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

    # Mutable global so tests can set the physical TP rank per connector.
    _TP_RANK = [0]

    class _Group:
        rank_in_group = 0
        world_size = 4

    distributed.get_dcp_group = lambda: _Group
    distributed.get_tensor_model_parallel_rank = lambda: _TP_RANK[0]
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
        "spark_cache_target_checkpoint_sha256": "1" * 64,
        "spark_cache_draft_checkpoint_sha256": "2" * 64,
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
        connector._worker_rank = lambda r=rank: r  # type: ignore[method-assign]
        connector._physical_rank = lambda r=rank: r  # type: ignore[method-assign]
    return connector


class CheckpointIdentityTests(unittest.TestCase):
    def test_mutable_target_identity_is_rejected_at_startup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                RuntimeError,
                "target checkpoint identity must be a 64-character lowercase SHA-256",
            ):
                _make_connector(
                    Path(directory),
                    0,
                    extra_config={
                        "spark_cache_target_checkpoint_sha256": "local/model/path"
                    },
                )

    def test_separate_draft_requires_its_own_immutable_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                RuntimeError,
                "separate draft checkpoint identity must be a 64-character"
                " lowercase SHA-256",
            ):
                _make_connector(
                    Path(directory),
                    0,
                    extra_config={"spark_cache_draft_checkpoint_sha256": ""},
                )

    def test_colocated_draft_derives_target_checkpoint_identity(self) -> None:
        target_digest = "a" * 64
        with tempfile.TemporaryDirectory() as directory:
            connector = _make_connector(
                Path(directory),
                0,
                extra_config={
                    "spark_cache_target_checkpoint_sha256": target_digest,
                    "spark_cache_draft_checkpoint_sha256": "",
                    "spark_cache_draft_policy": "colocated_target",
                },
            )

        self.assertEqual(connector._identity_base["target_checkpoint"], target_digest)
        self.assertEqual(connector._identity_base["draft_checkpoint"], target_digest)

    def test_replacing_checkpoint_identity_changes_cache_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = _make_connector(
                Path(directory),
                0,
                extra_config={"spark_cache_target_checkpoint_sha256": "a" * 64},
            )
            replacement = _make_connector(
                Path(directory),
                0,
                extra_config={"spark_cache_target_checkpoint_sha256": "b" * 64},
            )

        self.assertNotEqual(
            first._identity(0).storage_key, replacement._identity(0).storage_key
        )


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
    def test_streaming_snapshot_feature_is_disabled_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connector = _make_connector(Path(directory), 0)

        self.assertFalse(connector._streaming_snapshots_enabled)
        self.assertIsNone(connector._streaming_runtime)

    def test_streaming_snapshot_opt_in_fails_before_native_side_effects(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                RuntimeError, "runtime installation failed closed"
            ):
                _make_connector(
                    Path(directory),
                    0,
                    extra_config={"spark_cache_streaming_snapshots": "1"},
                )

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
            self.assertEqual(meta1.offers, [])
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
            self.assertEqual(meta2.offers, [])
            self.assertEqual(connector._store_progress, {})

    def test_full_quorum_suppresses_store_before_progress_is_created(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connector = _make_connector(Path(directory), 0, block_size=64)
            token_ids = list(range(1100))
            digest = connector._digest(token_ids, 1024)
            connector._quorum[digest] = {0, 1, 2, 3}
            step = types.SimpleNamespace(
                scheduled_new_reqs=[
                    types.SimpleNamespace(
                        req_id="already-cached",
                        prompt_token_ids=token_ids,
                        num_computed_tokens=0,
                        block_ids=([10, 11],),
                    )
                ],
                num_scheduled_tokens={"already-cached": 512},
                scheduled_cached_reqs=types.SimpleNamespace(
                    req_ids=[],
                    resumed_req_ids=set(),
                    num_computed_tokens=[],
                    new_block_ids=[],
                ),
            )

            metadata = connector.build_connector_meta(step)

            self.assertEqual(metadata.plans, [])
            self.assertEqual(connector._store_progress, {})
            self.assertEqual(connector.counters["store_skipped_quorum"], 1)

    def test_quorum_arriving_during_prefill_retires_pending_store(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connector = _make_connector(Path(directory), 0, block_size=64)
            token_ids = list(range(1100))
            digest = connector._digest(token_ids, 1024)
            first = types.SimpleNamespace(
                scheduled_new_reqs=[
                    types.SimpleNamespace(
                        req_id="becomes-cached",
                        prompt_token_ids=token_ids,
                        num_computed_tokens=0,
                        block_ids=([10, 11],),
                    )
                ],
                num_scheduled_tokens={"becomes-cached": 512},
                scheduled_cached_reqs=types.SimpleNamespace(
                    req_ids=[],
                    resumed_req_ids=set(),
                    num_computed_tokens=[],
                    new_block_ids=[],
                ),
            )
            self.assertEqual(connector.build_connector_meta(first).plans, [])
            self.assertIn("becomes-cached", connector._store_progress)
            connector._quorum[digest] = {0, 1, 2, 3}
            second = types.SimpleNamespace(
                scheduled_new_reqs=[],
                num_scheduled_tokens={"becomes-cached": 512},
                scheduled_cached_reqs=types.SimpleNamespace(
                    req_ids=["becomes-cached"],
                    resumed_req_ids=set(),
                    num_computed_tokens=[512],
                    new_block_ids=[([12, 13],)],
                ),
            )

            metadata = connector.build_connector_meta(second)

            self.assertEqual(metadata.plans, [])
            self.assertEqual(connector._store_progress, {})
            self.assertEqual(connector.counters["store_skipped_quorum"], 1)


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

    def test_present_digest_skips_snapshot_and_preserves_immutable_entry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connector = _make_connector(Path(directory), 0, 64)
            pool = _make_pools(8, 64)
            connector.register_kv_caches(pool)
            plan = _ReqPlan(
                "store-once",
                "7" * 64,
                1024,
                (3, 0, 5, 1),
                True,
            )
            connector.bind_connector_metadata(
                SparkCacheConnectorMetadata(plans=[plan])
            )
            connector.wait_for_save()
            self.assertTrue(connector.wait_for_pending_stores(timeout=5))
            self.assertIn(plan.digest, connector._held)

            for tensor in pool.values():
                tensor.zero_()
            connector.bind_connector_metadata(
                SparkCacheConnectorMetadata(
                    plans=[dataclasses.replace(plan, request_id="duplicate")]
                )
            )
            with mock.patch.object(
                connector,
                "_snapshot_store",
                wraps=connector._snapshot_store,
            ) as snapshot:
                connector.wait_for_save()

            snapshot.assert_not_called()
            self.assertEqual(connector.counters["store_committed"], 1)
            self.assertEqual(connector.counters["store_skipped_present"], 1)
            self.assertEqual(connector.counters["store_failed"], 0)
            self.assertTrue(
                connector._store.lookup(
                    connector._identity(0),
                    plan.digest,
                ).is_hit
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

    def test_duplicate_or_out_of_range_rank_reports_cannot_form_quorum(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            c = _make_connector(Path(directory) / "r0", 0, 64)
            digest = "e" * 64

            duplicates = types.SimpleNamespace(
                kv_connector_stats=types.SimpleNamespace(
                    data={
                        "reports": [
                            {"rank": 0, "held": [digest]},
                            {"rank": 0, "held": [digest]},
                            {"rank": 0, "held": [digest]},
                            {"rank": 0, "held": [digest]},
                        ]
                    }
                )
            )
            c._absorb_quorum(duplicates)
            self.assertEqual(c._quorum[digest], {0})
            self.assertFalse(c._has_full_quorum(digest))

            fabricated = types.SimpleNamespace(
                kv_connector_stats=types.SimpleNamespace(
                    data={
                        "reports": [
                            {"rank": -1, "held": [digest]},
                            {"rank": 4, "held": [digest]},
                            {"rank": 5, "held": [digest]},
                            {"rank": 6, "held": [digest]},
                        ]
                    }
                )
            )
            c._absorb_quorum(fabricated)
            self.assertEqual(c._quorum[digest], {0})
            self.assertFalse(c._has_full_quorum(digest))

            # Defense in depth for restored/legacy in-memory state: cardinality
            # alone is never proof that ranks 0..DCP-1 all confirmed.
            c._quorum[digest] = {-1, 4, 5, 6}
            self.assertFalse(c._has_full_quorum(digest))


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


class _FakeStreamingRuntime:
    def __init__(
        self,
        *,
        delay_free: bool = False,
        finished_sending: set[str] | None = None,
        events: list[str] | None = None,
    ) -> None:
        self.delay_free = delay_free
        self.finished_sending = set(finished_sending or ())
        self.events = events if events is not None else []
        self.preemption_metadata = None
        self.finished_request = None
        self.finished_blocks = None
        self.finished_filter = None
        self.completed_offers = []
        self.producer_streams = []
        self.poll_count = 0
        self.observed_metadata = []
        self.bind_count = 0

    def observe_metadata(self, metadata) -> None:
        self.observed_metadata.append(metadata)

    def bind_kv_caches(self) -> None:
        self.bind_count += 1

    def offer_completed(self, offer, *, producer_stream: int) -> None:
        self.completed_offers.append(offer)
        self.producer_streams.append(producer_stream)

    def poll(self) -> None:
        self.poll_count += 1

    def handle_preemptions(self, metadata) -> None:
        self.preemption_metadata = metadata
        self.events.append("streaming-preemptions-drained")

    def request_finished(
        self, request_id: str, block_ids: tuple[int, ...]
    ) -> bool:
        self.finished_request = request_id
        self.finished_blocks = block_ids
        return self.delay_free

    def take_finished(self, finished_request_ids: set[str]) -> set[str]:
        self.finished_filter = set(finished_request_ids)
        return self.finished_sending & finished_request_ids

    def shutdown(self) -> None:
        self.events.append("streaming-shutdown")


class StreamingLifecycleScaffoldingTests(unittest.TestCase):
    def test_metadata_carries_sorted_preempted_request_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connector = _make_connector(Path(directory), 0)
            output = _empty_scheduler_output()
            output.preempted_req_ids = {"request-z", "request-a"}

            metadata = connector.build_connector_meta(output)

        self.assertEqual(
            metadata.preempted_request_ids,
            ("request-a", "request-z"),
        )

    def test_default_none_preserves_connector_lifecycle_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connector = _make_connector(Path(directory), 0)
            metadata = SparkCacheConnectorMetadata()

            connector.handle_preemptions(metadata)
            self.assertEqual(
                connector.request_finished(
                    types.SimpleNamespace(request_id="finished"),
                    [1, 2],
                ),
                (False, None),
            )
            self.assertEqual(connector.get_finished({"finished"}), (None, None))

    def test_preemption_and_delayed_free_delegate_synchronously(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connector = _make_connector(Path(directory), 0)
            runtime = _FakeStreamingRuntime(delay_free=True)
            connector._streaming_runtime = runtime
            metadata = SparkCacheConnectorMetadata(
                preempted_request_ids=("preempted",)
            )

            connector.handle_preemptions(metadata)
            delayed = connector.request_finished(
                types.SimpleNamespace(request_id="finished"),
                [9, 3, 7],
            )

        self.assertIs(runtime.preemption_metadata, metadata)
        self.assertEqual(runtime.events, ["streaming-preemptions-drained"])
        self.assertEqual(runtime.finished_request, "finished")
        self.assertEqual(runtime.finished_blocks, (9, 3, 7))
        self.assertEqual(delayed, (True, None))

    def test_get_finished_merges_send_and_restore_completions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connector = _make_connector(Path(directory), 0)
            runtime = _FakeStreamingRuntime(
                finished_sending={"stream-done", "not-in-filter"}
            )
            connector._streaming_runtime = runtime
            connector._finished_load_reqs.add("restore-done")

            result = connector.get_finished({"stream-done"})

        self.assertEqual(result, ({"stream-done"}, {"restore-done"}))
        self.assertEqual(runtime.finished_filter, {"stream-done"})
        self.assertEqual(connector.get_finished(set()), (None, None))

    def test_shutdown_drains_streaming_before_native_close(self) -> None:
        events: list[str] = []
        with tempfile.TemporaryDirectory() as directory:
            connector = _make_connector(Path(directory), 0)
            connector._streaming_runtime = _FakeStreamingRuntime(events=events)
            native = mock.Mock()
            native.close.side_effect = lambda: events.append("native-close")
            connector._native_adapter = native

            connector.shutdown()

        self.assertEqual(events, ["streaming-shutdown", "native-close"])
        self.assertIsNone(connector._streaming_runtime)
        self.assertIsNone(connector._native_adapter)


class StreamingSnapshotConnectorSeamTests(unittest.TestCase):
    def setUp(self) -> None:
        connector_module.configure_streaming_snapshot_runtime(
            KVConnectorRole.SCHEDULER, None
        )
        connector_module.configure_streaming_snapshot_runtime(
            KVConnectorRole.WORKER, None
        )
        self.addCleanup(
            connector_module.configure_streaming_snapshot_runtime,
            KVConnectorRole.SCHEDULER,
            None,
        )
        self.addCleanup(
            connector_module.configure_streaming_snapshot_runtime,
            KVConnectorRole.WORKER,
            None,
        )

    def _scheduler(self, root: Path) -> SparkContextCacheConnector:
        runtime = _FakeStreamingRuntime()
        connector_module.configure_streaming_snapshot_runtime(
            KVConnectorRole.SCHEDULER, lambda connector: runtime
        )
        connector = _make_connector(
            root,
            0,
            block_size=64,
            role=KVConnectorRole.SCHEDULER,
            extra_config={"spark_cache_streaming_snapshots": "1"},
        )
        connector._test_streaming_runtime = runtime
        return connector

    def test_chunked_new_and_cached_steps_emit_promised_full_table_offers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connector = self._scheduler(Path(directory))
            tokens = list(range(1100))
            first = types.SimpleNamespace(
                scheduled_new_reqs=[
                    types.SimpleNamespace(
                        req_id="streamed",
                        prompt_token_ids=tokens,
                        num_computed_tokens=0,
                        block_ids=([10, 11],),
                    )
                ],
                num_scheduled_tokens={"streamed": 512},
                scheduled_cached_reqs=types.SimpleNamespace(
                    req_ids=[],
                    resumed_req_ids=set(),
                    num_computed_tokens=[],
                    new_block_ids=[],
                ),
            )
            first_meta = connector.build_connector_meta(first)
            second = types.SimpleNamespace(
                scheduled_new_reqs=[],
                num_scheduled_tokens={"streamed": 512},
                scheduled_cached_reqs=types.SimpleNamespace(
                    req_ids=["streamed"],
                    resumed_req_ids=set(),
                    num_computed_tokens=[512],
                    new_block_ids=[([12, 13],)],
                ),
            )
            second_meta = connector.build_connector_meta(second)

        self.assertEqual(first_meta.plans, [])
        self.assertEqual(len(first_meta.offers), 1)
        self.assertEqual(first_meta.offers[0].completed_tokens, 512)
        self.assertEqual(first_meta.offers[0].block_ids, (10, 11))
        self.assertEqual(second_meta.plans, [])
        self.assertEqual(len(second_meta.offers), 1)
        self.assertEqual(second_meta.offers[0].completed_tokens, 1024)
        self.assertEqual(second_meta.offers[0].block_ids, (10, 11, 12, 13))
        self.assertEqual(connector._store_progress, {})
        self.assertEqual(
            connector._test_streaming_runtime.observed_metadata,
            [first_meta, second_meta],
        )

    def test_resumed_and_final_tail_offers_never_fall_back_to_store_plans(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connector = self._scheduler(Path(directory))
            tokens = list(range(1100))
            start = types.SimpleNamespace(
                scheduled_new_reqs=[
                    types.SimpleNamespace(
                        req_id="resumed",
                        prompt_token_ids=tokens,
                        num_computed_tokens=0,
                        block_ids=([10, 11],),
                    )
                ],
                num_scheduled_tokens={"resumed": 512},
                scheduled_cached_reqs=types.SimpleNamespace(
                    req_ids=[],
                    resumed_req_ids=set(),
                    num_computed_tokens=[],
                    new_block_ids=[],
                ),
            )
            connector.build_connector_meta(start)
            resumed = types.SimpleNamespace(
                scheduled_new_reqs=[],
                preempted_req_ids={"resumed"},
                num_scheduled_tokens={"resumed": 512},
                scheduled_cached_reqs=types.SimpleNamespace(
                    req_ids=["resumed"],
                    resumed_req_ids={"resumed"},
                    num_computed_tokens=[512],
                    new_block_ids=[([20, 21, 22, 23],)],
                ),
            )
            resumed_meta = connector.build_connector_meta(resumed)
            tail = types.SimpleNamespace(
                scheduled_new_reqs=[
                    types.SimpleNamespace(
                        req_id="tail",
                        prompt_token_ids=tokens,
                        num_computed_tokens=0,
                        block_ids=([30, 31, 32, 33],),
                    )
                ],
                num_scheduled_tokens={"tail": 1100},
                scheduled_cached_reqs=types.SimpleNamespace(
                    req_ids=[],
                    resumed_req_ids=set(),
                    num_computed_tokens=[],
                    new_block_ids=[],
                ),
            )
            tail_meta = connector.build_connector_meta(tail)

        self.assertEqual(resumed_meta.plans, [])
        self.assertEqual(resumed_meta.preempted_request_ids, ("resumed",))
        self.assertEqual(resumed_meta.offers[0].block_ids, (20, 21, 22, 23))
        self.assertEqual(resumed_meta.offers[0].completed_tokens, 1024)
        self.assertEqual(tail_meta.plans, [])
        self.assertEqual(tail_meta.offers[0].completed_tokens, 1024)
        self.assertEqual(tail_meta.offers[0].block_ids, (30, 31, 32, 33))

    def test_worker_converts_offers_after_forward_and_polls_without_sync_store(self) -> None:
        runtime = _FakeStreamingRuntime()
        connector_module.configure_streaming_snapshot_runtime(
            KVConnectorRole.WORKER, lambda connector: runtime
        )
        with tempfile.TemporaryDirectory() as directory:
            connector = _make_connector(
                Path(directory),
                0,
                extra_config={"spark_cache_streaming_snapshots": "1"},
            )
            offer = connector_module._StreamingSnapshotOffer(
                "after-forward", "a" * 64, 1024, 512, (3, 0)
            )
            connector.bind_connector_metadata(
                SparkCacheConnectorMetadata(
                    plans=[_ReqPlan("legacy", "b" * 64, 1024, (3, 0, 5, 1), True)],
                    streaming_snapshot_offers=[offer],
                )
            )
            with mock.patch.object(connector, "_snapshot_store") as snapshot:
                with mock.patch.object(
                    torch.cuda,
                    "current_stream",
                    return_value=types.SimpleNamespace(cuda_stream=0xABC),
                ):
                    connector.wait_for_save()

        self.assertEqual(runtime.completed_offers, [offer])
        self.assertEqual(runtime.producer_streams, [0xABC])
        self.assertEqual(runtime.poll_count, 1)
        snapshot.assert_not_called()

    def test_worker_polls_empty_post_forward_callbacks(self) -> None:
        runtime = _FakeStreamingRuntime()
        connector_module.configure_streaming_snapshot_runtime(
            KVConnectorRole.WORKER, lambda connector: runtime
        )
        with tempfile.TemporaryDirectory() as directory:
            connector = _make_connector(
                Path(directory),
                0,
                extra_config={"spark_cache_streaming_snapshots": "1"},
            )
            connector.bind_connector_metadata(SparkCacheConnectorMetadata())
            connector.wait_for_save()

        self.assertEqual(runtime.completed_offers, [])
        self.assertEqual(runtime.poll_count, 1)

    def test_tiny_unoffered_request_is_not_reported_as_async_send_finished(
        self,
    ) -> None:
        """Mirror vLLM's scheduler/worker finish exchange for a tiny prompt.

        A request with no aligned snapshot span produces no streaming offer.
        The scheduler therefore frees it synchronously.  Worker ``get_finished``
        must not echo that ordinary finished ID back as an asynchronous send
        completion: vLLM's scheduler rightfully requires every reported send
        completion to still exist in its delayed-free request table.
        """

        from sparkcache.streaming.block_lease import (
            BlockLeaseRegistry,
            LeaseCapacity,
        )
        from sparkcache.streaming.factory import (
            SchedulerStreamingSnapshotAdapter,
            WorkerStreamingSnapshotAdapter,
        )

        worker_adapter: WorkerStreamingSnapshotAdapter | None = None

        def make_scheduler_runtime(connector):
            return SchedulerStreamingSnapshotAdapter(connector)

        def make_worker_runtime(connector):
            nonlocal worker_adapter
            worker_adapter = WorkerStreamingSnapshotAdapter(
                connector,
                settings=types.SimpleNamespace(),
            )
            # Finish reporting itself is GPU-free.  Keep the real adapter and
            # lease registry, while replacing unrelated native-ring progress.
            worker_adapter._bound = True
            worker_adapter._runtime = object()
            worker_adapter._leases = BlockLeaseRegistry(
                LeaseCapacity(max_active_leases=2, max_leased_blocks=16)
            )
            worker_adapter.poll = lambda: 0
            return worker_adapter

        connector_module.configure_streaming_snapshot_runtime(
            KVConnectorRole.SCHEDULER,
            make_scheduler_runtime,
        )
        connector_module.configure_streaming_snapshot_runtime(
            KVConnectorRole.WORKER,
            make_worker_runtime,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scheduler = _make_connector(
                root / "scheduler",
                0,
                role=KVConnectorRole.SCHEDULER,
                extra_config={"spark_cache_streaming_snapshots": "1"},
            )
            worker = _make_connector(
                root / "worker",
                0,
                role=KVConnectorRole.WORKER,
                extra_config={"spark_cache_streaming_snapshots": "1"},
            )
            output = types.SimpleNamespace(
                scheduled_new_reqs=[
                    types.SimpleNamespace(
                        req_id="tiny",
                        prompt_token_ids=list(range(32)),
                        num_computed_tokens=0,
                        block_ids=([10],),
                    )
                ],
                num_scheduled_tokens={"tiny": 32},
                scheduled_cached_reqs=types.SimpleNamespace(
                    req_ids=[],
                    resumed_req_ids=set(),
                    num_computed_tokens=[],
                    new_block_ids=[],
                ),
            )
            metadata = scheduler.build_connector_meta(output)
            self.assertEqual(metadata.streaming_snapshot_offers, [])

            worker.bind_connector_metadata(metadata)
            worker.wait_for_save()

            scheduler_requests = {"tiny": object()}
            delayed, _ = scheduler.request_finished(
                types.SimpleNamespace(request_id="tiny"),
                [10],
            )
            self.assertFalse(delayed)
            scheduler_requests.pop("tiny")

            finished_sending, _ = worker.get_finished({"tiny"})
            for request_id in finished_sending or ():
                # Exact ownership assertion in vLLM
                # Scheduler._update_from_kv_xfer_finished().
                self.assertIn(request_id, scheduler_requests)

        self.assertIsNotNone(worker_adapter)


class StreamingSnapshotProductionBoundaryTests(unittest.TestCase):
    def test_connector_runtime_publisher_commit_becomes_visible_only_after_poll(
        self,
    ) -> None:
        from sparkcache.streaming.factory import (
            ProductionStreamingSettings,
            WorkerStreamingSnapshotAdapter,
        )
        from sparkcache.streaming.native_ring import NativeSnapshotRing
        from sparkcache.streaming.test_factory import _glm_inventory_connector
        from sparkcache.streaming.test_publisher import FakeGlmRingBackend

        with tempfile.TemporaryDirectory() as directory:
            connector = _make_connector(
                Path(directory),
                0,
                block_size=64,
                extra_config={
                    "spark_cache_draft_policy": "colocated_target",
                    "spark_cache_draft_checkpoint_sha256": "1" * 64,
                },
            )
            inventory = _glm_inventory_connector()
            connector._plans = inventory._plans
            connector._layer_tensors = inventory._layer_tensors
            connector._rows_view = inventory._rows_view
            backends: list[FakeGlmRingBackend] = []
            append_started = threading.Event()
            allow_append = threading.Event()
            begin_context = connector._store.begin_context

            class BlockingTransaction:
                def __init__(self, transaction):
                    self._transaction = transaction

                def append_chunk(self, chunk):
                    append_started.set()
                    if not allow_append.wait(5):
                        raise TimeoutError("test append remained blocked")
                    return self._transaction.append_chunk(chunk)

                def commit_manifest(self):
                    return self._transaction.commit_manifest()

                def abort(self):
                    return self._transaction.abort()

            connector._store.begin_context = lambda **kwargs: BlockingTransaction(
                begin_context(**kwargs)
            )

            def build_ring(config, **_kwargs):
                backend = FakeGlmRingBackend()
                backends.append(backend)
                return NativeSnapshotRing(config, backend=backend)

            adapter = WorkerStreamingSnapshotAdapter(
                connector,
                settings=ProductionStreamingSettings(
                    native_library_path=Path("/opt/spark/lib/libspcc_snapshot.so"),
                    native_library_sha256="a" * 64,
                ),
                ring_builder=build_ring,
            )
            adapter.bind_kv_caches()
            connector._streaming_snapshots_enabled = True
            connector._streaming_runtime = adapter
            digest = "9" * 64
            offer = connector_module._StreamingSnapshotOffer(
                request_id="production-boundary",
                digest=digest,
                span_tokens=256,
                completed_tokens=256,
                block_ids=(10,),
            )
            identity = connector._identity(0)

            self.assertNotIn(digest, connector._held)
            self.assertFalse(connector._store.lookup(identity, digest).is_hit)
            connector.bind_connector_metadata(
                SparkCacheConnectorMetadata(
                    streaming_snapshot_offers=[offer],
                )
            )
            with mock.patch.object(
                torch.cuda,
                "current_stream",
                return_value=types.SimpleNamespace(cuda_stream=0xABC),
            ):
                connector.wait_for_save()

            self.assertTrue(append_started.wait(5))
            self.assertNotIn(digest, connector._held)
            self.assertFalse(connector._store.lookup(identity, digest).is_hit)
            allow_append.set()
            deadline = time.monotonic() + 5
            while digest not in connector._held and time.monotonic() < deadline:
                connector.bind_connector_metadata(SparkCacheConnectorMetadata())
                connector.wait_for_save()
                time.sleep(0.001)

            self.assertEqual(len(backends), 1)
            self.assertIn(digest, connector._held)
            self.assertTrue(connector._store.lookup(identity, digest).is_hit)
            stats = connector.get_kv_connector_stats()
            self.assertIsNotNone(stats)
            self.assertEqual(stats.data["reports"][0]["held"], [digest])
            streaming = stats.data["reports"][0]["streaming"]
            self.assertTrue(streaming["bound"])
            self.assertEqual(streaming["arena_mode"], "mapped_host")
            self.assertEqual(streaming["ring_depth"], 2)
            self.assertEqual(streaming["slot_bytes"], 64 * 1024 * 1024)
            self.assertEqual(adapter.status()["active_leases"], 0)
            self.assertEqual(adapter.status()["active_tickets"], 0)
            connector.shutdown()



class DCP2RoundTripTests(unittest.TestCase):
    """SparkCache DCP2 offline validation.

    The deployed NF3 variant runs TP4/DCP2 (dcp_rank_map [0,1,0,1]).
    These tests exercise the codec and connector's DCP2 code paths:
    interleave math, two-rank quorum, byte-exact store/restore on
    independent per-rank stores, and identity namespace separation
    from DCP4.

    BLOCKED: Live DCP2 acceptance requires resolving the TP/DCP
    identity gap (see test_dcp2_tp_ranks_sharing_dcp_rank_have_identical_storage_key)
    and read-only remote attestation of checkpoint identity, quant/rope
    layouts, and vLLM patch semantics.  These tests prove offline code
    paths only; they do NOT prove DCP2 is safe for live serving.
    """

    SPAN = 1024
    BLOCK_SIZE = 64

    def test_dcp2_owned_positions_cover_all_tokens(self) -> None:
        """DCP2 interleave: rank 0 owns even positions, rank 1 owns odd."""
        self.assertEqual(codec.owned_positions(8, 2, 0), (0, 2, 4, 6))
        self.assertEqual(codec.owned_positions(8, 2, 1), (1, 3, 5, 7))
        union: set[int] = set()
        for rank in range(2):
            union.update(codec.owned_positions(1024, 2, rank))
        self.assertEqual(union, set(range(1024)))
        # DCP2 has 128 local ordinals per 256-token chunk (vs 64 at DCP4)
        self.assertEqual(
            codec.local_slots_for_positions(
                codec.owned_positions(256, 2, 0),
                (0,),
                256,
                2,
            ),
            tuple(range(128)),
        )

    def test_dcp2_identity_namespace_differs_from_dcp4(self) -> None:
        """A cache entry written under DCP4 must never restore under DCP2."""
        with tempfile.TemporaryDirectory() as directory:
            dcp4_connector = _make_connector(
                Path(directory) / "dcp4",
                0,
                self.BLOCK_SIZE,
                extra_config={
                    "spark_cache_target_checkpoint_sha256": "a" * 64,
                    "spark_cache_draft_checkpoint_sha256": "b" * 64,
                    "spark_cache_draft_policy": "separate",
                },
            )
            # Override DCP degree to 2 for a second connector
            dcp2_values = {
                "spark_cache_root": str(Path(directory) / "dcp2"),
                "spark_cache_min_span_tokens": "256",
                "spark_cache_target_checkpoint_sha256": "a" * 64,
                "spark_cache_draft_checkpoint_sha256": "b" * 64,
                "spark_cache_draft_policy": "separate",
            }
            dcp2_kv_transfer = types.SimpleNamespace(
                get_from_extra_config=lambda key, default=None: dcp2_values.get(
                    key, default
                )
            )
            dcp2_vllm_config = types.SimpleNamespace(
                kv_transfer_config=dcp2_kv_transfer,
                cache_config=types.SimpleNamespace(block_size=self.BLOCK_SIZE),
                parallel_config=types.SimpleNamespace(
                    tensor_parallel_size=4, decode_context_parallel_size=2
                ),
                model_config=types.SimpleNamespace(model="test-target"),
            )
            dcp2_connector = SparkContextCacheConnector(
                vllm_config=dcp2_vllm_config,
                role=KVConnectorRole.WORKER,
                kv_cache_config=None,
            )
            dcp2_connector._worker_rank = lambda: 0  # type: ignore[method-assign]

        self.assertNotEqual(
            dcp4_connector._identity(0).storage_key,
            dcp2_connector._identity(0).storage_key,
        )
        self.assertEqual(dcp4_connector._dcp_degree, 4)
        self.assertEqual(dcp2_connector._dcp_degree, 2)

    def _make_dcp2_connector(
        self,
        directory: str,
        tp_rank: int,
        block_size: int = 64,
        extra_config: dict[str, object] | None = None,
    ) -> SparkContextCacheConnector:
        """Build a DCP2 connector with the TP-rank-to-DCP-rank map [0,1,0,1].

        In the deployed NF3 DCP2 variant, get_dcp_group().rank_in_group
        returns the DCP rank (0 or 1), not the TP rank (0-3).  We
        override _worker_rank to return the DCP rank so the store and
        restore paths use the correct shard identity.

        Each connector gets its own root directory (rank{tp_rank}) so
        stores are independent.  This does NOT model the case where
        two TP ranks share a single NVMe store — see
        test_dcp2_tp_ranks_sharing_dcp_rank_have_identical_storage_key.
        """
        dcp_rank_map = [0, 1, 0, 1]
        dcp_rank = dcp_rank_map[tp_rank]
        root = Path(directory) / f"rank{tp_rank}"
        values = {
            "spark_cache_root": str(root),
            "spark_cache_min_span_tokens": "256",
            "spark_cache_target_checkpoint_sha256": "1" * 64,
            "spark_cache_draft_checkpoint_sha256": "2" * 64,
            "spark_cache_draft_policy": "separate",
        }
        values.update(extra_config or {})
        kv_transfer_config = types.SimpleNamespace(
            get_from_extra_config=lambda key, default=None: values.get(
                key, default
            )
        )
        vllm_config = types.SimpleNamespace(
            kv_transfer_config=kv_transfer_config,
            cache_config=types.SimpleNamespace(block_size=block_size),
            parallel_config=types.SimpleNamespace(
                tensor_parallel_size=4,
                decode_context_parallel_size=2,
            ),
            model_config=types.SimpleNamespace(model="test-target"),
        )
        connector = SparkContextCacheConnector(
            vllm_config=vllm_config,
            role=KVConnectorRole.WORKER,
            kv_cache_config=None,
        )
        connector._worker_rank = lambda r=dcp_rank: r  # type: ignore[method-assign]
        connector._physical_rank = lambda r=tp_rank: r  # type: ignore[method-assign]
        return connector

    def test_dcp2_physical_and_dcp_local_rank_mapping(self) -> None:
        """TP4/DCP2: four physical TP ranks 0..3 map to DCP-local 0,1,0,1."""
        dcp_rank_map = [0, 1, 0, 1]
        with tempfile.TemporaryDirectory() as directory:
            connectors = [
                self._make_dcp2_connector(directory, tp, self.BLOCK_SIZE)
                for tp in range(4)
            ]
        for tp_rank, connector in enumerate(connectors):
            self.assertEqual(
                connector._physical_rank(), tp_rank,
                f"physical rank mismatch for tp_rank={tp_rank}",
            )
            self.assertEqual(
                connector._worker_rank(), dcp_rank_map[tp_rank],
                f"DCP-local rank mismatch for tp_rank={tp_rank}",
            )

    def test_dcp2_storage_keys_differ_for_tp_ranks_sharing_dcp_rank(self) -> None:
        """Under TP4/DCP2, TP ranks 0 and 2 share DCP rank 0 but must
        have distinct persistent storage keys because CacheIdentity now
        includes tp_shard_rank.

        This was the primary blocker (B1).  The tp_shard_rank field
        ensures each physical worker gets its own storage namespace,
        preventing cross-restore of complementary TP shards.
        """
        with tempfile.TemporaryDirectory() as directory:
            tp0 = self._make_dcp2_connector(directory, 0, self.BLOCK_SIZE)
            tp2 = self._make_dcp2_connector(directory, 2, self.BLOCK_SIZE)
            tp1 = self._make_dcp2_connector(directory, 1, self.BLOCK_SIZE)
            tp3 = self._make_dcp2_connector(directory, 3, self.BLOCK_SIZE)

        # Both TP0 and TP2 map to DCP rank 0.
        self.assertEqual(tp0._worker_rank(), 0)
        self.assertEqual(tp2._worker_rank(), 0)
        # But their physical ranks differ.
        self.assertEqual(tp0._physical_rank(), 0)
        self.assertEqual(tp2._physical_rank(), 2)
        # And their storage keys differ.
        id0 = tp0._identity(tp0._worker_rank())
        id2 = tp2._identity(tp2._worker_rank())
        self.assertNotEqual(
            id0.storage_key,
            id2.storage_key,
            "TP0 and TP2 must have distinct storage keys under DCP2.",
        )
        self.assertEqual(id0.tp_shard_rank, 0)
        self.assertEqual(id2.tp_shard_rank, 2)
        self.assertEqual(id0.dcp_shard_rank, 0)
        self.assertEqual(id2.dcp_shard_rank, 0)
        # TP1 and TP3 also differ.
        id1 = tp1._identity(tp1._worker_rank())
        id3 = tp3._identity(tp3._worker_rank())
        self.assertNotEqual(id1.storage_key, id3.storage_key)
        # All four are distinct.
        keys = {id0.storage_key, id1.storage_key, id2.storage_key, id3.storage_key}
        self.assertEqual(len(keys), 4, "all four physical workers must have distinct storage keys")

    def test_dcp2_reports_from_tp0_and_tp2_not_deduplicated(self) -> None:
        """Worker reports from TP0 (physical 0) and TP2 (physical 2)
        must not be deduplicated despite both having DCP-local rank 0.

        The scheduler's quorum set must contain both physical ranks.
        """
        with tempfile.TemporaryDirectory() as directory:
            scheduler = self._make_dcp2_scheduler(directory)
            digest = "e" * 64
            # TP0 reports (physical rank 0)
            scheduler.update_connector_output(
                self._make_report(0, [digest]))
            # TP2 reports (physical rank 2)
            scheduler.update_connector_output(
                self._make_report(2, [digest]))
            self.assertEqual(scheduler._quorum[digest], {0, 2})

    def test_dcp2_quorum_requires_all_four_physical_workers(self) -> None:
        """Quorum at DCP2 requires all four physical TP workers (0..3),
        not just the two DCP-local ranks."""
        with tempfile.TemporaryDirectory() as directory:
            scheduler = self._make_dcp2_scheduler(directory)
            digest = "e" * 64
            # Only physical 0 reports -> no quorum
            scheduler.update_connector_output(self._make_report(0, [digest]))
            self.assertFalse(scheduler._has_full_quorum(digest))
            # Physical 1 reports -> still no quorum
            scheduler.update_connector_output(self._make_report(1, [digest]))
            self.assertFalse(scheduler._has_full_quorum(digest))
            # Physical 2 reports -> still no quorum
            scheduler.update_connector_output(self._make_report(2, [digest]))
            self.assertFalse(scheduler._has_full_quorum(digest))
            # Physical 3 reports -> quorum
            scheduler.update_connector_output(self._make_report(3, [digest]))
            self.assertTrue(scheduler._has_full_quorum(digest))

    def test_dcp2_withdrawing_one_physical_worker_removes_quorum(self) -> None:
        """Withdrawing any one physical worker removes admission even if
        its paired DCP-local rank remains represented.

        Under TP4/DCP2, TP0 and TP2 share DCP rank 0.  If TP2 withdraws
        (holds=[]), quorum must break even though TP0 (same DCP rank 0)
        still holds the digest.
        """
        with tempfile.TemporaryDirectory() as directory:
            scheduler = self._make_dcp2_scheduler(directory)
            digest = "e" * 64
            for tp_rank in range(4):
                scheduler.update_connector_output(
                    self._make_report(tp_rank, [digest]))
            self.assertTrue(scheduler._has_full_quorum(digest))
            # TP2 (physical rank 2) withdraws
            scheduler.update_connector_output(
                self._make_report(2, []))
            self.assertFalse(scheduler._has_full_quorum(digest))
            # TP0 (physical rank 0, same DCP rank 0) still holds it
            self.assertIn(0, scheduler._quorum[digest])
            # But TP2 is gone
            self.assertNotIn(2, scheduler._quorum[digest])

    def test_dcp2_token_ownership_remains_based_on_dcp_rank(self) -> None:
        """DCP-local token position ownership must still use the DCP
        rank, not the physical TP rank.  This is the separation of
        concerns: identity uses physical rank, slicing uses DCP rank."""
        # DCP rank 0 owns even positions, DCP rank 1 owns odd positions
        self.assertEqual(codec.owned_positions(8, 2, 0), (0, 2, 4, 6))
        self.assertEqual(codec.owned_positions(8, 2, 1), (1, 3, 5, 7))
        # TP2 has physical rank 2 but DCP rank 0, so it owns even positions
        with tempfile.TemporaryDirectory() as directory:
            tp2 = self._make_dcp2_connector(directory, 2, self.BLOCK_SIZE)
            self.assertEqual(tp2._physical_rank(), 2)
            self.assertEqual(tp2._worker_rank(), 0)
            positions = codec.owned_positions(
                self.SPAN, 2, tp2._worker_rank())
            self.assertEqual(positions[0], 0)  # even
            self.assertEqual(positions[1], 2)  # even

    def test_dcp2_malformed_physical_rank_report_fails_closed(self) -> None:
        """Reports with out-of-range physical ranks (negative, >= tp_degree)
        must be rejected by _absorb_quorum and not affect the quorum set."""
        with tempfile.TemporaryDirectory() as directory:
            scheduler = self._make_dcp2_scheduler(directory)
            digest = "e" * 64
            # Negative rank
            scheduler.update_connector_output(self._make_report(-1, [digest]))
            # Rank >= tp_degree (4)
            scheduler.update_connector_output(self._make_report(4, [digest]))
            scheduler.update_connector_output(self._make_report(99, [digest]))
            # Non-integer rank
            scheduler.update_connector_output(
                types.SimpleNamespace(
                    invalid_block_ids=set(),
                    kv_connector_stats=types.SimpleNamespace(
                        data={"spark_context_cache": {"rank": "x", "held": [digest]}}
                    ),
                )
            )
            self.assertNotIn(digest, scheduler._quorum)
            self.assertFalse(scheduler._has_full_quorum(digest))

    def test_dcp2_independent_store_restore_is_byte_exact(self) -> None:
        """Full store/restore cycle with DCP2, using independent per-TP-rank
        store roots.

        This test proves the codec's DCP2 interleave math and the
        connector's store/restore path are byte-exact when each physical
        TP rank has its own NVMe store directory.

        DCP2 halves the interleave stride, so span 1024 produces 512
        local ordinals per DCP rank (vs 256 at DCP4).  The pool and
        block table must be doubled accordingly: 8 blocks of 64.
        """
        with tempfile.TemporaryDirectory() as directory:
            pools: list[dict[str, torch.Tensor]] = []
            originals: list[dict[str, torch.Tensor]] = []
            connectors = []
            dcp_rank_map = [0, 1, 0, 1]
            block_ids = (3, 0, 5, 1, 7, 2, 4, 6)  # 8 blocks for 512 slots
            for tp_rank in range(4):
                connector = self._make_dcp2_connector(
                    directory, tp_rank, self.BLOCK_SIZE
                )
                pool = _make_pools(8, self.BLOCK_SIZE)
                connector.register_kv_caches(pool)
                connectors.append(connector)
                pools.append(pool)
                originals.append({k: v.clone() for k, v in pool.items()})

            plan = _ReqPlan(
                request_id="req-dcp2",
                digest="d" * 64,
                span_tokens=self.SPAN,
                block_ids=block_ids,
                is_store=True,
            )
            store_meta = SparkCacheConnectorMetadata(plans=[plan])
            for connector in connectors:
                connector.bind_connector_metadata(store_meta)
                connector.wait_for_save()
                _drain_store(connector)
                self.assertEqual(connector.counters["store_committed"], 1)

            load_plan = dataclasses.replace(plan, is_store=False)
            load_meta = SparkCacheConnectorMetadata(plans=[load_plan])
            for tp_rank, connector in enumerate(connectors):
                for tensor in pools[tp_rank].values():
                    tensor.zero_()
                connector.bind_connector_metadata(load_meta)
                connector.start_load_kv(None)
                self.assertEqual(_drain(connector), {"req-dcp2"})
                self.assertEqual(connector.counters["load_verified"], 1)
                self.assertEqual(connector.get_block_ids_with_load_errors(), set())

            # Verify byte-exact restoration at the DCP2-owned slots.
            for tp_rank in range(4):
                dcp_rank = dcp_rank_map[tp_rank]
                slots = codec.local_slots_for_positions(
                    codec.owned_positions(self.SPAN, 2, dcp_rank),
                    block_ids,
                    self.BLOCK_SIZE,
                    2,
                )
                slot_tensor = torch.tensor(slots, dtype=torch.long)
                for name in _LAYERS:
                    restored = pools[tp_rank][name].reshape(-1, _LAYERS[name])
                    original = originals[tp_rank][name].reshape(-1, _LAYERS[name])
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

    def _make_dcp2_scheduler(self, directory: str) -> SparkContextCacheConnector:
        """Build a scheduler-side connector for DCP2 quorum tests."""
        sched_values = {
            "spark_cache_root": str(Path(directory) / "sched"),
            "spark_cache_min_span_tokens": "256",
            "spark_cache_target_checkpoint_sha256": "3" * 64,
            "spark_cache_draft_checkpoint_sha256": "4" * 64,
            "spark_cache_draft_policy": "separate",
        }
        sched_kv = types.SimpleNamespace(
            get_from_extra_config=lambda key, default=None: sched_values.get(
                key, default
            )
        )
        sched_vllm = types.SimpleNamespace(
            kv_transfer_config=sched_kv,
            cache_config=types.SimpleNamespace(block_size=64),
            parallel_config=types.SimpleNamespace(
                tensor_parallel_size=4, decode_context_parallel_size=2
            ),
            model_config=types.SimpleNamespace(model="test-target"),
        )
        return SparkContextCacheConnector(
            vllm_config=sched_vllm,
            role=KVConnectorRole.SCHEDULER,
            kv_cache_config=None,
        )

    @staticmethod
    def _make_report(physical_rank: int, held: list[str]) -> types.SimpleNamespace:
        """Build a connector_output with a single worker report."""
        return types.SimpleNamespace(
            invalid_block_ids=set(),
            kv_connector_stats=types.SimpleNamespace(
                data={
                    "spark_context_cache": {
                        "rank": physical_rank,
                        "held": held,
                    }
                }
            ),
        )



    def test_dcp2_identity_pins_dcp_degree_2(self) -> None:
        """The identity base must record dcp_degree=2 for the DCP2 variant.

        PENDING READ-ONLY REMOTE ATTESTATION: The quantization_layout
        and rope_layout strings are hard-coded in the connector and are
        provisionally consistent with the NF3 recipe docs, but have not
        been attested against the deployed image's actual KV config.
        """
        with tempfile.TemporaryDirectory() as directory:
            values = {
                "spark_cache_root": str(Path(directory)),
                "spark_cache_min_span_tokens": "256",
                "spark_cache_target_checkpoint_sha256": "5" * 64,
                "spark_cache_draft_checkpoint_sha256": "6" * 64,
                "spark_cache_draft_policy": "separate",
            }
            kv_transfer_config = types.SimpleNamespace(
                get_from_extra_config=lambda key, default=None: values.get(
                    key, default
                )
            )
            vllm_config = types.SimpleNamespace(
                kv_transfer_config=kv_transfer_config,
                cache_config=types.SimpleNamespace(block_size=64),
                parallel_config=types.SimpleNamespace(
                    tensor_parallel_size=4, decode_context_parallel_size=2
                ),
                model_config=types.SimpleNamespace(model="test-target"),
            )
            connector = SparkContextCacheConnector(
                vllm_config=vllm_config,
                role=KVConnectorRole.WORKER,
                kv_cache_config=None,
            )

        self.assertEqual(connector._identity_base["dcp_degree"], 2)
        self.assertEqual(connector._identity_base["tp_degree"], 4)
        # PENDING READ-ONLY REMOTE ATTESTATION: these layouts match the
        # recipe docs but have not been verified against the deployed image.
        self.assertEqual(
            connector._identity_base["quantization_layout"],
            "nvfp4_ds_mla-per-token-v1",
        )
        self.assertEqual(
            connector._identity_base["rope_layout"], "glm52-rope-v1"
        )

    def test_dcp2_checkpoint_identity_must_be_canonical_manifest(self) -> None:
        """The checkpoint identity must be a 64-character SHA-256, but a
        hard-coded hash computed from a subset of recipe pins is NOT a
        valid checkpoint identity.

        The connector correctly rejects non-SHA-256 strings (tested in
        CheckpointIdentityTests).  This test confirms that placeholder
        identities are accepted by the connector's format check but
        must be replaced by canonical manifest output before live use.

        A canonical manifest generator must hash every deployed weight
        shard and cache-affecting artifact, not just revision/config/index.
        The previous hand-computed hashes (4ae049... / 15202f...)
        only covered revision|config|index (and a few draft pins),
        omitting the target's 184 weight shards.  They have been
        removed and must NOT be used.
        """
        # Placeholder identities — format-valid but NOT attested.
        # The operator must replace these with canonical manifest output.
        placeholder_target = "a" * 64
        placeholder_draft = "b" * 64
        with tempfile.TemporaryDirectory() as directory:
            connector = _make_connector(
                Path(directory),
                0,
                extra_config={
                    "spark_cache_target_checkpoint_sha256": placeholder_target,
                    "spark_cache_draft_checkpoint_sha256": placeholder_draft,
                    "spark_cache_draft_policy": "separate",
                },
            )

        self.assertEqual(
            connector._identity_base["target_checkpoint"], placeholder_target
        )
        self.assertEqual(
            connector._identity_base["draft_checkpoint"], placeholder_draft
        )
        self.assertEqual(
            connector._identity_base["draft_kv_policy"], "separate"
        )



class DCP4CompatibilityTests(unittest.TestCase):
    """Verify that the tp_shard_rank extension is compatible with DCP4.

    Under DCP4, tp_degree==dcp_degree==4, so physical TP rank equals
    DCP rank.  The tp_shard_rank field is set to the physical TP rank,
    which equals the DCP rank.  This means DCP4 entries get a new
    storage_key (because to_wire() now includes tp_shard_rank).  Old
    entries written without tp_shard_rank will miss, causing a clean
    re-prefill — the fail-closed path.  This is intentional: entries
    written under the old schema did not distinguish physical workers
    and must NOT be silently reinterpreted.
    """

    def test_dcp4_physical_rank_equals_dcp_rank(self) -> None:
        """Under DCP4, physical TP rank == DCP rank for all four ranks."""
        with tempfile.TemporaryDirectory() as directory:
            for rank in range(4):
                connector = _make_connector(Path(directory), rank, 64)
                self.assertEqual(connector._physical_rank(), rank)
                self.assertEqual(connector._worker_rank(), rank)

    def test_dcp4_storage_keys_are_distinct_per_rank(self) -> None:
        """Under DCP4, all four ranks have distinct storage keys."""
        with tempfile.TemporaryDirectory() as directory:
            keys = set()
            for rank in range(4):
                connector = _make_connector(Path(directory), rank, 64)
                identity = connector._identity(rank)
                keys.add(identity.storage_key)
                self.assertEqual(identity.tp_shard_rank, rank)
                self.assertEqual(identity.dcp_shard_rank, rank)
        self.assertEqual(len(keys), 4)

    def test_dcp4_quorum_still_requires_all_four_physical_workers(self) -> None:
        """DCP4 quorum requires all four physical TP workers (0..3).

        This is unchanged from the old behavior (which required all 4
        DCP ranks) because under DCP4 physical rank == DCP rank.
        """
        with tempfile.TemporaryDirectory() as directory:
            c = _make_connector(Path(directory) / "r0", 0, 64)
            digest = "c" * 64
            for rank in (0, 1, 2):
                c.update_connector_output(
                    types.SimpleNamespace(
                        invalid_block_ids=set(),
                        kv_connector_stats=types.SimpleNamespace(
                            data={
                                "spark_context_cache": {
                                    "rank": rank, "held": [digest]
                                }
                            }
                        ),
                    )
                )
            self.assertFalse(c._has_full_quorum(digest))
            c.update_connector_output(
                types.SimpleNamespace(
                    invalid_block_ids=set(),
                    kv_connector_stats=types.SimpleNamespace(
                        data={
                            "spark_context_cache": {
                                "rank": 3, "held": [digest]
                            }
                        }
                    ),
                )
            )
            self.assertTrue(c._has_full_quorum(digest))

    def test_dcp4_legacy_entries_without_tp_shard_rank_miss(self) -> None:
        """Old entries written before the tp_shard_rank extension had
        to_wire() dicts that lacked the key entirely.  Their storage_key
        (SHA-256 of canonical JSON of the old wire dict) differs from
        any new identity that includes a concrete tp_shard_rank.

        This test constructs the explicit pre-extension wire dictionary
        (without the tp_shard_rank key) and proves its storage key
        differs from the new concrete identity.  The fail-closed claim
        is precise: old entries miss because the key set changed, not
        merely because the tp_shard_rank value differs.
        """
        import hashlib

        def _canonical_json(value: object) -> bytes:
            import json
            return json.dumps(
                value, sort_keys=True, separators=(",", ":"),
                ensure_ascii=True,
            ).encode("ascii")

        with tempfile.TemporaryDirectory() as directory:
            connector = _make_connector(Path(directory), 0, 64)
            new_identity = connector._identity(0)

            # Explicit pre-extension wire dict — no tp_shard_rank key.
            old_wire = {
                "target_checkpoint": "1" * 64,
                "draft_checkpoint": "2" * 64,
                "quantization_layout": "nvfp4_ds_mla-per-token-v1",
                "rope_layout": "glm52-rope-v1",
                "tp_degree": 4,
                "dcp_degree": 4,
                "chunk_tokens": 256,
                "dcp_shard_rank": 0,
                "boundary_hidden_policy": "live_forward",
                "draft_kv_policy": "separate",
            }
            old_storage_key = hashlib.sha256(
                _canonical_json(old_wire)
            ).hexdigest()

            # New identity's wire dict includes tp_shard_rank=0.
            new_wire = new_identity.to_wire()
            self.assertIn("tp_shard_rank", new_wire)
            self.assertNotIn("tp_shard_rank", old_wire)

            self.assertNotEqual(
                old_storage_key,
                new_identity.storage_key,
                "Old entries (wire dict without tp_shard_rank key) must"
                " have a different storage_key from new identities —"
                " fail-closed compatibility.",
            )
if __name__ == "__main__":
    unittest.main()
