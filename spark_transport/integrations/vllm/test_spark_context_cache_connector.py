"""GPU-free tests for the SparkRing persistent context-cache connector.

Simulates a four-rank DCP4 store -> pool wipe -> restore cycle with real
byte comparisons on CPU torch tensors, plus the fail-closed sabotage paths.
vLLM is stubbed the same way as the sibling backend suites.
"""

from __future__ import annotations

import dataclasses
import sys
import tempfile
import types
import unittest
from pathlib import Path

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
    metrics = types.ModuleType(
        "vllm.distributed.kv_transfer.kv_connector.v1.metrics"
    )
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
        plans = codec.build_layer_plans(
            {"a.attn": 8, "b.indexer": 4, "c.draft": 2}
        )
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


def _make_connector(
    root: Path, rank: int, block_size: int = 64
) -> SparkContextCacheConnector:
    kv_transfer_config = types.SimpleNamespace(
        get_from_extra_config=lambda key, default=None: {
            "spark_cache_root": str(root),
            "spark_cache_min_span_tokens": "256",
            "spark_cache_target_id": "test-target",
            "spark_cache_draft_id": "test-draft",
            "spark_cache_draft_policy": "separate",
        }.get(key, default)
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
        role=KVConnectorRole.WORKER,
        kv_cache_config=None,
    )
    connector._worker_rank = lambda: rank  # type: ignore[method-assign]
    return connector


_LAYERS = {
    "model.layers.0.self_attn.attn": 368,
    "model.layers.0.self_attn.indexer_cache": 128,
    "draft.layers.0.self_attn.attn": 368,
}


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
                originals.append(
                    {k: v.clone() for k, v in pool.items()}
                )
            store_meta = SparkCacheConnectorMetadata(plans=[self._plan()])
            for connector in connectors:
                connector.bind_connector_metadata(store_meta)
                connector.wait_for_save()
                self.assertEqual(connector.counters["store_committed"], 1)
                self.assertEqual(connector.counters["store_failed"], 0)

            load_plan = dataclasses.replace(self._plan(), is_store=False)
            load_meta = SparkCacheConnectorMetadata(plans=[load_plan])
            for rank, connector in enumerate(connectors):
                for tensor in pools[rank].values():
                    tensor.zero_()
                connector.bind_connector_metadata(load_meta)
                connector.start_load_kv(None)
                self.assertEqual(connector.counters["load_verified"], 1)
                self.assertEqual(
                    connector.get_block_ids_with_load_errors(), set()
                )

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
                    original = originals[rank][name].reshape(
                        -1, _LAYERS[name]
                    )
                    torch.testing.assert_close(
                        restored[slot_tensor],
                        original[slot_tensor],
                        rtol=0,
                        atol=0,
                    )
                    untouched = torch.ones(
                        restored.shape[0], dtype=torch.bool
                    )
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
            self.assertEqual(connector.counters["load_failed"], 1)
            self.assertEqual(
                connector.get_block_ids_with_load_errors(),
                set(load_plan.block_ids),
            )
            # errors drain once reported
            self.assertEqual(
                connector.get_block_ids_with_load_errors(), set()
            )

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
            manifests = sorted((root / "manifests").rglob("*.json"))
            self.assertTrue(manifests)
            payload = manifests[0].read_bytes()
            manifests[0].write_bytes(payload[: len(payload) // 2])

            load_plan = dataclasses.replace(self._plan(), is_store=False)
            connector.bind_connector_metadata(
                SparkCacheConnectorMetadata(plans=[load_plan])
            )
            connector.start_load_kv(None)
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
                    req_ids=[], resumed_req_ids=set(),
                    num_computed_tokens=[], new_block_ids=[],
                ),
            )
            meta1 = connector.build_connector_meta(step1)
            self.assertEqual(meta1.plans, [])
            step2 = types.SimpleNamespace(
                scheduled_new_reqs=[],
                num_scheduled_tokens={"req-c": 512},
                scheduled_cached_reqs=types.SimpleNamespace(
                    req_ids=["req-c"], resumed_req_ids=set(),
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


class SweepTests(unittest.TestCase):
    def test_sweep_invalidates_only_damaged_entries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rank1"
            connector = _make_connector(root, 1, 64)
            pool = _make_pools(8, 64)
            connector.register_kv_caches(pool)
            plan = _ReqPlan("req-s", "a" * 64, 1024, (3, 0, 5, 1), True)
            connector.bind_connector_metadata(
                SparkCacheConnectorMetadata(plans=[plan])
            )
            connector.wait_for_save()
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


class SchedulerRetirementTests(unittest.TestCase):
    def test_load_errors_retire_the_admitted_entry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rank0"
            connector = _make_connector(root, 0, 64)
            pool = _make_pools(8, 64)
            connector.register_kv_caches(pool)
            digest = "b" * 64
            plan = _ReqPlan("req-r", digest, 1024, (3, 0, 5, 1), True)
            connector.bind_connector_metadata(
                SparkCacheConnectorMetadata(plans=[plan])
            )
            connector.wait_for_save()
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
            connector._admitted["req-k"] = digest
            connector.update_connector_output(
                types.SimpleNamespace(invalid_block_ids=set())
            )
            self.assertTrue(
                connector._store.lookup(
                    connector._identity(0), digest
                ).is_hit
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
            self.assertEqual(c.get_num_new_matched_tokens(req, 0), (0, False))

            # three of four ranks confirm -> still not offered
            for rank in (0, 1, 2):
                c.update_connector_output(
                    types.SimpleNamespace(
                        invalid_block_ids=set(),
                        kv_connector_stats=types.SimpleNamespace(
                            data={"spark_context_cache": {
                                "rank": rank, "held": [digest]}}
                        ),
                    )
                )
            self.assertEqual(c.get_num_new_matched_tokens(req, 0), (0, False))

            # fourth rank confirms -> now offered
            c.update_connector_output(
                types.SimpleNamespace(
                    invalid_block_ids=set(),
                    kv_connector_stats=types.SimpleNamespace(
                        data={"spark_context_cache": {
                            "rank": 3, "held": [digest]}}
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
            for rank in range(4):
                c.update_connector_output(
                    types.SimpleNamespace(
                        invalid_block_ids=set(),
                        kv_connector_stats=types.SimpleNamespace(
                            data={"spark_context_cache": {
                                "rank": rank, "held": [digest]}}
                        ),
                    )
                )
            self.assertEqual(
                c.get_num_new_matched_tokens(req, 0)[0], 1024
            )

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
            self.assertIn(digest, c._held)

            chunk = sorted((root / "chunks").glob("*.spcc"))[0]
            payload = bytearray(chunk.read_bytes())
            payload[len(payload) // 2] ^= 0x10
            chunk.write_bytes(bytes(payload))
            c.sweep_integrity()
            self.assertNotIn(digest, c._held)


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
                    data={"reports": [
                        {"rank": r, "held": [digest]} for r in range(4)
                    ]}
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
        self.assertNotIn(
            "<locals>", connector_module.SparkCacheStats.__qualname__
        )


if __name__ == "__main__":
    unittest.main()
