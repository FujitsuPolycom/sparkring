"""DeepSeek-V4.1 groups recorded on release 2026.10.4, 2026-10-15.

The 17-group worker inventory is pinned in deepseek_v41_topology.json.
Seven encoder windows persist; one full-history group holds eight latent
and indexer caches; one compressor ring and eight decoder/draft windows
are request-private. Both decoder and draft reserve 128 replay tokens.
"""

from __future__ import annotations

import json
import tempfile
import types
import unittest
from pathlib import Path

import torch
import pytest

from sparkcache.request_cache_scope import UNSALTED_SCOPE
from sparkcache.spark_context_cache_profiles import (
    ProfileError,
    resolve_profile,
)
from sparkcache.test_spark_context_cache_config import _make_vllm_config
from sparkcache.test_qwen38_installer_topology import UniformTypeKVCacheSpecs
from sparkcache.test_spark_context_cache_connector import (
    KVConnectorRole,
    _ReqPlan,
    _deepseek_tp4_hma_config,
    _make_connector,
    SparkContextCacheConnector,
)

REPLAY_RESERVE = 128
CHUNK_TOKENS = 256


class _EngineSpec:
    """Fake with vLLM's KVCacheSpec property and field names."""

    block_size = 256
    storage_block_size = 256
    page_size_bytes = 0
    tokens_per_state = 1

    def __init__(
        self,
        *,
        prefix_cacheable: bool = True,
        prefill_replay_tokens: int = 0,
        block_size: int = 256,
        page_size_bytes: int = 0,
        tokens_per_state: int = 1,
    ) -> None:
        self._prefix_cacheable = prefix_cacheable
        self._prefill_replay_tokens = prefill_replay_tokens
        self.block_size = block_size
        self.storage_block_size = block_size
        self.page_size_bytes = page_size_bytes
        self.tokens_per_state = tokens_per_state

    @property
    def prefix_cacheable(self) -> bool:
        return self._prefix_cacheable

    @property
    def prefill_replay_tokens(self) -> int:
        return self._prefill_replay_tokens


class FullAttentionSpec(_EngineSpec):
    """MRO-name stand-in for vLLM's attention spec base."""


class SlidingWindowSpec(_EngineSpec):
    """MRO-name stand-in for vLLM's sliding-window spec base."""

    sliding_window = 128


class MLAAttentionSpec(FullAttentionSpec):
    """Latent and indexer pages share the engine's MLA spec class."""


class SlidingWindowMLASpec(SlidingWindowSpec):
    """Encoder and request-private decoder/draft sliding windows."""


class CircularBufferSpec(FullAttentionSpec):
    """Request-private compressor ring."""


OBSERVED_TOPOLOGY = json.loads(
    Path(__file__).with_name("deepseek_v41_topology.json").read_text()
)


def deepseek_v41_layout(*, small_pages: bool = False, worker: bool = False) -> types.SimpleNamespace:
    """Recorded manager packing with per-layer worker specs.

    Small pages reduce allocation bytes only. Manager blocks, compression
    quanta, replay reserves, layer names and ordering retain engine values.
    """
    classes = {cls.__name__: cls for cls in
               (MLAAttentionSpec, SlidingWindowMLASpec, CircularBufferSpec)}
    groups = []
    for row in OBSERVED_TOPOLOGY:
        members = {}
        for name in row["layers"]:
            ratio = 1
            page = row["page_size_bytes"]
            if row["reuse_policy"] == "full":
                ratio = 1 if ".20." in name else 2
                bytes_per_state = 68 if name.endswith("k_cache") else 288
                page = 256 // ratio * bytes_per_state
            if small_pages:
                page = (256 // ratio if not name.endswith("k_cache") else 64 // ratio) if row["reuse_policy"] == "full" else 32
            spec = classes[row["spec"]](
                prefix_cacheable=row["reuse_policy"] in ("full", "sliding"),
                prefill_replay_tokens=row.get("replay_tokens", 0),
                block_size=row["block_size"], page_size_bytes=page,
                tokens_per_state=ratio,
            )
            spec.storage_block_size = row["storage_block_size"]
            members[name] = spec
        representative = next(iter(members.values()))
        spec = representative
        if worker:
            spec = UniformTypeKVCacheSpecs(
                block_size=row["block_size"], dcp_replicated=False,
                page_size_bytes=sum(member.page_size_bytes for member in members.values()),
                kv_cache_specs=members,
            )
        groups.append(types.SimpleNamespace(
            kv_cache_spec=spec, layer_names=tuple(members), is_eagle_group=False,
        ))
    return types.SimpleNamespace(kv_cache_groups=tuple(groups), num_blocks=512)


@pytest.mark.parametrize("small_pages", [False, True])
def test_dsv41_hw_1_scheduler_worker_topology_and_context_digest(tmp_path, small_pages):
    from sparkcache import spark_context_cache_config as cfg

    scheduler_layout = deepseek_v41_layout(small_pages=small_pages)
    worker_layout = deepseek_v41_layout(small_pages=small_pages, worker=True)
    assert cfg.kv_group_topology(scheduler_layout) == cfg.kv_group_topology(worker_layout)
    assert cfg.kv_group_topology_digest(scheduler_layout) == cfg.kv_group_topology_digest(worker_layout)
    assert _v41_identity(scheduler_layout) == _v41_identity(worker_layout)
    scheduler = _v41_connector(tmp_path, scheduler_layout, role=KVConnectorRole.SCHEDULER)
    worker = _v41_connector(tmp_path, worker_layout)
    try:
        assert scheduler._context_digest_salt == worker._context_digest_salt
        assert scheduler._digest(list(range(2304)), 2048) == worker._digest(list(range(2304)), 2048)
        assert scheduler._block_size == worker._block_size == 256
        assert scheduler._identity(0) == worker._identity(0)
        for length in (*range(1, 4097), 127840, 524288):
            expected = max(0, (length - 128) // 256 * 256)
            assert scheduler._aligned_span(length) == worker._aligned_span(length) == expected
    finally:
        scheduler.shutdown()
        worker.shutdown()


def test_qwen_hw_1_v41_wrapper_validates_all_non_persisted_member_states():
    from sparkcache import spark_context_cache_config as cfg

    layout = deepseek_v41_layout(worker=True)
    members = list(layout.kv_cache_groups[-1].kv_cache_spec.kv_cache_specs.values())
    members[-1].tokens_per_state = 2
    with pytest.raises(RuntimeError, match="incompatible state geometry"):
        cfg.kv_group_topology(layout)


def span_tables(span: int) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    """Request block tables covering ``span`` for the V4.1 fixture groups."""

    from sparkcache import spark_context_cache_config as cfg

    topology = cfg.kv_group_topology(deepseek_v41_layout(small_pages=True))
    tables = []
    for group in topology:
        width = int(group["logical_tokens_per_block"])
        required = (span + width - 1) // width
        # Non-persisted request tables stay short: a compressor ring owns one
        # block and a request-private window covers only computed rows.
        if group["reuse_policy"] in ("transient", "replay"):
            required = 1
        tables.append(tuple(range(1, required + 1)))
    return tuple(tables), tuple(int(group["logical_tokens_per_block"])
                                for group in topology)


def _v41_connector(
    root: Path, layout: types.SimpleNamespace, *, role=KVConnectorRole.WORKER,
) -> SparkContextCacheConnector:
    return _make_connector(
        root,
        0,
        block_size=128 if role == KVConnectorRole.SCHEDULER else 256,
        extra_config={
            "spark_cache_model_profile": "deepseek-v41-b12x-hma",
            "spark_cache_min_span_tokens": str(CHUNK_TOKENS),
            "spark_cache_draft_policy": "colocated_target",
            "spark_cache_draft_checkpoint_sha256": "",
            "spark_cache_publication_schema": "tail-cow-v2",
        },
        tp=1,
        dcp=1,
        kv_cache_config=layout,
        role=role,
    )


def _v41_identity(layout: types.SimpleNamespace, **extra: object) -> object:
    from sparkcache import spark_context_cache_config as cfg

    vllm, transfer = _make_vllm_config(
        {
            "spark_cache_model_profile": "deepseek-v41-b12x-hma",
            "spark_cache_draft_policy": "colocated_target",
            "spark_cache_draft_checkpoint_sha256": "",
            **extra,
        },
        tp=1,
        dcp=1,
        block_size=CHUNK_TOKENS,
    )
    return cfg.parse_connector_config(
        vllm, transfer, layout
    ).build_identity(0, 0)


class DeepSeekV41TopologyTests(unittest.TestCase):
    """Group classification and topology-digest behavior for V4.1."""

    def test_group_inventory_matches_v41_cache_classes(self) -> None:
        from sparkcache import spark_context_cache_config as cfg

        topology = cfg.kv_group_topology(deepseek_v41_layout())
        self.assertEqual(json.loads(json.dumps(topology)), OBSERVED_TOPOLOGY)
        self.assertEqual(len(topology), 17)
        self.assertEqual(sum(len(group["layers"]) for group in topology), 54)
        self.assertEqual([group["reuse_policy"] for group in topology],
                         ["sliding"] * 7 + ["full", "transient"] + ["replay"] * 8)

    def test_non_persisted_groups_record_replay_and_state_geometry(self) -> None:
        from sparkcache import spark_context_cache_config as cfg

        topology = cfg.kv_group_topology(deepseek_v41_layout())
        for group in topology:
            if group["reuse_policy"] in ("transient", "replay"):
                self.assertEqual(
                    group["replay_tokens"],
                    REPLAY_RESERVE if group["reuse_policy"] == "replay" else 0,
                )
                self.assertEqual(group["tokens_per_state"], 1)
            else:
                self.assertNotIn("replay_tokens", group)
                self.assertNotIn("tokens_per_state", group)

    def test_digest_forks_on_replay_tokens_state_geometry_and_policy(self) -> None:
        from sparkcache import spark_context_cache_config as cfg

        layout = deepseek_v41_layout()
        original = cfg.kv_group_topology_digest(layout)
        layout.kv_cache_groups[9].kv_cache_spec._prefill_replay_tokens = 64
        self.assertNotEqual(cfg.kv_group_topology_digest(layout), original)

        layout = deepseek_v41_layout()
        original = cfg.kv_group_topology_digest(layout)
        layout.kv_cache_groups[8].kv_cache_spec.tokens_per_state = 2
        layout.kv_cache_groups[8].kv_cache_spec.page_size_bytes = 68 * 4
        self.assertNotEqual(cfg.kv_group_topology_digest(layout), original)

        layout = deepseek_v41_layout()
        original = cfg.kv_group_topology_digest(layout)
        layout.kv_cache_groups[9].kv_cache_spec._prefix_cacheable = True
        layout.kv_cache_groups[9].kv_cache_spec._prefill_replay_tokens = 0
        topology = cfg.kv_group_topology(layout)
        self.assertEqual(topology[9]["reuse_policy"], "sliding")
        self.assertNotEqual(cfg.kv_group_topology_digest(layout), original)

    def test_unknown_non_prefix_cacheable_spec_is_refused(self) -> None:
        from sparkcache import spark_context_cache_config as cfg

        class KpoolTailSpec(FullAttentionSpec):
            def __init__(self) -> None:
                super().__init__(
                    prefix_cacheable=False, block_size=8, page_size_bytes=64
                )

        layout = types.SimpleNamespace(
            kv_cache_groups=(
                types.SimpleNamespace(
                    kv_cache_spec=KpoolTailSpec(),
                    is_eagle_group=False,
                    layer_names=("scratch",),
                ),
            )
        )
        with self.assertRaisesRegex(RuntimeError, "unsupported block-page"):
            cfg.kv_group_topology(layout)

    def test_circular_buffer_requires_engine_private_properties(self) -> None:
        from sparkcache import spark_context_cache_config as cfg

        class CircularBufferSpec(FullAttentionSpec):
            def __init__(self) -> None:
                super().__init__(
                    prefix_cacheable=True, block_size=8, page_size_bytes=64
                )

        layout = types.SimpleNamespace(
            kv_cache_groups=(
                types.SimpleNamespace(
                    kv_cache_spec=CircularBufferSpec(),
                    is_eagle_group=False,
                    layer_names=("ring",),
                ),
            )
        )
        with self.assertRaisesRegex(RuntimeError, "circular-buffer"):
            cfg.kv_group_topology(layout)


class DeepSeekV41ProfileTests(unittest.TestCase):
    """The deepseek-v41-b12x-hma profile binds its cache contract."""

    def test_profile_fields_and_layout_names(self) -> None:
        profile = resolve_profile("deepseek-v41-b12x-hma")
        self.assertEqual(profile.storage_mode, "block_pages_v1")
        self.assertEqual(profile.chunk_tokens, CHUNK_TOKENS)
        self.assertEqual(profile.required_block_size, CHUNK_TOKENS)
        self.assertTrue(profile.requires_replay_group)
        self.assertFalse(profile.cuda_page_restore)
        self.assertTrue(profile.kv_replicated_across_tp)
        self.assertEqual(profile.default_draft_kv_policy, "colocated_target")
        self.assertEqual(
            profile.quantization_layout, "deepseek-v41-b12x-hma-block-pages-v1"
        )
        v4 = resolve_profile("deepseek-v4-fp8-hma")
        self.assertNotEqual(profile.quantization_layout, v4.quantization_layout)
        self.assertNotEqual(profile.rope_layout, v4.rope_layout)

    def test_profile_binds_block_size(self) -> None:
        profile = resolve_profile("deepseek-v41-b12x-hma")
        profile.validate_for_deployment(
            dcp_degree=1,
            block_size=CHUNK_TOKENS,
            min_span_tokens=CHUNK_TOKENS,
            cuda_restore=False,
        )
        with self.assertRaisesRegex(ProfileError, "block_size 512 must be 256"):
            profile.validate_for_deployment(
                dcp_degree=1,
                block_size=512,
                min_span_tokens=CHUNK_TOKENS,
                cuda_restore=False,
            )

    def test_other_profiles_refuse_non_persisted_groups(self) -> None:
        from sparkcache import spark_context_cache_config as cfg

        topology = cfg.kv_group_topology(deepseek_v41_layout())
        for name in ("qwen38-flash-next-hybrid", "deepseek-v4-fp8-hma"):
            profile = resolve_profile(name)
            with self.assertRaisesRegex(ProfileError, "non-persisted"):
                profile.validate_for_deployment(
                    dcp_degree=1,
                    block_size=CHUNK_TOKENS,
                    min_span_tokens=CHUNK_TOKENS,
                    cuda_restore=False,
                    group_topology=topology,
                )

    def test_profile_requires_a_replay_group(self) -> None:
        from sparkcache import spark_context_cache_config as cfg

        profile = resolve_profile("deepseek-v41-b12x-hma")
        profile.validate_for_deployment(
            dcp_degree=1,
            block_size=CHUNK_TOKENS,
            min_span_tokens=CHUNK_TOKENS,
            cuda_restore=False,
            group_topology=cfg.kv_group_topology(deepseek_v41_layout()),
        )
        class MambaSpec:
            block_size = CHUNK_TOKENS
            storage_block_size = CHUNK_TOKENS
            page_size_bytes = 8
            mamba_cache_mode = "align"
            tokens_per_state = 1
            num_speculative_blocks = 0
            num_prefill_checkpoint_blocks = 1

        recurrent_layout = types.SimpleNamespace(
            kv_cache_groups=(
                types.SimpleNamespace(
                    kv_cache_spec=MambaSpec(),
                    is_eagle_group=False,
                    layer_names=("recurrent",),
                ),
            )
        )
        with self.assertRaisesRegex(ProfileError, "no 'replay'"):
            profile.validate_for_deployment(
                dcp_degree=1,
                block_size=CHUNK_TOKENS,
                min_span_tokens=CHUNK_TOKENS,
                cuda_restore=False,
                group_topology=cfg.kv_group_topology(recurrent_layout),
            )

    def test_cuda_restore_refused_and_connector_binds_profile(self) -> None:
        profile = resolve_profile("deepseek-v41-b12x-hma")
        with self.assertRaisesRegex(ProfileError, "SparkCache CUDA restore"):
            profile.validate_for_deployment(
                dcp_degree=1,
                block_size=CHUNK_TOKENS,
                min_span_tokens=CHUNK_TOKENS,
                cuda_restore=True,
            )
        with tempfile.TemporaryDirectory() as directory:
            connector = _v41_connector(Path(directory), deepseek_v41_layout())
            self.assertEqual(connector._profile.name, "deepseek-v41-b12x-hma")
            self.assertEqual(connector._prefill_replay_tokens, REPLAY_RESERVE)
            self.assertEqual(connector._prefix_state_alignment, 2)
            self.assertEqual(connector._block_size, CHUNK_TOKENS)

    def test_v41_entry_never_matches_deepseek_v4_entry(self) -> None:
        v41 = _v41_identity(deepseek_v41_layout())
        v4 = _v41_identity(
            _deepseek_tp4_hma_config(),
            **{"spark_cache_model_profile": "deepseek-v4-fp8-hma"},
        )
        self.assertNotEqual(v41.storage_key, v4.storage_key)
        self.assertNotEqual(
            v41.to_wire()["quantization_layout"],
            v4.to_wire()["quantization_layout"],
        )

    def test_replay_reserve_state_geometry_and_pages_fork_the_entry(self) -> None:
        original = _v41_identity(deepseek_v41_layout())
        narrower = deepseek_v41_layout()
        narrower.kv_cache_groups[9].kv_cache_spec._prefill_replay_tokens = 64
        self.assertNotEqual(original.storage_key, _v41_identity(narrower).storage_key)
        coarser = deepseek_v41_layout()
        coarser.kv_cache_groups[8].kv_cache_spec.tokens_per_state = 2
        coarser.kv_cache_groups[8].kv_cache_spec.page_size_bytes = 68 * 4
        self.assertNotEqual(original.storage_key, _v41_identity(coarser).storage_key)
        larger_pages = deepseek_v41_layout()
        larger_pages.kv_cache_groups[0].kv_cache_spec.page_size_bytes = 288 * 256
        self.assertNotEqual(
            original.storage_key, _v41_identity(larger_pages).storage_key
        )

    def test_crc32_fast_entry_never_matches_sha256_entry(self) -> None:
        sha256_entry = _v41_identity(
            deepseek_v41_layout(),
            spark_cache_publication_schema="tail-cow-v2",
            spark_cache_payload_integrity="sha256",
        )
        fast_entry = _v41_identity(
            deepseek_v41_layout(),
            spark_cache_publication_schema="tail-cow-v2",
        )
        self.assertEqual(
            sha256_entry.to_wire()["publication_schema"], "page-tail-cow-v2"
        )
        self.assertEqual(
            fast_entry.to_wire()["publication_schema"], "page-tail-cow-v3-fast"
        )
        self.assertNotEqual(sha256_entry.storage_key, fast_entry.storage_key)



class DeepSeekV41SpanTests(unittest.TestCase):
    """Span selection leaves the replay reserve and stays state-aligned."""

    def _span_connector(
        self,
    ) -> tuple[SparkContextCacheConnector, tempfile.TemporaryDirectory]:
        lifetime = tempfile.TemporaryDirectory()
        connector = _v41_connector(Path(lifetime.name), deepseek_v41_layout())
        return connector, lifetime

    def test_aligned_span_property_over_prompt_lengths(self) -> None:
        connector, lifetime = self._span_connector()
        try:
            aligned = SparkContextCacheConnector._aligned_span
            for prompt_len in range(1, 2_000_001):
                span = aligned(connector, prompt_len)
                self.assertLessEqual(span, max(0, prompt_len - REPLAY_RESERVE))
                self.assertEqual(span % CHUNK_TOKENS, 0)
                self.assertEqual(span % 2, 0)
                if prompt_len < CHUNK_TOKENS + REPLAY_RESERVE:
                    self.assertEqual(span, 0)
                else:
                    self.assertGreater(span, 0)
        finally:
            lifetime.cleanup()

    def test_aligned_span_edges_around_the_reserve(self) -> None:
        connector, lifetime = self._span_connector()
        try:
            aligned = SparkContextCacheConnector._aligned_span
            self.assertEqual(aligned(connector, 1), 0)
            self.assertEqual(aligned(connector, REPLAY_RESERVE), 0)
            self.assertEqual(aligned(connector, REPLAY_RESERVE + 1), 0)
            self.assertEqual(
                aligned(connector, CHUNK_TOKENS + REPLAY_RESERVE), CHUNK_TOKENS
            )
            # A final chunk shorter than the replay reserve still recomputes
            # whole chunks: the span stops at the last chunk boundary that
            # leaves the full reserve.
            self.assertEqual(
                aligned(connector, CHUNK_TOKENS * 10 + 127), CHUNK_TOKENS * 9
            )
            self.assertEqual(
                aligned(connector, CHUNK_TOKENS * 10 + 128), CHUNK_TOKENS * 10
            )
            self.assertEqual(
                aligned(connector, CHUNK_TOKENS * 10 + 129), CHUNK_TOKENS * 10
            )
            self.assertEqual(aligned(connector, CHUNK_TOKENS * 11), CHUNK_TOKENS * 10)
        finally:
            lifetime.cleanup()


class DeepSeekV41StoreRestoreTests(unittest.TestCase):
    """Stores and restores publish only persisted groups' pages."""

    BASE_SPAN = CHUNK_TOKENS * 4
    RESULT_SPAN = CHUNK_TOKENS * 9

    def test_selection_returns_empty_non_persisted_selections(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connector = _v41_connector(
                Path(directory), deepseek_v41_layout(small_pages=True)
            )
            tables, _ = span_tables(self.RESULT_SPAN)
            selected = connector._select_group_blocks_for_span(
                tables, self.RESULT_SPAN
            )
            self.assertEqual(
                connector._group_block_counts_for_span(self.RESULT_SPAN),
                (1,) * 7 + (9,) + (0,) * 9,
            )
            self.assertEqual(selected[4], (tables[4][17],))
            self.assertEqual(selected[8:], ((),) * 9)

    def test_dsv41_hw_1_incremental_save_and_restore_across_roles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            layout = deepseek_v41_layout(small_pages=True)
            scheduler = _v41_connector(root, layout, role=KVConnectorRole.SCHEDULER)
            connector = _v41_connector(root, deepseek_v41_layout(small_pages=True, worker=True))
            self.addCleanup(scheduler.shutdown)
            self.addCleanup(connector.shutdown)
            pools: dict[str, torch.Tensor] = {}
            generator = torch.Generator().manual_seed(20261015)
            for group in deepseek_v41_layout(small_pages=True, worker=True).kv_cache_groups:
                for name, spec in group.kv_cache_spec.kv_cache_specs.items():
                    width = spec.page_size_bytes
                    pools[name] = torch.randint(
                        0, 256, (512, 1, width), dtype=torch.uint8,
                        generator=generator,
                    )
            connector.register_kv_caches(pools)

            tokens = tuple(range(self.RESULT_SPAN + REPLAY_RESERVE))
            base_digest = scheduler._digest(list(tokens), self.BASE_SPAN)
            result_digest = scheduler._digest(list(tokens), self.RESULT_SPAN)
            base_tables, _ = span_tables(self.BASE_SPAN)
            result_tables, _ = span_tables(self.RESULT_SPAN)

            connector._store_one(
                _ReqPlan(
                    "v41-base-store",
                    base_digest,
                    self.BASE_SPAN,
                    base_tables[0],
                    True,
                    block_ids_by_group=base_tables,
                    token_ids=tokens[: self.BASE_SPAN],
                    request_scope=UNSALTED_SCOPE,
                )
            )
            extension_plan = _ReqPlan(
                "v41-extension-store",
                result_digest,
                self.RESULT_SPAN,
                result_tables[0],
                True,
                block_ids_by_group=result_tables,
                token_ids=tokens,
                base_context_digest=base_digest,
                base_span_tokens=self.BASE_SPAN,
                request_scope=UNSALTED_SCOPE,
            )
            connector._store_one(extension_plan)

            lookup = connector._store.lookup(connector._identity(0), result_digest)
            self.assertTrue(lookup.is_hit, lookup.reason)
            self.assertEqual(lookup.root_kind, "page_delta")
            manifest = lookup._manifest
            assert manifest is not None
            self.assertEqual(
                manifest["base_block_counts"], [1] * 7 + [4] + [0] * 9
            )
            self.assertEqual(
                manifest["result_block_counts"], [1] * 7 + [9] + [0] * 9
            )

            destination = tuple(
                tuple(range(100 + 20 * index, 100 + 20 * index + len(table)))
                for index, table in enumerate(result_tables)
            )
            for index, table in enumerate(destination):
                for block in table:
                    for name in layout.kv_cache_groups[index].layer_names:
                        pools[name][block].zero_()
            self.assertTrue(
                connector._load_one(
                    _ReqPlan(
                        "v41-restore",
                        result_digest,
                        self.RESULT_SPAN,
                        destination[0],
                        False,
                        block_ids_by_group=destination,
                        request_scope=UNSALTED_SCOPE,
                    )
                )
            )
            expected = connector._select_group_blocks_for_span(
                result_tables, self.RESULT_SPAN
            )
            destination_selected = connector._select_group_blocks_for_span(
                destination, self.RESULT_SPAN
            )
            for index, group in enumerate(layout.kv_cache_groups):
                for source_block, destination_block in zip(
                    expected[index], destination_selected[index], strict=True
                ):
                    for name in group.layer_names:
                        self.assertTrue(
                            torch.equal(
                                pools[name][destination_block],
                                pools[name][source_block],
                            ),
                            f"group {index} layer {name} restored bytes differ",
                        )
            # Non-persisted groups contributed zero pages: their destination
            # blocks stay zeroed and their source blocks stay untouched.
            for index in range(8, 17):
                for block in destination[index]:
                    for name in layout.kv_cache_groups[index].layer_names:
                        self.assertTrue(
                            bool((pools[name][block] == 0).all()),
                            f"group {index} layer {name} was written on restore",
                        )


class DeepSeekV41LoadFailureTests(unittest.TestCase):
    """A failed full-group load recomputes within the replay cap."""

    SPAN = CHUNK_TOKENS * 9

    def _stored_connector(
        self, root: Path
    ) -> tuple[SparkContextCacheConnector, dict[str, torch.Tensor],
               tuple[tuple[int, ...], ...], str]:
        connector = _v41_connector(root, deepseek_v41_layout(small_pages=True, worker=True))
        pools: dict[str, torch.Tensor] = {}
        generator = torch.Generator().manual_seed(20261016)
        for group in deepseek_v41_layout(small_pages=True, worker=True).kv_cache_groups:
            for name, spec in group.kv_cache_spec.kv_cache_specs.items():
                width = spec.page_size_bytes
                pools[name] = torch.randint(
                    0, 256, (512, 1, width), dtype=torch.uint8,
                    generator=generator,
                )
        connector.register_kv_caches(pools)
        tables, _ = span_tables(self.SPAN)
        tokens = tuple(range(self.SPAN + REPLAY_RESERVE))
        digest = connector._digest(list(tokens), self.SPAN)
        connector._store_one(
            _ReqPlan(
                "v41-failure-store",
                digest,
                self.SPAN,
                tables[0],
                True,
                block_ids_by_group=tables,
                token_ids=tokens,
                request_scope=UNSALTED_SCOPE,
            )
        )
        return connector, pools, tables, digest

    def test_invalid_full_block_failure_recomputes_within_replay_cap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            connector, _pools, tables, digest = self._stored_connector(root)
            lookup = connector._store.lookup(connector._identity(0), digest)
            self.assertTrue(lookup.is_hit, lookup.reason)

            chunks = sorted(root.rglob("*.spcc"))
            self.assertTrue(chunks)
            payload = bytearray(chunks[0].read_bytes())
            payload[len(payload) // 2] ^= 0x40
            chunks[0].write_bytes(payload)

            self.assertFalse(
                connector._load_one(
                    _ReqPlan(
                        "v41-failure-restore",
                        digest,
                        self.SPAN,
                        tables[0],
                        False,
                        block_ids_by_group=tables,
                        request_scope=UNSALTED_SCOPE,
                    )
                )
            )
            self.assertFalse(
                connector._store.lookup(connector._identity(0), digest).is_hit,
                "a failed load must invalidate its entry",
            )
            # The retry boundary is the connector's own offer for the same
            # prompt. It must stay below the replay reserve, so the reset
            # request recomputes the request-private suffix itself.
            aligned = SparkContextCacheConnector._aligned_span
            prompt_len = self.SPAN + REPLAY_RESERVE
            for length in (prompt_len, self.SPAN, self.SPAN - 1):
                self.assertLessEqual(
                    aligned(connector, length), length - REPLAY_RESERVE
                )
            retry_span = aligned(connector, prompt_len)
            retry_tables, _ = span_tables(retry_span)
            retry_selection = connector._select_group_blocks_for_span(
                retry_tables, retry_span
            )
            self.assertEqual(retry_selection[8:], ((),) * 9)



@pytest.mark.parametrize("worker", [False, True])
def test_dsv41_hw_1_alignment_uses_group_lcm_and_chunk_quantum(worker):
    from sparkcache import spark_context_cache_config as cfg

    layout = deepseek_v41_layout(worker=worker)
    group = layout.kv_cache_groups[0]
    specs = (group.kv_cache_spec.kv_cache_specs.values() if worker
             else (group.kv_cache_spec,))
    for spec in specs:
        spec.block_size = 192
    block_size = cfg.cache_group_block_size(layout, 128 if not worker else 256)
    assert block_size == 768
    connector = types.SimpleNamespace(
        _block_size=block_size, _prefix_state_alignment=5, _chunk_tokens=256,
        _storage_mode="block_pages_v1", _dcp_degree=1, _prefill_replay_tokens=128,
        _group_topology=(),
    )
    assert SparkContextCacheConnector._aligned_span(connector, 10000) == 7680


@pytest.mark.parametrize("value", [0, -1, True, None])
def test_dsv41_hw_1_invalid_group_block_size_is_refused(value):
    from sparkcache import spark_context_cache_config as cfg

    layout = deepseek_v41_layout()
    layout.kv_cache_groups[7].kv_cache_spec.block_size = value
    with pytest.raises(RuntimeError, match="block size must be positive"):
        cfg.cache_group_block_size(layout, 256)


if __name__ == "__main__":
    unittest.main()
