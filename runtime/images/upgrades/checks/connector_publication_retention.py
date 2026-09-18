"""Replay connector publication hints through the selected producer's CPU methods."""

import ast
from collections import Counter
from collections.abc import Mapping
import hashlib
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


def source():
    return Path(os.environ["SPARKRING_TEST_SOURCE_ROOT"]) / "vllm"


def functions(path, names, env, owner=None):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    items = (
        tree.body
        if owner is None
        else next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner
        ).body
    )
    nodes = [n for n in items if isinstance(n, ast.FunctionDef) and n.name in names]
    for node in nodes:
        node.decorator_list = []
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future, *nodes], type_ignores=[])
            ),
            str(path),
            "exec",
        ),
        env,
    )


def seam(hints):
    root = source()
    spec = type(
        "MambaSpec", (), {"block_size": 1440, "num_prefill_checkpoint_blocks": 1}
    )
    env = dict(
        MambaSpec=spec,
        SlidingWindowSpec=type("SWA", (), {}),
        Mapping=Mapping,
        cdiv=lambda x, y: (x + y - 1) // y,
        logger=NS(warning=lambda *args: None),
    )
    functions(
        root / "v1/kv_cache_interface.py",
        {"get_mamba_prefill_checkpoint_position", "is_mamba_prefill_checkpoint_valid"},
        env,
    )
    single = root / "v1/core/single_type_kv_cache_manager.py"
    functions(
        single, {"reachable_hit_positions", "resolve_sparse_retention_inputs"}, env
    )
    functions(single, {"_expand_reachable_boundaries"}, env, "SingleTypeKVCacheManager")
    functions(single, {"reachable_block_mask"}, env, "MambaManager")
    functions(
        root / "v1/core/kv_cache_coordinator.py",
        {"get_replay_boundaries", "get_recurrent_publication_boundaries"},
        env,
        "KVCacheCoordinator",
    )
    functions(
        root / "v1/core/sched/scheduler.py",
        {"_mamba_block_aligned_split"},
        env,
        "Scheduler",
    )
    manager = NS(
        kv_cache_spec=spec(),
        block_size=1440,
        use_eagle=True,
        lookup_drops_eagle_block=True,
        cache_hit_alignment_tokens=1440,
        scheduler_block_size=1440,
    )
    manager._expand_reachable_boundaries = lambda b: env[
        "_expand_reachable_boundaries"
    ](manager, b)
    coordinator = NS(
        enable_partial_hash_hits=False,
        eagle_group_ids={0, 1},
        scheduler_block_size=1440,
        prefill_replay_tokens=0,
        single_type_managers=[manager],
        recurrent_publication_boundary_provider=hints,
    )
    coordinator.get_recurrent_publication_boundaries = lambda r: env.get(
        "get_recurrent_publication_boundaries", lambda *_: ()
    )(coordinator, r)
    coordinator.get_replay_boundaries = lambda r: env["get_replay_boundaries"](
        coordinator, r
    )
    scheduler = NS(
        _kda_coalescing_enabled=False,
        cache_config=NS(block_size=1440, prefix_cache_retention_interval=0),
        use_eagle_block_drop=True,
        hash_block_size=1440,
        mamba_has_prefill_checkpoint_blocks=True,
        mamba_prefill_checkpoint_alignment=16,
        max_num_scheduled_tokens=8192,
        scheduler_config=NS(long_prefill_token_threshold=0),
        mamba_partial_cache_hit=False,
        mamba_fine_grained_prefix_cache=False,
        kv_cache_manager=NS(coordinator=coordinator),
    )
    return env, coordinator, scheduler, manager


def request(count=7870, computed=0):
    return NS(
        request_id="r",
        num_prompt_tokens=count,
        num_tokens=count,
        num_computed_tokens=computed,
        use_boundary_checkpoints=False,
        shared_prefix_boundary=0,
    )


@pytest.mark.parametrize("count,target", [(7870, 7200), (8194, 7200), (32766, 31680)])
@pytest.mark.parametrize("tp", [2, 4])
def test_publication_boundary_is_materialized_retained_and_offered(count, target, tp):
    env, coordinator, scheduler, manager = seam(lambda r: (target,))
    req = request(count)
    retained = coordinator.get_replay_boundaries(req)
    assert target - 1440 in retained  # Native Eagle replay boundary is unchanged.
    assert target in retained
    reached = []
    while req.num_computed_tokens < count:
        step = env["_mamba_block_aligned_split"](
            scheduler, req, min(8192, count - req.num_computed_tokens)
        )
        assert step > 0
        req.num_computed_tokens += step
        reached.append(req.num_computed_tokens)
    assert target in reached
    mask = env["reachable_block_mask"](
        object, 0, count // 1440, 1440, manager.kv_cache_spec, False, 0, retained
    )
    assert mask[target // 1440 - 1]
    for rank in range(tp):
        blocks = [
            NS(
                block_id=rank * 100 + i + 1,
                is_null=(i + 1) * 1440 not in reached,
                block_hash=None,
                block_hash_num_tokens=None,
            )
            for i in range(count // 1440)
        ]
        # Use the actual hash-install and exact-block offload methods below.
        offers = committed_offers(env, blocks, mask, req)
        assert (0, blocks[target // 1440 - 1].block_id, target) in offers
        assert consumer_accepts(env, offers, target)


def committed_offers(
    env, blocks, mask, req, committed_tokens=None, cached_before=0, checkpoint=None
):
    root = source()
    functions(root / "v1/core/block_pool.py", {"cache_full_blocks"}, env, "BlockPool")
    env.update(
        resolve_block_hashes=lambda hashes, *_: hashes,
        make_block_hash_with_group_id=lambda h, g: (h, g),
    )

    def insert(key, block, num_tokens):
        block.block_hash = key
        block.block_hash_num_tokens = num_tokens

    pool = NS(
        hash_block_size=1440, enable_kv_cache_events=False, _insert_block_hash=insert
    )

    def evict(block):
        block.block_hash = None
        block.block_hash_num_tokens = None

    pool._maybe_evict_cached_block = evict
    req.block_hashes = list(range(len(blocks)))
    committed_tokens = (
        req.num_prompt_tokens if committed_tokens is None else committed_tokens
    )
    full_blocks = min(len(blocks), committed_tokens // 1440)
    env["cache_full_blocks"](
        pool,
        req,
        blocks,
        cached_before,
        full_blocks,
        1440,
        0,
        mask[cached_before:full_blocks],
    )
    # Execute the production Mamba handoff loop with its parent hash pass
    # supplied by the real BlockPool call above; no tensor values are inferred.
    tree = ast.parse((root / "v1/core/single_type_kv_cache_manager.py").read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MambaManager"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "cache_blocks"
    )
    partial = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_cache_partial_tail_block"
    )

    class Parent:
        def cache_blocks(self, request, *args, **kwargs):
            self.num_cached_block[request.request_id] = full_blocks

    env["Parent"] = Parent
    node = ast.ClassDef(
        name="SelectedManager",
        bases=[ast.Name(id="Parent", ctx=ast.Load())],
        keywords=[],
        body=[method, partial],
        decorator_list=[],
    )
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[])),
            "selected Mamba handoff",
            "exec",
        ),
        env,
    )
    obj = env["SelectedManager"]()
    obj.num_cached_block = {req.request_id: cached_before}
    obj.req_to_blocks = {"r": blocks}
    obj.mamba_cache_mode = "align"
    obj.cached_blocks_this_step = set()
    obj._pending_boundary_state_offloads = []
    obj.kv_cache_group_id = 0
    obj.block_pool = pool
    obj.block_size = 1440
    obj._checkpoints = {} if checkpoint is None else {req.request_id: checkpoint}
    obj.cache_blocks(req, committed_tokens, retention_interval=0)
    return [
        (group, block.block_id, boundary)
        for _, group, block, boundary in obj._pending_boundary_state_offloads
    ]


def consumer_accepts(env, offers, target, required=(0,)):
    configured = os.environ.get("SPARKRING_SPARKCACHE_SOURCE_ROOT")
    if configured:
        path = Path(configured) / "sparkcache/spark_context_cache_connector.py"
    else:
        module = importlib.util.find_spec("sparkcache")
        assert module and module.origin, "Exact installed SparkCache source is required"
        path = Path(module.origin).with_name("spark_context_cache_connector.py")
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        == "117c3dbae43936986a3061e06a17fb2701aaeda702fcab8d28f6df5ab5f018fe"
    )
    functions(
        path,
        {"_validated_recurrent_boundary_blocks"},
        env,
        "SparkContextCacheConnector",
    )
    connector = NS(
        _recurrent_group_indexes=set(required),
        _group_topology=[{"reuse_policy": "recurrent_align"}] * (max(required) + 1),
        _streaming_snapshots_enabled=False,
        _async_page_capture_enabled=True,
        _uses_capture_job_leases=lambda: True,
        counters=Counter(),
    )
    output = NS(kv_connector_block_state=NS(boundary_state_offloads={"r": offers}))
    return bool(
        env["_validated_recurrent_boundary_blocks"](connector, output, "r", target)
    )


@pytest.mark.parametrize(
    "hints",
    [
        None,
        lambda r: (),
        lambda r: True,
        lambda r: (True,),
        lambda r: ("7200",),
        lambda r: (-1440,),
        lambda r: (0,),
        lambda r: (7201,),
        lambda r: (8640,),
        lambda r: (7870,),
        lambda r: tuple(range(100)),
        lambda r: iter((7200,)),
    ],
)
def test_absent_or_invalid_provider_preserves_native_retention(hints):
    _, coordinator, _, _ = seam(hints)
    assert coordinator.get_replay_boundaries(request()) == (5760,)


def test_provider_exception_and_withdrawal_do_not_latch_hints():
    def fail(req):
        raise RuntimeError("optional publication unavailable")

    _, coordinator, _, _ = seam(fail)
    assert coordinator.get_replay_boundaries(request()) == (5760,)
    coordinator.recurrent_publication_boundary_provider = lambda r: (7200,)
    assert 7200 in coordinator.get_replay_boundaries(request())
    coordinator.recurrent_publication_boundary_provider = lambda r: ()
    assert coordinator.get_replay_boundaries(request()) == (5760,)


def test_late_hint_cannot_resurrect_a_previously_skipped_boundary():
    _, coordinator, _, _ = seam(lambda r: (7200,))
    assert coordinator.get_replay_boundaries(request(computed=7200)) == (5760,)


@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("eagle", [False, True])
def test_hint_merge_keeps_every_native_replay_branch(partial, eagle):
    _, coordinator, _, _ = seam(None)
    coordinator.enable_partial_hash_hits = partial
    coordinator.eagle_group_ids = {0} if eagle else set()
    coordinator._cache_hit_alignment_tokens = 1440
    native = coordinator.get_replay_boundaries(request())
    coordinator.recurrent_publication_boundary_provider = lambda r: (7200,)
    merged = coordinator.get_replay_boundaries(request())
    assert merged[: len(native)] == native
    assert 7200 in merged


@pytest.mark.parametrize("committed,missing", [(7199, False), (7200, True)])
def test_hint_does_not_offer_uncommitted_or_unreferenced_bytes(committed, missing):
    env, _, _, _ = seam(lambda r: (7200,))
    blocks = [
        NS(
            block_id=i + 1,
            is_null=(missing and i == 4),
            block_hash=None,
            block_hash_num_tokens=None,
        )
        for i in range(5)
    ]
    offers = committed_offers(env, blocks, [False] * 4 + [True], request(), committed)
    assert not consumer_accepts(env, offers, 7200)
    assert blocks[4].block_hash is None


def test_missing_group_or_wrong_boundary_cannot_be_published():
    env, _, _, _ = seam(None)
    assert not consumer_accepts(env, [(0, 10, 7200)], 7200, required=(0, 1))
    assert not consumer_accepts(env, [(0, 10, 5760)], 7200)
    assert consumer_accepts(env, [(0, 10, 7200), (1, 11, 7200)], 7200, required=(0, 1))


def test_scheduler_binds_metadata_provider_during_initialization():
    tree = ast.parse((source() / "v1/core/sched/scheduler.py").read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler"
    )
    init = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    assert any(
        isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Attribute)
            and t.attr == "recurrent_publication_boundary_provider"
            for t in n.targets
        )
        for n in ast.walk(init)
    )


@pytest.mark.parametrize("count,target", [(7870, 7200), (8194, 7200), (32766, 31680)])
def test_each_chunk_requeries_before_hashing_and_preserves_issued_boundary(
    count, target
):
    env, coordinator, scheduler, manager = seam(lambda r: (target,))
    req = request(count)
    blocks = [
        NS(block_id=i + 1, is_null=True, block_hash=None, block_hash_num_tokens=None)
        for i in range((count + 1439) // 1440)
    ]
    cached_before = 0
    issued = []
    while req.num_computed_tokens < count:
        start = req.num_computed_tokens
        end = start + env["_mamba_block_aligned_split"](
            scheduler, req, min(8192, count - start)
        )
        blocks[(end - 1) // 1440].is_null = False
        position = env["get_mamba_prefill_checkpoint_position"](end, 1440, True)
        valid = env["is_mamba_prefill_checkpoint_valid"](
            start, end, position, 1440, 1440, 16
        )
        checkpoint = None
        if valid:
            column = (end + 1439) // 1440 - 2
            blocks[column].is_null = False
            checkpoint = (position, column)
        retained = coordinator.get_replay_boundaries(req)
        mask = env["reachable_block_mask"](
            object, 0, len(blocks), 1440, manager.kv_cache_spec, False, 0, retained
        )
        offers = committed_offers(
            env, blocks, mask, req, end, cached_before, checkpoint
        )
        issued.extend(offers)
        cached_before = end // 1440
        if end == target:
            assert consumer_accepts(env, offers, target)
        req.num_computed_tokens = end
    assert target not in coordinator.get_recurrent_publication_boundaries(req)
    assert blocks[target // 1440 - 1].block_hash_num_tokens == target
    assert consumer_accepts(env, issued, target)
