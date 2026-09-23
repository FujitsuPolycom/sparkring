# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU behavior of exact source methods, without importing GPU-only vLLM modules.

The fixture models allocation references; it does not execute the full scheduler,
native CUDA copies, or the packed recurrent checkpoint allocator.
"""

from __future__ import annotations

import ast
from abc import ABC, abstractmethod
import logging
import os
import sys
import time
from collections import deque
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS

import pytest

ROOT = Path(os.environ['SPARKRING_TEST_SOURCE_ROOT'])


@pytest.fixture(autouse=True)
def current_vmm_capability_api(monkeypatch):
    """Load the selected source's extracted capability API without GPU imports."""
    path = ROOT / "vllm/distributed/kv_transfer/kv_connector/v1/base.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body
             if isinstance(node, (ast.ClassDef, ast.FunctionDef))
             and node.name in {"SupportsVmmSafeTransfers", "supports_vmm_safe_transfers"}]
    if not nodes:
        return
    namespace = {"ABC": ABC, "abstractmethod": abstractmethod}
    module = ast.Module(body=[ast.ImportFrom(module="__future__",
        names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    api = ModuleType("vllm.distributed.kv_transfer.kv_connector.v1")
    api.supports_vmm_safe_transfers = namespace["supports_vmm_safe_transfers"]
    monkeypatch.setitem(sys.modules, api.__name__, api)
    factory = ModuleType("vllm.distributed.kv_transfer.kv_connector.factory")
    factory.KVConnectorFactory = NS(get_connector_class=lambda _: NS())
    monkeypatch.setitem(sys.modules, factory.__name__, factory)


class Status:
    RUNNING = "running"
    WAITING = "waiting"
    WAITING_FOR_REMOTE_KVS = "receiving"
    PREEMPTED = "preempted"


class Request(NS):
    __hash__ = object.__hash__


def source_class(relative, name, methods, **namespace):
    path = ROOT / relative
    tree = ast.parse(path.read_text(encoding="utf-8"))
    original = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name
    )
    # Include helpers extracted upstream from the methods under test; execute
    # their real selected-source implementations rather than mocking behavior.
    dependencies = {'VllmConfig': {'_connector_supports_vmm_safe_transfers'},
                    'Scheduler': {'_request_blocks_can_be_freed'},
                    'ActiveKVConnector': {'_start_load_kv', 'finish_forward'}}
    available = {n.name for n in original.body if isinstance(n, ast.FunctionDef)}
    methods = methods | (dependencies.get(name, set()) & available)
    selected = [
        n for n in original.body if isinstance(n, ast.FunctionDef) and n.name in methods
    ]
    assert {n.name for n in selected} == set(methods)
    for node in selected:
        node.decorator_list = []
    cls = ast.ClassDef(
        name=name, bases=[], keywords=[], body=selected, decorator_list=[]
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            cls,
        ],
        type_ignores=[],
    )
    env = {
        "RequestStatus": Status,
        "logger": logging.getLogger(__name__),
        "time": time,
        **namespace,
    }
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), env)
    return env[name]


def scheduler_case(dcp=1, *, asynchronous=True, policy="recompute", single=False):
    methods = {
        "_handle_invalid_blocks",
        "_update_requests_with_invalid_blocks",
        "_update_waiting_for_remote_kv",
        "_preempt_request",
        "_free_request_blocks",
        "_drain_deferred_frees",
    }
    scheduler = source_class("vllm/v1/core/sched/scheduler.py", "Scheduler", methods)()
    scheduler._kda_coalescing_enabled = False
    scheduler._kda_coalescing_origins = {}
    span = 8192
    sizes = [512 * dcp] if single else [512 * dcp, 512]
    groups = [list(range(1, span // sizes[0] + 1))]
    if not single:
        groups.append([0] * 14 + [40, 41, 42, 43])
    tables = {"request": tuple(groups)}
    owned = {bid for group in groups for bid in group if bid}
    refs = {bid: 1 for bid in owned}
    events = []

    def release(ids):
        for bid in ids:
            refs[bid] -= 1
            assert refs[bid] >= 0
        events.append("release")

    def pop(request):
        return [bid for group in tables.pop(request.request_id) for bid in group if bid]

    cache = NS(
        num_kv_cache_groups=len(groups),
        coordinator=NS(single_type_managers=[NS(block_size=size) for size in sizes]),
        block_pool=NS(null_block=NS(block_id=0), free_blocks=release),
        get_block_ids=lambda rid: tables.get(rid, tuple([] for _ in groups)),
        evict_blocks=lambda ids: events.append(("evict", set(ids))),
        free=lambda request: release(pop(request)),
        pop_blocks_for_free=pop,
        cache_blocks=lambda *args: events.append("cache"),
    )
    request = Request(
        request_id="request",
        num_computed_tokens=span,
        num_tokens=span,
        status=Status.WAITING_FOR_REMOTE_KVS if asynchronous else Status.RUNNING,
        last_sched_seq=1,
        spec_token_ids=[],
        drop_stale_output=False,
        num_stale_output_tokens=0,
        num_in_flight_tokens=4,
        num_output_placeholders=1,
        num_preemptions=0,
    )
    scheduler.kv_cache_manager = cache
    scheduler.block_size = sizes[0]
    scheduler.recompute_kv_load_failures = policy == "recompute"
    scheduler.skipped_waiting = [request] if asynchronous else []
    scheduler.running = [] if asynchronous else [request]
    scheduler.waiting = NS(
        prepend_request=lambda req: events.append(("preempt", req.request_id))
    )
    scheduler.failed_recving_kv_req_ids = set()
    scheduler.finished_recving_kv_req_ids = set()
    scheduler.connector = object()
    scheduler.needs_kv_cache_zeroing = False
    scheduler.encoder_cache_manager = NS(free=lambda req: None)
    scheduler._inflight_prefills = set()
    scheduler.reset_preempted_req_ids = set()
    scheduler.defer_block_free = True
    scheduler.processed_step_seq = 0
    scheduler.sched_step_seq = 1
    scheduler.deferred_frees = deque()
    scheduler.log_stats = False
    return scheduler, request, groups, refs, events, tables


@pytest.mark.parametrize("dcp", [1, 2, 4])
@pytest.mark.parametrize("group", [0, 1])
def test_failed_hybrid_group_invalidates_dependencies_and_waits_for_receive(dcp, group):
    scheduler, request, groups, refs, events, _ = scheduler_case(dcp)
    failed = groups[group][0 if group == 0 else 15]
    assert scheduler._handle_invalid_blocks({failed}, {}) == set()
    assert request.num_computed_tokens == 0
    assert scheduler.failed_recving_kv_req_ids == {request.request_id}
    assert events == [("evict", set(refs))]
    assert set(refs.values()) == {1}
    scheduler.finished_recving_kv_req_ids.add(request.request_id)
    scheduler._update_waiting_for_remote_kv(request)
    assert "cache" not in events
    assert set(refs.values()) == {0}


@pytest.mark.parametrize("dcp", [1, 2, 4])
def test_padding_and_future_recurrent_scratch_do_not_fail_a_restore(dcp):
    scheduler, request, _, refs, events, _ = scheduler_case(dcp)
    assert scheduler._handle_invalid_blocks({0, 42, 43, 999}, {}) == set()
    assert request.num_computed_tokens == 8192
    assert set(refs.values()) == {1}
    assert events == []


@pytest.mark.parametrize("dcp", [1, 2, 4])
def test_effective_attention_grid_excludes_current_scheduled_suffix(dcp):
    scheduler, request, groups, _, _, _ = scheduler_case(dcp)
    frontier = 4096
    n = frontier // (512 * dcp)
    scheduled = {request.request_id: 8192 - frontier}
    assert scheduler._update_requests_with_invalid_blocks(
        [request], {groups[0][n]}, scheduled
    ) == (set(), 0, set())
    failed, recomputed, _ = scheduler._update_requests_with_invalid_blocks(
        [request], {groups[0][n - 1]}, scheduled
    )
    assert failed == {request.request_id} and recomputed == frontier
    assert request.num_computed_tokens == 0


@pytest.mark.parametrize("coalescing", [False, True])
def test_sync_hybrid_failure_uses_real_preemption_and_deferred_free_methods(coalescing):
    scheduler, request, _, refs, events, _ = scheduler_case(asynchronous=False)
    scheduler._kda_coalescing_enabled = coalescing
    if coalescing:
        scheduler._kda_coalescing_origins[request.request_id] = 0
    assert scheduler._handle_invalid_blocks({40}, {request.request_id: 4}) == {
        request.request_id
    }
    assert request.status == Status.PREEMPTED
    assert request.drop_stale_output and request.num_stale_output_tokens == 4
    assert request.num_output_placeholders == request.num_computed_tokens == 0
    assert scheduler.running == []
    assert request.request_id not in scheduler._kda_coalescing_origins
    assert ("preempt", request.request_id) in events
    scheduler._drain_deferred_frees()
    assert set(refs.values()) == {1}
    scheduler.processed_step_seq = 1
    scheduler._drain_deferred_frees()
    assert set(refs.values()) == {0}


@pytest.mark.parametrize("asynchronous", [True, False])
def test_fail_policy_invalidates_identity_without_releasing_owned_pages(asynchronous):
    scheduler, request, _, refs, events, _ = scheduler_case(
        asynchronous=asynchronous, policy="fail"
    )
    assert scheduler._handle_invalid_blocks({40}, {}) == {request.request_id}
    assert events == [("evict", set(refs))]
    assert set(refs.values()) == {1}


def test_unrelated_hybrid_request_is_not_reset():
    scheduler, request, _, _, _, tables = scheduler_case()
    other = Request(
        request_id="other",
        num_computed_tokens=8192,
        status=Status.WAITING_FOR_REMOTE_KVS,
    )
    tables["other"] = ([101] * 16, [0] * 15 + [102, 103, 104])
    scheduler.skipped_waiting.append(other)
    scheduler._handle_invalid_blocks({40}, {})
    assert request.num_computed_tokens == 0 and other.num_computed_tokens == 8192
    assert scheduler.failed_recving_kv_req_ids == {request.request_id}


def test_single_attention_group_keeps_its_proven_prefix():
    scheduler, request, groups, _, events, _ = scheduler_case(single=True)
    assert scheduler._handle_invalid_blocks({groups[0][3]}, {}) == set()
    assert request.num_computed_tokens == 1536
    assert not events


@pytest.mark.parametrize(
    "capability,external,allowed",
    [(True, True, True), (False, True, False), (1, True, False), (True, False, False)],
)
def test_vmm_opt_in_requires_external_connector_and_literal_capability(
    monkeypatch, capability, external, allowed
):
    config_class = source_class(
        "vllm/config/vllm.py", "VllmConfig", {"_verify_kv_transfer_compat"}, os=os
    )
    cfg = config_class()
    cfg.model_config = NS(enable_cumem_allocator=False)
    cfg.kv_transfer_config = NS(
        kv_connector="SparkContextCacheConnector",
        kv_connector_extra_config={},
        kv_connector_module_path="sparkcache.spark_context_cache_connector"
        if external
        else None,
    )
    module = ModuleType("vllm.distributed.kv_transfer.kv_connector.factory")
    module.KVConnectorFactory = NS(
        get_connector_class=lambda _: NS(supports_cuda_vmm=capability)
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if allowed:
        cfg._verify_kv_transfer_compat()
    else:
        with pytest.raises(ValueError, match="incompatible"):
            cfg._verify_kv_transfer_compat()


@pytest.mark.parametrize(
    "connector", ["LMCacheMPConnector", "LMCacheRecurrentCheckpointConnector"]
)
def test_engine_driven_lmcache_remains_accepted(monkeypatch, connector):
    config_class = source_class(
        "vllm/config/vllm.py", "VllmConfig", {"_verify_kv_transfer_compat"}, os=os
    )
    cfg = config_class()
    cfg.model_config = NS(enable_cumem_allocator=False)
    cfg.kv_transfer_config = NS(
        kv_connector=connector,
        kv_connector_extra_config={"lmcache.mp.mp_transfer_mode": "engine_driven"},
    )
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cfg._verify_kv_transfer_compat()


def test_aligned_connector_cannot_claim_atomic_request_boundary_adapter():
    base = source_class(
        "vllm/distributed/kv_transfer/kv_connector/v1/base.py",
        "KVConnectorBase_V1",
        {"supports_request_boundary_checkpoints"},
    )
    assert base().supports_request_boundary_checkpoints(None) is False


def test_actual_worker_idle_callback_does_not_invoke_save():
    calls = []
    output_cls = NS
    connector_cls = source_class(
        "vllm/v1/worker/gpu/kv_connector.py",
        "ActiveKVConnector",
        {"pre_forward", "post_forward", "no_forward"},
        KVConnectorOutput=output_cls,
        ModelRunnerOutput=NS(with_kv_conn_output_only=lambda x: x),
        is_forward_context_available=lambda: True,
        get_forward_context=lambda: None,
    )
    worker = connector_cls()
    worker._disabled = False
    worker._pending_load_kwargs = None
    worker.kv_connector = NS(
        handle_preemptions=lambda _: calls.append("preempt"),
        bind_connector_metadata=lambda _: calls.append("bind"),
        start_load_kv=lambda _: calls.append("load"),
        wait_for_save=lambda: calls.append("save"),
        get_finished=lambda _: (set(), set()),
        get_block_ids_with_load_errors=lambda: set(),
        get_kv_connector_stats=lambda: None,
        get_kv_connector_kv_cache_events=lambda: None,
        build_connector_worker_meta=lambda: calls.append("metadata"),
        clear_connector_metadata=lambda: calls.append("clear"),
        finish_forward=lambda: calls.append('finish'),
        get_transfer_results=lambda _: NS(finished_sending=set(), finished_recving=set(), failed_recving=set()),
    )
    worker.post_forward(set())
    assert calls == ["save", "metadata", "clear"]
    calls.clear()
    worker.no_forward(NS(kv_connector_metadata=object(), finished_req_ids=set(), has_sync_kv_loads=False))
    expected = ['preempt', 'bind']
    if hasattr(worker, 'finish_forward'):
        expected.append('finish')
    assert calls == [*expected, "load", "metadata", "clear"]
