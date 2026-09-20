"""Kraken cache-agreement source oracle v2; CPU metadata protocol only.

Runs source-owned protocol types, cache reconciliation, PreparationJob state
transitions, and vLLM authorization. Compilation, timing, installation, and
physical cache persistence are the explicit fixture seams. See the accompanying
contract document for source prerequisites and limits.
"""
from __future__ import annotations

import ast
from contextlib import nullcontext
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace as NS

import pytest


def source_root(variable):
    root = Path(os.environ[variable]).resolve(strict=True)
    assert root.is_dir()
    return root


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def execute_nodes(path, names, environment, methods=None):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = [node for node in tree.body if getattr(node, "name", None) in names]
    assert {node.name for node in selected} == set(names)
    if methods is not None:
        job = next(node for node in selected if node.name == "PreparationJob")
        job.body = [node for node in job.body if getattr(node, "name", None) in methods]
        assert {node.name for node in job.body} == methods
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[]))
    exec(compile(module, str(path), "exec"), environment)
    return environment


@pytest.fixture(scope="module")
def protocol():
    b12x = source_root("SPARKRING_B12X_SOURCE_ROOT") / "b12x/preparation"
    vllm = source_root("SPARKRING_VLLM_SOURCE_ROOT") / "vllm/v1/worker/b12x_startup.py"
    types = load_module("_kraken_cache_v2_types", b12x / "types.py")
    startup = load_module("_kraken_cache_v2_startup", vllm)
    env = dict(vars(types), json=json, hashlib=hashlib, Path=Path, time=time,
               nullcontext=nullcontext, _ADVANCE_SECONDS=1,
               _coalesce_requests=lambda requests: [(request,) for request in requests])
    execute_nodes(b12x / "_cache.py", {"_json", "digest", "SelectionCache"}, env)
    execute_nodes(b12x / "session.py", {"PreparationJob", "_TuningBatch"}, env, {
        "_choice_key", "_selection", "_lookup", "_consolidate", "_run", "_advance",
        "_warmup_only",
    })
    return NS(types=types, startup=startup, **{key: env[key] for key in (
        "PreparationJob", "SelectionCache", "_TuningBatch",
    )})


def record(width):
    return dict(assignment={"width": width}, config={"width": width},
                coverage=dict(cartesian_count=4, legal_count=4, effective_count=4,
                              measured_count=4), programs=[["cute", "local-object"]])


def make_job(protocol, tmp_path, rank, ranks, width=None, *, query=128, cache_only=False):
    types = protocol.types
    frozen = types.FrozenMapping
    job = protocol.PreparationJob()
    contract = NS(component_id="gemm.oracle", query_schema_version=1, config_schema_version=1,
                  semantic_version=1, candidate_contract_version=1, config_payload=frozen,
                  _lower=lambda query, device, assignment: dict(assignment))
    plan = NS(contract=contract, component_id=contract.component_id, invocation=frozen())
    request = NS(name="projection", dependencies=(), plan=plan, collective=None)
    comm = NS(name="collective", dependencies=(), plan=plan,
              collective=types.CollectiveRequirement("comm", ranks))
    configuration = NS(encoded_query=frozen({"tokens": query}), pinned=None,
                       space=NS(validate=lambda assignment: None), query=query, device=None,
                       default={"width": 1})
    race = NS(request=request, requests=(request,), configuration=configuration,
              selection=None, ready=False, candidates=list(range(4)), planned_candidates=0,
              coverage={})
    fixed = NS(request=comm, requests=(comm,), configuration=configuration,
               selection=types.Selection("comm", frozen(), {}, "fixed"), ready=False,
               candidates=[0], planned_candidates=0)
    identity = dict(schema_version=5, tuning_cache_version=1,
                    measurement="stream_gated_events_v1", namespace={"model": "oracle"},
                    device_name="NVIDIA GB10")
    cache = protocol.SelectionCache(tmp_path / str(rank), identity)
    stop = threading.Event()
    job.session = NS(state="PREPARING", cache_only=cache_only, _tuning_ranks=ranks,
                     _tuning_cache_synchronized=False, _selection_cache=lambda: cache,
                     _stop=stop, _pool=None, _check_thread=lambda: None)
    job.requests = (request, comm)
    job.autotune = True
    job._timing = NS(span=lambda *a: nullcontext(), record=lambda *a, **kw: None,
                     add=lambda *a: None)
    job._cache_hits = 0
    job._error = job._result = job._blocked = None
    job._closed = False
    job._phase = "initial"
    job._expand = lambda: (job.requests, {})
    job._progress = lambda gpu, compilation, ready, done, ready_tuning=(), **kw: NS(
        ready=ready, done=done, ready_tuning=ready_tuning, ready_cache=kw.get("ready_cache"))
    installed, races = [], []

    def configure(groups):
        yield from ()
        return [race, fixed]

    def race_candidates(obligation):
        races.append(rank)
        yield from ()
        return types.TuningRequirement(obligation.key, ranks, {"width": rank + 2},
                                       float(len(ranks) - rank), rank)

    def install(obligation, selections):
        if obligation.request.collective is not None:
            yield obligation.request.collective
        installed.append(obligation.selection)
        selections[obligation.request.name] = obligation.selection, contract

    job._configure, job._race, job._install_obligation = configure, race_candidates, install
    job._steps = job._run()
    job.close = job._steps.close
    key = job._choice_key(race, {})
    if width is not None:
        cache.records = cache._validate({key: record(width)})
    return NS(job=job, cache=cache, stop=stop, race=race, key=key,
              installed=installed, races=races)


def decision(protocol, ranks, progresses, *, cancelled=False):
    coordinator = protocol.startup.B12xPreparationCoordinator.__new__(
        protocol.startup.B12xPreparationCoordinator)
    coordinator.world_ranks, coordinator._round, coordinator._control = ranks, 0, None
    payloads = []
    for rank, progress in zip(ranks, progresses, strict=True):
        contributions = tuple((protocol.startup._scoped_key(item.key, item.ranks),
                               item.ranks, item.assignment, item.latency_us,
                               item.candidate_index) for item in progress.ready_tuning)
        payloads.append(dict(global_rank=rank, world_ranks=ranks, round=0, error=None,
                             stop=cancelled and rank == ranks[-1], local_done=False,
                             cleanup_complete=False, tuning=contributions,
                             cache=progress.ready_cache,
                             ready=tuple((item.key, item.ranks) for item in progress.ready)))
    return coordinator._decision(payloads)


def agree(protocol, ranks, jobs):
    progress = [item.job._advance() for item in jobs]
    assert all(isinstance(item.ready_cache, protocol.types.TuningCacheRequirement)
               for item in progress)
    assert all(not item.races and not item.installed for item in jobs)
    authorization = decision(protocol, ranks, progress)
    snapshots = authorization["caches"][ranks]
    assert snapshots == tuple(item.ready_cache for item in progress)
    next_progress = [item.job._advance(cache=snapshots) for item in jobs]
    assert all(item.job.session._tuning_cache_synchronized for item in jobs)
    assert all(item.cache._agreed_records == jobs[0].cache._agreed_records for item in jobs)
    return next_progress


def assert_equal_installed(jobs, source, width):
    assert all(len(item.installed) == 1 for item in jobs)
    assert {item.installed[0].source for item in jobs} == {source}
    assert {item.installed[0].config["width"] for item in jobs} == {width}


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("mode", ["cold", "warm", "mixed", "conflicting"])
def test_cache_agreement_precedes_rank_sharded_tuning(protocol, tmp_path, world, mode):
    ranks = tuple(range(world))
    widths = {"cold": [None] * world, "warm": [2] * world,
              "mixed": [None] * (world - 1) + [2],
              "conflicting": list(range(2, world + 2))}[mode]
    jobs = [make_job(protocol, tmp_path, rank, ranks, width)
            for rank, width in enumerate(widths)]
    try:
        progress = agree(protocol, ranks, jobs)
        # The reconciled candidate intentionally reraces distributed autotune.
        assert all(item.ready_tuning and not item.ready for item in progress)
        authorization = decision(protocol, ranks, progress)
        assert len(authorization["tuning"]) == 1
        key, participants, assignment, latency, index = authorization["tuning"][0]
        winner = protocol.types.TuningRequirement(
            protocol.startup._unscoped_key(key), participants, assignment, latency, index)
        ready = [item.job._advance(tuning=(winner,)) for item in jobs]
        assert all(item.ready[0].key == "comm" for item in ready)
        assert_equal_installed(jobs, "tuned", world + 1)
        assert all(item.races == [rank] for rank, item in enumerate(jobs))
    finally:
        for item in jobs:
            item.job.close()


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("mode", ["mixed", "conflicting"])
def test_cache_only_uses_same_rank_ordered_selection(protocol, tmp_path, world, mode):
    ranks = tuple(range(world))
    widths = ([None] * (world - 1) + [3] if mode == "mixed" else list(range(2, world + 2)))
    jobs = [make_job(protocol, tmp_path, rank, ranks, width, cache_only=True)
            for rank, width in enumerate(widths)]
    try:
        ready = agree(protocol, ranks, jobs)
        assert all(item.ready[0].key == "comm" for item in ready)
        assert_equal_installed(jobs, "cached", 3 if mode == "mixed" else 2)
        assert all(not item.races for item in jobs)
    finally:
        for item in jobs:
            item.job.close()


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("boundary", ["cache", "tuning"])
def test_one_rank_cancellation_discards_cache_or_partial_winners(
    protocol, tmp_path, world, boundary,
):
    ranks = tuple(range(world))
    jobs = [make_job(protocol, tmp_path, rank, ranks, 2 if rank == 0 else None)
            for rank in ranks]
    try:
        progress = ([item.job._advance() for item in jobs] if boundary == "cache"
                    else agree(protocol, ranks, jobs))
        authorization = decision(protocol, ranks, progress, cancelled=True)
        assert authorization["stop"] and not authorization["caches"] and not authorization["tuning"]
        for item in jobs:
            item.stop.set()  # vLLM broadcasts stop before resuming each B12X job.
            ready = item.job._advance(**{boundary: ()})
            assert ready.ready[0].key == "comm"
            assert item.job.session._tuning_cache_synchronized is (boundary == "tuning")
        assert_equal_installed(jobs, "default", 1)
    finally:
        for item in jobs:
            item.job.close()


@pytest.mark.parametrize("failure", ["missing-rank", "wrong-ranks", "identity", "incomplete", "bad-lowering"])
def test_bad_agreement_fails_before_installation(protocol, tmp_path, failure):
    ranks = (0, 1)
    jobs = [make_job(protocol, tmp_path, rank, ranks, 2, cache_only=True) for rank in ranks]
    try:
        if failure == "identity":
            jobs[1].cache.identity["device_name"] = "different"
        elif failure == "incomplete":
            jobs[1].cache.records[jobs[1].key]["coverage"]["measured_count"] = 1
        elif failure == "bad-lowering":
            jobs[0].cache.records[jobs[0].key]["config"]["width"] = 99
        progress = [item.job._advance() for item in jobs]
        if failure == "missing-rank":
            progress[1].ready_cache = None
            with pytest.raises(RuntimeError, match="cache boundaries"):
                decision(protocol, ranks, progress)
        else:
            snapshots = tuple(item.ready_cache for item in progress)
            if failure == "wrong-ranks":
                snapshots = (snapshots[0], protocol.types.TuningCacheRequirement(
                    (1,), jobs[1].cache.identity, jobs[1].cache.records))
            for item in jobs:
                with pytest.raises(ValueError):
                    item.job._advance(cache=snapshots)
        assert all(not item.installed and not item.races for item in jobs)
    finally:
        for item in jobs:
            item.job.close()


def test_incompatible_geometry_cannot_authorize_partial_winners(protocol, tmp_path):
    ranks = (0, 1)
    jobs = [make_job(protocol, tmp_path, rank, ranks, query=128 * (rank + 1)) for rank in ranks]
    try:
        progress = agree(protocol, ranks, jobs)
        with pytest.raises(RuntimeError, match="incompatible tuning boundaries"):
            decision(protocol, ranks, progress)
        assert all(not item.installed for item in jobs)
    finally:
        for item in jobs:
            item.job.close()


@pytest.mark.parametrize("world", [2, 4])
def test_cold_cache_only_fails_on_every_rank(protocol, tmp_path, world):
    ranks = tuple(range(world))
    jobs = [make_job(protocol, tmp_path, rank, ranks, cache_only=True) for rank in ranks]
    try:
        progress = [item.job._advance() for item in jobs]
        snapshots = decision(protocol, ranks, progress)["caches"][ranks]
        for item in jobs:
            with pytest.raises(LookupError, match="no completed selection"):
                item.job._advance(cache=snapshots)
            assert not item.installed and not item.races
    finally:
        for item in jobs:
            item.job.close()


def test_empty_agreement_requires_cancellation(protocol, tmp_path):
    item = make_job(protocol, tmp_path, 0, (0, 1))
    try:
        item.job._advance()
        with pytest.raises(ValueError, match="participating ranks"):
            item.job._advance(cache=())
        assert not item.job.session._tuning_cache_synchronized
        assert not item.installed and not item.races
    finally:
        item.job.close()


def test_completed_session_agreement_is_reused_by_next_job(protocol, tmp_path):
    ranks = (0, 1)
    jobs = [make_job(protocol, tmp_path, rank, ranks, 2) for rank in ranks]
    later = []
    try:
        agree(protocol, ranks, jobs)
        for rank, item in enumerate(jobs):
            item.job.close()
            following = make_job(protocol, tmp_path / "later", rank, ranks)
            later.append(following)
            following.job.session = item.job.session
            progress = following.job._advance()
            assert progress.ready_cache is None and progress.ready_tuning
    finally:
        for item in (*jobs, *later):
            item.job.close()


def test_oracle_rejects_mutation_that_retains_rank_local_choices(protocol, tmp_path, monkeypatch):
    def keep_local(cache, snapshots):
        cache._agreed_records = cache.records

    monkeypatch.setattr(protocol.SelectionCache, "reconcile", keep_local)
    jobs = [make_job(protocol, tmp_path, rank, (0, 1), rank + 2, cache_only=True)
            for rank in range(2)]
    try:
        with pytest.raises(AssertionError):
            agree(protocol, (0, 1), jobs)
        with pytest.raises(AssertionError):
            assert_equal_installed(jobs, "cached", 2)
    finally:
        for item in jobs:
            item.job.close()
