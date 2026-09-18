"""Exercise distributed selection-cache boundaries without compiling kernels.

Execute the selected source's preparation state machine. Only physical cache
storage, candidate timing and kernel execution are replaced by host fixtures.
Local cache hits must not remove one rank from a shared tuning boundary.
"""

from __future__ import annotations

import ast
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


class FrozenMapping(dict):
    def to_dict(self):
        return dict(self)


@dataclass
class Selection:
    component_id: str
    query: object
    config: object
    source: str
    assignment: object


@dataclass
class TuningRequirement:
    key: str
    ranks: tuple
    assignment: object
    latency_us: float
    candidate_index: int


@dataclass
class _TuningBatch:
    contributions: tuple


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_job():
    path = (
        Path(os.environ["SPARKRING_B12X_SOURCE_ROOT"]) / "b12x/preparation/session.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    original = next(n for n in tree.body if getattr(n, "name", "") == "PreparationJob")
    methods = {"_choice_key", "_selection", "_lookup", "_consolidate", "_run"}
    original.body = [n for n in original.body if getattr(n, "name", "") in methods]
    assert {n.name for n in original.body} == methods
    env = dict(
        FrozenMapping=FrozenMapping,
        Selection=Selection,
        TuningRequirement=TuningRequirement,
        _TuningBatch=_TuningBatch,
        digest=digest,
        _coalesce_requests=lambda requests: [(r,) for r in requests],
    )
    node = ast.Module(body=[original], type_ignores=[])
    exec(compile(ast.fix_missing_locations(node), str(path), "exec"), env)
    return env["PreparationJob"]


def make_job(
    rank, ranks, cached, *, query=128, autotune=True, cache_only=False, stopped=False
):
    job = load_job()()
    contract = NS(
        component_id="gemm.example",
        query_schema_version=1,
        config_schema_version=1,
        semantic_version=1,
        candidate_contract_version=1,
        config_payload=lambda config: FrozenMapping(config),
        _lower=lambda query, device, assignment: dict(assignment),
    )
    plan = NS(
        contract=contract,
        component_id=contract.component_id,
        invocation=FrozenMapping(),
    )
    request = NS(name="model.projection", dependencies=(), plan=plan, collective=None)
    collective = NS(key="distributed.roce.collectives", ranks=ranks)
    comm_plan = NS(
        contract=contract, component_id="comm.roce", invocation=FrozenMapping()
    )
    comm = NS(name="comm.roce", dependencies=(), plan=comm_plan, collective=collective)
    configuration = NS(
        encoded_query=FrozenMapping(tokens=query),
        pinned=None,
        space=NS(validate=lambda value: None),
        query=query,
        device=None,
    )
    race = NS(
        request=request,
        requests=(request,),
        configuration=configuration,
        selection=None,
        ready=False,
        candidates=[0, 1, 2, 3],
        planned_candidates=0,
    )
    fixed = NS(
        request=comm,
        requests=(comm,),
        configuration=configuration,
        selection=Selection("comm.roce", FrozenMapping(), {}, "fixed", {}),
        ready=False,
        candidates=[0],
        planned_candidates=0,
    )
    job.requests = (request, comm)
    record = {
        "assignment": {"width": 2},
        "config": {"width": 2},
        "coverage": {"measured_count": 4},
    }
    cache = NS(get=lambda key: record if cached else None)
    job.session = NS(
        state="PREPARING",
        cache_only=cache_only,
        _tuning_ranks=ranks,
        _selection_cache=lambda: cache,
        _stop=NS(is_set=lambda: stopped),
    )
    job._timing = NS(span=lambda *a: nullcontext(), record=lambda *a, **k: None)
    job._cache_hits = 0
    job._warmup_only = False
    job.autotune = autotune
    job._expand = lambda: (job.requests, {})

    def configure(groups):
        yield from ()
        return [race, fixed]

    def run_race(obligation):
        yield from ()
        return TuningRequirement(
            obligation.key, ranks, FrozenMapping(width=2), 1.0, rank
        )

    def install(obligation, selections):
        if obligation.request.collective:
            yield obligation.request.collective
        selections[obligation.request.name] = (obligation.selection, contract)

    job._configure = configure
    job._race = run_race
    job._install_obligation = install
    return job, race


def first_boundary(rank, ranks, cached, *, query=128):
    job, _ = make_job(rank, ranks, cached, query=query)
    iterator = job._run()
    boundary = next(iterator)
    iterator.close()
    return boundary


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("mode", ["cold", "warm", "mixed"])
def test_rank_local_cache_state_does_not_change_shared_tuning_boundary(world, mode):
    ranks = tuple(range(world))
    hits = {
        "cold": [False] * world,
        "warm": [True] * world,
        "mixed": [True] + [False] * (world - 1),
    }[mode]
    boundaries = [first_boundary(rank, ranks, hit) for rank, hit in enumerate(hits)]
    assert all(isinstance(item, _TuningBatch) for item in boundaries)
    contributions = [item.contributions[0] for item in boundaries]
    assert len({item.key for item in contributions}) == 1
    assert all(item.ranks == ranks for item in contributions)
    assert len({item.candidate_index for item in contributions}) == world


def test_distinct_geometry_keeps_distinct_tuning_identity():
    boundaries = [
        first_boundary(rank, (0, 1), hit, query=query)
        for rank, (hit, query) in enumerate([(True, 128), (False, 256)])
    ]
    assert all(isinstance(item, _TuningBatch) for item in boundaries)
    assert boundaries[0].contributions[0].key != boundaries[1].contributions[0].key


@pytest.mark.parametrize(
    "options",
    [
        {"ranks": (0,)},
        {"autotune": False},
        {"cache_only": True},
        {"stopped": True},
    ],
)
def test_non_racing_modes_retain_cached_selection(options):
    options = {"ranks": (0, 1), **options}
    job, obligation = make_job(0, cached=True, **options)
    job._lookup(obligation, {})
    assert obligation.selection.source == "cached"
    assert job._cache_hits == 1


def test_cache_only_miss_does_not_invent_a_selection():
    job, obligation = make_job(0, (0, 1), False, cache_only=True)
    job._lookup(obligation, {})
    assert obligation.selection is None
    assert obligation.key is not None


def test_existing_selection_does_not_reenter_lookup():
    job, obligation = make_job(0, (0, 1), True)
    chosen = object()
    obligation.selection = chosen
    job._lookup(obligation, {})
    assert obligation.selection is chosen
    assert job._cache_hits == 0
