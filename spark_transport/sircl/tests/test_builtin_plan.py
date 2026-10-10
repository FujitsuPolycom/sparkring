"""The built-in plans of the cabled pair and the cycle of eight as tuning tables (tuning.builtin_document,
torch-free)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from sparkring_sircl import pieces, tuning

FACTS = {"shape": "pair", "world": 2, "lanes": 2, "max_relays": 0, "native": "n" * 16, "kernels": "k" * 16,
         "sircl": "0/abi9"}


def table(**kwargs):
    document = tuning.builtin_document(FACTS, **kwargs)
    return tuning.Table(document, "builtin:pair", builtin=True)


def test_the_pair_plan_decides_ring_ops_from_its_sizes_in_both_modes():
    plan = table()
    assert plan.builtin and not plan.mismatches(FACTS) and plan.settings == {}
    for mode in tuning.MODES:
        assert plan.decide("all_reduce", (3 << 20) - 16, mode) is None
        assert plan.decide("all_reduce", 3 << 20, mode) == tuning.Choice(schedule="ring", piece=256 << 10)
        assert plan.decide("all_reduce", 1 << 30, mode) == tuning.Choice(schedule="ring", piece=256 << 10)
        assert plan.decide("all_gather", (2 << 20) - 16, mode) is None
        assert plan.decide("all_gather", 2 << 20, mode) == tuning.Choice(schedule="ring", piece=256 << 10)
        assert plan.decide("all_gather", 16 << 20, mode) == tuning.Choice(schedule="ring", piece=512 << 10)
        assert plan.decide("reduce_scatter", (8 << 20) - 16, mode) is None
        assert plan.decide("reduce_scatter", 32 << 20, mode) == tuning.Choice(schedule="ring", piece=256 << 10)
        assert plan.decide("reduce_scatter", 64 << 20, mode) == tuning.Choice(schedule="ring", piece=512 << 10)
        assert plan.decide("all_to_all", 64 << 20, mode) is None
        assert plan.backend("all_reduce", 4 << 20, mode) == "sircl"
    assert {choice.order(c) for c, _, choice in plan.decided() if c != "all_gather"} == {"ring"}


def test_collectives_and_configured_pieces_narrow_the_plan():
    plan = table(collectives=("all_gather",))
    assert plan.decide("all_reduce", 4 << 20, "eager") is None
    assert plan.decide("all_gather", 4 << 20, "eager") == tuning.Choice(schedule="ring", piece=256 << 10)
    unpieced = table(configured_pieces=("all_gather", "reduce_scatter"))
    assert unpieced.decide("all_gather", 2 << 20, "eager") == tuning.Choice(schedule="ring")
    assert unpieced.decide("all_gather", 16 << 20, "eager") == tuning.Choice(schedule="ring")
    assert [len(entry["intervals"]) for entry in unpieced.document["decisions"]
            if entry["collective"] == "all_gather"] == [1, 1]
    assert unpieced.decide("all_reduce", 3 << 20, "graph") == tuning.Choice(schedule="ring", piece=256 << 10)
    assert tuning.builtin_document(FACTS, collectives=()) is None


CYCLE8 = {**FACTS, "shape": "cycle:8", "world": 8}


def test_the_cycle_of_eight_runs_the_ring_all_reduce_above_the_two_shot_capacity():
    plan = tuning.Table(tuning.builtin_document(CYCLE8), "builtin:cycle:8", builtin=True)
    assert plan.builtin and not plan.mismatches(CYCLE8)
    for mode in tuning.MODES:
        # At 2 MiB, the two-shot capacity, the two-shot op and the ring tied: the rules keep two-shot.
        assert plan.decide("all_reduce", 2 << 20, mode) is None
        assert plan.decide("all_reduce", (2 << 20) + 16, mode) == tuning.Choice(schedule="ring", piece=128 << 10)
        assert plan.decide("all_reduce", 3 << 20, mode) == tuning.Choice(schedule="ring", piece=128 << 10)
        assert plan.decide("all_reduce", (4 << 20) - 16, mode) == tuning.Choice(schedule="ring", piece=128 << 10)
        assert plan.decide("all_reduce", 4 << 20, mode) == tuning.Choice(schedule="ring", piece=256 << 10)
        assert plan.decide("all_reduce", 16 << 20, mode) == tuning.Choice(schedule="ring", piece=512 << 10)
        assert plan.decide("all_reduce", 1 << 30, mode) == tuning.Choice(schedule="ring", piece=512 << 10)
        for collective in ("all_gather", "reduce_scatter", "all_to_all"):
            assert plan.decide(collective, 8 << 20, mode) is None
    # A 3 MiB all-reduce under the plan is one ring op, never two-shot pieces; at 2 MiB the rules' plan (no
    # chain below its 8 MiB minimum) is one two-shot op.
    ring = pieces.reduce_plan(3 << 20, 4 << 20, None, ring_from=16 * 8, ring_world=8)
    assert ring and all(piece.ring for piece in ring)
    rules = pieces.reduce_plan(2 << 20, 4 << 20, 8 << 20, ring_from=None, ring_world=8)
    assert rules and not any(piece.ring or piece.chain for piece in rules)


def test_cycle8_direct_capacity_override_matches_executed_schedule():
    # A whole cycle of eight with a 4 MiB capacity: the direct all_reduce of 3 MiB is one two-shot launch, so the
    # built-in plan's ring choice for 3 MiB (all_reduce_large's) is neither applied nor counted there.
    pytest.importorskip("torch")
    pytest.importorskip("numpy")
    tests = Path(__file__).resolve().parent
    env = {key: value for key, value in os.environ.items() if not key.startswith("SIRCL_")}
    env["PYTHONPATH"] = os.pathsep.join([str(tests.parent), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    process = subprocess.run([sys.executable, str(tests / "direct_driver.py"), str(4 << 20), str(3 << 20)],
                             capture_output=True, text=True, env=env, timeout=300)
    assert process.returncode == 0, process.stderr[-3000:]
    found = json.loads(process.stdout.strip().splitlines()[-1])
    assert found["large"] == ["ring"]                     # the plan's choice where it runs
    assert found["launched"] == ["twoshot"]
    assert not any(label.startswith("all_reduce/") and "ring" in label for label in found["decisions"]), found


def test_shapes_without_a_plan_have_none_and_the_plan_is_deterministic():
    for shape in ("path:4", "cycle:4", "path:3", "strided:cycle:8:0,2"):
        assert tuning.builtin_document({**FACTS, "shape": shape}) is None
    assert table().hash == table().hash
    assert tuning.Table(tuning.builtin_document(FACTS), "x").builtin is False
    assert "ring piece 262144" in tuning.render(table().document)
