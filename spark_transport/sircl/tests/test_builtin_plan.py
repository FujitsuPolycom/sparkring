"""The built-in pair plan as a tuning table (tuning.builtin_document, torch-free)."""

from __future__ import annotations

from sparkring_sircl import tuning

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


def test_shapes_without_a_plan_have_none_and_the_plan_is_deterministic():
    for shape in ("path:4", "cycle:8", "path:3", "strided:cycle:8:0,2"):
        assert tuning.builtin_document({**FACTS, "shape": shape}) is None
    assert table().hash == table().hash
    assert tuning.Table(tuning.builtin_document(FACTS), "x").builtin is False
    assert "ring piece 262144" in tuning.render(table().document)
