"""Blocks per role chosen per op: the kernels' tail rule, tuning tables that name blocks (schema v2), and the
ring harness's blocks and rotated buffers (torch-free)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sparkring_sircl import tuning
from sparkring_sircl.ring import cli, plan as plan_mod, summary, worker

PACKAGE = Path(tuning.__file__).resolve().parent
FACTS = {"shape": "pair", "world": 2, "lanes": 2, "max_relays": 0, "native": "n" * 16, "kernels": "k" * 16,
         "sircl": "0/abi9"}


def last_arrivals(grids, rule):
    """The arrival (0-based) each launch's block takes as the last of its launch, counting on one tail word:
    ``modulo`` (the last arrival is the one that makes the count a multiple of the grid) or ``reset`` (the
    arrival that completes the grid, which then returns the word to 0)."""
    word, picked = 0, []
    for grid in grids:
        for arrival in range(grid):
            prior, word = word, word + 1
            if rule == "modulo" and (prior + 1) % grid == 0:
                picked.append(arrival)
            if rule == "reset" and prior + 1 == grid:
                picked.append(arrival)
                word = 0
    return picked


def test_the_reset_rule_names_the_true_last_block_when_grids_change():
    grids = [2, 4, 4, 2, 6, 3]
    assert last_arrivals(grids, "reset") == [grid - 1 for grid in grids]
    # Counting modulo the grid on a shared word picks a block that is not the last once a grid of 4 follows
    # a grid of 2: that block would advance the item bases while others of its launch still run.
    assert last_arrivals([2, 4, 4], "modulo") == [1, 1, 1]
    assert last_arrivals([4, 4, 4], "modulo") == [3, 3, 3]


def test_every_link_and_chain_kernel_uses_the_reset_rule():
    links = (PACKAGE / "oneshot" / "_links_cute.py").read_text(encoding="utf-8")
    chain = (PACKAGE / "oneshot" / "_chain_cute.py").read_text(encoding="utf-8")
    assert links.count("if prior + Uint32(1) == Uint32(gdim):") == 3
    assert chain.count("if prior + Uint32(1) == Uint32(gdim):") == 1
    assert "% Uint32(gdim)" not in links + chain
    for offset in ("20", "24", "_RING_TAILS[self._mode]"):
        assert f"st_release_gpu_u32(counters + Int64({offset}), Uint32(0))" in links
    assert "st_release_gpu_u32(counters + Int64(12), Uint32(0))" in chain


def test_choices_name_blocks_of_chain_and_ring_schedules():
    ring = tuning.Choice(schedule="ring", piece=262144, blocks=2)
    assert ring.to_json() == {"schedule": "ring", "piece": 262144, "blocks": 2}
    assert tuning.Choice.from_json({"schedule": "chain", "piece": 65536, "blocks": "4"}).blocks == 4
    assert ring.label() == "ring piece 262144 blocks 2"
    for bad in ({"schedule": "ring", "blocks": 0}, {"schedule": "ring", "blocks": 65},
                {"schedule": "pieces", "blocks": 1}, {"algorithm": "twoshot", "blocks": 1}):
        with pytest.raises(tuning.TuningError, match="blocks"):
            tuning.Choice(**bad)
    with pytest.raises(tuning.TuningError, match="blocks|NCCL"):
        tuning.Choice(backend="nccl", blocks=1)


def document(schema=tuning.SCHEMA, choice=None, conditions=None):
    doc = {"schema": schema, "key": {**FACTS, "image": ""}, "run_id": "t", "created": "", "measurements": [],
           "decisions": [{"collective": "all_reduce", "mode": "eager",
                          "intervals": [{"from": 1 << 20, "nccl": False,
                                         "choice": choice or {"schedule": "ring", "piece": 262144}}]}]}
    if conditions is not None:
        doc["conditions"] = conditions
    return doc


def test_schema_two_adds_blocks_and_conditions_and_reads_schema_one():
    assert tuning.SCHEMA == "sircl-tuning-table/v2" and tuning.SCHEMAS[0] == "sircl-tuning-table/v1"
    table = tuning.Table(document(choice={"schedule": "ring", "piece": 262144, "blocks": 1},
                                  conditions={"rotate_buffers": 8}))
    assert table.decide("all_reduce", 4 << 20, "eager").blocks == 1 and table.conditions == {"rotate_buffers": 8}
    old = tuning.Table(document(schema=tuning.SCHEMAS[0]))
    assert old.decide("all_reduce", 4 << 20, "eager") == tuning.Choice(schedule="ring", piece=262144)
    assert old.conditions == {}
    with pytest.raises(tuning.TuningError, match="blocks need schema"):
        tuning.Table(document(schema=tuning.SCHEMAS[0], choice={"schedule": "ring", "blocks": 1}))
    with pytest.raises(tuning.TuningError, match="conditions"):
        tuning.Table(document(schema=tuning.SCHEMAS[0], conditions={"rotate_buffers": 8}))
    for bad in ({"rotate_buffers": 0}, {"rotate_buffers": True}, {"warmups": 3}):
        with pytest.raises(tuning.TuningError, match="conditions"):
            tuning.Table(document(conditions=bad))
    with pytest.raises(tuning.TuningError, match="schema must be"):
        tuning.Table(document(schema="sircl-tuning-table/v3"))


def test_built_documents_record_their_conditions():
    rows = [{"collective": "all_reduce", "mode": "eager", "bytes": size, "p50_us": time,
             "choice": {"schedule": "ring", "piece": 262144, "blocks": blocks}}
            for size, time, blocks in ((1 << 20, 80.0, 1), (4 << 20, 212.0, 1), (1 << 20, 85.0, 2),
                                       (4 << 20, 220.0, 2))]
    built = tuning.build_document({**FACTS, "image": ""}, rows, conditions={"rotate_buffers": 8})
    assert built["schema"] == tuning.SCHEMA and built["conditions"] == {"rotate_buffers": 8}
    assert tuning.Table(built).decide("all_reduce", 4 << 20, "eager").blocks == 1
    assert "measured with rotate_buffers 8" in tuning.render(built)
    assert "conditions" not in tuning.build_document({**FACTS, "image": ""}, rows)


def test_variants_name_the_kernel_whose_blocks_they_set():
    assert worker.variant_kernel("all_reduce_large", {"schedule": "ring", "link_blocks": 2}) == "ring_reduce"
    assert worker.variant_kernel("all_reduce_large", {"schedule": "chain", "chain_blocks": 2}) == "chain_reduce"
    assert worker.variant_kernel("all_reduce_large", {"schedule": "chain", "link_blocks": 2}) is None
    assert worker.variant_kernel("all_gather_large", {"gather_schedule": "chain", "link_blocks": 1}) == "chain_gather"
    assert worker.variant_kernel("all_gather_large", {"gather_schedule": "ring", "link_blocks": 1}) == "ring_gather"
    assert worker.variant_kernel("reduce_scatter", {"scatter_schedule": "ring", "link_blocks": 4}) == "ring_scatter"
    assert worker.variant_kernel("reduce_scatter", {"scatter_schedule": "pieces", "link_blocks": 4}) is None
    assert worker.forced({"schedule": "ring", "link_blocks": 1})


class _Session:
    max_size = 2 << 20
    max_gather_bytes = 16 << 20
    chain_available = True
    chain_slot_bytes = 1 << 20
    link_available = True
    ring_available = True
    _available = {"oneshot": True, "twoshot": True, "swing": False}


def harness(**options):
    built = worker.Harness.__new__(worker.Harness)
    built.session, built.world, built.swing_through, built.scatter_through = _Session(), 2, None, "session"
    built.options = {"tune_grids": [8], "tune_pieces": [262144], "tune_staggers": [0], "tune_large_from": 262144,
                     **options}
    return built


def test_tune_candidates_cross_pieces_with_blocks_per_role():
    blocked = harness(tune_link_blocks=[1, 2], tune_chain_blocks=[1, 4])
    reduce = [(choice, variant) for choice, _, _, variant in blocked.tune_candidates("all_reduce", 8 << 20)]
    assert ({"schedule": "chain", "piece": 262144, "blocks": 4},
            {"schedule": "chain", "chunk": 262144, "chain_blocks": 4}) in reduce
    assert ({"schedule": "ring", "piece": 262144, "stagger": 0, "gather_stagger": 0, "blocks": 2},
            {"schedule": "ring", "link_chunk": 262144, "stagger": 0, "gather_stagger": 0, "link_blocks": 2}) in reduce
    assert sum(1 for choice, _ in reduce if choice.get("schedule") == "ring") == 2
    gathers = {json.dumps(choice, sort_keys=True) for choice, *_ in blocked.tune_candidates("all_gather", 1 << 20)}
    assert '{"blocks": 1, "piece": 262144, "schedule": "chain"}' in gathers
    assert '{"blocks": 2, "gather_stagger": 0, "piece": 262144, "schedule": "ring"}' in gathers
    scatters = {json.dumps(variant, sort_keys=True) for _, _, _, variant in blocked.tune_candidates("reduce_scatter", 1 << 20)}
    assert '{"link_blocks": 2, "link_chunk": 262144, "scatter_schedule": "ring", "stagger": 0}' in scatters
    # 0, or no list, keeps the session's blocks and names none.
    for plain in (harness(), harness(tune_link_blocks=[0], tune_chain_blocks=[0])):
        assert all("blocks" not in choice for choice, *_ in plain.tune_candidates("all_reduce", 8 << 20))


def test_rotated_windows_stay_within_their_memory():
    rotated = harness(rotate_buffers=8)
    assert rotated._windows((1 << 19,), [1 << 19]) == 8                 # 2 MiB per window
    assert rotated._windows((1 << 28,), [1 << 28]) == 4                 # 1 GiB per window, 4 GiB in all
    assert rotated._windows((1 << 31,), [1 << 31]) == 1
    assert harness()._windows((1 << 19,), [1 << 19]) == 1


def _site(tmp_path):
    from tests.test_ring_harness import _site_document

    site = tmp_path / "site.json"
    site.write_text(json.dumps(_site_document()))
    return str(site)


def test_run_and_tune_options_carry_rotation_and_blocks(tmp_path, capsys):
    site = _site(tmp_path)
    assert cli.main(["plan", "--site", site, "--groups", "0-1", "--name", "pair", "--large", "--rotate-buffers", "3",
                     "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["options"]["rotate_buffers"] == 3
    assert cli.main(["plan", "--site", site, "--groups", "0-1", "--name", "pair", "--tune", "--tune-link-blocks",
                     "0,2", "--json"]) == 0
    options = json.loads(capsys.readouterr().out)["options"]
    assert options["rotate_buffers"] == plan_mod.TUNE_ROTATE_BUFFERS == 8
    assert options["tune_link_blocks"] == [0, 2] and options["tune_chain_blocks"] == [1, 2, 4]
    assert {3 << 19, 3 << 20, 3 << 21} <= set(options["tune_sizes"])
    for bad in (["--rotate-buffers", "65"], ["--tune", "--tune-link-blocks", "65"]):
        assert cli.main(["plan", "--site", site, "--groups", "0-1", "--name", "pair", "--large", *bad]) == 2
        capsys.readouterr()


def test_summaries_name_the_rotation_and_tables_record_it():
    result = {"configuration": "c", "run_id": "r", "status": "passed", "cases": [], "rotate_buffers": 8,
              "problems": [], "warnings": [], "crossover": [], "post_orders": [], "tuning": []}
    assert "cycled through 8 input and output windows" in summary.table(result).splitlines()[0]
    assert "cycled" not in summary.table({**result, "rotate_buffers": 1}).splitlines()[0]
