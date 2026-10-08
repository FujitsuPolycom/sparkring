"""CPU tests of the tuning table (``sparkring_sircl.tuning``): candidates, group shapes and facts, the
decisions a table derives from measurements (measured fastest at measured sizes, the fitted cost models
between them, NCCL where it was faster), lookups, hashes and the refusals of malformed tables."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sparkring_sircl import routes, tuning


def _key(**overrides):
    key = {"shape": "cycle:8", "world": 8, "lanes": 2, "max_relays": 3, "native": "n" * 16, "kernels": "k" * 16,
           "sircl": "0.2.0/abi8", "image": "sha256:0123"}
    key.update(overrides)
    return key


def _row(collective, mode, nbytes, choice, micros):
    return {"collective": collective, "mode": mode, "bytes": nbytes, "choice": choice, "p50_us": micros}


def test_choices_validate_and_round_trip():
    choice = tuning.Choice(schedule="ring", piece=524288, stagger=1)
    assert tuning.Choice.from_json(choice.to_json()) == choice
    assert choice.label() == "ring piece 524288 stagger 1"
    assert tuning.Choice(algorithm="twoshot", grid=8).label() == "twoshot grid 8"
    both = tuning.Choice(schedule="ring", piece=1 << 20, stagger=1, gather_stagger=2)
    assert tuning.Choice.from_json(both.to_json()) == both
    assert both.label() == "ring piece 1048576 stagger 1 gather stagger 2"
    assert tuning.Choice(backend="nccl").to_json() == {"backend": "nccl"} and tuning.Choice().to_json() == {}
    for bad in ({"algorithm": "fourshot"}, {"schedule": "tree"}, {"grid": 6}, {"piece": 100}, {"stagger": 9},
                {"gather_stagger": 5}, {"backend": "nccl", "gather_stagger": 1},
                {"algorithm": "oneshot", "schedule": "ring"}, {"backend": "nccl", "grid": 8}, {"color": "red"}):
        with pytest.raises(tuning.TuningError):
            tuning.Choice.from_json(bad)
    assert tuning.Choice(schedule="ring", piece=1 << 20).items("all_reduce", 64 << 20, 8) == 2 * 7 * 8
    assert tuning.Choice(schedule="ring", piece=1 << 20).items("all_gather", 8 << 20, 8) == 7 * 8
    assert tuning.Choice(schedule="chain", piece=1 << 20).items("all_reduce", 3 << 20, 8) == 3
    assert tuning.Choice(algorithm="oneshot").items("all_reduce", 1 << 20, 8) == 0


@pytest.mark.parametrize("text, shape", [("ring:8", "cycle:8"), ("path:0-3", "path:4"), ("path:2-3", "pair"),
                                         ("ring:8:4,5", "pair"), ("ring:8:7,0,1", "path:3"),
                                         ("ring:8:0,2,4,6", "strided:cycle:8:0,2,4,6"),
                                         ("ring:8:0,1,2,3,4", "strided:cycle:8:0,1,2,3,4")])
def test_group_shapes(text, shape):
    assert tuning.shape_of(routes.Layout.parse(text).identity()) == shape


def test_facts_of_a_layout_name_the_build():
    facts = tuning.facts_for_layout("path:0-3", 2)
    assert facts["shape"] == "path:4" and facts["world"] == 4 and facts["max_relays"] == 2
    assert len(facts["native"]) == 16 and len(facts["kernels"]) == 16 and "/abi" in facts["sircl"]
    assert tuning.facts_for_layout("path:4-7", 2) == facts


def test_measured_fastest_at_measured_sizes_and_models_between():
    rows = []
    for nbytes, one, two in ((4096, 20.0, 31.0), (16384, 24.0, 33.0), (65536, 40.0, 36.0), (262144, 110.0, 60.0)):
        rows.append(_row("all_reduce", "graph", nbytes, {"algorithm": "oneshot"}, one))
        rows.append(_row("all_reduce", "graph", nbytes, {"algorithm": "twoshot", "grid": 8}, two))
    document = tuning.build_document(_key(), rows, run_id="r1")
    table = tuning.Table(document)
    assert table.decide("all_reduce", 4096, "graph") == tuning.Choice(algorithm="oneshot")
    assert table.decide("all_reduce", 16384, "graph") == tuning.Choice(algorithm="oneshot")
    assert table.decide("all_reduce", 65536, "graph") == tuning.Choice(algorithm="twoshot", grid=8)
    assert table.decide("all_reduce", 1 << 30, "graph") == tuning.Choice(algorithm="twoshot", grid=8)
    assert table.decide("all_reduce", 4080, "graph") is None and table.decide("all_reduce", 4096, "eager") is None
    # The crossover falls between the measured 16 KiB and 64 KiB, where the models cross.
    starts = [interval["from"] for interval in document["decisions"][0]["intervals"]]
    assert len(starts) == 2 and 16384 < starts[1] <= 65536
    assert table.backend("all_reduce", 65536, "graph") == "sircl"


def test_nccl_marks_the_sizes_where_it_was_faster():
    rows = [_row("all_gather", "eager", nbytes, {"schedule": "pieces", "grid": 8}, micros)
            for nbytes, micros in ((1 << 20, 100.0), (4 << 20, 500.0))]
    rows += [_row("all_gather", "eager", nbytes, {"backend": "nccl"}, micros)
             for nbytes, micros in ((1 << 20, 140.0), (4 << 20, 400.0))]
    table = tuning.Table(tuning.build_document(_key(), rows))
    assert table.decide("all_gather", 4 << 20, "eager") == tuning.Choice(schedule="pieces", grid=8)
    assert table.backend("all_gather", 1 << 20, "eager") == "sircl"
    assert table.backend("all_gather", 4 << 20, "eager") == "nccl"


def test_a_pruned_candidate_competes_only_where_it_was_measured():
    rows = [_row("reduce_scatter", "graph", 1 << 20, {"schedule": "chain", "piece": 262144}, 50.0),
            _row("reduce_scatter", "graph", 1 << 20, {"schedule": "pieces", "grid": 8}, 80.0),
            _row("reduce_scatter", "graph", 4 << 20, {"schedule": "pieces", "grid": 8}, 300.0)]
    table = tuning.Table(tuning.build_document(_key(), rows))
    assert table.decide("reduce_scatter", 1 << 20, "graph").schedule == "chain"
    assert table.decide("reduce_scatter", 2 << 20, "graph").schedule == "pieces"


def test_fit_recovers_a_linear_cost():
    candidate = tuning._Candidate(tuning.Choice(schedule="ring", piece=1 << 20), {})
    for nbytes in (4 << 20, 8 << 20, 16 << 20, 32 << 20):
        items = candidate.choice.items("all_reduce", nbytes, 8)
        candidate.points[nbytes] = 30.0 + nbytes / 25000.0 + 5.0 * items
    a, b, c = tuning.fit(candidate, "all_reduce", 8)
    assert abs(a - 30.0) < 1e-6 and abs(b * 25000.0 - 1.0) < 1e-9 and abs(c - 5.0) < 1e-6


def test_tables_hash_canonically_and_name_their_mismatches(tmp_path):
    rows = [_row("all_to_all", "graph", 65536, {"grid": 8}, 40.0)]
    document = tuning.build_document(_key(), rows)
    path = tmp_path / "table.json"
    path.write_text(json.dumps(document, indent=3))
    table = tuning.Table.load(path)
    assert table.hash == tuning.document_hash(document) and table.source == str(path)
    assert table.mismatches(_key()) == []
    problems = table.mismatches(_key(shape="path:4", native="x" * 16))
    assert len(problems) == 2 and problems[0].startswith("shape: table 'cycle:8'")
    assert table.chosen() == [tuning.Choice(grid=8)]
    assert "all_to_all     graph" in tuning.render(document)


@pytest.mark.parametrize("damage, message", [
    (lambda d: d.update(schema="other"), "schema"),
    (lambda d: d["key"].pop("kernels"), "key needs"),
    (lambda d: d["decisions"][0]["intervals"].append(dict(d["decisions"][0]["intervals"][0])), "increasing"),
    (lambda d: d["decisions"][0]["intervals"][0].update(choice={"backend": "nccl"}), "SIRCL"),
    (lambda d: d["decisions"].append(dict(d["decisions"][0])), "twice"),
    (lambda d: d["decisions"][0].update(mode="replay"), "unknown"),
])
def test_malformed_tables_are_refused(damage, message):
    document = tuning.build_document(_key(), [_row("all_reduce", "eager", 8192, {"algorithm": "oneshot"}, 9.0)])
    damage(document)
    with pytest.raises(tuning.TuningError, match=message):
        tuning.Table(document)


def test_a_session_selects_the_table_of_its_own_shape(tmp_path):
    paths = []
    for name, shape in (("ring", "cycle:8"), ("pair", "pair")):
        document = tuning.build_document(_key(shape=shape), [_row("all_reduce", "eager", 8192, {"grid": 4}, 9.0)])
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(document))
        paths.append(str(path))
    assert tuning.table_paths(f" {paths[0]} ,, {paths[1]} ") == paths
    table, unmatched = tuning.select_table(paths, _key(shape="pair"))
    assert table is not None and table.source == paths[1] and table.key["shape"] == "pair"
    assert list(unmatched) == [paths[0]] and unmatched[paths[0]][0].startswith("shape: table 'cycle:8'")
    table, unmatched = tuning.select_table(paths, _key(shape="path:4"))
    assert table is None and sorted(unmatched) == sorted(paths)
    copy = tmp_path / "pair-copy.json"
    copy.write_text(Path(paths[1]).read_text())
    assert tuning.select_table([paths[1], str(copy)], _key(shape="pair"))[0].source == paths[1]
    other = tuning.build_document(_key(shape="pair"), [_row("all_reduce", "eager", 8192, {"grid": 8}, 9.0)])
    (tmp_path / "pair-other.json").write_text(json.dumps(other))
    with pytest.raises(tuning.TuningError, match="several different"):
        tuning.select_table([paths[1], str(tmp_path / "pair-other.json")], _key(shape="pair"))


def test_an_unreadable_table_is_refused(tmp_path):
    with pytest.raises(tuning.TuningError, match="cannot read"):
        tuning.Table.load(tmp_path / "missing.json")
