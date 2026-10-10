"""Copying a measured row into the default SIRCL tuning table; offline, on a copy of the repository files."""
import hashlib
import json

import pytest

from runtime.common import fabric_document, transport
from runtime.common.test_image_lock import sircl_block, sircl_lock
from runtime.common.test_transport import document, measured_table, pair_table
from runtime.host import fabric_tune
from scripts import promote_sircl_tuning as promote
from spark_transport.sircl.sparkring_sircl import tuning
from spark_transport.sircl.sparkring_sircl.ring import summary


@pytest.fixture
def measured(tmp_path):
    """A measured table of a four-Spark cycle with its cycle-4 SIRCL table, and the directory holding the table."""
    image = sircl_lock()
    cycle = document("cycle", 4)
    digest, data = pair_table(image, cycle, positions=(0, 1, 2, 3))
    host = tmp_path / "node-a"
    value = measured_table(host, cycle, image, rows={"cycle-4": {"link_slots": 12, "link_slot": 1048576}},
                           tables={digest: data})
    return value, host / transport.HOST_TABLES.lstrip("/"), digest, data, image, cycle


def test_a_measured_cycle4_row_replaces_the_rule_its_groups_took(tmp_path, measured):
    value, tables, digest, data, image, cycle = measured
    defaults = transport.load_tuning()
    assert transport.tuning_row(defaults, "cycle", 4)[0] == "cycle"
    promoted, written, relative = promote.promote(defaults, value, "cycle-4", tables)
    assert written == data and relative == "runtime/common/sircl-tuning/cycle-4.json"
    row = promoted["layouts"]["cycle-4"]
    assert {key: row[key] for key in ("source", "settings")} == {"source": "measured",
                                                                 "settings": {"link_slots": 12, "link_slot": 1048576}}
    # The evidence names the fabric, the image and its SIRCL build, the drivers and the measured table.
    assert row["evidence"].startswith(f"sparkring fabric tune on fabric {cycle['id'][7:19]} with image {image['name']} "
                                      f"(SIRCL {image['sircl']['version']}, ABI 9), position 0 GPU driver 580.95.05")
    assert row["evidence"].endswith(f"table {tuning.document_hash(json.loads(data))}")
    assert promoted["tables"] == [{"path": relative, "sha256": digest}] and promoted["measured_at"] == "2026-10-09"
    assert promoted["source"] == "defaults" and promoted["fabric"] is None and promoted["image_id"] is None
    repository = tmp_path / "repository"
    (repository / relative).parent.mkdir(parents=True)
    (repository / relative).write_bytes(written)
    transport.validate_tuning(promoted, root=repository)
    # A cycle-4 deployment on the image the row was measured with takes the row and the table.
    section = transport.section(image, cycle, [0, 1, 2, 3], nccl="never", tuning=promoted, root=repository)
    assert (section["tuning"]["row"], section["tuning"]["row_source"]) == ("cycle-4", "measured")
    assert section["tuning"]["tables"][0]["path"] == relative
    assert transport.plan_lines(section)[0] == "Transport: sircl on every collective, NCCL off (default table, cycle-4)"
    # Promoting again replaces the shipped table instead of adding a second one for the same groups.
    again, _, _ = promote.promote(promoted, value, "cycle-4", tables, root=repository)
    assert again["tables"] == promoted["tables"]


def test_a_row_the_table_did_not_measure_or_a_missing_table_is_refused(measured, tmp_path):
    value, tables, digest, _, _, _ = measured
    with pytest.raises(ValueError, match="no measured pair row; it measured cycle-4"):
        promote.promote(transport.load_tuning(), value, "pair", tables)
    with pytest.raises(ValueError, match="not pair, path-<n> or cycle-<n>"):
        promote.promote(transport.load_tuning(), value, "ring-4", tables)
    (tables / f"{digest}.json").write_text("{}")
    with pytest.raises(ValueError, match="differs from the measured table's SHA-256"):
        promote.promote(transport.load_tuning(), value, "cycle-4", tables)


def test_without_write_the_command_prints_the_change_and_writes_nothing(measured, tmp_path, capsys):
    value, tables, _, _, _, _ = measured
    path = tmp_path / "sircl-tuning.json"
    path.write_text(json.dumps(value))
    before = transport.TUNING_DEFAULTS.read_bytes()
    assert promote.main(["--measured", str(path), "--tables", str(tables), "--row", "cycle-4"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("cycle-4: no row (the shape row applies) -> ") and "Review, then repeat with --write." in out
    assert transport.TUNING_DEFAULTS.read_bytes() == before


# Ring-harness tunes of two separate rings of four.

SESSION = {"link_slots": 12, "link_slot_bytes": 1048576, "chain_slot_bytes": 1048576, "large_piece_bytes": 4194304}
CHOICES = {"oneshot": {"algorithm": "oneshot", "grid": 8}, "twoshot": {"algorithm": "twoshot", "grid": 8},
           "wide": {"algorithm": "twoshot", "grid": 16},
           "ring": {"schedule": "ring", "piece": 524288, "stagger": 0, "gather_stagger": 0},
           "chain": {"schedule": "chain", "piece": 524288}}
# Ring a: the ring all-reduce is the faster 8 MiB candidate; ring b: the chain. Both measured both; ring a alone
# measured the 16-block two-shot op, its fastest 8 KiB candidate.
TIMES_A = {("graph", 8192, "oneshot"): 14.0, ("graph", 8192, "twoshot"): 30.0, ("graph", 8192, "wide"): 9.0,
           ("graph", 8388608, "ring"): 800.0, ("graph", 8388608, "chain"): 900.0}
TIMES_B = {("graph", 8192, "oneshot"): 15.0, ("graph", 8192, "twoshot"): 31.0,
           ("graph", 8388608, "ring"): 850.0, ("graph", 8388608, "chain"): 700.0}
DRIVERS = {"gpu": "580.95.05", "kernel": "6.11.0-1016-nvidia"}
NEWER = {"gpu": "580.178.04", "kernel": "7.0.0-1019-nvidia"}


def choice(name):
    return tuning.Choice.from_json(CHOICES[name]).to_json()


def other_ring(value):
    """The fabric document of another cycle of four: the same cabling on Sparks of other node IDs."""
    value = json.loads(json.dumps(value))
    for row in value["positions"]:
        row["node_id"] = f"00000000-0000-0000-0000-0000000001{row['position']:02d}"
    value["id"] = fabric_document.identity(value)
    return fabric_document.validate(value)


def ring_tune(root, label, value, image, times, *, session=SESSION, status="passed", drivers=None):
    """A ring's --ring directory: its fabric document, drivers and one tune run of its whole cycle whose exact
    cases took ``times`` ({(mode, bytes, choice name): p50 us}), with the table the harness builds from them."""
    directory = root / f"ring-{label}"
    run = f"20261010-0100{label}-tune4"
    folder = directory / "results" / run / "ring"
    folder.mkdir(parents=True)
    (directory / "fabric.json").write_text(json.dumps(value), encoding="utf-8")
    size = int(value["size"])
    (directory / "facts.json").write_text(json.dumps({"positions": drivers or [DRIVERS] * size}), encoding="utf-8")
    topology = transport.group_topology(transport.sircl_layout(value), list(range(size)))
    plan = {"run_id": run, "image": image["image_id"], "options": {"rotate_buffers": 8, "tune_sizes": [8192, 8388608]},
            "groups": [{"index": 0, "global_ranks": list(range(size)), "layout": topology.session_layout(),
                        "lanes": 2, "max_relays": topology.max_relays()}]}
    cases = [{"group": 0, "collective": "all_reduce", "mode": mode, "bytes": nbytes, "correct": True,
              "slowest_p50_us": micros, "tune": {"collective": "all_reduce", "choice": CHOICES[name]}}
             for (mode, nbytes, name), micros in sorted(times.items())]
    result = {"status": status, "problems": [] if status == "passed" else ["group 0: a case differs"], "cases": cases,
              "ranks": [{"global_rank": rank, "session": dict(session)} for rank in range(size)]}
    key = {**fabric_tune.expected_facts(value, list(range(size)), image), "image": image["image_id"]}
    table = tuning.build_document(key, summary.tuning_rows(result)[0], run_id=run, created="2026-10-10T01:00:00+0000",
                                  session=summary.tune_session(plan, result, 0), conditions={"rotate_buffers": 8})
    (folder / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    (folder / "result.json").write_text(json.dumps(result), encoding="utf-8")
    (folder / "tuning-group0.json").write_bytes(promote.table_bytes(table))
    return directory


@pytest.fixture
def rings(tmp_path):
    image = sircl_lock()
    a = document("cycle", 4)
    b = other_ring(a)
    # Ring b's first two Sparks run a newer driver and kernel than its others and every Spark of ring a.
    return image, a, b, [ring_tune(tmp_path, "a", a, image, TIMES_A),
                         ring_tune(tmp_path, "b", b, image, TIMES_B, drivers=[NEWER, NEWER, DRIVERS, DRIVERS])]


def test_two_rings_merge_into_one_cycle4_row_judged_by_the_slower_ring(tmp_path, rings):
    image, a, b, directories = rings
    defaults = transport.load_tuning()
    promoted, data, relative, lines = promote.promote_rings(defaults, directories, "cycle-4", image)
    table = json.loads(data)
    merged = tuning.Table(table)
    # 8 MiB: the ring's slower time is 850 us and the chain's 900 us, so the ring decides though ring b's own table
    # chose the chain; 8 KiB: the one-shot op at its slower 15 us, because the 16-block two-shot op, ring a's
    # fastest, was measured on ring a only and is dropped.
    assert merged.decide("all_reduce", 8388608, "graph").to_json() == choice("ring")
    assert merged.decide("all_reduce", 8192, "graph").to_json() == choice("oneshot")
    times = {(row["bytes"], json.dumps(row["choice"], sort_keys=True)): row["p50_us"] for row in table["measurements"]}
    assert times[(8388608, json.dumps(choice("ring"), sort_keys=True))] == 850.0
    assert times[(8388608, json.dumps(choice("chain"), sort_keys=True))] == 900.0
    assert len(table["measurements"]) == 4 and table["run_id"] == "20261010-0100a-tune4+20261010-0100b-tune4"
    assert any(line.startswith("all_reduce graph 8388608 B: ring-a ") and line.endswith("; merged "
               + tuning.Choice.from_json(CHOICES["ring"]).label()) for line in lines)
    assert any(line.startswith("all_reduce graph 8192 B: ring-a ") for line in lines)
    row = promoted["layouts"]["cycle-4"]
    assert row["source"] == "measured" and row["settings"] == transport.measured_row(defaults, "cycle-4", {}, table)
    for text in (a["id"][7:19], b["id"][7:19], "20261010-0100a-tune4", "20261010-0100b-tune4",
                 f"SIRCL {image['sircl']['version']}, ABI 9",
                 f"fabric {a['id'][7:19]}: positions 0-3 GPU driver 580.95.05, kernel 6.11.0-1016-nvidia; fabric "
                 f"{b['id'][7:19]}: positions 0-1 GPU driver 580.178.04, kernel 7.0.0-1019-nvidia; positions 2-3 GPU "
                 "driver 580.95.05, kernel 6.11.0-1016-nvidia",
                 "4 measurements exact on every ring, 1 dropped", f"table {tuning.document_hash(table)}"):
        assert text in row["evidence"], text
    assert promoted["tables"] == [{"path": relative, "sha256": hashlib.sha256(data).hexdigest()}]
    assert promoted["measured_at"] == "2026-10-10"
    repository = tmp_path / "repository"
    (repository / relative).parent.mkdir(parents=True)
    (repository / relative).write_bytes(data)
    transport.validate_tuning(promoted, root=repository)
    # A cycle-4 deployment on either ring with the measured build takes the row and the merged table.
    for value in (a, b):
        section = transport.section(image, value, [0, 1, 2, 3], nccl="never", tuning=promoted, root=repository)
        assert (section["tuning"]["row"], section["tuning"]["row_source"]) == ("cycle-4", "measured")
        assert [entry["path"] for entry in section["tuning"]["tables"]] == [relative]
    # A session of a compatible build takes the row's settings; the table's choices keep their own build's key.
    version, abi = transport.tuning_builds(promoted)[1]
    older = sircl_lock(sircl=sircl_block(version, abi))
    section = transport.section(older, a, [0, 1, 2, 3], nccl="never", tuning=promoted, root=repository)
    assert section["tuning"]["applies"] and section["tuning"]["row_source"] == "measured"
    assert section["tuning"]["tables"] == []


@pytest.mark.parametrize("make, message", [
    (lambda root, image, a, b: [ring_tune(root, "a", a, image, TIMES_A, status="failed"),
                                ring_tune(root, "b", b, image, TIMES_B)], "did not pass: group 0: a case differs"),
    (lambda root, image, a, b: [ring_tune(root, "a", a, image, TIMES_A), ring_tune(root, "b", a, image, TIMES_B)],
     "measured the same fabric"),
    (lambda root, image, a, b: [ring_tune(root, "a", a, image, TIMES_A),
                                ring_tune(root, "b", b, image, TIMES_B, session=dict(SESSION, link_slots=16))],
     "ran under different settings"),
    (lambda root, image, a, b: [ring_tune(root, "a", document("cycle", 8), image, TIMES_A)],
     "the cycle-4 row serves the whole cycle of 4"),
    (lambda root, image, a, b: [], "names no tune"),
])
def test_a_ring_tune_that_cannot_make_the_row_is_refused(tmp_path, make, message):
    image = sircl_lock()
    a = document("cycle", 4)
    directories = make(tmp_path, image, a, other_ring(a))
    with pytest.raises(ValueError, match=message):
        promote.promote_rings(transport.load_tuning(), directories, "cycle-4", image)


def test_a_table_of_another_build_or_one_its_run_does_not_rebuild_is_refused(rings):
    image, _, _, directories = rings
    other = sircl_lock(sircl=dict(image["sircl"], tuning_key=dict(image["sircl"]["tuning_key"], kernels="b" * 16)))
    with pytest.raises(ValueError, match="its key does not match a deployment on the whole ring"):
        promote.promote_rings(transport.load_tuning(), directories, "cycle-4", other)
    path = next(directories[1].glob("results/*/*/tuning-group0.json"))
    table = json.loads(path.read_text(encoding="utf-8"))
    table["decisions"] = [dict(entry, intervals=entry["intervals"][:1]) for entry in table["decisions"]]
    path.write_bytes(promote.table_bytes(table))
    with pytest.raises(ValueError, match="do not rebuild this table"):
        promote.promote_rings(transport.load_tuning(), directories, "cycle-4", image)


def test_the_ring_command_prints_where_the_rings_differ_and_writes_nothing(tmp_path, rings, capsys):
    image, _, _, directories = rings
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps(image), encoding="utf-8")
    before = transport.TUNING_DEFAULTS.read_bytes()
    argv = ["--row", "cycle-4", "--image-lock", str(lock)] + [arg for path in directories for arg in ("--ring", str(path))]
    assert promote.main(argv) == 0
    out = capsys.readouterr().out
    assert "rings differ: all_reduce graph 8388608 B: " in out and "Review, then repeat with --write." in out
    assert transport.TUNING_DEFAULTS.read_bytes() == before
    with pytest.raises(SystemExit):
        promote.main(["--row", "cycle-4", "--ring", str(directories[0])])
