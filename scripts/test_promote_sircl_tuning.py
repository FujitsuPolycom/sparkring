"""Copying a measured row into the default SIRCL tuning table; offline, on a copy of the repository files."""
import json

import pytest

from runtime.common import transport
from runtime.common.test_image_lock import sircl_lock
from runtime.common.test_transport import document, measured_table, pair_table
from scripts import promote_sircl_tuning as promote


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
    assert promoted["layouts"]["cycle-4"] == {"source": "measured", "settings": {"link_slots": 12, "link_slot": 1048576}}
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
