"""Copy a row of a measured SIRCL tuning table into the release's default table (maintainers; OFFLINE).

The default table, ``runtime/common/sircl-tuning-defaults.json``, applies on
every cluster that has not run ``sudo sparkring fabric tune``. Its rows are
``measured`` where the owner's checklist run measured the group shape, and
SIRCL's own ``rules`` or a row ``inherited:<row>`` elsewhere. After the owner
runs ``sudo sparkring fabric tune --execute`` on a qualified fabric, this
command replaces one default row with that measurement:

    python scripts/promote_sircl_tuning.py --measured sircl-tuning.json --tables DIR --row cycle-4
    python scripts/promote_sircl_tuning.py --measured sircl-tuning.json --tables DIR --row cycle-4 --write

``--measured`` is a copy of Node A's ``/var/lib/sparkring/controller/sircl-tuning.json``
and ``DIR`` a copy of ``/etc/sparkring/fabric/sircl-tuning/``, which holds its
SIRCL tables (``sircl-tuning-table/v1``) under their SHA-256 names. Without
``--write`` the command prints the change. With it, the row's SIRCL table is
written byte for byte to ``runtime/common/sircl-tuning/<row>.json``, the row
becomes ``source: measured`` with the measured settings, the table is named in
``tables`` in place of any default table that serves the same groups, and
``measured_at`` becomes the later of the two dates.

The default table names no fabric and no image. The measured table must be of
the SIRCL version and ABI the default table names; a build the default table
lists as ``compatible`` is refused, because the compatibility reason covers
the default rows, not a new measurement. The promoted table keeps its key,
which names the SIRCL build and the image it was measured with: sessions on
another SIRCL build do not take it and keep SIRCL's rules for those choices. Record the evidence (fabric, image, driver, harness run) in the
release's performance record before the release cites the row.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.common import transport  # noqa: E402

TABLE_DIRECTORY = "runtime/common/sircl-tuning"
_ROW = re.compile(r"(path|cycle)-([3-9]|1[0-6])")


def row_identity(row):
    """``(shape, world)`` of the SIRCL table key that serves the groups of the tuning row ``row``."""
    if row == "pair":
        return "pair", 2
    found = _ROW.fullmatch(row)
    if not found:
        raise ValueError(f"--row {row} is not pair, path-<n> or cycle-<n>")
    return f"{found.group(1)}:{found.group(2)}", int(found.group(2))


def measured_tables(measured, tables):
    """``{sha256: bytes}`` of the measured table's SIRCL tables, read from ``tables`` and checked."""
    found = {}
    for entry in measured["tables"]:
        if not entry["path"].startswith(transport.HOST_TABLES + "/"):
            continue
        path = Path(tables) / f"{entry['sha256']}.json"
        data = path.read_bytes() if path.is_file() else b""
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError(f"{path} is missing or differs from the measured table's SHA-256")
        found[entry["sha256"]] = data
    return found


def promote(defaults, measured, row, tables, *, root=ROOT):
    """``(new default table, SIRCL table bytes, repository path)`` with ``row`` of ``measured`` in place."""
    shape, world = row_identity(row)
    found = measured_tables(measured, tables)
    with tempfile.TemporaryDirectory() as host:
        for digest, data in found.items():
            path = Path(host) / transport.HOST_TABLES.lstrip("/") / f"{digest}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        transport.validate_tuning(measured, root=root, host_root=host)
    if measured["source"] != "measured":
        raise ValueError("--measured names a default table; it needs one sparkring fabric tune wrote")
    named = transport.tuning_builds(defaults)[0]
    if (measured["sircl"]["version"], measured["sircl"]["abi_version"]) != named:
        raise ValueError(f"The measured table is for SIRCL {measured['sircl']['version']} (ABI "
                         f"{measured['sircl']['abi_version']}), the default table for {named[0]} (ABI {named[1]}); "
                         "promote a measurement of the build the default table names")
    measured_row = measured["layouts"].get(row)
    if measured_row is None or measured_row["source"] != "measured":
        raise ValueError(f"The measured table has no measured {row} row; it measured "
                         + ", ".join(name for name, value in sorted(measured["layouts"].items())
                                     if value["source"] == "measured"))
    matching = [(digest, data) for digest, data in found.items()
                if transport.table_identity(json.loads(data))[:2] == (shape, world)]
    if len(matching) != 1:
        raise ValueError(f"The measured table holds {len(matching)} SIRCL tables for {row} groups; it needs one")
    digest, data = matching[0]
    identity = transport.table_identity(json.loads(data))
    relative = f"{TABLE_DIRECTORY}/{row}.json"
    value = copy.deepcopy(defaults)
    value["layouts"][row] = {"source": "measured", "settings": dict(measured_row["settings"])}
    kept = []
    for entry in value["tables"]:
        if entry["path"] == relative:
            continue
        shipped = json.loads((Path(root) / entry["path"]).read_text(encoding="utf-8"))
        if transport.table_identity(shipped) != identity:
            kept.append(entry)
    value["tables"] = sorted(kept + [{"path": relative, "sha256": digest}], key=lambda entry: entry["path"])
    value["layouts"] = dict(sorted(value["layouts"].items()))
    value["measured_at"] = max(value["measured_at"], measured["measured_at"])
    return value, data, relative


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python scripts/promote_sircl_tuning.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--measured", required=True, type=Path, help="a measured sparkring-sircl-tuning/v1 table")
    parser.add_argument("--tables", required=True, type=Path, help="the directory of its SIRCL tables")
    parser.add_argument("--row", required=True, help="the row to promote: pair, path-<n> or cycle-<n>")
    parser.add_argument("--write", action="store_true", help="write the default table and the SIRCL table")
    args = parser.parse_args(argv)
    try:
        defaults = transport.load_tuning()
        measured = json.loads(args.measured.read_text(encoding="utf-8"))
        value, data, relative = promote(defaults, measured, args.row, args.tables)
    except (OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    before = defaults["layouts"].get(args.row)
    print(f"{args.row}: {json.dumps(before, sort_keys=True) if before else 'no row (the shape row applies)'} -> "
          f"{json.dumps(value['layouts'][args.row], sort_keys=True)}")
    print(f"tables: {', '.join(entry['path'] for entry in value['tables'])}")
    print(f"measured_at: {defaults['measured_at']} -> {value['measured_at']}")
    if not args.write:
        print("Review, then repeat with --write.")
        return 0
    target = ROOT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    transport.TUNING_DEFAULTS.write_text(transport.encoded(value), encoding="utf-8", newline="\n")
    transport.load_tuning()
    print(f"Wrote {relative} and {transport.TUNING_DEFAULTS.relative_to(ROOT).as_posix()}; the default table's SHA-256 "
          f"is now {transport.tuning_digest(value)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
