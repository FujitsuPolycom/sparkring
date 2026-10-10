"""Copy a measured SIRCL tuning row into the release's default table (maintainers; OFFLINE).

The default table, ``runtime/common/sircl-tuning-defaults.json``, applies on
every cluster that has not run ``sudo sparkring fabric tune``. Its rows are
``measured`` where the owner's checklist run measured the group shape, and
SIRCL's own ``rules`` or a row ``inherited:<row>`` elsewhere. This command
replaces one default row with a measurement, from either of two sources.

A measured table that ``sudo sparkring fabric tune --execute`` wrote on a
qualified fabric:

    python scripts/promote_sircl_tuning.py --measured sircl-tuning.json --tables DIR --row cycle-4

``--measured`` is a copy of Node A's ``/var/lib/sparkring/controller/sircl-tuning.json``
and ``DIR`` a copy of ``/etc/sparkring/fabric/sircl-tuning/``, which holds its
SIRCL tables (``sircl-tuning-table/v1``) under their SHA-256 names.

Or SIRCL ring-harness tunes of the row's whole group on one or more rings of
that shape, such as two independent cycles of four Sparks, merged into one
table:

    python scripts/promote_sircl_tuning.py --row cycle-4 --ring RING_A --ring RING_B --image-lock LOCK

Each ``--ring`` directory holds the ring's fabric document (``fabric.json``,
the bytes ``sudo sparkring setup`` wrote to ``/etc/sparkring/fabric/topology.json``),
its Sparks' GPU drivers and kernels by position (``facts.json``:
``{"positions": [{"gpu", "kernel"}, ...]}``), and one harness tune run of the
ring's whole cycle under ``results/<run id>/<configuration>/`` (``plan.json``,
``result.json``, ``tuning-group0.json``). ``--image-lock`` is the v3 lock of
the image the tune ran on. Each run must have passed, cover the ring's every
position, and carry a table whose key is the one a deployment on the whole ring
checks with that image's SIRCL build (``runtime.host.fabric_tune.expected_facts``)
and which the run's own measurements rebuild. The rings' tables must share one
key, their tune sessions one set of settings and one rotation of buffers.
The merged table judges every candidate by its slower ring: a measurement
(collective, mode, per-rank size, choice) that every ring measured exactly
keeps the largest of the rings' medians, one that some ring did not measure
exactly is dropped, and ``sparkring_sircl.tuning.build_document`` decides each
size from those times. Where the rings' own tables choose differently, the
merged choice is the one with the best slower-ring time. A size at which no
candidate was measured exactly on every ring takes the choice that
``build_document`` carries from the neighbouring sizes, as in one ring's
table: a table's decisions are contiguous intervals, so a size cannot be left
to SIRCL's rules between decided ones. The command prints every measured size
at which the rings' own tables choose differently.

Without ``--write`` the command prints the change. With it, the row's SIRCL
table is written to ``runtime/common/sircl-tuning/<row>.json``, the row
becomes ``source: measured`` with the measured settings (``transport.measured_row``)
and an ``evidence`` text naming the fabrics, runs, image, SIRCL build and
drivers, the table is named in ``tables`` in place of any default table that
serves the same groups, and ``measured_at`` becomes the later of the dates.

The default table names no fabric and no image. A measurement must be of the
SIRCL version and ABI the default table names; a build the default table lists
as ``compatible`` is refused, because the compatibility reason covers the
default rows, not a new measurement. Sessions of a compatible build take the
promoted row's settings; its SIRCL table keeps its key, which names the build
it was measured with, so only sessions of that build take its choices and
sessions of another build keep SIRCL's rules for them. Record the evidence in
the release's performance record before the release cites the row.
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

from runtime.common import fabric_document, image_lock, transport  # noqa: E402

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


def table_bytes(document):
    """A SIRCL table's bytes as the ring harness writes them (``tuning-group<index>.json``)."""
    return (json.dumps(document, indent=1, sort_keys=True) + "\n").encode()


def _check_build(defaults, version, abi):
    named = transport.tuning_builds(defaults)[0]
    if (version, abi) != named:
        raise ValueError(f"The measurement is of SIRCL {version} (ABI {abi}), the default table names {named[0]} "
                         f"(ABI {named[1]}); promote a measurement of the build the default table names")


def _drivers_text(drivers):
    """Each position's GPU driver and kernel, of driver rows ``{"gpu", "kernel"}`` in position order, with
    consecutive positions of one driver and kernel joined: ``positions 0-1 GPU driver D, kernel K; position 2 ...``."""
    runs = []
    for position, row in enumerate(drivers):
        pair = (str(row.get("gpu") or "unrecorded"), str(row.get("kernel") or "unrecorded"))
        if runs and runs[-1][2] == pair:
            runs[-1][1] = position
        else:
            runs.append([position, position, pair])
    return "; ".join(f"{'position' if first == last else 'positions'} {first}{'' if first == last else f'-{last}'} GPU "
                     f"driver {pair[0]}, kernel {pair[1]}" for first, last, pair in runs)


def _place(defaults, row, settings, evidence, data, measured_at, *, root=ROOT):
    """``(new default table, repository path)`` with ``row`` measured and its SIRCL table ``data`` named."""
    identity = transport.table_identity(json.loads(data))
    relative = f"{TABLE_DIRECTORY}/{row}.json"
    value = copy.deepcopy(defaults)
    value["layouts"][row] = {"source": "measured", "settings": dict(settings), "evidence": evidence}
    kept = []
    for entry in value["tables"]:
        if entry["path"] == relative:
            continue
        shipped = json.loads((Path(root) / entry["path"]).read_text(encoding="utf-8"))
        if transport.table_identity(shipped) != identity:
            kept.append(entry)
    value["tables"] = sorted(kept + [{"path": relative, "sha256": hashlib.sha256(data).hexdigest()}],
                             key=lambda entry: entry["path"])
    value["layouts"] = dict(sorted(value["layouts"].items()))
    value["measured_at"] = max(value["measured_at"], measured_at)
    return value, relative


# A measured table of sparkring fabric tune.

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
    from spark_transport.sircl.sparkring_sircl import tuning

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
    _check_build(defaults, measured["sircl"]["version"], measured["sircl"]["abi_version"])
    measured_row = measured["layouts"].get(row)
    if measured_row is None or measured_row["source"] != "measured":
        raise ValueError(f"The measured table has no measured {row} row; it measured "
                         + ", ".join(name for name, value in sorted(measured["layouts"].items())
                                     if value["source"] == "measured"))
    matching = [(digest, data) for digest, data in found.items()
                if transport.table_identity(json.loads(data))[:2] == (shape, world)]
    if len(matching) != 1:
        raise ValueError(f"The measured table holds {len(matching)} SIRCL tables for {row} groups; it needs one")
    _, data = matching[0]
    binding = measured["binding"]
    runs = ", ".join(f"{name} {run}" for name, run in sorted((binding["harness"].get("runs") or {}).items()))
    evidence = (f"sparkring fabric tune on fabric {measured['fabric'][7:19]} with image {binding['image']} (SIRCL "
                f"{measured['sircl']['version']}, ABI {measured['sircl']['abi_version']}), "
                f"{_drivers_text([binding['drivers'][key] for key in sorted(binding['drivers'], key=int)])}, "
                f"{binding['harness'].get('sizes')} sweep, harness "
                f"runs {runs or 'unrecorded'}; table {tuning.document_hash(json.loads(data))}")
    value, relative = _place(defaults, row, measured_row["settings"], evidence, data, measured["measured_at"],
                             root=root)
    return value, data, relative


# Ring-harness tunes of one or more rings.

def ring_tune(directory, row, image_value):
    """One ring's tune (``--ring``), checked: its fabric, drivers, run and table, and the run's measurements."""
    from runtime.host import fabric_tune
    from spark_transport.sircl.sparkring_sircl import tuning
    from spark_transport.sircl.sparkring_sircl.ring import summary

    directory = Path(directory)
    shape, size = transport.row_group(row)
    document = fabric_document.validate(json.loads((directory / "fabric.json").read_text(encoding="utf-8")))
    if (document["shape"], int(document["size"])) != (shape, size):
        raise ValueError(f"{directory}: the fabric is a {document['shape']} of {document['size']}, and the {row} row "
                         f"serves the whole {shape} of {size}")
    drivers = json.loads((directory / "facts.json").read_text(encoding="utf-8"))["positions"]
    if len(drivers) != size:
        raise ValueError(f"{directory}: facts.json lists {len(drivers)} positions, the fabric {size}")
    folders = sorted(path.parent for path in directory.glob("results/*/*/plan.json"))
    if len(folders) != 1:
        raise ValueError(f"{directory}: needs one harness run under results/<run id>/<configuration>/, found "
                         f"{len(folders)}")
    folder = folders[0]
    plan = json.loads((folder / "plan.json").read_text(encoding="utf-8"))
    result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    if result.get("status") != "passed":
        raise ValueError(f"{folder}: the tune run did not pass: " + "; ".join(result.get("problems") or ["no detail"]))
    groups = plan["groups"]
    if len(groups) != 1 or sorted(groups[0]["global_ranks"]) != list(range(size)):
        raise ValueError(f"{folder}: the run measured {len(groups)} groups, not the whole ring of {size}")
    table_path = folder / f"tuning-group{groups[0]['index']}.json"
    if not table_path.is_file():
        raise ValueError(f"{folder}: no {table_path.name}; rebuild it with the harness's tune-table --results")
    table = json.loads(table_path.read_text(encoding="utf-8"))
    tuning.Table(table)
    expected = fabric_tune.expected_facts(document, list(range(size)), image_value)
    differs = [field for field in tuning.KEY_FIELDS if table["key"].get(field) != expected.get(field)]
    if differs or table["key"].get("image") != image_value["image_id"]:
        raise ValueError(f"{table_path}: its key does not match a deployment on the whole ring with image "
                         f"{image_value['name']}: " + "; ".join(
                             [f"{field} {table['key'].get(field)!r}, expected {expected.get(field)!r}" for field in differs]
                             + ([f"image {table['key'].get('image')!r}"] if table["key"].get("image")
                                != image_value["image_id"] else [])))
    index = groups[0]["index"]
    rows = summary.tuning_rows(result).get(index, [])
    session = summary.tune_session(plan, result, index)
    conditions = {"rotate_buffers": int((plan.get("options") or {}).get("rotate_buffers", 1))}
    rebuilt = tuning.build_document(table["key"], rows, session=session, conditions=conditions)
    if (rebuilt["decisions"], rebuilt.get("settings")) != (table["decisions"], table.get("settings")):
        raise ValueError(f"{table_path}: the run's measurements do not rebuild this table")
    sizes = list((plan.get("options") or {}).get("tune_sizes") or ())
    return {"directory": directory, "fabric": document["id"], "drivers": drivers, "run": str(plan.get("run_id", "")),
            "table": table, "rows": rows, "session": session, "conditions": conditions,
            "created": str(table.get("created", "")), "sizes": sizes}


def _row_key(row):
    return row["collective"], row["mode"], int(row["bytes"]), json.dumps(row["choice"], sort_keys=True)


def merge(rings):
    """``(table document, kept, dropped)``: the rings' tables merged, every candidate judged by its slower ring."""
    from spark_transport.sircl.sparkring_sircl import tuning

    first = rings[0]
    for ring in rings[1:]:
        for field, what in (("session", "tune sessions ran under different settings"),
                            ("conditions", "tune sessions rotated different numbers of buffers")):
            if ring[field] != first[field]:
                raise ValueError(f"The rings' {what}: {first[field]} and {ring[field]}")
        if ring["table"]["key"] != first["table"]["key"]:
            raise ValueError(f"The rings' tables have different keys: {first['table']['key']} and {ring['table']['key']}")
    fabrics = [ring["fabric"] for ring in rings]
    if len(set(fabrics)) != len(fabrics):
        raise ValueError("Two --ring directories measured the same fabric")
    measured = [{_row_key(row): row for row in ring["rows"]} for ring in rings]
    common = set(measured[0]).intersection(*measured[1:])
    every = set().union(*measured)
    rows = [dict(measured[0][key], p50_us=max(float(found[key]["p50_us"]) for found in measured))
            for key in sorted(common)]
    created = max(ring["created"] for ring in rings)
    document = tuning.build_document(first["table"]["key"], rows, run_id="+".join(ring["run"] for ring in rings),
                                     created=created, session=first["session"], conditions=first["conditions"])
    return document, len(common), len(every - common)


def differences(rings, document):
    """``[text]``: every measured collective, mode and size at which the rings' own tables choose differently,
    with the merged table's choice there."""
    from spark_transport.sircl.sparkring_sircl import tuning

    tables = [tuning.Table(ring["table"]) for ring in rings]
    merged = tuning.Table(document)
    points = sorted({(row["collective"], row["mode"], int(row["bytes"])) for ring in rings for row in ring["rows"]})
    lines = []
    for collective, mode, size in points:
        chosen = [table.decide(collective, size, mode) for table in tables]
        labels = [choice.label() if choice is not None else "rules" for choice in chosen]
        if len(set(labels)) > 1:
            final = merged.decide(collective, size, mode)
            lines.append(f"{collective} {mode} {size} B: " + ", ".join(
                f"{ring['directory'].name} {label}" for ring, label in zip(rings, labels))
                + f"; merged {final.label() if final is not None else 'rules'}")
    return lines


def promote_rings(defaults, directories, row, image_value, *, root=ROOT):
    """``(new default table, SIRCL table bytes, repository path, difference lines)``: the rings' tunes merged into
    ``row``."""
    from spark_transport.sircl.sparkring_sircl import tuning

    shape, world = row_identity(row)
    sircl = image_lock.sircl(image_value)
    if sircl is None:
        raise ValueError(f"image {image_value.get('name')} carries no SIRCL layer")
    _check_build(defaults, sircl["version"], sircl["abi_version"])
    if not directories:
        raise ValueError("--ring names no tune")
    rings = [ring_tune(directory, row, image_value) for directory in directories]
    document, kept, dropped = merge(rings)
    if transport.table_identity(document)[:2] != (shape, world):
        raise ValueError(f"The merged table serves {transport.table_identity(document)[:2]}, not the {row} groups")
    if not document["decisions"]:
        raise ValueError("No candidate was measured exactly on every ring; the merged table would decide nothing")
    lines = differences(rings, document)
    points = len({(row["collective"], row["mode"], int(row["bytes"])) for ring in rings for row in ring["rows"]})
    data = table_bytes(document)
    settings = transport.measured_row(defaults, row, {}, document)
    sizes = rings[0]["sizes"]
    kind, count = transport.row_group(row)
    sweep = (f"{len(sizes)} per-rank sizes of {sizes[0]} to {sizes[-1]} bytes" if sizes else "sizes unrecorded")
    drivers = "; ".join(f"fabric {ring['fabric'][7:19]}: {_drivers_text(ring['drivers'])}" for ring in rings)
    evidence = (f"SIRCL ring harness tune (python -m sparkring_sircl.ring tune, {sweep}, "
                f"{rings[0]['conditions']['rotate_buffers']} rotated buffers) of the whole {kind} of {count} Sparks on "
                f"{len(rings)} separate rings: fabrics {', '.join(ring['fabric'][7:19] for ring in rings)}, runs "
                f"{', '.join(ring['run'] for ring in rings)}; image {image_value['name']} "
                f"({image_value['image_id'][7:19]}, SIRCL {sircl['version']}, ABI {sircl['abi_version']}), "
                f"each Spark's driver and kernel by position: {drivers}. Each candidate's time at a size is the slower ring's median, the choice the "
                f"fastest of those: {kept} measurements exact on every ring, {dropped} dropped as exact on fewer; the "
                f"rings' own tables choose differently at {len(lines)} of {points} measured points; table "
                f"{tuning.document_hash(document)}")
    measured_at = max(ring["created"][:10] or defaults["measured_at"] for ring in rings)
    value, relative = _place(defaults, row, settings, evidence, data, measured_at, root=root)
    return value, data, relative, lines


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python scripts/promote_sircl_tuning.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--measured", type=Path, help="a measured sparkring-sircl-tuning/v1 table")
    parser.add_argument("--tables", type=Path, help="the directory of its SIRCL tables")
    parser.add_argument("--ring", type=Path, action="append", default=[],
                        help="a ring-harness tune of the row's whole group (repeatable, one per ring)")
    parser.add_argument("--image-lock", type=Path, help="the v3 lock of the image the --ring tunes ran on")
    parser.add_argument("--row", required=True, help="the row to promote: pair, path-<n> or cycle-<n>")
    parser.add_argument("--write", action="store_true", help="write the default table and the SIRCL table")
    args = parser.parse_args(argv)
    if bool(args.ring) == bool(args.measured or args.tables):
        parser.error("name either --measured and --tables, or --ring (with --image-lock)")
    if args.ring and args.image_lock is None:
        parser.error("--ring needs --image-lock")
    if args.measured and args.tables is None:
        parser.error("--measured needs --tables")
    try:
        defaults = transport.load_tuning()
        lines = []
        if args.ring:
            image_value = json.loads(args.image_lock.read_text(encoding="utf-8"))
            value, data, relative, lines = promote_rings(defaults, args.ring, args.row, image_value)
        else:
            measured = json.loads(args.measured.read_text(encoding="utf-8"))
            value, data, relative = promote(defaults, measured, args.row, args.tables)
    except (OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    before = defaults["layouts"].get(args.row)
    print(f"{args.row}: {json.dumps(before, sort_keys=True) if before else 'no row (the shape row applies)'} -> "
          f"{json.dumps(value['layouts'][args.row], sort_keys=True)}")
    for line in lines:
        print(f"rings differ: {line}")
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
