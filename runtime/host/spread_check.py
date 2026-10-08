"""``sudo sparkring fabric spread-check``: spread test files from Node A to every Spark along the cables and time it.

Status: implemented and tested offline (``runtime/host/test_spread.py``). It
is the hardware check of the self-spreading install (``runtime/host/spread.py``)
on any recorded fabric of 2 to 8 Sparks, including layouts that no installer
profile serves yet.

Class: MUTATES HOST. Node A writes ``--files`` files of ``--size`` bytes to
``CHECK_ROOT/source``; their content is fixed by their number, so a repeated
check reuses them. Every other Spark receives them into
``CHECK_ROOT/received`` through the same hop and source programs and the same
executor as an install, and places each file only after its SHA-256 equals
Node A's. The check prints, per direction, when each Spark finished and how
long after the Spark before it, and saves ``spread-check-<time>.json``
(``sparkring-spread-check/v1``) with the fabric reports. Once every Spark holds
every file, both directories are removed on every Spark. A Spark that stops
answering stops the check with the message an install prints; the same
command then resumes. The check refuses to run while a model serves unless
``--while-serving`` is given.
"""
import hashlib
import json
import os
from pathlib import Path
import time

from runtime.common import fabric_layout, installer
from runtime.host import fabric_ssh, fabric_stream, install_assets, spread
from runtime.host.install_errors import NeedsInput

CHECK_ROOT = "/var/lib/sparkring/spread/check"
SCHEMA = "sparkring-spread-check/v1"
MIB = 1024 ** 2
BLOCK = 4 * MIB


class Incomplete(ValueError):
    """A Spark did not receive every test file; the files stay for a repeated check."""


def write_files(root, count, size):
    """``count`` files of ``size`` bytes below ``root`` on this Spark; returns ``{name: [size, sha256]}``.

    File ``i`` repeats a block derived from ``i``, so a repeated check writes
    and pins the same bytes; a file of the right size is hashed, not rewritten.
    """
    os.makedirs(root, mode=0o700, exist_ok=True)
    files = {}
    for index in range(count):
        name = f"check-{index:03d}.bin"
        path = os.path.join(root, name)
        seed = hashlib.sha256(f"sparkring spread check {index}".encode()).digest()
        block = (seed * (BLOCK // len(seed) + 1))[:BLOCK]
        digest = hashlib.sha256()
        if os.path.isfile(path) and os.path.getsize(path) == size:
            with open(path, "rb") as stream:
                while data := stream.read(BLOCK):
                    digest.update(data)
        else:
            with open(path + ".part", "wb") as stream:
                remaining = size
                while remaining:
                    data = block[:min(remaining, BLOCK)]
                    stream.write(data)
                    digest.update(data)
                    remaining -= len(data)
            os.replace(path + ".part", path)
        files[name] = [size, digest.hexdigest()]
    return files


def remove(root, base="/var/lib/sparkring/spread/check"):
    """Remove the check's ``source`` or ``received`` directory below ``base``; runs on each Spark."""
    import os
    import shutil
    if os.path.dirname(root) != base or os.path.basename(root) not in ("source", "received"):
        raise ValueError("Only the spread check's own directories are removed")
    if os.path.lexists(root):
        shutil.rmtree(root)
    return True


def report_lines(outcome, started, document):
    """Per direction, each Spark's finish time after the start and after the Spark before it."""
    lines, rows = [], {}
    for row in outcome["received"]:
        rows.setdefault((row["round"], row["direction"], row["start"]), {})[row["target"]] = row
    for (round_, direction, start), targets in sorted(rows.items()):
        ordered = sorted(targets.values(), key=lambda row: row["hop"])
        parts, before = [], None
        for row in ordered:
            at = row["completed_at"] - started
            gap = "" if before is None else f", +{(row['completed_at'] - before) * 1000:.0f} ms"
            parts.append(f"{row['target']} {at:.2f} s{gap}")
            before = row["completed_at"]
        name = spread.hostname(document, start)
        lines.append(f"  Pass {round_}, {direction} from {name} (position {start}): " + "; ".join(parts))
    return lines


def run(state, *, count, size, allow_serving=False, say=print, transport=None, assets=None, root=CHECK_ROOT,
        now=time.time):
    """Run the check on the cluster recorded in ``state``; returns the ``sparkring-spread-check/v1`` report."""
    from runtime.host import fabric, fabric_bandwidth
    cluster = installer.read(Path(state) / "cluster.json")
    document = spread.recorded(state, cluster)
    if document is None:
        raise ValueError("No fabric document describes this cluster; sudo sparkring setup records it")
    sparks = document["size"]
    if not allow_serving and fabric_bandwidth.serving(state, sparks):
        raise ValueError("A model is serving on this fabric; the spread check would slow it. Stop it first, or add "
                         "--while-serving")
    if transport is None:
        transport = fabric_ssh.Transport(cluster, Path(state) / "bulk-ssh")
        transport.document = document
    current = assets or install_assets.Assets(transport, Path(state) / "spread-check")
    source, received = root + "/source", root + "/received"
    say(f"Write {count} test files of {size / MIB:.0f} MiB on Node A ({source})")
    files = write_files(source, count, size)
    positions = list(range(sparks))
    value = spread.plan(document, [{"asset": "check", "source": 0, "writes": {p: count * size for p in positions[1:]}}])
    for line in spread.describe(value, document):
        say(line)

    def start(rank, listen, peers, names, want):
        return current.launch(rank, fabric_stream.hop_source("directory", {"root": received}, listen, peers, names,
                                                             want, current.pipeline))

    def send(rank, sources, addresses, ports, groups, token):
        program = fabric_stream.push_source(source if rank == 0 else received, sources, addresses, ports, groups,
                                            current.pipeline)
        return current.finish_program(rank, program, token)

    executor = current.spreader({position: position for position in positions}, start=start, send=send)
    started = now()
    outcome = executor.run("check", files, 0, {position: set(files) for position in positions[1:]})
    report = {"schema": SCHEMA, "fabric_id": document["id"], "layout": fabric_layout.name(
        fabric_layout.layout(document["shape"], sparks)), "files": count, "file_bytes": size,
        "started_at": started, "rounds": outcome["rounds"], "received": outcome["received"],
        "remaining": {str(p): sorted(names) for p, names in outcome["remaining"].items()},
        "stopped": sorted(outcome["dead"]), "down": sorted(outcome["down"]),
        "fallbacks": {str(cable): group for cable, group in outcome["fallbacks"].items()},
        "failures": {str(p): text for p, text in outcome["failures"].items()}}
    report["result"] = "complete" if not outcome["remaining"] else "incomplete"
    for line in report_lines(outcome, started, document):
        say(line)
    reports = Path(state) / fabric.REPORTS
    reports.mkdir(parents=True, exist_ok=True)
    path = reports / f"spread-check-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime(now()))}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    say(f"Report: {path}")
    spread.raise_stopped(document, "check", outcome, positions, len(files))
    if outcome["remaining"]:
        position = min(outcome["remaining"])
        reason = outcome["failures"].get(position) or (
            "a down cable keeps it from the fabric" if any(position in group for group in outcome["fallbacks"].values())
            else "the files did not arrive")
        raise Incomplete(f"{spread.hostname(document, position)} (position {position}) did not receive every test "
                         f"file ({reason}); the files stay for a repeated check")
    for rank in positions[1:]:
        current.remote(rank, remove, received, base=root)
    remove(source, base=root)
    say(f"Every Spark holds the {count} test files with Node A's SHA-256; the test files are removed.")
    return report


def main(args, state):
    """``sudo sparkring fabric spread-check``: the exit status (0 complete, 1 stopped or incomplete)."""
    try:
        run(state, count=args.files, size=args.size * MIB, allow_serving=args.while_serving)
    except (NeedsInput, Incomplete) as error:
        print("SparkRing: " + str(error))
        return 1
    return 0
