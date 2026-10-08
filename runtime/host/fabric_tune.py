"""Measure SIRCL's tuning table on this cluster's own fabric: ``sudo sparkring fabric tune``.

Every SIRCL deployment takes its session settings and its measured choices
(per collective, size and mode: the fastest SIRCL algorithm, schedule, piece
and launch grid) from a tuning table (``runtime/common/transport.py``). The
release's default table holds the owner's measurements; this command replaces
it, for this cluster, with measurements on the cluster's own Sparks and
cables. NCCL is not measured: a table chooses only among SIRCL's settings.

The measurement is SIRCL's ring harness (``python -m sparkring_sircl.ring``,
``spark_transport/sircl/RUNBOOK.md``), run as a subprocess from this package.
Its ``tune`` command times every candidate of the all-reduce, all-gather,
reduce-scatter and all-to-all at every per-rank size from 4 KiB to 128 MiB,
eagerly and in CUDA graph replay: the latency cases (one-shot and two-shot
at each launch grid) and the crossover cases (from 256 KiB, two-shot pieces,
tiles and scatter ops, the chain at each piece and the ring at each piece and
stagger). It writes one ``sircl-tuning-table/v1`` per group from the exact
cases; ``tune-table`` rebuilds a table from a run's results.

What is measured: one group of each shape a deployment can use on the
fabric (``layouts``), at the positions starting at 0. A group of ``k``
consecutive Sparks of a larger fabric is a ``pair`` or ``path-<k>``, the
whole cycle is ``cycle-<n>``; a group whose lanes would cross more relays
than SIRCL's qualified limit serves no model and is not listed. A table is
keyed by shape, size, lanes and relays, not by positions, so the deployment
on positions 4-7 of a cycle of eight takes the ``path-4`` table measured on
positions 0-3. The ring harness describes the Sparks as a cycle in cabling
order, so it cannot measure a pair cabled port 0 to port 0, nor the group of
every Spark of a path; those rows stay the default table's.

The command is dry-run first: without ``--execute`` it prints the plan and
contacts nothing. With ``--execute`` it holds Node A's installation lock
(``install.lock``) throughout, so no installation, model start, setup or
automatic recovery runs beside it, and it refuses while a model serves unless
``--stop-serving`` stops the serving models first. Then it:

1. reads each Spark's GPU driver and kernel (``sparkring node tuning-facts``)
   and refuses when they differ between Sparks;
2. verifies the fabric as ``sparkring fabric verify`` does;
3. writes the harness's site file (``sircl-ring-site/v1``) from the cluster
   record, has the harness copy the package's SIRCL sources to every Spark and
   build their native library inside the image (``stage``);
4. per layout: on a cycle, the harness's read-only ``preflight``; then
   ``tune``; then checks the run passed and that its table's key matches the
   deployment group and the image's SIRCL build;
5. copies every measured table to ``/etc/sparkring/fabric/sircl-tuning/`` on
   every Spark (``sparkring node sircl-tuning``) and writes the measured table
   ``/var/lib/sparkring/controller/sircl-tuning.json``.

Each run has a deadline per step: the harness's container wait per layout
(``--layout-timeout``, its rank watchdog ends a rank after 1,500 s), the
stage and preflight limits per Spark, and an optional limit for the whole run
(``--max-hours``) after which layouts not started stay pending. Progress is
kept per binding (``fabric-tune/<binding>/progress.json``): a repeated
command skips the layouts already measured, collects a finished run whose
table was not built, and measures the rest; ``--fresh`` measures every named
layout again.

The harness measures this package's SIRCL sources built inside the image, so
the command refuses when their native and kernel source hashes or SIRCL
version differ from the image's SIRCL layer (its lock's ``tuning_key``):
sessions would not take such a table. The measured table is bound to the
fabric document's identity, the image and its SIRCL build, and each Spark's
GPU driver and kernel; ``transport.tuning_in_effect`` uses it only while they
hold, and ``summary`` tells ``sparkring status`` when they no longer do.

Measured rows also set the session settings the measured choices ran under:
the tune session's link slot (the largest swept piece above SIRCL's 512 KiB
default slot) and its link slot count (``row_settings``). A SIRCL build whose
tables record their own session settings makes that derivation unnecessary.
"""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import threading
import time

from runtime.common import fabric_document, fabric_layout, image_lock, installer, process_lock, transport
from runtime.host import fabric

STATE = fabric.STATE
# The root of Node A's own filesystem, which holds its copies of the measured tables (Node A is position 0).
HOST_ROOT = "/"
WORK = "fabric-tune"
PROGRESS = "progress.json"
PLAN_SCHEMA = "sparkring-fabric-tune-plan/v1"
PROGRESS_SCHEMA = "sparkring-fabric-tune/v1"
RESULT_SCHEMA = "sparkring-fabric-tune-result/v1"
TABLES_SCHEMA = "sparkring-sircl-tables/v1"
FACTS_SCHEMA = "sparkring-tuning-facts/v1"
SITE_SCHEMA = "sircl-ring-site/v1"
HARNESS = "sparkring_sircl.ring"
# The directory of this package that holds the importable ``sparkring_sircl``.
SIRCL_SOURCE = "spark_transport/sircl"
# The harness's working directory on every Spark: staged sources, the native build cache and run files.
REMOTE_DIR = "/var/tmp/sparkring-sircl-ring"
CONTROL_PORT = 29650
LAYOUT_TIMEOUT = 1800
# The harness's own limits: per Spark, copying the sources (120 s) and building the native library (900 s),
# and preflight's reads (about 120 s); after a configuration's container wait, reading the results and
# removing the containers.
STAGE_SECONDS_PER_SPARK = 1020
PREFLIGHT_SECONDS_PER_SPARK = 120
RUN_MARGIN = 300
TABLE_SECONDS = 120
TIMED_OUT = 124
# SIRCL's default link slot (SIRCL_LINK_SLOT_BYTES): pieces above it need a larger slot.
DEFAULT_LINK_SLOT = 512 << 10
PAIR_REASON = ("the ring harness describes the Sparks as a cycle in cabling order, and a pair is cabled port 0 to "
               "port 0")
WHOLE_PATH_REASON = ("the ring harness plans a group of every Spark of its site as a cycle, and a path has no "
                     "cable between its ends")
FAILURES = (ValueError, KeyError, TypeError, OSError, RuntimeError, subprocess.SubprocessError)


def _sircl_tuning():
    from spark_transport.sircl.sparkring_sircl import tuning
    return tuning


def _harness_plan():
    from spark_transport.sircl.sparkring_sircl.ring import plan
    return plan


def _max_relays():
    from spark_transport.sircl.sparkring_sircl.vllm import fabric as sircl_fabric
    return sircl_fabric.DEFAULT_MAX_RELAYS


def _iso(seconds):
    return datetime.datetime.fromtimestamp(float(seconds), datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _stamp(seconds):
    return datetime.datetime.fromtimestamp(float(seconds), datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write(path, text, mode=0o600):
    path = Path(path)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.chmod(mode)
    temporary.replace(path)


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


# What can be measured here.

def layouts(document):
    """Every group shape a deployment can use on the fabric of ``document``.

    Rows ``{"name", "positions", "unmeasurable"}``: the group name the tuning
    table's rows use, the positions the harness measures it on, and why the
    harness cannot measure it (None when it can).
    """
    value = fabric_document.layout(document)
    shape, size = value["shape"], value["size"]
    if shape == fabric_layout.PAIR:
        return [{"name": "pair", "positions": [0, 1], "unmeasurable": PAIR_REASON}]
    largest = _max_relays() + 2
    rows = [{"name": "pair" if count == 2 else f"path-{count}", "positions": list(range(count)), "unmeasurable": None}
            for count in range(2, min(size - 1, largest) + 1)]
    if shape == fabric_layout.CYCLE:
        rows.append({"name": f"cycle-{size}", "positions": list(range(size)), "unmeasurable": None})
    elif size <= largest:
        rows.append({"name": f"path-{size}", "positions": list(range(size)), "unmeasurable": WHOLE_PATH_REASON})
    return rows


def package_key():
    """The SIRCL tuning key of this package's sources, which the harness stages and measures."""
    tuning = _sircl_tuning()
    return {"native": tuning.native_hash(), "kernels": tuning.kernels_hash(), "sircl": tuning.sircl_version()}


def chosen_image(name=None):
    """The image lock ``sparkring install`` uses, or the one ``--image NAME`` names."""
    path = image_lock.lock_path(name) if name else None
    return installer.read(path) if path is not None else image_lock.default()


def lan_choice(cluster, requested=None):
    """``(interface, addresses)``: the interface of the harness's control exchange and each position's address
    on it, or None when the addresses are read from each Spark at the start (``tuning-facts --interface``).

    Without ``requested`` it is the management interface every Spark records, which must have one name.
    """
    hosts = cluster["plan"]["spec"]["hosts"]
    if requested is None:
        names = sorted({host["management_netdev"] for host in hosts})
        if len(names) != 1:
            raise ValueError("The Sparks reach each other on management interfaces of different names ("
                             + ", ".join(names) + "); the ring harness's control exchange uses one interface name on "
                             "every Spark. Name an interface every Spark has with --lan-interface")
        requested = names[0]
    if all(host["management_netdev"] == requested for host in hosts):
        return requested, [host["management_address"] for host in hosts]
    return requested, None


def docker_command(target):
    """The Docker command of an SSH target: ``docker`` for root, else through passwordless sudo, which
    SparkRing's SSH accounts have."""
    return "docker" if target.split("@", 1)[0] == "root" else "sudo -n docker"


def site_document(cluster, document, image_value, *, lan_interface, addresses, control_port=CONTROL_PORT,
                  remote_dir=REMOTE_DIR):
    """The harness's ``sircl-ring-site/v1`` site file: every Spark in position order, which is cabling order."""
    hosts = cluster["plan"]["spec"]["hosts"]
    ring = [{"name": document["positions"][position]["hostname"], "ssh": host["host"],
             "lan_address": addresses[position], "docker": docker_command(host["host"])}
            for position, host in enumerate(hosts)]
    return {"schema": SITE_SCHEMA, "image": image_value["image_id"], "lan_interface": lan_interface,
            "control_port": int(control_port), "remote_dir": remote_dir, "ring": ring}


def check_site(value):
    """Raise ValueError unless the harness accepts the site file ``value``."""
    from spark_transport.sircl.sparkring_sircl.ring.site import Site, SiteError
    try:
        Site.from_json(value)
    except SiteError as error:
        raise ValueError(f"The ring harness refuses the site file made from this cluster: {error}") from None


def expected_facts(document, positions, image_value):
    """The key a SIRCL session of a deployment on ``positions`` checks a table against (``tuning.facts``),
    with the image's SIRCL build."""
    topology = transport.group_topology(transport.sircl_layout(document), positions)
    facts = _sircl_tuning().facts_for_layout(topology.session_layout(), topology.lane_count)
    return {**facts, **image_lock.sircl(image_value)["tuning_key"]}


def blockers(document, image_value, *, key=None):
    """Why nothing can be measured with this image on this fabric; empty when the measurement can run."""
    found = []
    sircl = image_lock.sircl(image_value)
    reason = transport.unavailable(image_value, document)
    if reason:
        found.append(f"SIRCL cannot run here: {reason}")
    elif not fabric_document.default_names(document):
        found.append("the Sparks name their fabric functions otherwise than DGX OS does, and the ring harness's "
                     "preflight and route maps use the DGX OS names")
    if sircl is not None:
        key = key or package_key()
        if key != sircl["tuning_key"]:
            found.append(f"this package's SIRCL sources ({key['sircl']}, native {key['native']}, kernels "
                         f"{key['kernels']}) differ from image {image_value['name']}'s SIRCL layer "
                         f"({sircl['tuning_key']['sircl']}, native {sircl['tuning_key']['native']}, kernels "
                         f"{sircl['tuning_key']['kernels']}). The harness measures the package's sources inside the "
                         "image, and the image's sessions would not take such a table: install the SparkRing package "
                         "the image was built with, or name its image with --image")
    if not any(row["unmeasurable"] is None for row in layouts(document)):
        found.append("the ring harness can measure no group of this fabric: "
                     + "; ".join(f"{row['name']}: {row['unmeasurable']}" for row in layouts(document)))
    return found


# The plan.

def plan(cluster, document, image_value, *, names=None, quick=False, lan_interface=None, control_port=CONTROL_PORT,
         layout_timeout=LAYOUT_TIMEOUT, max_hours=None, busy=None, key=None, state=STATE):
    """The ``sparkring-fabric-tune-plan/v1`` document of a measurement: what runs where, its limits and blockers."""
    rows = layouts(document)
    measurable = [row for row in rows if row["unmeasurable"] is None]
    if names:
        known = {row["name"]: row for row in rows}
        unknown = [name for name in names if name not in known]
        if unknown:
            raise ValueError(f"--layouts {','.join(unknown)}: this fabric's group shapes are "
                             + ", ".join(known))
        refused = [f"{name}: {known[name]['unmeasurable']}" for name in names if known[name]["unmeasurable"]]
        if refused:
            raise ValueError("The ring harness cannot measure " + "; ".join(refused))
        selected = [row for row in measurable if row["name"] in names]
    else:
        selected = measurable
    found = blockers(document, image_value, key=key)
    try:
        interface, addresses = lan_choice(cluster, lan_interface)
    except ValueError as error:
        interface, addresses = lan_interface, None
        found.append(str(error))
    size = document["size"]
    cycle = document["shape"] == fabric_layout.CYCLE
    used = sorted({position for row in selected for position in row["positions"]})
    stage = STAGE_SECONDS_PER_SPARK * len(used) + 60
    per_layout = {row["name"]: (PREFLIGHT_SECONDS_PER_SPARK * len(row["positions"]) + 60 if cycle else 0)
                  + layout_timeout + RUN_MARGIN + TABLE_SECONDS for row in selected}
    harness_plan = _harness_plan()
    sizes = harness_plan.TUNE_QUICK_SIZES if quick else harness_plan.TUNE_SIZES
    options = harness_plan.Options()
    return {
        "schema": PLAN_SCHEMA, "fabric": {"id": document["id"], "layout": fabric_layout.name(fabric_document.layout(document)),
                                          "size": size},
        "image": {"name": image_value["name"], "image_id": image_value["image_id"],
                  "sircl": (image_lock.sircl(image_value) or {}).get("tuning_key")},
        "layouts": [dict(row, measure=row in selected) for row in rows],
        "sweep": {"sizes": "quick" if quick else "full", "bytes": list(sizes), "grids": list(options.tune_grids),
                  "pieces": list(options.tune_pieces), "staggers": list(options.tune_staggers),
                  "collectives": list(options.tune_collectives), "modes": list(options.tune_modes)},
        "harness": {"lan_interface": interface, "addresses": "recorded" if addresses is not None else "read at start",
                    "control_port": int(control_port), "remote_dir": REMOTE_DIR, "preflight": cycle,
                    "positions": used},
        "limits": {"stage_seconds": stage, "layout_seconds": per_layout, "layout_timeout": layout_timeout,
                   "worst_case_seconds": stage + sum(per_layout.values()), "max_hours": max_hours},
        "serving": [dict(row, rank=rank) for rank, row in sorted((busy or {}).items())],
        "table": str(Path(state) / transport.MEASURED_TUNING),
        "blockers": found,
    }


def _hours(seconds):
    return f"{seconds / 3600:.1f} h" if seconds >= 5400 else f"{round(seconds / 60)} min"


def _bytes(value):
    for unit, size in (("MiB", 1 << 20), ("KiB", 1 << 10)):
        if value >= size and value % size == 0:
            return f"{value // size} {unit}"
    return f"{value} B"


def plan_lines(value, progress=None):
    """What ``sparkring fabric tune`` prints before it measures."""
    image = value["image"]
    sweep = value["sweep"]
    lines = [f"Fabric: {value['fabric']['layout']} (fabric {value['fabric']['id'][7:19]}); image {image['name']}"
             + (f" (SIRCL {image['sircl']['sircl']})" if image["sircl"] else "")]
    measured = [row for row in value["layouts"] if row["measure"]]
    if measured:
        lines.append("Measures one group of each shape, one after another, with SIRCL's ring harness "
                     "(python -m sparkring_sircl.ring tune):")
        binding = (progress or {}).get("binding") or {}
        # The last run's progress applies when it measured this fabric with this image and sweep; a driver
        # change since then starts the measurement afresh.
        same = (binding.get("fabric"), binding.get("image_id"), binding.get("sizes")) == (
            value["fabric"]["id"], value["image"]["image_id"], sweep["sizes"])
        recorded = progress["layouts"] if same else {}
        for row in measured:
            done = recorded.get(row["name"], {})
            where = f"Sparks {row['positions'][0]}-{row['positions'][-1]}"
            note = (f"; measured {done['measured_at'][:10]}, kept while the drivers are unchanged"
                    if done.get("state") == "measured" else f"; last run {done['state']}" if done.get("state") else "")
            lines.append(f"  {row['name']:<8} {where}{note}")
    skipped = [row for row in value["layouts"] if row["unmeasurable"]]
    for row in skipped:
        lines.append(f"Not measured: {row['name']}, because {row['unmeasurable']}; its row stays the default table's.")
    others = [row["name"] for row in value["layouts"] if not row["measure"] and not row["unmeasurable"]]
    if others:
        lines.append("Not named by --layouts: " + ", ".join(others) + "; their rows stay as recorded.")
    lines.append(f"Cases: {', '.join(sweep['collectives'])}; {_bytes(sweep['bytes'][0])} to {_bytes(sweep['bytes'][-1])} "
                 f"per rank ({len(sweep['bytes'])} sizes{', --quick' if sweep['sizes'] == 'quick' else ''}), eager "
                 "and in CUDA graph replay. Latency: one-shot and two-shot at launch grids "
                 f"{', '.join(map(str, sweep['grids']))}. Crossover from 256 KiB: two-shot pieces, tiles and scatter "
                 f"ops, the chain at pieces {', '.join(_bytes(p) for p in sweep['pieces'])}, the ring at each piece "
                 f"and stagger {', '.join(map(str, sweep['staggers']))}. NCCL is not measured.")
    limits = value["limits"]
    harness = value["harness"]
    lines.append(f"Limits: staging the SIRCL sources and building the native library at most "
                 f"{_hours(limits['stage_seconds'])}; each group at most {_hours(limits['layout_timeout'])} of "
                 f"measurement; worst case {_hours(limits['worst_case_seconds'])}"
                 + (f", stopped after {limits['max_hours']:g} h with the rest pending" if limits["max_hours"] else "")
                 + ".")
    lines.append(f"Host changes: one container per rank on GPU 0 of Sparks "
                 f"{', '.join(map(str, harness['positions']))}, files under {harness['remote_dir']} on them, the "
                 f"measured tables under {transport.HOST_TABLES}/ on every Spark, and {value['table']} on Node A.")
    lines.append(f"Needs image {image['name']} on every measured Spark; sudo sparkring install places it there.")
    lines.append(f"Control exchange: {harness['lan_interface']} (addresses {harness['addresses']}), TCP port "
                 f"{harness['control_port']}."
                 + ("" if harness["preflight"] else " On a path the harness's preflight would report the free ports "
                                                     "at its ends, so sparkring's fabric verification stands for it."))
    if value["serving"]:
        ranks = {}
        for row in value["serving"]:
            ranks.setdefault(row["profile"], []).append(str(row["rank"]))
        lines.append("Serving: " + "; ".join(f"{profile} on Sparks {', '.join(found)}" for profile, found in
                                             sorted(ranks.items()))
                     + ". The measurement does not run beside a model; --stop-serving stops it first.")
    lines.append("The result binds this fabric, this image and its SIRCL build, and each Spark's GPU driver and "
                 "kernel; installations made afterwards use it while they hold.")
    for blocker in value["blockers"]:
        lines.append("BLOCKER: " + blocker)
    return lines


# Facts of each Spark (sparkring node tuning-facts).

def _output(argv, run):
    try:
        done = run(argv, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def _ipv4(interface, run):
    text = _output(["ip", "-j", "-4", "address", "show", "dev", interface], run)
    try:
        for link in json.loads(text or "[]"):
            for entry in link.get("addr_info") or []:
                if entry.get("family") == "inet" and entry.get("local"):
                    return entry["local"]
    except (ValueError, AttributeError, TypeError):
        return None
    return None


def local_facts(interface=None, *, run=subprocess.run):
    """This Spark's facts that a measured table binds: its GPU driver and kernel release, and with
    ``interface`` that interface's IPv4 address."""
    gpu = _output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], run)
    value = {"schema": FACTS_SCHEMA, "gpu": gpu.splitlines()[0].strip() if gpu else None,
             "kernel": platform.release() or None}
    if interface:
        value["interface"] = {"name": interface, "address": _ipv4(interface, run)}
    return value


def drivers_of(facts):
    """``{"<position>": {"gpu", "kernel"}}`` of the Sparks' facts."""
    return {str(position): {field: row.get(field) for field in transport.DRIVER_FIELDS}
            for position, row in enumerate(facts)}


def read_facts(access, size, interface=None):
    argv = ["tuning-facts"] + (["--interface", interface] if interface else [])

    def one(rank):
        return json.loads(access.node(rank, argv))
    with concurrent.futures.ThreadPoolExecutor(max_workers=size) as pool:
        return list(pool.map(one, range(size)))


def uniform_drivers(facts):
    """Raise ValueError when the Sparks report different GPU drivers or kernels."""
    seen = {(row.get("gpu"), row.get("kernel")) for row in facts}
    if len(seen) > 1:
        raise ValueError("The Sparks run different GPU drivers or kernels ("
                         + "; ".join(f"position {position}: {row.get('gpu')}, {row.get('kernel')}"
                                     for position, row in enumerate(facts))
                         + "). A measured table binds one software state: update them alike, then measure")


# Each Spark's copies of the measured tables (sparkring node sircl-tuning).

def install_tables(text, *, root="/"):
    """Write the measured SIRCL tables Node A sends as this Spark's copies under ``HOST_TABLES``.

    Each table is named by the SHA-256 of its bytes and written once; a copy
    whose bytes differ from its name is replaced by the sent bytes.
    """
    from runtime.host import node
    value = json.loads(text)
    if not isinstance(value, dict) or value.get("schema") != TABLES_SCHEMA or not isinstance(value.get("tables"), list):
        raise ValueError(f"Expected a {TABLES_SCHEMA} document of tables")
    directory = node.location(root, transport.HOST_TABLES)
    directory.mkdir(parents=True, exist_ok=True, mode=0o755)
    written, present = [], []
    for entry in value["tables"]:
        data = str(entry["text"]).encode("utf-8")
        digest = _sha256(data)
        if digest != entry["sha256"]:
            raise ValueError(f"Table {entry['sha256'][:12]} arrived with other bytes")
        _sircl_tuning().Table(json.loads(data))
        path = directory / f"{digest}.json"
        if path.is_file() and _sha256(path.read_bytes()) == digest:
            present.append(digest)
            continue
        temporary = path.with_name(path.name + ".writing")
        temporary.write_bytes(data)
        temporary.chmod(0o644)
        temporary.replace(path)
        written.append(digest)
    return {"directory": str(directory), "written": written, "present": present, "sha256": sorted(written + present)}


def distribute(access, size, tables, *, say=print):
    """Copy ``tables`` (SHA-256 to bytes) to every Spark and confirm each copy."""
    payload = json.dumps({"schema": TABLES_SCHEMA, "tables": [{"sha256": digest, "text": tables[digest].decode("utf-8")}
                                                             for digest in sorted(tables)]})

    def one(rank):
        return json.loads(access.node(rank, ["sircl-tuning"], data=payload))
    with concurrent.futures.ThreadPoolExecutor(max_workers=size) as pool:
        confirmed = list(pool.map(one, range(size)))
    for rank, result in enumerate(confirmed):
        if sorted(result.get("sha256") or []) != sorted(tables):
            raise ValueError(f"Position {rank} did not confirm its copies of the measured tuning tables")
    say(f"Measured tuning tables on {size} Sparks: {', '.join(digest[:12] for digest in sorted(tables))}")


# Serving models.

def serving(state, size):
    from runtime.host import fabric_bandwidth
    return fabric_bandwidth.serving(state, size)


def refusal(busy):
    models = sorted({(row["profile"], tuple(row["placement"]) if row["placement"] else None) for row in busy.values()},
                    key=str)
    from runtime.host import placement
    where = "; ".join(f"{profile} serves on " + ("every Spark" if slot is None else placement.text(slot))
                      for profile, slot in models)
    return (f"{where}. The measurement runs containers on GPU 0 of every measured Spark and fills the fabric, so it "
            "never runs beside a model. Stop it first (sudo sparkring down --execute), or add --stop-serving to stop "
            "it as part of this command")


def stop_models(state, size, *, say=print, apply=None):
    """Stop every serving deployment as ``sparkring down --execute`` does; the caller holds ``install.lock``."""
    from runtime.host import placement, recovery, retained_source
    apply = apply or retained_source.apply
    stopped = []
    for slot, directory in placement.actives(state).items():
        if placement.stopped(directory):
            continue
        profile = placement.profile_of(directory)
        say(f"Stop {profile} on {placement.text(slot)} for the measurement")
        recovery.forget_attempt(directory)
        apply(directory, "down", cache=Path(state) / "retained-sources")
        if not placement.stopped(directory):
            raise ValueError(f"{profile} did not stop; sparkring status names its state. Nothing was measured")
        stopped.append({"profile": profile, "placement": list(slot) if slot else None, "deployment": str(directory)})
    return stopped


# The harness.

def run_harness(argv, *, cwd, timeout, log):
    """Run ``python -m sparkring_sircl.ring ARGV`` from this package within ``timeout`` seconds, its output
    appended to ``log``; returns its exit code, or ``TIMED_OUT`` after ending it at the deadline."""
    env = dict(os.environ, PYTHONPATH=str(installer.ROOT / SIRCL_SOURCE), PYTHONDONTWRITEBYTECODE="1")
    command = [sys.executable, "-m", HARNESS, *argv]
    with open(log, "a", encoding="utf-8") as stream:
        stream.write("$ " + shlex.join(command) + "\n")
        stream.flush()
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT)
        ended = threading.Event()

        def end():
            ended.set()
            process.kill()
        timer = threading.Timer(timeout, end)
        timer.start()
        try:
            code = process.wait()
        finally:
            timer.cancel()
    return TIMED_OUT if ended.is_set() else code


def _tail(log, lines=12):
    try:
        return Path(log).read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    except OSError:
        return []


def row_settings(table_document, options, world):
    """The session settings a measured row sets so that its table's link choices run as they were measured.

    The tune session ran with a link slot that holds the largest swept piece
    when that exceeds SIRCL's default slot, and with link slots for the
    largest swept stagger, at least the harness's ``TUNE_LINK_SLOTS``
    (``ring/worker.py``). A table that chooses no chain or ring candidate
    needs neither.
    """
    from spark_transport.sircl.sparkring_sircl import protocol
    links = [interval["choice"] for entry in table_document.get("decisions", ())
             for interval in entry["intervals"] if interval["choice"].get("schedule") in ("chain", "ring")]
    if not links:
        return {}
    defaults = _harness_plan().Options()
    pieces = options.get("tune_pieces") or defaults.tune_pieces
    staggers = options.get("tune_staggers") or defaults.tune_staggers
    settings = {"link_slots": max([protocol.ring_stagger_slots(world, stagger) for stagger in staggers]
                                  + [_harness_plan().TUNE_LINK_SLOTS])}
    if max(pieces) > DEFAULT_LINK_SLOT:
        settings["link_slot"] = int(max(pieces))
    return settings


class Measurement:
    """One ``--execute`` run: the work directory of its binding, its progress and its harness calls."""

    def __init__(self, state, cluster, document, image_value, value, *, access, harness, say, now, quick, fresh,
                 layout_timeout, max_hours):
        self.state, self.cluster, self.document, self.image = Path(state), cluster, document, image_value
        self.plan, self.access, self.harness, self.say, self.now = value, access, harness, say, now
        self.quick, self.fresh, self.layout_timeout = quick, fresh, layout_timeout
        self.deadline = now() + max_hours * 3600 if max_hours else None
        self.size = document["size"]
        self.cycle = document["shape"] == fabric_layout.CYCLE

    # The binding and its progress.

    def bind(self, facts, addresses):
        sircl = image_lock.sircl(self.image)
        self.binding = {"fabric": self.document["id"], "image_id": self.image["image_id"],
                        "tuning_key": sircl["tuning_key"], "drivers": drivers_of(facts),
                        "sizes": "quick" if self.quick else "full"}
        identity = _sha256(json.dumps(self.binding, sort_keys=True).encode())[:16]
        self.work = self.state / WORK / identity
        self.work.mkdir(parents=True, exist_ok=True, mode=0o700)
        (self.work / "tables").mkdir(exist_ok=True, mode=0o700)
        self.log = self.work / "harness.log"
        harness = self.plan["harness"]
        site = site_document(self.cluster, self.document, self.image, lan_interface=harness["lan_interface"],
                             addresses=addresses, control_port=harness["control_port"])
        check_site(site)
        self.site = self.work / "site.json"
        _write(self.site, json.dumps(site, indent=2) + "\n")
        path = self.work / PROGRESS
        progress = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        if not isinstance(progress, dict) or progress.get("schema") != PROGRESS_SCHEMA:
            progress = {"schema": PROGRESS_SCHEMA, "binding": self.binding, "layouts": {}}
        self.progress = progress
        self.save()
        return identity

    def save(self):
        self.progress["updated_at"] = _iso(self.now())
        _write(self.work / PROGRESS, json.dumps(self.progress, indent=2, sort_keys=True) + "\n")

    def call(self, argv, timeout):
        return self.harness([*argv], cwd=str(self.work), timeout=timeout, log=str(self.log))

    # Steps.

    def verify(self):
        """The fabric verification of ``sparkring fabric verify``, saved with the others."""
        report = fabric.gather(self.document, self.cluster["plan"], access=self.access, say=self.say)
        name = fabric.save_report(self.state, report, now=self.now)
        if report["result"] != "healthy":
            raise ValueError("\n".join(fabric.report_lines(report)) + f"\nNothing was measured (report {name})")
        self.say(f"Fabric verified before the measurement ({name})")

    def stage(self, rows):
        used = sorted({position for row in rows for position in row["positions"]})
        groups = f"{used[0]}-{used[-1]}"
        run = f"{_stamp(self.now())}-stage"
        self.say(f"Stage this package's SIRCL sources on Sparks {groups} and build their native library in the image")
        code = self.call(["stage", "--site", str(self.site), "--groups", groups, "--name", "stage", "--run-id", run],
                         STAGE_SECONDS_PER_SPARK * len(used) + 60)
        if code:
            raise ValueError("The ring harness could not stage the SIRCL sources: "
                             + " | ".join(_tail(self.log, 4)) + f" (log {self.log}). Nothing was measured")

    def pending(self, rows):
        """The layouts of ``rows`` that still need a measurement or a table."""
        def kept(entry):
            table = entry.get("table") or {}
            return (entry.get("state") == "measured" and bool(table.get("sha256"))
                    and (self.work / "tables" / f"{table['sha256']}.json").is_file())
        return [row for row in rows if self.fresh or not kept(self.progress["layouts"].get(row["name"], {}))]

    def collectable(self, row):
        """Whether an earlier command saw this layout's harness run exit 0 and did not record its table."""
        earlier = self.progress["layouts"].get(row["name"], {})
        return (not self.fresh and earlier.get("state") in ("running", "measured") and earlier.get("exit") == 0
                and bool(earlier.get("results")) and (self.work / earlier["results"] / "result.json").is_file())

    def measure(self, row):
        """Measure one layout, or collect the run an earlier command finished; returns its progress entry."""
        name = row["name"]
        groups = f"{row['positions'][0]}-{row['positions'][-1]}"
        if self.collectable(row):
            earlier = self.progress["layouts"][name]
            self.say(f"{name}: collect the finished run {earlier['run']}")
            return self.collect(row, earlier)
        run = f"{_stamp(self.now())}-{name}"
        entry = {"state": "running", "run": run, "positions": row["positions"], "results": f"results/{run}/{name}",
                 "started_at": _iso(self.now())}
        self.progress["layouts"][name] = entry
        self.save()
        if self.cycle:
            code = self.call(["preflight", "--site", str(self.site), "--groups", groups, "--name", name],
                             PREFLIGHT_SECONDS_PER_SPARK * len(row["positions"]) + 60)
            if code:
                return self.fail(name, "the ring harness's preflight found blockers: "
                                 + " | ".join(line for line in _tail(self.log, 20) if "BLOCKER" in line)[:600])
        self.say(f"{name}: measure on Sparks {groups} (at most {_hours(self.layout_timeout)}; log {self.log})")
        argv = ["tune", "--site", str(self.site), "--groups", groups, "--name", name, "--run-id", run,
                "--output", str(self.work / "results"), "--timeout", str(self.layout_timeout)]
        code = self.call(argv + (["--quick"] if self.quick else []), self.layout_timeout + RUN_MARGIN)
        entry["exit"] = code
        self.save()
        if code == TIMED_OUT:
            self.call(["cleanup", "--site", str(self.site)], 60 * self.size)
            return self.fail(name, f"the harness did not finish within {_hours(self.layout_timeout + RUN_MARGIN)}; its "
                                   "containers were removed")
        if code:
            return self.fail(name, f"the ring harness's tune run exited with {code}: "
                             + " | ".join(_tail(self.log, 4))[:600])
        return self.collect(row, entry)

    def collect(self, row, entry):
        name = row["name"]
        folder = self.work / entry["results"]
        try:
            merged = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return self.fail(name, "the harness wrote no result: " + " | ".join(_tail(self.log, 4)))
        if merged.get("status") != "passed":
            return self.fail(name, "the harness run failed: " + "; ".join(merged.get("problems") or ["no detail"])[:600])
        table_path = folder / "tuning-group0.json"
        if not table_path.is_file():
            code = self.call(["tune-table", "--results", str(folder)], TABLE_SECONDS)
            if code or not table_path.is_file():
                return self.fail(name, "the run has no exact tune measurement, so no tuning table")
        data = table_path.read_bytes()
        table_document = json.loads(data)
        table = _sircl_tuning().Table(table_document)
        expected = expected_facts(self.document, row["positions"], self.image)
        mismatches = table.mismatches(expected)
        if table_document["key"].get("image") != self.image["image_id"]:
            mismatches.append(f"image: table {table_document['key'].get('image')!r}, here {self.image['image_id']!r}")
        if mismatches:
            return self.fail(name, "its table would not match the deployment group: " + "; ".join(mismatches))
        digest = _sha256(data)
        stored = self.work / "tables" / f"{digest}.json"
        if not stored.is_file():
            stored.write_bytes(data)
        options = json.loads((folder / "plan.json").read_text(encoding="utf-8"))["options"]
        finished = {**entry, "state": "measured", "table": {"sha256": digest, "hash": table.hash},
                    "settings": row_settings(table_document, options, len(row["positions"])),
                    "measured_at": _iso(self.now())}
        finished.pop("reason", None)
        self.progress["layouts"][name] = finished
        self.save()
        self.say(f"{name}: measured; table {table.hash} ({len(table_document['decisions'])} decisions)")
        return finished

    def fail(self, name, reason):
        entry = {**self.progress["layouts"].get(name, {}), "state": "failed", "reason": reason}
        self.progress["layouts"][name] = entry
        self.save()
        self.say(f"{name}: not measured: {reason}")
        return entry

    def publish(self):
        """Copy every measured table to every Spark, then write the measured table on Node A; None when no
        layout is measured."""
        measured = {name: entry for name, entry in self.progress["layouts"].items() if entry.get("state") == "measured"}
        if not measured:
            return None
        tables, documents = {}, {}
        for entry in measured.values():
            digest = entry["table"]["sha256"]
            data = (self.work / "tables" / f"{digest}.json").read_bytes()
            if _sha256(data) != digest:
                raise ValueError(f"The kept table {digest[:12]} differs from its SHA-256; measure again with --fresh")
            tables[digest], documents[digest] = data, json.loads(data)
        distribute(self.access, self.size, tables, say=self.say)
        binding = {"image": self.image["name"], "tuning_key": self.binding["tuning_key"],
                   "drivers": self.binding["drivers"],
                   "harness": {"sizes": self.binding["sizes"],
                               "runs": {name: entry["run"] for name, entry in sorted(measured.items())}}}
        value = transport.measured_document(
            transport.load_tuning(), {name: entry["settings"] for name, entry in measured.items()}, documents,
            fabric=self.document["id"], image_value=self.image,
            measured_at=max(entry["measured_at"] for entry in measured.values())[:10], binding=binding)
        return value


def execute(state, cluster, document, image_value, value, *, access, harness=run_harness, say=print, now=time.time,
            stop_serving=False, quick=False, fresh=False, layout_timeout=LAYOUT_TIMEOUT, max_hours=None,
            host_root="/"):
    """Measure, under ``install.lock``; returns the ``sparkring-fabric-tune-result/v1`` document."""
    if value["blockers"]:
        raise ValueError("; ".join(value["blockers"]))
    state = Path(state)
    rows = [row for row in value["layouts"] if row["measure"]]
    with process_lock.hold(state / "install.lock"):
        busy = serving(state, document["size"])
        stopped = []
        if busy:
            if not stop_serving:
                raise ValueError(refusal(busy))
            stopped = stop_models(state, document["size"], say=say)
            if serving(state, document["size"]):
                raise ValueError("A model still serves after --stop-serving; nothing was measured")
        measurement = Measurement(state, cluster, document, image_value, value, access=access, harness=harness,
                                  say=say, now=now, quick=quick, fresh=fresh, layout_timeout=layout_timeout,
                                  max_hours=max_hours)
        interface = value["harness"]["lan_interface"]
        _, addresses = lan_choice(cluster, interface)
        facts = read_facts(access, document["size"], None if addresses is not None else interface)
        uniform_drivers(facts)
        if addresses is None:
            addresses = [(row.get("interface") or {}).get("address") for row in facts]
            missing = [str(position) for position, address in enumerate(addresses) if not address]
            if missing:
                raise ValueError(f"{interface} has no IPv4 address on position(s) {', '.join(missing)}; name an "
                                 "interface every Spark has with --lan-interface")
        identity = measurement.bind(facts, addresses)
        say(f"Progress: {measurement.work / PROGRESS}")
        todo = measurement.pending(rows)
        for row in rows:
            if row not in todo:
                entry = measurement.progress["layouts"][row["name"]]
                say(f"{row['name']}: measured {entry['measured_at'][:10]} (table {entry['table']['hash']}); kept")
        runs = [row for row in todo if not measurement.collectable(row)]
        if runs:
            measurement.verify()
            measurement.stage(runs)
        for row in todo:
            limit = value["limits"]["layout_seconds"][row["name"]]
            if measurement.deadline is not None and now() + limit > measurement.deadline:
                entry = {**measurement.progress["layouts"].get(row["name"], {}), "state": "pending",
                         "reason": f"not started within --max-hours {max_hours:g}"}
                measurement.progress["layouts"][row["name"]] = entry
                measurement.save()
                say(f"{row['name']}: pending; the next sudo sparkring fabric tune --execute measures it")
                continue
            measurement.measure(row)
        table = measurement.publish()
        written = None
        if table is not None:
            transport.validate_tuning(table, host_root=host_root)
            _write(state / transport.MEASURED_TUNING, transport.encoded(table), 0o644)
            written = {"path": str(state / transport.MEASURED_TUNING), "sha256": transport.tuning_digest(table),
                       "measured_at": table["measured_at"],
                       "rows": sorted(name for name, row in table["layouts"].items() if row["source"] == "measured")}
    layouts_ = {row["name"]: {key: measurement.progress["layouts"].get(row["name"], {}).get(key)
                              for key in ("state", "reason", "run", "measured_at")} for row in rows}
    for name in layouts_:
        entry = measurement.progress["layouts"].get(name, {})
        if entry.get("table"):
            layouts_[name]["table"] = entry["table"]
    return {"schema": RESULT_SCHEMA, "binding": identity, "work": str(measurement.work), "layouts": layouts_,
            "table": written, "stopped": stopped,
            "complete": all(entry["state"] == "measured" for entry in layouts_.values())}


def result_lines(result):
    lines = []
    for name, entry in result["layouts"].items():
        text = {"measured": f"measured, table {(entry.get('table') or {}).get('hash')}",
                "failed": f"failed: {entry.get('reason')}", "pending": f"pending: {entry.get('reason')}"}
        lines.append(f"  {name}: " + text.get(entry["state"], str(entry["state"])))
    table = result["table"]
    if table:
        lines.append(f"Measured tuning table: {table['path']} (sha256 {table['sha256'][:12]}, measured "
                     f"{table['measured_at']}; rows {', '.join(table['rows'])}).")
        lines.append("Installations made from now on use it while this fabric, image and the Sparks' drivers stay; "
                     "a deployment made before keeps the table it recorded, and sudo sparkring install makes one on "
                     "the measured table.")
    else:
        lines.append("No layout was measured; installations keep the default table.")
    for row in result["stopped"]:
        lines.append(f"Stopped for the measurement: {row['profile']}; sudo sparkring up starts it again on the "
                     "table it recorded, sudo sparkring install installs it on the measured table.")
    if not result["complete"]:
        lines.append(f"Repeat sudo sparkring fabric tune --execute to measure the rest; progress in {result['work']}.")
    return lines


def redistribute(state, document, *, access, say=print, host_root="/"):
    """``--distribute``: copy the recorded measured table's SIRCL tables to every Spark again."""
    path = Path(state) / transport.MEASURED_TUNING
    if not path.exists():
        raise ValueError("No measured tuning table is recorded; sudo sparkring fabric tune --execute measures one")
    value = json.loads(path.read_text(encoding="utf-8"))
    tables = {}
    for entry in value.get("tables") or []:
        if not entry["path"].startswith(transport.HOST_TABLES + "/"):
            continue
        candidates = [Path(host_root) / entry["path"].lstrip("/"),
                      *sorted((Path(state) / WORK).glob(f"*/tables/{entry['sha256']}.json"))]
        for candidate in candidates:
            if candidate.is_file() and _sha256(candidate.read_bytes()) == entry["sha256"]:
                tables[entry["sha256"]] = candidate.read_bytes()
                break
        else:
            raise ValueError(f"Node A holds no copy of table {entry['sha256'][:12]}; sudo sparkring fabric tune "
                             "--execute --fresh measures again")
    with process_lock.hold(Path(state) / "install.lock"):
        distribute(access, document["size"], tables, say=say)
    transport.validate_tuning(value, host_root=host_root)
    return {"tables": sorted(tables)}


# sparkring status.

def latest_progress(state):
    """The progress record of the most recent run, or None."""
    found = []
    for path in sorted((Path(state) / WORK).glob(f"*/{PROGRESS}")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(value, dict) and value.get("schema") == PROGRESS_SCHEMA:
            found.append(value)
    return max(found, key=lambda value: value.get("updated_at") or "", default=None)


def bound_image(measured):
    """The lock of the image a measured table names, from this package's catalog; None when it lists none."""
    for row in image_lock.catalog():
        if row["lock"].get("image_id") == measured["image_id"]:
            return row["lock"]
    return None


def summary(state, *, facts=None, host_root=None):
    """What ``sparkring status`` says about SIRCL tuning: the table installations use and why."""
    state = Path(state)
    value = {"state": "default"}
    progress = latest_progress(state)
    if progress is not None:
        value["last_run"] = {name: {"state": entry.get("state"), "reason": entry.get("reason")}
                             for name, entry in sorted(progress["layouts"].items())}
    path = state / transport.MEASURED_TUNING
    if not path.exists():
        return value
    try:
        measured = transport.load_tuning(path, host_root=host_root or HOST_ROOT)
    except transport.TransportError as error:
        return dict(value, state="unreadable", error=str(error))
    try:
        document = fabric.read_document(state)
    except (OSError, ValueError):
        document = None
    image_value = bound_image(measured)
    problems = transport.measured_problems(measured, document,
                                           image_value or {"image_id": measured["image_id"]},
                                           drivers={0: facts or local_facts()})
    if image_value is None:
        problems.append(f"this package no longer lists image {measured['binding']['image']}")
    try:
        default = image_lock.default()
    except (OSError, ValueError, KeyError):
        default = {}
    value.update(state="stale" if problems else "measured", problems=problems, measured_at=measured["measured_at"],
                 sha256=transport.tuning_digest(measured), image=measured["binding"]["image"],
                 rows=sorted(name for name, row in measured["layouts"].items() if row["source"] == "measured"),
                 default_image=default.get("name"), covers_default=default.get("image_id") == measured["image_id"])
    return value


def status_line(value):
    """The ``SIRCL tuning:`` line of ``sparkring status``."""
    failed = [name for name, entry in (value.get("last_run") or {}).items() if entry["state"] in ("failed", "pending")]
    tail = f"; last measurement left {', '.join(failed)} unmeasured" if failed else ""
    if value["state"] == "default":
        return "SIRCL tuning: the default table; sudo sparkring fabric tune measures this fabric" + tail
    if value["state"] == "unreadable":
        return (f"SIRCL tuning: the measured table cannot be used ({value['error']}); installations use the default "
                "table")
    if value["state"] == "stale":
        return (f"SIRCL tuning: the table measured {value['measured_at']} no longer applies ({value['problems'][0]}); "
                "installations use the default table; sudo sparkring fabric tune measures again" + tail)
    text = (f"SIRCL tuning: measured on this fabric {value['measured_at']} for {', '.join(value['rows'])} with image "
            f"{value['image']}")
    if not value["covers_default"] and value.get("default_image"):
        text += f"; installations without --image use {value['default_image']}, which it does not cover"
    return text + tail


# sparkring fabric tune.

def add_arguments(parser):
    parser.add_argument("--layouts", metavar="NAMES",
                        help="group shapes to measure, comma-separated (default: every one the ring harness can "
                             "measure on this fabric, such as pair,path-3,path-4,path-5,cycle-8)")
    parser.add_argument("--image", metavar="NAME", help="measure with this installer image (default: the one sparkring "
                                                        "install uses)")
    parser.add_argument("--quick", action="store_true", help="every fourth size, 4 KiB to 64 MiB: about a quarter of "
                                                             "the measurement time")
    parser.add_argument("--lan-interface", metavar="NAME", help="the interface of the harness's control exchange on "
                                                                "every Spark (default: the management interface)")
    parser.add_argument("--control-port", type=int, default=CONTROL_PORT, metavar="PORT",
                        help=f"the TCP port of the control exchange on the first Spark of each group (default "
                             f"{CONTROL_PORT})")
    parser.add_argument("--layout-timeout", type=int, default=LAYOUT_TIMEOUT, metavar="SECONDS",
                        help=f"the longest measurement of one group (default {LAYOUT_TIMEOUT})")
    parser.add_argument("--max-hours", type=float, metavar="HOURS",
                        help="start no group after this time; the rest stay pending for the next run")
    parser.add_argument("--stop-serving", action="store_true", help="stop serving models before measuring")
    parser.add_argument("--fresh", action="store_true", help="measure every named group again")
    parser.add_argument("--distribute", action="store_true",
                        help="copy the recorded measured tables to every Spark again; measures nothing")
    parser.add_argument("--execute", action="store_true", help="measure; without it the plan is printed")
    parser.add_argument("--json", action="store_true", help="print the plan or the result as JSON")


def command(args, *, state=STATE, access=None, harness=run_harness, now=time.time, host_root=None, key=None):
    """``sudo sparkring fabric tune``: the exit status (0 measured or planned, 1 some groups unmeasured)."""
    state = Path(state)
    cluster = fabric._cluster(state)
    if not (state / "fabric.json").exists():
        raise ValueError("This cluster has no fabric document; sudo sparkring setup records it")
    document = fabric.read_document(state)
    if [row["node_id"] for row in document["positions"]] != [host["node_id"] for host in cluster["plan"]["spec"]["hosts"]]:
        raise ValueError("The fabric document does not describe the recorded cluster; run sudo sparkring setup")
    say = (lambda line: print(line, file=sys.stderr)) if args.json else print
    access = access or fabric.Access(cluster["plan"])
    host_root = host_root or HOST_ROOT
    if args.distribute:
        result = redistribute(state, document, access=access, say=say, host_root=host_root)
        if args.json:
            print(json.dumps(result, indent=2))
        return 0
    image_value = chosen_image(args.image)
    names = [name.strip() for name in args.layouts.split(",") if name.strip()] if args.layouts else None
    value = plan(cluster, document, image_value, names=names, quick=args.quick, lan_interface=args.lan_interface,
                 control_port=args.control_port, layout_timeout=args.layout_timeout, max_hours=args.max_hours,
                 busy=serving(state, document["size"]), key=key, state=state)
    if not args.execute:
        if args.json:
            print(json.dumps(value, indent=2))
        else:
            for line in plan_lines(value, latest_progress(state)):
                print(line)
            print("Review, then repeat with --execute." if not value["blockers"] else "Nothing can be measured until "
                  "the blockers are resolved.")
        return 2 if value["blockers"] else 0
    for line in plan_lines(value, latest_progress(state)):
        say(line)
    result = execute(state, cluster, document, image_value, value, access=access, harness=harness, say=say, now=now,
                     stop_serving=args.stop_serving, quick=args.quick, fresh=args.fresh,
                     layout_timeout=args.layout_timeout, max_hours=args.max_hours, host_root=host_root)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for line in result_lines(result):
            print(line)
    return 0 if result["complete"] else 1


def main(argv=None):
    """``sparkring fabric tune`` alone (``runtime/host/fabric.py`` dispatches to ``command``)."""
    parser = argparse.ArgumentParser(prog="sparkring fabric tune")
    add_arguments(parser)
    args = parser.parse_args(argv)
    try:
        if not hasattr(os, "geteuid") or os.geteuid() != 0:
            raise ValueError("Run sudo sparkring fabric tune: it reads root-only state on every Spark")
        return command(args)
    except FAILURES as error:
        print("SparkRing: " + str(error), file=sys.stderr)
        return 2
