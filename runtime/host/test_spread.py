"""The self-spreading install: bootstrap order, spread plans and pipelines through simulated Sparks.

Each simulated Spark is a temporary directory. Its hop and source programs run
as real ``python -I`` processes, the programs ``fabric_stream.hop_source`` and
``push_source`` that ship to Sparks, or as threads where a test replaces a sink
or a connection. Fabric documents come from the synthetic setup plans of
``test_fabric_layouts`` with their ``198.18.0.0`` addresses moved to
``127.18.0.0``, so every cable function has its own loopback /24 and each
stream binds and checks the addresses a Spark would. Throttled tests pace each
link (``fabric_stream.Throttle``) to measure the pipeline's timing.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import threading
import time

import pytest

from runtime.common import fabric_document, fabric_layout
from runtime.host import bootstrap, checkpoint_plan, fabric_stream, install_assets, spread
from runtime.host import checkpoint_place as place
from runtime.host.install_errors import NeedsInput
from runtime.host.test_fabric import document_of
from runtime.host.test_fabric_stream import REPOSITORY, REVISION, checkpoint, journal, mount_table, prepare, state  # noqa: F401

linux = pytest.mark.skipif(not sys.platform.startswith("linux"),
                           reason="checkpoint placement uses Linux hard links, /proc/self/fd, O_NOATIME and flock")
FAST = {"accept_seconds": 20, "read_seconds": 30}


def loopback(shape, size, *, failed=()):
    """The fabric document of a synthetic setup plan with loopback addresses; ``failed`` cables recorded down.

    The identity covers each cable's subnets, so it is computed again for the loopback ones.
    """
    _, value, _ = document_of(fabric_layout.layout(shape, size))
    value = json.loads(json.dumps(value).replace("198.18.", "127.18."))
    for cable in failed:
        value["cables"][cable]["health"] = {"state": "failed"}
    value["id"] = fabric_document.identity(value)
    return fabric_document.validate(value)


def write_files(root, count, size, *, seed=7):
    """``count`` checkpoint files of about ``size`` bytes below ``root``; returns their ``{name: [size, sha256]}``."""
    generator, files = random.Random(seed), {}
    for index in range(count):
        data = generator.randbytes(size + index)
        name = f"model-{index + 1:05d}-of-{count:05d}.safetensors"
        (Path(root) / name).parent.mkdir(parents=True, exist_ok=True)
        (Path(root) / name).write_bytes(data)
        files[name] = [len(data), hashlib.sha256(data).hexdigest()]
    return files


def write_blobs(root, count, size, *, seed=11):
    """Image blobs below ``root``, each named by its SHA-256 as the registry relay stores them."""
    generator, files = random.Random(seed), {}
    Path(root).mkdir(parents=True, exist_ok=True)
    for index in range(count):
        data = generator.randbytes(size + index)
        name = hashlib.sha256(data).hexdigest()
        (Path(root) / name).write_bytes(data)
        files[name] = [len(data), name]
    return files


class ThreadProcess:
    """A hop run in a thread behind the stdin and stdout pipes of a process, for tests that replace parts of it."""

    def __init__(self, target):
        read_in, write_in = os.pipe()
        read_out, write_out = os.pipe()
        self.stdin, self.stdout = os.fdopen(write_in, "wb"), os.fdopen(read_out, "rb")
        self.returncode, self.errors = None, None

        def main():
            with os.fdopen(read_in, "rb") as stdin, os.fdopen(write_out, "w") as stdout:
                try:
                    target(stdin, stdout)
                    self.returncode = 0
                except BaseException:  # noqa: BLE001 - the executor reads the missing result
                    self.returncode = 1
        self.thread = threading.Thread(target=main, daemon=True)
        self.thread.start()

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.thread.join(timeout)
        return self.returncode

    def kill(self):
        pass


class Fleet:
    """Simulated Sparks for ``spread.Spread``: one directory each, power switched off and on by the test.

    Hop programs run as ``python -I`` processes unless ``sink`` or ``connect``
    (each a function of the position) replace parts of a hop, which then runs
    in a thread. The source program reads ``source_root`` on ``source`` and
    each other Spark's own directory.
    """

    def __init__(self, tmp_path, document, source, source_root, *, options=None, sink=None, connect=None):
        self.tmp, self.document, self.source, self.source_root = Path(tmp_path), document, source, Path(source_root)
        self.options = {**FAST, **(options or {})}
        self.sink, self.connect = sink, connect
        self.off, self.processes, self.said, self.guard = set(), {}, [], threading.Lock()

    def store(self, position):
        return self.tmp / f"spark{position}"

    def start(self, position, listen, peers, files, want):
        if position in self.off:
            # A powered-off Spark: its SSH command fails at once.
            return subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(255)"], stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        if self.sink is not None or self.connect is not None:
            return self.threaded(position, listen, peers, files, want)
        program = fabric_stream.hop_source("directory", {"root": str(self.store(position))}, listen, peers, files,
                                           want, self.options)
        errors = tempfile.TemporaryFile()
        process = subprocess.Popen([sys.executable, *fabric_stream.BOOT[1:]], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=errors)
        process.errors = errors
        process.stdin.write(fabric_stream.boot_input(program))
        process.stdin.flush()
        with self.guard:
            self.processes.setdefault(position, []).append(process)
        return process

    def threaded(self, position, listen, peers, files, want):
        sink = (self.sink or (lambda position, files: fabric_stream.DirectorySink(str(self.store(position)), files)))
        connect = self.connect(position) if self.connect else None
        options = {key: value for key, value in self.options.items() if key in fabric_stream.HOP_OPTIONS}

        def target(stdin, stdout):
            guard = threading.Lock()

            def emit(value):
                with guard:
                    stdout.write(json.dumps(value) + "\n")
                    stdout.flush()
            token = stdin.read(32)
            result = fabric_stream.hop(listen, peers, files, sink(position, files), token=token, want=want,
                                       downstream=lambda: json.loads(stdin.readline() or "null"), announce=emit,
                                       progress=emit, connect=connect, **options)
            emit({"result": result})
        return ThreadProcess(target)

    def send(self, position, sources, addresses, ports, groups, token):
        root = self.source_root if position == self.source else self.store(position)
        program = fabric_stream.push_source(str(root), sources, addresses, ports, groups, self.options)
        done = subprocess.run([sys.executable, *fabric_stream.BOOT[1:]], input=fabric_stream.boot_input(program) + token,
                              capture_output=True, timeout=120)
        if done.returncode:
            raise ValueError(done.stderr.decode(errors="replace").strip()[-800:])
        return json.loads(done.stdout.decode().strip().splitlines()[-1])["result"]

    def alive(self, position):
        return position not in self.off

    def spreader(self, **options):
        return spread.Spread(self.document, start=self.start, send=self.send, alive=self.alive, say=self.said.append,
                             announce_seconds=30, finish_seconds=120, **options)

    def power_off_after(self, position, placed, names):
        """Power ``position`` off once it has placed ``placed`` of ``names``: its processes are killed."""
        def watch():
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if sum((self.store(position) / name).is_file() for name in names) >= placed:
                    self.off.add(position)
                    with self.guard:
                        running = list(self.processes.get(position, ()))
                    for process in running:
                        if process.poll() is None:
                            process.kill()
                    return
                time.sleep(0.002)
        thread = threading.Thread(target=watch, daemon=True)
        thread.start()
        return thread

    def missing(self, files, positions):
        """What each position lacks, as a repeated install's survey finds it."""
        needs = {}
        for position in positions:
            lacking = {name for name, (size, digest) in files.items()
                       if not (self.store(position) / name).is_file()
                       or hashlib.sha256((self.store(position) / name).read_bytes()).hexdigest() != digest}
            if lacking:
                needs[position] = lacking
        return needs


def assert_identical(fleet, files, positions):
    for position in positions:
        for name, (size, digest) in files.items():
            data = (fleet.store(position) / name).read_bytes()
            assert (len(data), hashlib.sha256(data).hexdigest()) == (size, digest), (position, name)
            assert data == (fleet.source_root / name).read_bytes()
        assert not [path for path in fleet.store(position).rglob("*.part")], position


# Bootstrap order --------------------------------------------------------------

@pytest.mark.parametrize("shape, size, expected", [
    ("pair", 2, [(1, 0, 1)]),
    ("path", 4, [(1, 0, 1), (2, 1, 2), (3, 2, 3)]),
    ("cycle", 4, [(1, 0, 1), (3, 0, 1), (2, 1, 2)]),
    ("path", 8, [(p, p - 1, p) for p in range(1, 8)]),
    ("cycle", 8, [(1, 0, 1), (7, 0, 1), (2, 1, 2), (6, 7, 2), (3, 2, 3), (5, 6, 3), (4, 3, 4)]),
])
def test_bootstrap_order_reaches_each_spark_by_the_fewest_cables_through_the_one_before_it(shape, size, expected):
    document = loopback(shape, size)
    rows = spread.bootstrap_order(document)
    assert [(row["position"], row["via"], row["hops"]) for row in rows] == expected
    assert spread.hop_limit(fabric_document.layout(document)) == size - 1 == bootstrap.hop_limit(size)


def test_the_hop_limit_is_one_less_than_the_spark_count_and_the_far_end_of_an_eight_path_reaches_it():
    for size in range(2, 9):
        for shape in (("pair",) if size == 2 else ("path", "cycle")):
            rows = spread.bootstrap_order(loopback(shape, size))
            farthest = max(row["hops"] for row in rows)
            assert farthest <= size - 1
            assert farthest == (size - 1 if shape != "cycle" else size // 2)
    assert bootstrap.hop_limit() == bootstrap.MAX_HOPS == 7
    with pytest.raises(ValueError):
        bootstrap.hop_limit(9)


# Neighbours and trees from the fabric document ----------------------------------

def test_neighbours_come_from_the_fabric_document_for_paths_and_cycles():
    path = loopback("path", 4)
    assert fabric_stream.neighbours(path, 0) == [
        {"position": 1, "port": 0, "cable": 0, "links": [("127.18.0.1", "127.18.0.2"), ("127.18.1.1", "127.18.1.2")]}]
    assert [(row["position"], row["port"], row["cable"]) for row in fabric_stream.neighbours(path, 1)] == [
        (2, 0, 1), (0, 1, 0)]
    assert [row["position"] for row in fabric_stream.neighbours(path, 3)] == [2]
    cycle = loopback("cycle", 8)
    assert [(row["position"], row["port"], row["cable"]) for row in fabric_stream.neighbours(cycle, 0)] == [
        (1, 0, 0), (7, 1, 7)]
    assert fabric_stream.cable_links(cycle, 0, 7) == [("127.18.14.2", "127.18.14.1"), ("127.18.15.2", "127.18.15.1")]
    with pytest.raises(ValueError, match="share no fabric cable"):
        fabric_stream.cable_links(cycle, 0, 2)


def test_trees_follow_the_layout_and_equal_the_ring_tree_on_pairs_and_four_rings():
    for donor in range(2):
        assert fabric_stream.tree(2, donor, fabric_layout.layout("pair", 2)) == fabric_stream.tree(2, donor)
    for donor in range(4):
        assert fabric_stream.tree(4, donor, fabric_layout.layout("cycle", 4)) == fabric_stream.tree(4, donor)
    # A path does not wrap round: its far end is three cables from Node A.
    assert fabric_stream.tree(4, 0, fabric_layout.layout("path", 4)) == [[(0, 1)], [(1, 2)], [(2, 3)]]
    sizes = {"a": 10, "b": 20}
    holdings = [set(), {"a"}, set(), {"b"}]
    ring = checkpoint_plan._distribute(4, holdings, sizes)
    assert checkpoint_plan._distribute(4, holdings, sizes, fabric_layout.layout("cycle", 4)) == ring
    line = checkpoint_plan._distribute(4, holdings, sizes, fabric_layout.layout("path", 4))
    assert [(item["source"], item["transport"]) for item in line["pool"]] == [(1, "fabric"), (3, "rsync")]
    assert [(item["source"], item["target"]) for item in line["receive"]] == [(0, 1), (1, 2), (2, 3)]


# Pipeline plans -------------------------------------------------------------------

@pytest.mark.parametrize("shape, size, source, expected", [
    ("pair", 2, 0, [("next", 0, [1], [0])]),
    ("pair", 2, 1, [("next", 1, [0], [0])]),
    ("path", 4, 0, [("next", 0, [1, 2, 3], [0, 1, 2])]),
    ("path", 8, 0, [("next", 0, [1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6])]),
    ("path", 8, 3, [("next", 3, [4, 5, 6, 7], [3, 4, 5, 6]), ("previous", 3, [2, 1, 0], [2, 1, 0])]),
    ("cycle", 4, 0, [("next", 0, [1, 2], [0, 1]), ("previous", 0, [3], [3])]),
    ("cycle", 8, 0, [("next", 0, [1, 2, 3, 4], [0, 1, 2, 3]), ("previous", 0, [7, 6, 5], [7, 6, 5])]),
    ("cycle", 8, 5, [("next", 5, [6, 7, 0, 1], [5, 6, 7, 0]), ("previous", 5, [4, 3, 2], [4, 3, 2])]),
])
def test_pipelines_leave_the_donor_both_ways_round_a_cycle_and_along_a_path(shape, size, source, expected):
    document = loopback(shape, size)
    layout = fabric_document.layout(document)
    found, unreached = spread.chains(layout, source, {p: True for p in range(size) if p != source})
    assert [(c["direction"], c["start"], c["hops"], c["cables"]) for c in found] == expected and unreached == []
    value = spread.plan(document, [{"asset": "checkpoint", "source": source,
                                    "writes": {p: 100 + p for p in range(size) if p != source}}])
    for pipeline, (direction, start, hops, cables) in zip(value["pipelines"], expected, strict=True):
        assert (pipeline["direction"], pipeline["source_position"]) == (direction, start)
        assert [(hop["from"], hop["to"], hop["cable"]) for hop in pipeline["hops"]] == list(
            zip([start, *hops], hops, cables))
        assert all(hop["functions"] == ["primary", "secondary"] for hop in pipeline["hops"])
        assert pipeline["bytes_written_per_position"] == {str(p): 100 + p for p in hops}
        assert pipeline["forwarded_only_positions"] == []


def test_chains_go_round_a_stopped_spark_or_a_down_cable_and_start_at_the_nearest_holder():
    cycle = fabric_layout.layout("cycle", 8)
    everyone = {p: True for p in range(1, 8)}
    found, unreached = spread.chains(cycle, 0, everyone, dead={3})
    assert [(c["start"], c["hops"]) for c in found] == [(0, [1, 2]), (0, [7, 6, 5, 4])] and unreached == [3]
    found, unreached = spread.chains(cycle, 0, everyone, down={2})
    assert [(c["start"], c["hops"]) for c in found] == [(0, [1, 2]), (0, [7, 6, 5, 4, 3])] and unreached == []
    # A repair starts at the last Spark before the first one that lacks something.
    found, _ = spread.chains(cycle, 0, {3: True}, holders={1, 2})
    assert [(c["start"], c["hops"]) for c in found] == [(2, [3])]
    # A Spark that only forwards (an image already in Docker) stays on the chain without writing.
    found, _ = spread.chains(cycle, 0, {2: True}, relays=set(range(8)))
    assert [(c["start"], c["hops"]) for c in found] == [(0, [1, 2])]
    # A Spark serving a named copy in place holds the checkpoint but cannot forward: it starts a chain.
    found, _ = spread.chains(cycle, 0, {2: True, 3: True}, holders={1}, relays=set(range(8)) - {1})
    assert [(c["start"], c["hops"]) for c in found] == [(1, [2, 3])]
    path = fabric_layout.layout("path", 6)
    found, unreached = spread.chains(path, 0, {p: True for p in range(1, 6)}, down={2})
    assert [(c["start"], c["hops"]) for c in found] == [(0, [1, 2])] and unreached == [3, 4, 5]
    assert spread.blocking(path, 0, 4, down={2}) == ("cable", 2)
    assert spread.blocking(path, 0, 4, dead={3}) == ("spark", 3)
    assert spread.blocking(cycle, 0, 4, down={2}) == ("cable", 2)


def install_plan():
    """A checkpoint plan of a blank cycle-8 whose Node A downloads, as ``checkpoint_plan.plan`` writes its nodes."""
    nodes = []
    for rank in range(8):
        action = "hub" if rank == 0 else "receive"
        basis = {0: "present", 1: "present", 2: "layers"}.get(rank, "whole")
        nodes.append({"rank": rank, "mode": "owned",
                      "files": {"a": {"action": action, "size": 3 * spread.GIB},
                                "b": {"action": action, "size": 5 * spread.GIB}},
                      "storage": {"image": {"basis": basis, "download_bytes": 2 * spread.GIB}}})
    return {"nodes": nodes, "distribution": {"donor": 0}}


def test_the_install_plan_spreads_the_image_and_the_checkpoint_and_names_what_each_spark_writes():
    document = loopback("cycle", 8)
    card = {"download_bytes": 14 * spread.GIB}
    value = spread.plan(document, spread.assets(document, install_plan(), positions=list(range(8)), card=card))
    assert value["schema"] == "sparkring-spread-plan/v1" and value["fabric_id"] == document["id"]
    assert (value["layout"], value["hop_limit"]) == ("cycle-8", 7)
    assert value["sha256"] == spread._digest(value)
    image = [p for p in value["pipelines"] if p["asset"] == "image"]
    assert [(p["source_position"], [h["to"] for h in p["hops"]]) for p in image] == [(0, [1, 2, 3, 4]),
                                                                                      (0, [7, 6, 5, 4][:3])]
    # Spark 1 holds the image and forwards; Spark 2 holds leading layers and loads the rest through the relay.
    assert image[0]["forwarded_only_positions"] == [1, 2]
    assert image[0]["bytes_written_per_position"] == {"3": 14 * spread.GIB, "4": 14 * spread.GIB}
    assert value["fallbacks"] == [{"asset": "image", "position": 2, "path": "admin", "reason": "layers",
                                   "cable": None, "bytes": 2 * spread.GIB}]
    checkpoints = [p for p in value["pipelines"] if p["asset"] == "checkpoint"]
    assert [p["bytes_written_per_position"] for p in checkpoints] == [
        {str(p): 8 * spread.GIB for p in (1, 2, 3, 4)}, {str(p): 8 * spread.GIB for p in (7, 6, 5)}]
    assert spread.writes(value)[3] == 22 * spread.GIB and spread.writes(value)[2] == 10 * spread.GIB
    lines = spread.describe(value, document)
    assert lines[1].startswith("  Node A reaches the Sparks in cable order (1 via 0, 7 via 0, 2 via 1, 6 via 7")
    assert "the farthest is 4 of at most 7 cables away" in lines[1]
    assert any("0 → 1 → 2 → 3 → 4 and 0 → 7 → 6 → 5; positions 1-7 each write 8.0 GiB" in line for line in lines)
    assert any("positions 1-2 forward without writing" in line for line in lines)
    assert any(line.startswith("  Position 2 already holds leading image layers") for line in lines)


def test_a_down_cable_on_a_path_sends_the_sparks_beyond_it_over_the_administration_network():
    document = loopback("path", 4, failed=(1,))
    value = spread.plan(document, [{"asset": "checkpoint", "source": 0, "writes": {1: 10, 2: 10, 3: 10}}])
    assert [[h["to"] for h in p["hops"]] for p in value["pipelines"]] == [[1]]
    assert value["fallbacks"] == [{"asset": "checkpoint", "position": p, "path": "admin", "reason": "cable",
                                   "cable": 1, "bytes": 10} for p in (2, 3)]
    assert ("  cable 1 (spark1 port 0 ↔ spark2 port 1) is down: positions 2-3 receive the checkpoint over the "
            "administration network instead of the fabric (slower).") in spread.describe(value, document)
    # On a cycle the same cable is routed round.
    cycle = spread.plan(loopback("cycle", 4, failed=(1,)), [{"asset": "checkpoint", "source": 0,
                                                              "writes": {1: 10, 2: 10, 3: 10}}])
    assert [[h["to"] for h in p["hops"]] for p in cycle["pipelines"]] == [[1], [3, 2]] and cycle["fallbacks"] == []


def test_a_reviewed_spread_plan_bounds_the_fresh_one_with_the_checkpoint_plan():
    document = loopback("cycle", 4)
    assets = [{"asset": "checkpoint", "source": 0, "writes": {1: spread.GIB, 2: spread.GIB, 3: spread.GIB}}]
    reviewed = spread.plan(document, assets)
    assert spread.envelope(reviewed, spread.plan(document, assets)) == []
    grown = spread.plan(document, [{**assets[0], "writes": {1: spread.GIB, 2: 3 * spread.GIB, 3: spread.GIB}}])
    assert [item["position"] for item in spread.envelope(reviewed, grown)] == [2]
    cut = spread.plan(loopback("cycle", 4, failed=(1, 2)), assets)
    assert [item["text"] for item in spread.envelope(reviewed, cut)] == [
        "position 2 would receive the checkpoint over the administration network instead of the fabric"]
    other = spread.plan(loopback("path", 4), assets)
    assert spread.envelope(reviewed, other)[0]["text"].startswith("the fabric changed")
    node = {"rank": 0, "host": "spark0", "hostname": "spark0", "write_bytes": 0, "mode": "owned", "sources": []}
    base = {"repository": "r", "revision": "v", "pins_sha256": "p", "hub_files": [], "required": {"sizes": {}},
            "nodes": [node]}
    assert checkpoint_plan.envelope({**base, "spread": reviewed}, {**base, "spread": reviewed}) == []
    assert checkpoint_plan.envelope({**base, "spread": reviewed}, {**base, "spread": grown})[0]["kind"] == "spread"


def test_install_assets_place_a_half_of_a_ring_by_its_positions():
    document = loopback("cycle", 4)
    plan = {"nodes": [{"rank": 0, "mode": "owned", "files": {"a": {"action": "present", "size": 7}},
                       "storage": {"image": {"basis": "present"}}},
                      {"rank": 1, "mode": "owned", "files": {"a": {"action": "receive", "size": 7}},
                       "storage": {"image": {"basis": "whole"}}}],
            "distribution": {"donor": 0}}
    image, checkpoints = spread.assets(document, plan, positions=[2, 3], card={"download_bytes": 9})
    # Sparks 1 and 3 were not surveyed by this deployment and are planned as lacking the image.
    assert image == {"asset": "image", "source": 0, "writes": {1: 9, 3: 9}, "admin": []}
    assert checkpoints == {"asset": "checkpoint", "source": 2, "writes": {3: 7}, "members": [2, 3], "holders": [2],
                           "relays": [2, 3]}
    assert spread.deployment_layout(fabric_layout.layout("cycle", 4), [2, 3]) == fabric_layout.layout("pair", 2)
    assert spread.deployment_layout(fabric_layout.layout("cycle", 8), range(4)) == fabric_layout.layout("path", 4)


def test_messages_name_the_spark_the_positions_and_what_they_hold():
    document = loopback("cycle", 8)
    text = spread.stopped_message(document, "checkpoint", {4}, {4: set(range(30))}, range(8), 53)
    assert text == ("spark4 (position 4) stopped answering during the checkpoint spread; positions 0-3 and 5-7 hold "
                    "the complete checkpoint, position 4 holds 23 of 53 files. Power it on and repeat the same "
                    "command; the spread resumes.")
    assert spread.cable_message(document, 3, [4, 5, 6, 7]) == (
        "cable 3 (spark3 port 0 ↔ spark4 port 1) is down; positions 4-7 receive over the admin network instead of "
        "the fabric (slower). Reseat the cable and run sudo sparkring fabric verify.")
    problem = spread.first_unreached(document, lambda position: position != 6)
    assert isinstance(problem, NeedsInput) and problem.field == "spark"
    # Bootstrap order 1, 7, 2, 6, ...: Spark 6 comes after 0, 1, 7 and 2.
    assert str(problem) == ("positions 0-2 and 7 are set up; spark6 (position 6) was not reached over cable 6; "
                            "check the cable and repeat the same command.")
    assert spread.first_unreached(document, lambda position: True) is None


# Pipeline mechanics ---------------------------------------------------------------

class Recording(fabric_stream.DirectorySink):
    """A directory sink that records when each write of each file happened."""

    def __init__(self, root, files, log):
        super().__init__(root, files)
        self.log, self.names = log, {}

    def open(self, name, size):
        handle = super().open(name, size)
        self.names[handle] = name
        return handle

    def write(self, handle, data):
        self.log.append((self.root, self.names[handle], time.monotonic()))
        super().write(handle, data)


def test_a_chunk_pipeline_through_three_receivers_forwards_each_chunk_while_writing(tmp_path):
    source = tmp_path / "source"
    files = write_files(source, 2, 1 << 20)
    log = []
    fleet = Fleet(tmp_path, loopback("path", 4), 0, source, options={"chunk": 64 << 10, "limit": 4 << 20},
                  sink=lambda position, names: Recording(str(fleet.store(position)), names, log))
    outcome = fleet.spreader().run("checkpoint", files, 0, {p: set(files) for p in (1, 2, 3)})
    assert outcome["remaining"] == {} and outcome["rounds"] == 1
    assert_identical(fleet, files, (1, 2, 3))
    for name in files:
        first = [at for root, written, at in log if written == name and root == str(fleet.store(1))]
        last = [at for root, written, at in log if written == name and root == str(fleet.store(3))]
        # The third Spark writes its first chunk of a file before the first Spark writes its last.
        assert min(last) < max(first)
    rows = {row["target"]: row for row in outcome["received"]}
    assert [rows[p]["source"] for p in (1, 2, 3)] == [0, 1, 2] and all(rows[p]["hop"] == p for p in (1, 2, 3))


class Corrupting:
    """A downstream connection that flips one byte of the first data block it sends."""

    def __init__(self, connection, state):
        self.connection, self.state = connection, state

    def sendall(self, data):
        if not self.state["done"] and len(data) >= 4096:
            data = bytes([data[0] ^ 1]) + bytes(data[1:])
            self.state["done"] = True
        return self.connection.sendall(data)

    def __getattr__(self, name):
        return getattr(self.connection, name)


def test_a_hash_mismatch_at_hop_two_is_requested_again_from_hop_one(tmp_path):
    source = tmp_path / "source"
    files = write_files(source, 4, 200_000)
    flipped = {"done": False}

    def connect(position):
        if position != 1:
            return None
        return lambda *args: Corrupting(__import__("socket").create_connection(*args), flipped)
    fleet = Fleet(tmp_path, loopback("path", 4), 0, source, options={"chunk": 64 << 10}, connect=connect)
    outcome = fleet.spreader().run("checkpoint", files, 0, {p: set(files) for p in (1, 2, 3)})
    assert flipped["done"] and outcome["remaining"] == {} and outcome["rounds"] == 2
    assert_identical(fleet, files, (1, 2, 3))
    first = {row["target"]: set(row["names"]) for row in outcome["received"] if row["round"] == 1}
    again = [row for row in outcome["received"] if row["round"] == 2]
    damaged = set(files) - first[2]
    assert len(damaged) == 1 and first[1] == set(files) and first[3] == first[2]
    # Spark 1 verified its copy; the second pass starts there and carries only the damaged file.
    assert [(row["start"], row["source"], row["target"], set(row["names"])) for row in again] == [
        (1, 1, 2, damaged), (1, 2, 3, damaged)]


class Stalling(fabric_stream.DirectorySink):
    """A directory sink whose first write blocks for ``seconds``, once per test."""

    def __init__(self, root, files, state, seconds):
        super().__init__(root, files)
        self.state, self.seconds = state, seconds

    def write(self, handle, data):
        if not self.state["slept"]:
            self.state["slept"] = True
            time.sleep(self.seconds)
        super().write(handle, data)


def test_a_slow_writer_defers_its_files_while_the_chain_runs_on_and_they_arrive_in_the_next_pass(tmp_path):
    source = tmp_path / "source"
    files = write_files(source, 6, 256 << 10)
    asleep = {"slept": False}

    def sink(position, names):
        if position == 1:
            return Stalling(str(fleet.store(1)), names, asleep, 3.0)
        return fabric_stream.DirectorySink(str(fleet.store(position)), names)
    fleet = Fleet(tmp_path, loopback("path", 4), 0, source,
                  options={"chunk": 16 << 10, "buffer": 2, "stall": 0.2}, sink=sink)
    started = time.time()
    outcome = fleet.spreader().run("checkpoint", files, 0, {p: set(files) for p in (1, 2, 3)})
    assert outcome["remaining"] == {}
    assert_identical(fleet, files, (1, 2, 3))
    first = [row for row in outcome["received"] if row["round"] == 1]
    far = next(row for row in first if row["target"] == 3)
    # The far Spark finished while the slow writer still slept: the stall cost at most 0.2 s per file.
    assert set(far["names"]) == set(files) and far["completed_at"] - started < 3.0
    later = [row for row in outcome["received"] if row["round"] > 1]
    assert {row["target"] for row in later} == {1}


def test_a_repeated_pass_keeps_placed_files_and_replaces_a_partial_one(tmp_path):
    source = tmp_path / "source"
    files = write_files(source, 3, 100_000)
    fleet = Fleet(tmp_path, loopback("pair", 2), 0, source)
    names = sorted(files)
    fleet.store(1).mkdir()
    for name in names[:2]:
        (fleet.store(1) / name).write_bytes((source / name).read_bytes())
    (fleet.store(1) / (names[2] + ".part")).write_bytes(b"bytes of an interrupted pass")
    outcome = fleet.spreader().run("checkpoint", files, 0, {1: set(files)})
    assert outcome["remaining"] == {} and [row["names"] for row in outcome["received"]] == [[names[2]]]
    assert outcome["received"][0]["bytes"] == files[names[2]][0]
    assert_identical(fleet, files, (1,))


def hop_thread(target, files, listen, peers, *, want=None, sink=None):
    """Run ``hop_checkpoint`` (or ``hop`` into ``sink``) in a thread; returns ``(thread, announcements, result)``."""
    import queue
    announced, result = queue.Queue(), {}

    def main():
        options = dict(token=b"t" * 32, want=want, downstream=None, announce=announced.put, accept_seconds=20)
        try:
            if sink is None:
                result["value"] = fabric_stream.hop_checkpoint(target, REPOSITORY, REVISION, listen, peers, files,
                                                               **options)
            else:
                result["value"] = fabric_stream.hop(listen, peers, files, sink, **options)
        except Exception as error:  # noqa: BLE001 - surfaced to the test thread
            result["error"] = error
            announced.put(None)
    thread = threading.Thread(target=main)
    thread.start()
    return thread, announced, result


def single_hop(source, target, files, *, want=None):
    listen, peers = ["127.18.0.2", "127.18.1.2"], ["127.18.0.1", "127.18.1.1"]
    thread, announced, result = hop_thread(target, files, listen, peers, want=want)
    offer = announced.get(timeout=30)
    assert offer is not None, result
    sizes = {name: value[0] for name, value in files.items()}
    groups = [[[name, sizes[name], 1] for name in group] for group in fabric_stream.balance(offer["needed"], sizes, 2)]
    fabric_stream.push(peers, listen, offer["ports"], groups, fabric_stream.files_in(str(source)), token=b"t" * 32)
    thread.join(timeout=30)
    assert "error" not in result, result
    return result["value"]


@linux
def test_the_checkpoint_sink_resumes_from_its_journal_with_two_files_placed_and_one_partial(tmp_path, mount_table):  # noqa: F811
    source = tmp_path / "source"
    files = write_files(source, 3, 150_000)
    names = sorted(files)
    target = checkpoint(tmp_path)
    prepare(target)
    first = single_hop(source, target, files, want=names[:2])
    assert first["placed"] == names[:2]
    (state(target) / "receive" / (names[2] + ".part")).write_bytes(b"bytes of an interrupted pass")
    second = single_hop(source, target, files)
    assert (second["held"], second["needed"], second["placed"]) == (names[:2], names[2:], names[2:])
    recorded = journal(target)
    for name in names:
        assert Path(target, name).read_bytes() == (source / name).read_bytes()
        assert (recorded[name]["state"], recorded[name]["origin"], recorded[name]["sha256"]) == (
            "placed", "fabric", files[name][1])
    assert not (state(target) / "receive" / (names[2] + ".part")).exists()


# Blank fabrics end to end -------------------------------------------------------------

def run_blank(tmp_path, shape, size, *, options=None, count=8, file_bytes=512 << 10):
    source = tmp_path / "source"
    files = write_files(source, count, file_bytes)
    fleet = Fleet(tmp_path, loopback(shape, size), 0, source, options=options)
    started = time.time()
    outcome = fleet.spreader().run("checkpoint", files, 0, {p: set(files) for p in range(1, size)})
    return fleet, files, outcome, started


def chain_times(outcome):
    """Per direction, each hop's completion time in hop order."""
    times = {}
    for row in outcome["received"]:
        times.setdefault(row["direction"], {})[row["hop"]] = row["completed_at"]
    return {direction: [rows[hop] for hop in sorted(rows)] for direction, rows in times.items()}


@pytest.mark.parametrize("shape, size", [("cycle", 8), ("path", 8)])
def test_a_blank_fabric_spreads_identical_files_and_the_farthest_spark_finishes_one_chunk_per_cable_later(
        tmp_path, shape, size):
    # Each link carries 2 MiB/s per stream: one 512 KiB file takes 0.25 s, one 64 KiB chunk 31 ms, and each
    # stream's 2 MiB take 1 s over one cable. Store and forward would take 1 s more per cable.
    limit, chunk, file_bytes = 2 << 20, 64 << 10, 512 << 10
    fleet, files, outcome, started = run_blank(tmp_path, shape, size, options={"chunk": chunk, "limit": limit},
                                               file_bytes=file_bytes)
    assert outcome["remaining"] == {} and outcome["rounds"] == 1 and outcome["dead"] == set()
    assert_identical(fleet, files, range(1, size))
    file_time, single = file_bytes / limit, 4 * file_bytes / limit
    for direction, times in chain_times(outcome).items():
        assert times == sorted(times) or max(times) - min(times) < file_time
        # The farthest Spark completes within one file's transfer time after the Spark before it,
        # and the whole chain within one file's time after its first Spark.
        if len(times) > 1:
            assert times[-1] - times[-2] < file_time, (direction, times)
        assert times[-1] - times[0] < file_time, (direction, times)
    farthest = max(row["completed_at"] for row in outcome["received"])
    hops = max(len(times) for times in chain_times(outcome).values())
    # Pipelined: about one single-cable time, far below ``hops`` of them (process start-up included).
    assert farthest - started < single + 1.5 < hops * single


def test_a_blank_cycle8_receives_the_image_blobs_and_the_checkpoint_at_once_byte_for_byte(tmp_path):
    document = loopback("cycle", 8)
    blobs = write_blobs(tmp_path / "relay", 4, 300_000)
    files = write_files(tmp_path / "donor", 6, 200_000)
    image = Fleet(tmp_path / "image", document, 0, tmp_path / "relay")
    weights = Fleet(tmp_path / "checkpoint", document, 0, tmp_path / "donor")
    outcomes = {}

    def run(name, fleet, names):
        outcomes[name] = fleet.spreader().run(name, names, 0, {p: set(names) for p in range(1, 8)})
    threads = [threading.Thread(target=run, args=("image", image, blobs)),
               threading.Thread(target=run, args=("checkpoint", weights, files))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=180)
    assert all(outcomes[name]["remaining"] == {} and outcomes[name]["rounds"] == 1 for name in ("image", "checkpoint"))
    assert_identical(image, blobs, range(1, 8))
    assert_identical(weights, files, range(1, 8))
    for outcome in outcomes.values():
        assert sorted((row["source"], row["target"]) for row in outcome["received"]) == [
            (0, 1), (0, 7), (1, 2), (2, 3), (3, 4), (6, 5), (7, 6)]


@pytest.mark.parametrize("size", [3, 5])
def test_blank_paths_of_any_length_receive_identical_files(tmp_path, size):
    fleet, files, outcome, _ = run_blank(tmp_path, "path", size, count=5, file_bytes=100_000)
    assert outcome["remaining"] == {} and outcome["rounds"] == 1
    assert_identical(fleet, files, range(1, size))
    assert sorted((row["source"], row["target"]) for row in outcome["received"]) == [
        (p - 1, p) for p in range(1, size)]


# Failures -------------------------------------------------------------------------------

def test_a_spark_powered_off_mid_spread_stops_with_needs_input_and_the_same_command_resumes(tmp_path):
    source = tmp_path / "source"
    files = write_files(source, 12, 256 << 10)
    total = len(files)
    fleet = Fleet(tmp_path, loopback("cycle", 8), 0, source, options={"chunk": 32 << 10, "limit": 1 << 20})
    fleet.power_off_after(3, 3, files)
    executor = fleet.spreader()
    outcome = executor.run("checkpoint", files, 0, {p: set(files) for p in range(1, 8)})
    assert outcome["dead"] == {3} and set(outcome["remaining"]) == {3}
    # Spark 4 lost its stream with Spark 3 and was sent the rest from the other direction.
    again = [row for row in outcome["received"] if row["target"] == 4 and row["round"] > 1]
    assert again and all(row["direction"] == "previous" and row["source"] == 5 for row in again)
    assert_identical(fleet, files, [0, 1, 2, 4, 5, 6, 7][1:])
    with pytest.raises(NeedsInput) as stopped:
        spread.raise_stopped(fleet.document, "checkpoint", outcome, range(8), total)
    held = total - len(outcome["remaining"][3])
    assert stopped.value.field == "spark" and held >= 3
    assert str(stopped.value) == (f"spark3 (position 3) stopped answering during the checkpoint spread; positions "
                                  f"0-2 and 4-7 hold the complete checkpoint, position 3 holds {held} of {total} "
                                  "files. Power it on and repeat the same command; the spread resumes.")
    assert "spark3 (position 3) stopped answering during the checkpoint spread." in fleet.said
    # Power it on and repeat: only Spark 3 receives, only what it lacks, from Spark 2.
    fleet.off.clear()
    needs = fleet.missing(files, range(1, 8))
    assert set(needs) == {3} and len(needs[3]) <= total - 3
    resumed = fleet.spreader().run("checkpoint", files, 0, needs, holders=set(range(8)) - set(needs))
    assert resumed["remaining"] == {} and resumed["dead"] == set()
    assert {(row["start"], row["source"], row["target"]) for row in resumed["received"]} == {(2, 2, 3)}
    assert set().union(*(set(row["names"]) for row in resumed["received"])) == needs[3]
    assert_identical(fleet, files, range(1, 8))


def test_on_a_path_the_sparks_after_a_powered_off_one_wait_and_the_repeat_finishes_them(tmp_path):
    source = tmp_path / "source"
    files = write_files(source, 12, 256 << 10)
    fleet = Fleet(tmp_path, loopback("path", 5), 0, source, options={"chunk": 32 << 10, "limit": 1 << 20})
    fleet.power_off_after(2, 3, files)
    outcome = fleet.spreader().run("checkpoint", files, 0, {p: set(files) for p in range(1, 5)})
    assert outcome["dead"] == {2} and set(outcome["remaining"]) == {2, 3, 4} and outcome["fallbacks"] == {}
    with pytest.raises(NeedsInput, match=r"positions 0-1 hold the complete checkpoint, position 2 holds \d+ of 12 "
                                         r"files, and positions 3-4 wait for it\. Power it on"):
        spread.raise_stopped(fleet.document, "checkpoint", outcome, range(5), len(files))
    fleet.off.clear()
    needs = fleet.missing(files, range(1, 5))
    resumed = fleet.spreader().run("checkpoint", files, 0, needs, holders=set(range(5)) - set(needs))
    assert resumed["remaining"] == {}
    assert sorted((row["start"], row["target"]) for row in resumed["received"]) == [(1, 2), (1, 3), (1, 4)]
    assert_identical(fleet, files, range(1, 5))


def test_a_recorded_down_cable_on_a_path_leaves_the_sparks_beyond_it_to_the_administration_path(tmp_path):
    source = tmp_path / "source"
    files = write_files(source, 3, 50_000)
    fleet = Fleet(tmp_path, loopback("path", 4, failed=(1,)), 0, source)
    outcome = fleet.spreader().run("checkpoint", files, 0, {p: set(files) for p in (1, 2, 3)})
    assert outcome["fallbacks"] == {1: [2, 3]} and set(outcome["remaining"]) == {2, 3}
    assert [row["target"] for row in outcome["received"]] == [1] and set(fleet.processes) == {1}
    assert fleet.said == [spread.cable_message(fleet.document, 1, [2, 3])]


@pytest.mark.parametrize("shape, routed", [("path", False), ("cycle", True)])
def test_a_cable_that_fails_during_the_spread_is_found_and_avoided(tmp_path, shape, routed):
    source = tmp_path / "source"
    files = write_files(source, 3, 50_000)

    def refuse(*args):
        raise ConnectionRefusedError("the cable to the next Spark carries nothing")
    fleet = Fleet(tmp_path, loopback(shape, 4), 0, source, options={"accept_seconds": 2},
                  connect=lambda position: refuse if position == 1 else None)
    outcome = fleet.spreader().run("checkpoint", files, 0, {p: set(files) for p in (1, 2, 3)})
    assert 1 in outcome["down"] and outcome["dead"] == set()
    if routed:
        # Round the cycle: Spark 2 is reached from Spark 3.
        assert outcome["remaining"] == {} and outcome["fallbacks"] == {}
        assert [(row["source"], row["target"]) for row in outcome["received"] if row["round"] == 2] == [(3, 2)]
        assert_identical(fleet, files, (1, 2, 3))
    else:
        assert outcome["fallbacks"] == {1: [2, 3]} and set(outcome["remaining"]) == {2, 3}
        assert spread.cable_message(fleet.document, 1, [2, 3]) in fleet.said


# The serving image's layer archive ------------------------------------------------------

FAKE_DOCKER = r'''
import json, shutil, sys
state, args = sys.argv[1], sys.argv[2:]
if args[:1] == ["info"]:
    print(state)
elif args[:2] == ["image", "load"]:
    with open(state + "/loaded.tar", "wb") as archive:
        shutil.copyfileobj(sys.stdin.buffer, archive)
    print("Loaded image")
elif args[:2] == ["image", "inspect"]:
    print(json.dumps([{"Id": args[2], "Architecture": "arm64", "Os": "linux"}]))
else:
    sys.exit(2)
'''


def image_fixture(directory):
    """A configuration and three layer blobs with real digests, stored as the relay stores them."""
    blobs = write_blobs(directory, 3, 70_000)
    layers = [("sha256:" + hashlib.sha256(name.encode()).hexdigest(), "sha256:" + name, size)
              for name, (size, _) in blobs.items()]
    config = json.dumps({"rootfs": {"diff_ids": [diff for diff, _, _ in layers]}}).encode()
    image = hashlib.sha256(config).hexdigest()
    (Path(directory) / image).write_bytes(config)
    return "sha256:" + image, config, layers


def fake_docker(tmp_path, name):
    state_dir = tmp_path / ("docker-" + name)
    state_dir.mkdir(exist_ok=True)
    script = tmp_path / "fake_docker.py"
    script.write_text(FAKE_DOCKER, encoding="utf-8")
    return [sys.executable, str(script), str(state_dir)], state_dir


def test_staged_blobs_are_verified_and_load_as_the_archive_that_write_layer_archive_writes(tmp_path):
    root = tmp_path / "staged"
    image_id, config, layers = image_fixture(root)
    docker, state_dir = fake_docker(tmp_path, "a")
    result = spread.load_staged_image(str(root), image_id, [list(layer) for layer in layers], "127.0.0.1:5255/x:r",
                                      0, docker=docker)
    assert result == {"loaded": True, "image_id": image_id, "free_bytes": result["free_bytes"]}
    expected = io.BytesIO()
    install_assets.write_layer_archive(expected, image_id, config, layers, 0,
                                       lambda digest: root / digest.removeprefix("sha256:"), "127.0.0.1:5255/x:r")
    assert (state_dir / "loaded.tar").read_bytes() == expected.getvalue()
    # A blob that changed after it was verified is removed and named; Docker sees nothing.
    damaged = root / layers[1][1].removeprefix("sha256:")
    damaged.write_bytes(b"changed")
    (state_dir / "loaded.tar").unlink()
    assert spread.load_staged_image(str(root), image_id, [list(layer) for layer in layers], None, 0,
                                    docker=docker) == {"loaded": False, "differs": [damaged.name]}
    assert not damaged.exists() and not (state_dir / "loaded.tar").exists()
    assert spread.discard_staged(str(root / ("0" * 64)), base=str(root)) is False
    with pytest.raises(ValueError, match="spread directories"):
        spread.discard_staged(str(tmp_path), base=str(root))


# A host program wrapper that gives each simulated Spark its own directories for the paths
# the shipped programs use on a real Spark.
HOST = r'''
import sys
replacements = [arg.split("=", 1) for arg in sys.argv[1:]]
length = int(sys.stdin.buffer.readline())
program = sys.stdin.buffer.read(length).decode()
for old, new in replacements:
    program = program.replace(old, new)
exec(compile(program, "<sparkring-pipeline>", "exec"))
'''


class Hosts:
    """A transport over simulated Sparks: pipeline programs run locally with each Spark's directories."""

    mode = "fiber-ssh"

    def __init__(self, tmp_path, document, replacements):
        self.document, self.positions = document, list(range(document["size"]))
        self.hosts = [{"rank": rank} for rank in self.positions]
        self.wrapper = tmp_path / "host.py"
        self.wrapper.write_text(HOST, encoding="utf-8")
        self.replacements = replacements

    def local(self, rank):
        return rank == 0

    def command(self, rank, argv):
        argv = argv[2:] if argv[:2] == ["sudo", "-n"] else list(argv)
        if argv == ["true"]:
            return [sys.executable, "-c", "pass"]
        if argv == fabric_stream.BOOT:
            return [sys.executable, str(self.wrapper), *(f"{old}={new}" for old, new in self.replacements(rank))]
        return ["node" + str(rank), *argv]

    def forwarded(self, rank, remote, local, argv):
        return ["node" + str(rank), f"forward:{remote}->{local}", *argv]


def test_the_layer_archive_rides_the_pipeline_to_a_spark_without_layers_and_skips_one_that_holds_them(
        tmp_path, monkeypatch):
    """Cycle-4: Spark 1 holds the image and forwards, Spark 2 holds no layer, Spark 3 holds two of three."""
    relay_root = tmp_path / "relay-source"
    image_id, config, layers = image_fixture(relay_root)
    sparks = {rank: tmp_path / f"spark{rank}" for rank in range(4)}
    for directory in sparks.values():
        directory.mkdir()

    class Relay:
        port, repository = 5255, "owner/image"

        def __init__(self, reference, directory):
            self.directory = Path(directory)
            self.directory.mkdir(parents=True, exist_ok=True)
            self.fetched = []

        def reference(self, port=None):
            return f"127.0.0.1:{port or self.port}/owner/image@sha256:" + "b" * 64

        def image(self, image_id):
            self.blob(image_id)
            return config, layers

        def blob(self, digest):
            self.fetched.append(digest)
            target = self.directory / digest.removeprefix("sha256:")
            target.write_bytes((relay_root / digest.removeprefix("sha256:")).read_bytes())
            return target

        def close(self):
            pass
    monkeypatch.setattr(install_assets.registry_relay, "Relay", Relay)
    card = {"image_id": image_id, "image_reference": "ghcr.io/owner/image@sha256:" + "b" * 64,
            "image_bytes": 1 << 30, "download_bytes": 1 << 20, "release": "dev-test"}
    images = spread.IMAGE_ROOT
    transport = Hosts(tmp_path, loopback("cycle", 4), lambda rank: [(images, sparks[rank].as_posix())])
    present, held, loaded = {0, 1}, {2: 0, 3: 2}, {}
    docker = {rank: fake_docker(tmp_path, str(rank)) for rank in range(4)}

    class Loader:
        def __init__(self, argv, **kwargs):
            self.rank = int(argv[0][-1])
            self.stdin, self.stdout = io.BytesIO(), io.BytesIO(b"{}")
            self.stdin.close = lambda: loaded.setdefault(self.rank, self.stdin.getvalue())

        def wait(self, timeout=None):
            present.add(self.rank)
            return 0

    def popen(argv, **kwargs):
        return subprocess.Popen(argv, **kwargs) if argv[0] == sys.executable else Loader(argv, **kwargs)
    current = install_assets.Assets(transport, tmp_path / "assets", popen=popen, pipeline=FAST)

    def remote(rank, function, *args, **kwargs):
        if function is install_assets.image_probe:
            return {"present": rank in present, "image_id": image_id, "size_bytes": 1 << 30, "free_bytes": 1 << 40}
        if function is install_assets.layer_prefix:
            return held[rank]
        mine = args[0].replace(images, sparks[rank].as_posix())
        if function is spread.load_staged_image:
            result = spread.load_staged_image(mine, *args[1:], docker=docker[rank][0])
            present.update({rank} if result["loaded"] else set())
            return result
        if function is spread.discard_staged:
            return spread.discard_staged(mine, base=sparks[rank].as_posix())
        raise AssertionError(function)
    current.remote = remote
    result = current.images(card)
    assert result["spread_ranks"] == [2] and result["relayed_ranks"] == [3] and result["reused_ranks"] == [0, 1]
    assert result["spread"]["received"][0]["source"] == 1 and result["spread"]["admin_ranks"] == []
    # Spark 2 imported the whole archive from its verified blobs, which are then removed.
    expected = io.BytesIO()
    install_assets.write_layer_archive(expected, image_id, config, layers, 0,
                                       lambda digest: relay_root / digest.removeprefix("sha256:"),
                                       "127.0.0.1:5255/owner/image:dev-test")
    assert (docker[2][1] / "loaded.tar").read_bytes() == expected.getvalue()
    assert not (sparks[2] / image_id.removeprefix("sha256:")).exists()
    # Spark 1 forwarded without writing; Spark 3 loaded only the layer it lacked through the relay.
    assert not (sparks[1] / image_id.removeprefix("sha256:")).exists()
    assert not (docker[1][1] / "loaded.tar").exists() and not (docker[3][1] / "loaded.tar").exists()
    import tarfile
    with tarfile.open(fileobj=io.BytesIO(loaded[3])) as archive:
        assert [member.name for member in archive.getmembers()][2:] == [
            layers[2][0].removeprefix("sha256:") + "/layer.tar"]
    assert present == {0, 1, 2, 3}


# The checkpoint through the asset path -------------------------------------------------------

class Runner:
    """``model-transfer-prepare`` and ``model-transfer-complete`` of the rank operations, on claimed directories."""

    def __init__(self, rows, required):
        self.rows, self.required, self.calls = rows, set(required), []

    def remote(self, rank, name, data=None):
        manifest = json.loads(data)
        self.calls.append((rank, name))
        with place.claim(self.rows[rank]["model"], REPOSITORY, REVISION) as claimed:
            if name == "model-transfer-prepare":
                os.close(place.staging(claimed, "receive", empty=True))
                present = set(place.listing(claimed.dir_fd)[0])
                return {"ok": True, "needed": sorted(set(manifest["files"]) - present)}
            placed = {n for n, entry in place.journal_load(claimed).files.items() if entry["state"] == "placed"}
            return {"ok": True, "complete": self.required <= placed, "missing": sorted(self.required - placed)}


@linux
def test_the_checkpoint_spreads_through_the_asset_path_into_claimed_directories(tmp_path, mount_table):  # noqa: F811
    source = tmp_path / "source"
    files = write_files(source, 5, 120_000)
    rows = [{"rank": 0, "host": "spark0", "model": str(source)}]
    for rank in (1, 2, 3):
        rows.append({"rank": rank, "host": f"spark{rank}", "model": checkpoint(tmp_path / f"spark{rank}")})
        prepare(rows[rank]["model"])
    records = tmp_path / "records"
    transport = Hosts(tmp_path, loopback("cycle", 4), lambda rank: [
        ('"/proc/self/mountinfo"', json.dumps(str(mount_table))), ('"/var/lib/sparkring/checkpoints/files"',
                                                                   json.dumps(str(records)))])
    current = install_assets.Assets(transport, tmp_path / "assets", pipeline=FAST)
    manifest = {"repository": REPOSITORY, "revision": REVISION, "files": {n: v[1] for n, v in files.items()},
                "sizes": {n: v[0] for n, v in files.items()}}
    runner, complete = Runner(rows, files), {0}
    receives = [{"source": 0, "target": rank, "names": sorted(files), "level": 0, "transport": "fabric"}
                for rank in (1, 2, 3)]
    sent = current.distribute(runner, rows, manifest, 0, receives, complete)
    assert complete == {0, 1, 2, 3}
    assert sorted((row["source"], row["target"], row["level"]) for row in sent) == [(0, 1, 0), (0, 3, 0), (1, 2, 1)]
    for rank in (1, 2, 3):
        recorded = journal(rows[rank]["model"])
        for name in files:
            assert Path(rows[rank]["model"], name).read_bytes() == (source / name).read_bytes()
            assert recorded[name]["origin"] == "fabric"
    assert [call for call in runner.calls if call[1] == "model-transfer-complete"] == [
        (1, "model-transfer-complete"), (2, "model-transfer-complete"), (3, "model-transfer-complete")]


def test_package_updates_follow_the_bootstrap_order_and_name_the_first_spark_not_reached(tmp_path, monkeypatch):
    monkeypatch.setattr(install_assets.distribution, "identity", lambda root: "a" * 40)
    document = loopback("cycle", 8)
    transport = Hosts(tmp_path, document, lambda rank: [])
    current = install_assets.Assets(transport, tmp_path / "assets")
    asked = []

    def remote(rank, function, *args, **kwargs):
        asked.append(rank)
        if rank == 3:
            raise subprocess.CalledProcessError(255, ["ssh"])
        return "a" * 40
    current.remote = remote
    with pytest.raises(NeedsInput) as stopped:
        current.sync_packages()
    assert asked == [1, 7, 2, 6, 3]
    assert str(stopped.value) == ("positions 0-2 and 6-7 are set up; spark3 (position 3) was not reached over cable 2; "
                                  "check the cable and repeat the same command.")


def test_the_spread_check_spreads_test_files_to_every_spark_reports_the_timing_and_removes_them(tmp_path):
    from runtime.host import spread_check
    document = loopback("path", 4)
    state = tmp_path / "state"
    state.mkdir()
    (state / "fabric.json").write_text(fabric_document.encoded(document), encoding="utf-8")
    hosts = [{"node_id": row["node_id"], "host": row["hostname"]} for row in document["positions"]]
    (state / "cluster.json").write_text(json.dumps({"plan": {"spec": {"hosts": hosts}}}), encoding="utf-8")
    root = (tmp_path / "check").as_posix()
    sparks = {rank: tmp_path / f"spark{rank}" for rank in range(4)}
    transport = Hosts(tmp_path, document, lambda rank: [(root, sparks[rank].as_posix())] if rank else [])
    current = install_assets.Assets(transport, tmp_path / "assets", pipeline=FAST)

    def remote(rank, function, *args, **kwargs):
        assert function is spread_check.remove
        return function(args[0].replace(root, sparks[rank].as_posix()), base=sparks[rank].as_posix())
    current.remote = remote
    lines = []
    report = spread_check.run(state, count=3, size=150_000, allow_serving=True, say=lines.append, transport=transport,
                              assets=current, root=root)
    assert report["schema"] == "sparkring-spread-check/v1" and report["result"] == "complete"
    assert sorted((row["source"], row["target"]) for row in report["received"]) == [(0, 1), (1, 2), (2, 3)]
    assert any(line.startswith("  Pass 1, next from spark0 (position 0): 1 ") and line.count("ms") == 2
               for line in lines)
    assert any("The test files from position 0" in line for line in lines)
    saved = json.loads(next((state / "fabric-reports").glob("spread-check-*.json")).read_text(encoding="utf-8"))
    assert saved["received"] == report["received"]
    # Every Spark verified each file against Node A's SHA-256 on arrival; then every copy was removed.
    assert not any((sparks[rank] / "received").exists() for rank in (1, 2, 3)) and not Path(root, "source").exists()
