"""``sudo sparkring fabric tune`` offline: SIRCL's ring harness runs in this process on simulated Sparks.

The harness is SIRCL's own command line (``sparkring_sircl.ring.cli.main``):
its site checks, plans, run loop, result merging and table building run as
they do on Node A. Only each Spark's SSH answers are simulated (``Sparks``):
its facts and routes come from the fabric document and the relay plan setup
records, and every rank's result carries synthetic timings of the candidates
the tune command times (``timing``). The timings are a cost model chosen so
that the fastest candidate changes with the size; they are not measurements.

The tests follow a table from the simulated measurement to its use: the
measured table SparkRing writes, its binding and digest, its copies on every
Spark, the deployment section and containers the transport adapter renders
from it, the SIRCL session's choice of table, the receipt check and the
staleness checks.
"""
import argparse
import contextlib
import hashlib
import ipaddress
import json
import re
from pathlib import Path

import pytest

from runtime.common import fabric_document, fabric_layout, image_lock, installer, transport
from runtime.common.test_image_lock import sircl_block, sircl_lock
from runtime.common.test_transport import TP4, install_site
from runtime.host import fabric, fabric_tune, relays
from runtime.host.test_fabric_layouts import plan_of
from runtime.host.test_transport_receipts import receipt, report
from spark_transport.sircl.sparkring_sircl import tuning as sircl_tuning
from spark_transport.sircl.sparkring_sircl.ring import cli, remote

MARKER = {"binary": relays.MARKER_BINARY, "sha256": "ab" * 32}
GPU, KERNEL = "580.95.05", "6.11.0-1016-nvidia"
NOW = 1791540000.0   # 2026-10-09T09:20:00Z


def image():
    """A v3 image lock whose SIRCL layer is this package's SIRCL build, as the harness measures it."""
    key = fabric_tune.package_key()
    version, abi = key["sircl"].split("/abi")
    block = sircl_block(version, int(abi))
    block["native"] = {"path": f"{image_lock.LIBRARY_DIRECTORY}/roce_proxy-{key['native']}.so", "sha256": "2" * 64,
                       "source_digest": key["native"]}
    block["tuning_key"] = key
    return sircl_lock(sircl=block)


def recorded(shape, size):
    plan = plan_of(fabric_layout.layout(shape, size))
    document, relay_plan = fabric.prepare(plan, cluster="test", marker=MARKER)
    return {"name": "test", "plan": plan}, document, relay_plan


# Synthetic timings: a fixed cost, a cost per byte, and a cost per link item.

def timing(collective, choice, nbytes):
    grid = choice.get("grid")
    penalty = abs((grid or 16).bit_length() - 5) * 0.5
    if choice.get("algorithm") == "oneshot":
        return 8 + nbytes / 2000 + penalty
    if choice.get("algorithm") == "twoshot":
        return 14 + nbytes / 4000 + penalty
    schedule = choice.get("schedule")
    if schedule == "pieces":
        return 30 + nbytes / 5000 + penalty
    if schedule == "chain":
        return 60 + nbytes / 9000 + nbytes / choice["piece"] * 0.5
    if schedule == "ring":
        return 80 + nbytes / 12000 + nbytes / choice["piece"] * 0.8 + 5 * choice.get("stagger", 0)
    return (6 if collective == "all_gather" else 9) + nbytes / 3000 + penalty


def candidates(collective, nbytes, options):
    grids, pieces, staggers = options["tune_grids"], options["tune_pieces"], options["tune_staggers"]
    if nbytes < options["tune_large_from"]:
        if collective == "all_reduce":
            return [{"algorithm": name, "grid": grid} for name in ("oneshot", "twoshot") for grid in grids]
        if collective == "all_gather":
            return [{"grid": grid} for grid in grids]
        return []
    found = [{"schedule": "pieces", "grid": grid} for grid in grids]
    found += [{"schedule": "chain", "piece": piece} for piece in pieces]
    found += [{"schedule": "ring", "piece": piece, "stagger": stagger, "gather_stagger": stagger}
              for piece in pieces for stagger in staggers]
    return found


def rank_result(plan, global_rank):
    options = plan["options"]
    runs = []
    for collective in ("all_reduce", "all_gather", "reduce_scatter"):
        for mode in options["tune_modes"]:
            for nbytes in options["tune_sizes"]:
                for choice in candidates(collective, nbytes, options):
                    runs.append({"collective": f"{collective}_tune", "mode": mode, "shape": [nbytes // 2], "dim": 0,
                                 "bytes": nbytes, "correct": True, "mismatched_calls": 0, "checked": 1, "counters": {},
                                 "times_us": [round(timing(collective, choice, nbytes), 3)],
                                 "tune": {"collective": collective, "choice": choice}})
    rank = plan["ranks"][global_rank]
    return {"global_rank": global_rank, "group": rank["group"], "host": rank["host"], "runs": runs, "error": None,
            "exit_code": 0}


class Sparks:
    """Each Spark's answers to the harness's SSH commands (``ring/remote.py``)."""

    def __init__(self, cluster, document, relay_plan, *, failing=()):
        self.document, self.failing = document, set(failing)
        self.targets = {host["host"]: position for position, host in enumerate(cluster["plan"]["spec"]["hosts"])}
        self.files, self.runs, self.commands = {}, [], []
        self.routes = []
        for position, row in enumerate(document["positions"]):
            table = []
            for entry in row["ports"].values():
                for function in entry["functions"].values():
                    if function["address"]:
                        table.append((ipaddress.IPv4Interface(function["address"]).network, function["netdev"],
                                      str(ipaddress.IPv4Interface(function["address"]).ip)))
            for route in relay_plan["positions"][position]["routes"]:
                table.append((ipaddress.IPv4Network(route["dst"]), route["dev"], route["src"]))
            self.routes.append(sorted(table, key=lambda item: -item[0].prefixlen))
        self.management = {host["host"]: host["management_address"] for host in cluster["plan"]["spec"]["hosts"]}

    def ok(self, text=""):
        return remote.Result(0, text, "")

    def facts(self, target):
        position = self.targets[target]
        lines = [f"hostname\tspark{position}", "docker\t27.3.1", f"image\t{image()['image_id']}", "gpu\t0, NVIDIA GB10",
                 f"lan\t{self.management[target]}/24", f"address:enP7s7\t{self.management[target]}/24"]
        for row in self.document["positions"][position]["ports"].values():
            for function in row["functions"].values():
                state = "ACTIVE" if function["address"] else "DOWN"
                lines.append(f"device:{function['rdma']}\t{state} {function['netdev']}")
                if function["address"]:
                    lines.append(f"address:{function['netdev']}\t{function['address']}")
        return "\n".join(lines) + "\n"

    def route(self, target, destination):
        address = ipaddress.IPv4Address(destination)
        for network, netdev, source in self.routes[self.targets[target]]:
            if address in network:
                return f"route:{destination}\t{destination} dev {netdev} src {source} uid 0"
        return f"route:{destination}\t"

    def __call__(self, target, command, *, timeout=60, input_bytes=None, binary="ssh"):
        self.commands.append((target, command.split()[0]))
        if "hostname\t" in command:
            return self.ok(self.facts(target))
        if "ps --format" in command:
            return self.ok("")
        if command.startswith('echo "route:'):
            return self.ok("\n".join(self.route(target, item) for item in re.findall(r'echo "route:([0-9.]+)', command)))
        if "tar -C" in command or "sparkring_sircl.build" in command or command.startswith("test -d"):
            return self.ok("built\n")
        if " cat > " in command:
            path = command.rsplit("cat > ", 1)[1].strip("'")
            self.files[path] = input_bytes
            return self.ok()
        if " run -d " in command:
            name = re.search(r"--name (\S+)", command).group(1)
            self.runs.append(name)
            if any(f"-{layout}-r" in name for layout in self.failing):
                return remote.Result(125, "", "docker: device busy")
            return self.ok("container-id\n")
        if "inspect -f" in command:
            ranks = re.findall(r'echo "(\d+) ', command)
            return self.ok("".join(f"{rank} exited 0\n" for rank in ranks))
        if command.startswith("cat "):
            path = command[4:].split(" 2>", 1)[0].strip("'")
            if path.endswith(".log"):
                return self.ok("rank log\n")
            if any(f"/{layout}/rank-" in path for layout in self.failing):
                return remote.Result(1, "", "")
            plan = json.loads(self.files[path.rsplit("/", 1)[0] + "/plan.json"])
            rank = int(re.search(r"rank-(\d+)\.json$", path).group(1))
            return self.ok(json.dumps(rank_result(plan, rank)))
        if "ps -aq --filter" in command:
            return self.ok("0\n")
        raise AssertionError(command)


class Harness:
    """``python -m sparkring_sircl.ring`` in this process; its SSH reaches ``Sparks``."""

    def __init__(self, sparks, monkeypatch, *, interrupt=None):
        self.sparks, self.calls, self.interrupt = sparks, [], interrupt
        monkeypatch.setattr(remote, "ssh", sparks)

    def __call__(self, argv, *, cwd, timeout, log):
        name = argv[argv.index("--name") + 1] if "--name" in argv else None
        self.calls.append((argv[0], name))
        if self.interrupt and (argv[0], name) == self.interrupt:
            raise KeyboardInterrupt
        with open(log, "a", encoding="utf-8") as stream, contextlib.redirect_stdout(stream), \
                contextlib.redirect_stderr(stream):
            return cli.main(list(argv))


class Fleet:
    """``fabric.Access`` over simulated Sparks: each runs the package's own node commands under its root."""

    def __init__(self, root, size, *, facts=None):
        self.root, self.size, self.calls = Path(root), size, []
        self.facts = facts or [{"schema": fabric_tune.FACTS_SCHEMA, "gpu": GPU, "kernel": KERNEL}] * size

    def host(self, rank):
        return self.root / f"spark{rank}"

    def node(self, rank, argv, *, data=None):
        self.calls.append((rank, argv[0]))
        if argv[0] == "tuning-facts":
            return json.dumps(self.facts[rank])
        if argv[0] == "sircl-tuning":
            return json.dumps(fabric_tune.install_tables(data, root=self.host(rank)))
        if argv[0] == "fabric-check":
            return json.dumps({"schema": fabric.CHECK_SCHEMA, "position": rank,
                               "rows": [{"kind": "interfaces", "what": "fabric functions", "state": "ok"}]})
        raise AssertionError(argv)


def arguments(*extra):
    parser = argparse.ArgumentParser()
    fabric_tune.add_arguments(parser)
    return parser.parse_args(list(extra))


@pytest.fixture
def cycle4(tmp_path, monkeypatch):
    """A recorded four-Spark cycle on Node A, its simulated Sparks and the harness."""
    cluster, document, relay_plan = recorded("cycle", 4)
    state = tmp_path / "controller"
    state.mkdir()
    (state / "cluster.json").write_text(json.dumps(cluster))
    (state / "fabric.json").write_text(fabric_document.encoded(document))
    monkeypatch.setattr(fabric_tune, "chosen_image", lambda name=None: image())
    sparks = Sparks(cluster, document, relay_plan)
    return {"state": state, "cluster": cluster, "document": document, "sparks": sparks,
            "harness": Harness(sparks, monkeypatch), "fleet": Fleet(tmp_path / "hosts", 4)}


def tune(setup, *extra, harness=None):
    return fabric_tune.command(arguments("--execute", "--quick", *extra), state=setup["state"], access=setup["fleet"],
                               harness=harness or setup["harness"], now=lambda: NOW,
                               host_root=setup["fleet"].host(0))


# What can be measured.

@pytest.mark.parametrize("shape, size, measured, unmeasured", [
    ("pair", 2, [], ["pair"]),
    ("path", 3, ["pair"], ["path-3"]),
    ("path", 5, ["pair", "path-3", "path-4"], ["path-5"]),
    ("path", 8, ["pair", "path-3", "path-4", "path-5"], []),
    ("cycle", 3, ["pair", "cycle-3"], []),
    ("cycle", 4, ["pair", "path-3", "cycle-4"], []),
    ("cycle", 8, ["pair", "path-3", "path-4", "path-5", "cycle-8"], []),
])
def test_every_group_a_deployment_can_use_is_listed_with_what_the_harness_cannot_measure(shape, size, measured,
                                                                                          unmeasured):
    _, document, _ = recorded(shape, size)
    rows = fabric_tune.layouts(document)
    assert [row["name"] for row in rows if row["unmeasurable"] is None] == measured
    assert [row["name"] for row in rows if row["unmeasurable"]] == unmeasured
    for row in rows:
        assert row["positions"] == list(range(len(row["positions"])))


def test_the_plan_contacts_nothing_and_names_layouts_limits_and_host_changes(cycle4, capsys):
    assert fabric_tune.command(arguments(), state=cycle4["state"], access=cycle4["fleet"],
                               harness=cycle4["harness"]) == 0
    out = capsys.readouterr().out
    assert cycle4["harness"].calls == [] and cycle4["fleet"].calls == []
    assert "Fabric: cycle-4" in out and "  pair     Sparks 0-1" in out and "  cycle-4  Sparks 0-3" in out
    assert "each group at most 30 min of measurement" in out and "NCCL is not measured" in out
    assert f"files under {fabric_tune.REMOTE_DIR}" in out and "Review, then repeat with --execute." in out


def test_a_pair_fabric_and_another_sircl_build_block_the_measurement(tmp_path, monkeypatch):
    cluster, document, _ = recorded("pair", 2)
    value = fabric_tune.plan(cluster, document, image())
    assert any("cabled port 0 to port 0" in blocker for blocker in value["blockers"])
    other = sircl_lock()
    value = fabric_tune.plan(*recorded("cycle", 4)[:2], other)
    assert any("differ from image" in blocker and "--image" in blocker for blocker in value["blockers"])
    with pytest.raises(ValueError, match="cannot measure path-5"):
        fabric_tune.plan(*recorded("path", 5)[:2], image(), names=["path-5"])
    with pytest.raises(ValueError, match="this fabric's group shapes are pair, path-3, cycle-4"):
        fabric_tune.plan(*recorded("cycle", 4)[:2], image(), names=["cycle-8"])


# The measurement, the table and its binding.

def test_a_measured_table_is_produced_bound_digested_distributed_and_used_by_the_adapter(cycle4, capsys):
    assert tune(cycle4) == 0
    out = capsys.readouterr().out
    state, document = cycle4["state"], cycle4["document"]
    measured = transport.load_tuning(state / transport.MEASURED_TUNING, host_root=cycle4["fleet"].host(0))
    # Bound to this fabric, this image and its SIRCL build, and the Sparks' drivers.
    assert measured["source"] == "measured" and measured["fabric"] == document["id"]
    assert measured["image_id"] == image()["image_id"] and measured["measured_at"] == "2026-10-09"
    assert measured["binding"]["tuning_key"] == fabric_tune.package_key()
    assert measured["binding"]["drivers"] == {str(p): {"gpu": GPU, "kernel": KERNEL} for p in range(4)}
    assert measured["binding"]["defaults_sha256"] == transport.tuning_digest(transport.load_tuning())
    assert {name: row["source"] for name, row in measured["layouts"].items()} == {
        "pair": "measured", "path-3": "measured", "cycle-4": "measured", "cycle": "default:rules",
        "cycle-8": "default:measured", "path": "default:rules"}
    # One SIRCL table per layout, the same bytes on every Spark under their SHA-256.
    assert len(measured["tables"]) == 3
    for entry in measured["tables"]:
        assert entry["path"] == f"{transport.HOST_TABLES}/{entry['sha256']}.json"
        for rank in range(4):
            copy = cycle4["fleet"].host(rank) / entry["path"].lstrip("/")
            assert hashlib.sha256(copy.read_bytes()).hexdigest() == entry["sha256"]
    assert "Measured tuning table:" in out and "rows cycle-4, pair, path-3" in out
    # The harness ran preflight and tune per layout, after one stage.
    assert [call for call in cycle4["harness"].calls if call[0] != "preflight"] == [
        ("stage", "stage"), ("tune", "pair"), ("tune", "path-3"), ("tune", "cycle-4")]
    assert ("preflight", "cycle-4") in cycle4["harness"].calls
    image_value = image()
    table, notes = transport.tuning_in_effect(state, document, image_value, host_root=cycle4["fleet"].host(0),
                                              drivers={0: {"gpu": GPU, "kernel": KERNEL}})
    assert table == measured and notes == []
    for positions, name in (([0, 1, 2, 3], "cycle-4"), ([1, 2, 3], "path-3"), ([2, 3], "pair")):
        section = transport.section(image_value, document, positions, nccl="never", tuning=table,
                                    host_root=cycle4["fleet"].host(0))
        tuning = section["tuning"]
        assert (tuning["source"], tuning["row"], tuning["row_source"]) == ("measured", name, "measured")
        assert tuning["sha256"] == transport.tuning_digest(measured) and tuning["measured_at"] == "2026-10-09"
        assert tuning["settings"] == {"link_slots": 12, "link_slot": 1048576}
        (entry,) = tuning["tables"]
        on_spark = cycle4["fleet"].host(0) / entry["path"].lstrip("/")
        # The session's own choice: the table whose key matches its group and build (its setup agreement
        # then carries the table's hash).
        topology = transport.group_topology(section["group"]["layout"], positions)
        own = sircl_tuning.facts_for_layout(topology.session_layout(), topology.lane_count)
        chosen, _ = sircl_tuning.select_table([str(on_spark)], own)
        assert chosen is not None and chosen.hash == entry["hash"]
        lines = transport.plan_lines(section)
        assert lines[0] == f"Transport: sircl on every collective, NCCL off (measured on this fabric 2026-10-09, {name})"
        assert any(f"SIRCL tuning table {entry['hash']}" in line for line in lines)
        assert "  SIRCL settings: link_slot 1048576, link_slots 12" in lines


def test_a_deployment_on_the_measured_table_mounts_it_and_its_receipts_are_checked_against_it(cycle4):
    assert tune(cycle4) == 0
    image_value = image()
    table, _ = transport.tuning_in_effect(cycle4["state"], cycle4["document"], image_value,
                                          host_root=cycle4["fleet"].host(0))
    section = transport.section(image_value, cycle4["document"], [0, 1, 2, 3], nccl="never", tuning=table,
                                host_root=cycle4["fleet"].host(0))
    lock = installer.make_lock(TP4, install_site(4), "1" * 40, "2" * 64, image_runtime=image_lock.v2_view(image_value),
                               transport=section)
    (entry,) = section["tuning"]["tables"]
    for spec in installer.specifications(lock):
        environment = spec.environment
        assert environment["SIRCL_TUNING_TABLE"] == f"{transport.TABLE_TARGET}/{entry['hash']}.json"
        assert (environment["SIRCL_LINK_SLOTS"], environment["SIRCL_LINK_SLOT_BYTES"]) == ("12", "1048576")
        mounts = {mount.target: mount for mount in spec.mounts}
        mounted = mounts[f"{transport.TABLE_TARGET}/{entry['hash']}.json"]
        assert mounted.source == entry["path"] and mounted.read_only
    stats = {"link_slots": 12, "link_slot_bytes": 1048576}
    good = [report(rank, [dict(receipt(rank, tuning=entry["hash"]), session_stats=stats)]) for rank in range(4)]
    from runtime.host import transport_receipts
    verdict = transport_receipts.evaluate(lock, good, now=lambda: 0)
    assert verdict["verdict"] == "as-expected", verdict["problems"]
    assert any("the sessions report the row's link_slot_bytes 1048576, link_slots 12" in line
               for line in verdict["lines"])
    other = [report(rank, [dict(receipt(rank, tuning="0" * 16), session_stats=dict(stats, link_slots=8))])
             for rank in range(4)]
    verdict = transport_receipts.evaluate(lock, other, now=lambda: 0)
    assert verdict["verdict"] == "differs"
    assert any("tuning table 0000000000000000, the plan matched " + entry["hash"] in problem
               for problem in verdict["problems"])
    assert any("the session's link_slots is 8, the tuning row sets 12" in problem for problem in verdict["problems"])


def test_each_spark_holds_the_tables_its_deployment_mounts(cycle4):
    assert tune(cycle4) == 0
    image_value = image()
    table, _ = transport.tuning_in_effect(cycle4["state"], cycle4["document"], image_value,
                                          host_root=cycle4["fleet"].host(0))
    section = transport.section(image_value, cycle4["document"], [0, 1], nccl="never", tuning=table,
                                host_root=cycle4["fleet"].host(0))
    host = cycle4["fleet"].host(1)
    path = host / fabric_document.HOST_PATH.lstrip("/")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(fabric_document.encoded(cycle4["document"]))
    assert transport.check_host_document(section, root=host)["ok"]
    (entry,) = section["tuning"]["tables"]
    (host / entry["path"].lstrip("/")).unlink()
    with pytest.raises(transport.TransportError, match="fabric tune --distribute"):
        transport.check_host_document(section, root=host)
    assert fabric_tune.command(arguments("--distribute"), state=cycle4["state"], access=cycle4["fleet"],
                               host_root=cycle4["fleet"].host(0)) == 0
    assert transport.check_host_document(section, root=host)["ok"]


# Resuming, limits and refusals.

def test_an_interrupted_measurement_resumes_at_the_first_layout_without_a_table(cycle4, monkeypatch):
    interrupted = Harness(cycle4["sparks"], monkeypatch, interrupt=("tune", "path-3"))
    with pytest.raises(KeyboardInterrupt):
        tune(cycle4, harness=interrupted)
    assert not (cycle4["state"] / transport.MEASURED_TUNING).exists()
    progress = fabric_tune.latest_progress(cycle4["state"])
    assert progress["layouts"]["pair"]["state"] == "measured" and progress["layouts"]["path-3"]["state"] == "running"
    assert tune(cycle4) == 0
    tuned = [call for call in cycle4["harness"].calls if call[0] == "tune"]
    assert tuned == [("tune", "path-3"), ("tune", "cycle-4")]
    measured = json.loads((cycle4["state"] / transport.MEASURED_TUNING).read_text())
    assert sorted(name for name, row in measured["layouts"].items() if row["source"] == "measured") == [
        "cycle-4", "pair", "path-3"]
    # Everything measured: a repeat distributes and records again and measures nothing.
    del cycle4["harness"].calls[:]
    assert tune(cycle4) == 0 and cycle4["harness"].calls == []
    assert tune(cycle4, "--fresh", "--layouts", "pair") == 0
    assert [call for call in cycle4["harness"].calls if call[0] == "tune"] == [("tune", "pair")]


def test_a_finished_run_without_its_table_is_collected_with_tune_table(cycle4):
    assert tune(cycle4, "--layouts", "pair") == 0
    progress_path = next((cycle4["state"] / fabric_tune.WORK).glob(f"*/{fabric_tune.PROGRESS}"))
    progress = json.loads(progress_path.read_text())
    entry = progress["layouts"]["pair"]
    folder = progress_path.parent / entry["results"]
    (folder / "tuning-group0.json").unlink()
    # The harness run exited 0; the command stopped before it recorded the table.
    progress["layouts"]["pair"] = {key: entry[key] for key in ("run", "positions", "results", "exit")} | {
        "state": "running"}
    progress_path.write_text(json.dumps(progress))
    del cycle4["harness"].calls[:]
    assert tune(cycle4, "--layouts", "pair") == 0
    assert cycle4["harness"].calls == [("tune-table", None)]
    assert (folder / "tuning-group0.json").is_file()


def test_a_failed_layout_leaves_the_others_measured_and_the_command_exits_1(cycle4, monkeypatch, capsys):
    cycle4["sparks"].failing.add("path-3")
    assert tune(cycle4) == 1
    out = capsys.readouterr().out
    assert "path-3: failed: the ring harness's tune run exited with 1" in out
    assert "Repeat sudo sparkring fabric tune --execute" in out
    measured = json.loads((cycle4["state"] / transport.MEASURED_TUNING).read_text())
    # A path-3 group then takes the default table's row of its shape.
    assert "path-3" not in measured["layouts"] and measured["layouts"]["path"]["source"] == "default:rules"
    assert measured["layouts"]["cycle-4"]["source"] == "measured"


def test_layouts_beyond_the_time_limit_stay_pending(cycle4, capsys):
    assert tune(cycle4, "--max-hours", "0.01") == 1
    out = capsys.readouterr().out
    assert "pair: pending: not started within --max-hours 0.01" in out
    assert [call for call in cycle4["harness"].calls if call[0] == "tune"] == []
    assert not (cycle4["state"] / transport.MEASURED_TUNING).exists()


def test_a_serving_model_stops_the_measurement_unless_stop_serving(cycle4, monkeypatch, capsys):
    busy = {0: {"profile": TP4, "placement": None}, 1: {"profile": TP4, "placement": None}}
    stopped = []
    monkeypatch.setattr(fabric_tune, "serving", lambda state, size: {} if stopped else busy)
    monkeypatch.setattr(fabric_tune, "stop_models", lambda state, size, say=print: stopped.append(TP4) or [
        {"profile": TP4, "placement": None, "deployment": "/var/lib/sparkring/controller/deployments/x"}])
    with pytest.raises(ValueError, match="serves on every Spark.*--stop-serving"):
        tune(cycle4)
    assert cycle4["harness"].calls == []
    assert tune(cycle4, "--stop-serving") == 0
    assert stopped == [TP4] and "Stopped for the measurement: " + TP4 in capsys.readouterr().out


def test_the_lock_keeps_other_operations_out_while_it_measures(cycle4, monkeypatch):
    from runtime.common import process_lock
    with process_lock.hold(cycle4["state"] / "install.lock"):
        with pytest.raises(ValueError, match="Another operation is active"):
            tune(cycle4)


def test_sparks_with_different_drivers_are_refused(cycle4):
    cycle4["fleet"].facts = [{"gpu": GPU, "kernel": KERNEL}] * 3 + [{"gpu": "580.65.06", "kernel": KERNEL}]
    with pytest.raises(ValueError, match="different GPU drivers or kernels.*position 3: 580.65.06"):
        tune(cycle4)


def test_tables_arrive_once_and_by_their_digest(tmp_path):
    table = sircl_tuning.build_document(
        {"shape": "pair", "world": 2, "lanes": 2, "max_relays": 0, "native": "0" * 16, "kernels": "1" * 16,
         "sircl": "0.2.0/abi9", "image": "img"},
        [{"collective": "all_reduce", "mode": "eager", "bytes": 8192, "choice": {"algorithm": "oneshot"},
          "p50_us": 9.0}])
    text = json.dumps(table) + "\n"
    digest = hashlib.sha256(text.encode()).hexdigest()
    payload = json.dumps({"schema": fabric_tune.TABLES_SCHEMA, "tables": [{"sha256": digest, "text": text}]})
    assert fabric_tune.install_tables(payload, root=tmp_path)["written"] == [digest]
    assert fabric_tune.install_tables(payload, root=tmp_path)["present"] == [digest]
    (tmp_path / transport.HOST_TABLES.lstrip("/") / f"{digest}.json").write_text("{}")
    assert fabric_tune.install_tables(payload, root=tmp_path)["written"] == [digest]
    wrong = json.dumps({"schema": fabric_tune.TABLES_SCHEMA, "tables": [{"sha256": "0" * 64, "text": text}]})
    with pytest.raises(ValueError, match="other bytes"):
        fabric_tune.install_tables(wrong, root=tmp_path)


def test_the_facts_of_a_spark_are_its_gpu_driver_kernel_and_interface_address():
    def run(argv, **kwargs):
        if argv[0] == "nvidia-smi":
            return type("Done", (), {"returncode": 0, "stdout": f"{GPU}\n"})()
        return type("Done", (), {"returncode": 0, "stdout": json.dumps(
            [{"addr_info": [{"family": "inet", "local": "192.0.2.10", "prefixlen": 24}]}])})()
    value = fabric_tune.local_facts("enP7s7", run=run)
    assert value["gpu"] == GPU and value["interface"] == {"name": "enP7s7", "address": "192.0.2.10"}
    missing = fabric_tune.local_facts(run=lambda argv, **kwargs: (_ for _ in ()).throw(FileNotFoundError()))
    assert missing["gpu"] is None


# Staleness.

@pytest.mark.parametrize("change, reason", [
    ("fabric", "it was measured on fabric"),
    ("image", "it was measured with image"),
    ("sircl", "it was measured with SIRCL"),
    ("driver", "the GPU driver of position 0 changed from 580.95.05 to 590.10.01"),
    ("kernel", "the kernel of position 0 changed"),
])
def test_a_stale_table_gives_way_to_the_default_and_says_why(cycle4, change, reason):
    assert tune(cycle4) == 0
    document, image_value = cycle4["document"], image()
    drivers = {0: {"gpu": GPU, "kernel": KERNEL}}
    if change == "fabric":
        document = recorded("cycle", 8)[1]
    elif change == "image":
        image_value = dict(image_value, image_id="sha256:" + "9" * 64, name="another-image")
    elif change == "sircl":
        block = dict(image_value["sircl"], version="0.3.0", tuning_key=dict(image_value["sircl"]["tuning_key"],
                                                                          sircl="0.3.0/abi9"))
        image_value = dict(image_value, sircl=block)
    elif change == "driver":
        drivers = {0: {"gpu": "590.10.01", "kernel": KERNEL}}
    else:
        drivers = {0: {"gpu": GPU, "kernel": "6.14.0-1001-nvidia"}}
    table, notes = transport.tuning_in_effect(cycle4["state"], document, image_value,
                                              host_root=cycle4["fleet"].host(0), drivers=drivers)
    assert table == transport.load_tuning()
    assert reason in notes[0] and notes[0].endswith("sudo sparkring fabric tune measures this fabric again")


def test_status_says_which_table_installations_use_and_when_it_went_stale(cycle4, monkeypatch):
    state, host = cycle4["state"], cycle4["fleet"].host(0)
    value = fabric_tune.summary(state, facts={"gpu": GPU, "kernel": KERNEL}, host_root=host)
    assert fabric_tune.status_line(value).startswith("SIRCL tuning: the default table; sudo sparkring fabric tune")
    assert tune(cycle4) == 0
    image_value = image()
    monkeypatch.setattr(fabric_tune.image_lock, "catalog", lambda: [{"lock": image_value}])
    monkeypatch.setattr(fabric_tune.image_lock, "default", lambda: image_value)
    value = fabric_tune.summary(state, facts={"gpu": GPU, "kernel": KERNEL}, host_root=host)
    assert fabric_tune.status_line(value) == (f"SIRCL tuning: measured on this fabric 2026-10-09 for cycle-4, pair, "
                                              f"path-3 with image {image_value['name']}")
    value = fabric_tune.summary(state, facts={"gpu": "590.10.01", "kernel": KERNEL}, host_root=host)
    assert value["state"] == "stale"
    assert fabric_tune.status_line(value).startswith(
        "SIRCL tuning: the table measured 2026-10-09 no longer applies (the GPU driver of position 0 changed from "
        f"{GPU} to 590.10.01); installations use the default table")
    monkeypatch.setattr(fabric_tune.image_lock, "default", lambda: dict(image_value, image_id="sha256:" + "8" * 64,
                                                                        name="newer-image"))
    value = fabric_tune.summary(state, facts={"gpu": GPU, "kernel": KERNEL}, host_root=host)
    assert fabric_tune.status_line(value).endswith("installations without --image use newer-image, which it does "
                                                   "not cover")


def test_row_settings_hold_the_tune_sessions_link_slot_and_slots():
    options = {"tune_pieces": [262144, 524288, 1048576], "tune_staggers": [0, 1]}
    ring = {"decisions": [{"collective": "all_reduce", "mode": "graph", "intervals": [
        {"from": 4096, "choice": {"algorithm": "oneshot", "grid": 8}, "nccl": False},
        {"from": 1 << 22, "choice": {"schedule": "ring", "piece": 524288, "stagger": 1}, "nccl": False}]}]}
    assert fabric_tune.row_settings(ring, options, 8) == {"link_slots": 12, "link_slot": 1048576}
    assert fabric_tune.row_settings(ring, dict(options, tune_staggers=[0, 2]), 8) == {"link_slots": 16,
                                                                                       "link_slot": 1048576}
    assert fabric_tune.row_settings(ring, dict(options, tune_pieces=[262144]), 4) == {"link_slots": 12}
    small = {"decisions": [{"collective": "all_reduce", "mode": "graph", "intervals": [
        {"from": 4096, "choice": {"algorithm": "twoshot", "grid": 16}, "nccl": False}]}]}
    assert fabric_tune.row_settings(small, options, 8) == {}


def test_the_harness_runs_as_a_subprocess_from_this_package_with_a_deadline(cycle4, tmp_path):
    assert tune(cycle4, "--layouts", "pair") == 0
    folder = next((cycle4["state"] / fabric_tune.WORK).glob("*/results/*/pair"))
    table = (folder / "tuning-group0.json").read_bytes()
    (folder / "tuning-group0.json").unlink()
    log = tmp_path / "harness.log"
    # tune-table reads the run's results only (OFFLINE).
    assert fabric_tune.run_harness(["tune-table", "--results", str(folder)], cwd=str(tmp_path), timeout=120,
                                   log=str(log)) == 0
    rebuilt = json.loads((folder / "tuning-group0.json").read_bytes())
    assert sircl_tuning.document_hash(dict(rebuilt, created="")) == sircl_tuning.document_hash(
        dict(json.loads(table), created=""))
    assert "-m sparkring_sircl.ring tune-table" in log.read_text() and "tuning table of group 0" in log.read_text()
    assert fabric_tune.run_harness(["tune-table", "--results", str(folder)], cwd=str(tmp_path), timeout=0.001,
                                   log=str(log)) == fabric_tune.TIMED_OUT


def test_the_commands_route_to_this_module(tmp_path, monkeypatch, capsys):
    import io
    import os
    import sys
    from scripts import sparkring, sparkring_node
    monkeypatch.setattr(os, "geteuid", lambda: 0, raising=False)
    seen = []
    monkeypatch.setattr(fabric_tune, "command", lambda args: seen.append((args.execute, args.layouts)) or 0)
    assert sparkring.main(["fabric", "tune", "--layouts", "pair,cycle-4"]) == 0
    assert seen == [(False, "pair,cycle-4")]
    monkeypatch.setattr(fabric_tune, "local_facts", lambda interface=None: {"gpu": GPU, "interface": interface})
    assert sparkring_node.main(["tuning-facts", "--interface", "enP7s7"]) == 0
    assert json.loads(capsys.readouterr().out) == {"gpu": GPU, "interface": "enP7s7"}
    monkeypatch.setattr(fabric_tune, "install_tables", lambda text: {"sha256": [json.loads(text)["schema"]]})
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"schema": fabric_tune.TABLES_SCHEMA})))
    assert sparkring_node.main(["sircl-tuning"]) == 0
    assert json.loads(capsys.readouterr().out) == {"sha256": [fabric_tune.TABLES_SCHEMA]}
