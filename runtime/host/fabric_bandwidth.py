"""Measure the RDMA bandwidth of each fabric cable of the recorded pair or four-Spark ring.

Status: implemented. The measurement is the one in
``performance/records/transport/fabric-cable-bandwidth-20261002.md``; no
hardware run of this command is on record.

Each Spark has two ConnectX ports, and each port appears as two network
functions that share its cable (``cabling``). The two PCIe Gen5 x4 links
``enp1s0`` and ``enP2p1s0`` each carry one function of each port (``f0`` and
``f1``) and limit a function to about 109 Gb/s each way, so a bidirectional
RDMA write test of a healthy cable reaches about 213 Gb/s per function. A
cable can stay up and count no errors (no RoCE retransmits, no uncorrectable
FEC) while it carries far less in both directions at once, about 119 or
26 Gb/s per function, although a one-way test still reaches about 109 Gb/s.
Model decode looks normal over such a cable; prefill is slower. Rebooting
both Sparks on the cable restores full speed; a link reset or a driver
restart does not. FEC corrected-bit counts differ by cable model and are not
a signal.

The check runs ``ib_write_bw`` (Debian package ``perftest``) between the two
ends of each function, one function at a time: functions that share a PCIe
link share its bandwidth, so concurrent tests would disturb each other. The
server end runs ``serve``, sent over the administration access as Python source: it
picks a free TCP port from ``PORTS``, starts the server under ``timeout``,
announces the port once the server listens, and stops the server when Node A
closes the program's input. The client end runs ``ib_write_bw`` with the
server's fabric address. Both ends use RoCE GID index 3, which SparkRing
keeps equal to each function's IPv4 address; the check reads that entry on
both ends first and reports a function whose entry holds something else
instead of testing it.

The measured cables are the pair's cable between the ports 0, or the four
ring cables from port 0 of rank r to port 1 of rank r+1. Every function's
ends and addresses come from the setup plan in the cluster record. A pair's
cable between the ports 1 has no fabric addresses and is not measured.

The test fills a cable for several seconds per function and would slow a
model that uses it, so cables that touch a serving model's Sparks are skipped
unless the caller allows them.

``check`` returns a ``sparkring-fabric-bandwidth/v1`` document and ``lines``
renders it. ``save`` keeps the latest in the controller directory, where
``sparkring status`` reads it (``summary``, ``status_lines``) without
measuring. ``after_setup`` is the last step of ``sparkring setup``;
``command`` is ``sudo sparkring cabling --bandwidth``.
"""
import contextlib
import inspect
import ipaddress
import json
import os
from pathlib import Path
import queue
import re
import shlex
import subprocess
import sys
import threading
import time

from runtime.common import installer
from runtime.host import cabling, discovery, node, placement

SCHEMA = "sparkring-fabric-bandwidth/v1"
# The latest result, in the controller directory beside cluster.json.
RECORD = "fabric-bandwidth.json"
# A healthy cable measures about 213 Gb/s per function in this test: the
# PCIe Gen5 x4 link behind a function carries about 109 Gb/s each way. Degraded
# cables measured about 119 and 26 Gb/s. 190 Gb/s leaves about 10% below the
# healthy value for run-to-run variation and stays far above degraded values.
HEALTHY_GBPS = 190.0
# SparkRing keeps each fabric function's IPv4 address at this RoCE v2 GID index.
GID_INDEX = 3
MESSAGE_BYTES = 1048576
SECONDS = 5
# TCP ports for the test's connection setup, below Linux's ephemeral port
# range (32768-60999), so no outgoing connection holds one.
PORTS = tuple(range(18620, 18640))
# Time limits in seconds.
READY_SECONDS = 20    # a server listens within this after it starts
SERVER_SECONDS = 60   # timeout(1) ends a test server that still runs after this
CLIENT_SECONDS = 30   # timeout(1) ends a test client that still runs after this
SSH_SECONDS = 20      # SSH sign-in and session time on top of a remote limit
STOP_SECONDS = 10     # Node A waits this long for a server to end after closing its input
SSH_OPTIONS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=5",
               "-o", "ServerAliveCountMax=3")
COMMAND = "sudo sparkring cabling --bandwidth"
FAILURES = (ValueError, KeyError, TypeError, OSError, RuntimeError, subprocess.SubprocessError)

HEALTHY, DEGRADED, FAILED, SKIPPED = "healthy", "degraded", "failed", "skipped"
# The columns of an ib_write_bw result line: #bytes, #iterations, BW peak,
# BW average and MsgRate.
RESULT = re.compile(r"^\s*(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", re.MULTILINE)


# The cables and what may be measured.

def hostnames(plan):
    """Each rank's hostname from the setup plan's inspection records, else its SSH target."""
    hosts = plan["spec"]["hosts"]
    nodes = plan.get("nodes") or []
    names = []
    for rank, host in enumerate(hosts):
        record = nodes[rank] if rank < len(nodes) and isinstance(nodes[rank], dict) else {}
        names.append(record.get("hostname") or host["host"])
    return names


def cables(plan):
    """The measured cables of a setup plan, with both functions of each.

    Each cable is ``{"ends": [{"rank", "hostname", "port"}, ...],
    "functions": [{"function", "server", "client"}, ...]}``. ``function`` is
    ``primary`` (``enp1s0f*``) or ``secondary`` (``enP2p1s0f*``); ``server``
    is the end on the cable's first Spark and ``client`` the end on its
    second, each ``{"rank", "hostname", "netdev", "rdma_device", "address"}``
    with the IPv4 address without its prefix. A function whose two recorded
    addresses are not on one subnet also carries ``problem``.
    """
    hosts = plan["spec"]["hosts"]
    size = len(hosts)
    if size == 2:
        layout = [(0, "cw", 1, "cw")]
    elif size == 4:
        layout = [(rank, "cw", (rank + 1) % 4, "ccw") for rank in range(4)]
    else:
        raise ValueError("The cluster record holds neither a pair nor a four-Spark ring")
    names = hostnames(plan)
    result = []
    for low, low_side, high, high_side in layout:
        roles = [{port["role"]: port for port in hosts[rank]["data_interfaces"]} for rank in (low, high)]
        functions = []
        for function in ("primary", "secondary"):
            ends = []
            for rank, side, ports in ((low, low_side, roles[0]), (high, high_side, roles[1])):
                port = ports[side + "_" + function]
                ends.append({"rank": rank, "hostname": names[rank], "netdev": port["netdev"],
                             "rdma_device": port["rdma_device"], "address": port["address"]})
            networks = [ipaddress.IPv4Interface(end["address"]).network for end in ends]
            for end in ends:
                end["address"] = str(ipaddress.IPv4Interface(end["address"]).ip)
            row = {"function": function, "server": ends[0], "client": ends[1]}
            if networks[0] != networks[1]:
                row["problem"] = (f"the recorded addresses of {ends[0]['netdev']} on {ends[0]['hostname']} and "
                                  f"{ends[1]['netdev']} on {ends[1]['hostname']} are not on one subnet; "
                                  "sudo sparkring setup reviews the fabric")
            functions.append(row)
        result.append({"ends": [{"rank": low, "hostname": names[low], "port": 0},
                                {"rank": high, "hostname": names[high], "port": 0 if size == 2 else 1}],
                       "functions": functions})
    return result


def unmeasured(plan):
    """Notes about cables between the recorded Sparks that the check cannot measure.

    A pair's cable between the ports 1, as setup's LLDP observations recorded
    it, carries only the administration network's fallback path and has no
    fabric addresses.
    """
    hosts = plan["spec"]["hosts"]
    nodes = plan.get("nodes") or []
    if len(hosts) != 2 or len(nodes) != 2:
        return []
    from runtime.host import topology
    try:
        sparks = [cabling.from_inspection(record, topology.endpoints(record)) for record in nodes]
        found = cabling.diagnose(sparks, nodes[0]["node_id"])
    except FAILURES:
        return []
    if any([end["port"] for end in cable["ends"]] == [1, 1] for cable in found["cables"]):
        return ["The cable between the ports 1 is not measured: pair models do not use it, and SparkRing gives "
                "it no fabric addresses."]
    return []


def serving(state_root, size):
    """``{rank: {"profile", "placement"}}`` of the Sparks that a model may be serving on.

    A slot's active deployment serves unless its last operation is a
    completed ``down`` (``placement.stopped``). The whole cluster's model
    (placement None) uses every Spark; a ring half's model uses that half's
    two Sparks.
    """
    found = {}
    for slot, directory in placement.actives(state_root, size).items():
        if placement.stopped(directory):
            continue
        for rank in range(size) if slot is None else slot:
            found[rank] = {"profile": placement.profile_of(directory), "placement": slot}
    return found


def cable_text(cable):
    """``spark-a port 0 ↔ spark-b port 1``."""
    a, b = cable["ends"]
    return f"{a['hostname']} port {a['port']} ↔ {b['hostname']} port {b['port']}"


def _users(cable, busy):
    """``(text, count)``: ``PROFILE serves on spark-a and spark-b`` for the serving models on a cable's Sparks.

    ``count`` is the number of models; the text is None when no model serves there.
    """
    profiles = {}
    for end in cable["ends"]:
        if end["rank"] in busy:
            profiles.setdefault(busy[end["rank"]]["profile"], []).append(end["hostname"])
    text = "; ".join(f"{profile} serves on {' and '.join(names)}" for profile, names in profiles.items())
    return text or None, len(profiles)


# Programs that run on the Sparks.

def read_gids(devices, index):
    """Run on a Spark: print ``{device: [GID, GID type]}`` of RoCE GID entry ``index`` of each RDMA device.

    Each value is the sysfs text, or None where it cannot be read.
    """
    import json
    result = {}
    for device in devices:
        values = []
        for name in (f"gids/{index}", f"gid_attrs/types/{index}"):
            try:
                with open(f"/sys/class/infiniband/{device}/ports/1/{name}", encoding="utf-8") as stream:
                    values.append(stream.read().strip())
            except OSError:
                values.append(None)
        result[device] = values
    print(json.dumps(result))


def serve(argv, ports, ready_seconds, limit, *, stop=None):
    """Run on the server Spark: start the test server ``argv`` on a free TCP port of ``ports``.

    Prints one JSON line as soon as the server listens, ``{"port": PORT}``,
    and after the server ends ``{"returncode": CODE, "output": [LINES]}``
    with its last output lines. When the server does not listen within
    ``ready_seconds``, or stops first, the only line is ``{"error": TEXT,
    "returncode": CODE or None, "output": [LINES]}``. ``timeout -k 5 LIMIT``
    ends a server that runs longer than ``limit`` seconds; the server also
    ends when ``stop`` returns, by default when the program's input ends:
    Node A closes it when it is done, and it also ends with the SSH session.
    """
    import json
    import subprocess
    import sys
    import threading
    import time

    def listening():
        """The TCP ports with a listening socket, from the kernel's tables."""
        found = set()
        for table in ("/proc/net/tcp", "/proc/net/tcp6"):
            try:
                with open(table, encoding="ascii") as stream:
                    rows = stream.read().splitlines()[1:]
            except OSError:
                continue
            for row in rows:
                fields = row.split()
                if len(fields) > 3 and fields[3] == "0A":
                    found.add(int(fields[1].rsplit(":", 1)[1], 16))
        return found

    def report(value):
        print(json.dumps(value), flush=True)

    busy = listening()
    port = next((number for number in ports if number not in busy), None)
    if port is None:
        report({"error": f"TCP ports {ports[0]}-{ports[-1]} are all in use", "returncode": None, "output": []})
        return
    process = subprocess.Popen(["timeout", "-k", "5", str(limit), *argv, "-p", str(port)], stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    output = []
    reader = threading.Thread(target=lambda: output.extend(process.stdout), daemon=True)
    reader.start()

    def end():
        (stop or sys.stdin.read)()
        if process.poll() is None:
            process.terminate()

    threading.Thread(target=end, daemon=True).start()
    deadline = time.monotonic() + ready_seconds
    while True:
        ready = port in listening()
        exited = process.poll() is not None
        if ready or exited or time.monotonic() >= deadline:
            break
        time.sleep(0.1)
    if ready:
        report({"port": port})
    elif not exited:
        process.terminate()
    try:
        process.wait(limit + 10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    reader.join(5)
    lines = [line.strip() for line in output if line.strip()][-20:]
    if ready:
        report({"returncode": process.returncode, "output": lines})
    elif exited:
        report({"error": "the test server stopped before it listened", "returncode": process.returncode,
                "output": lines})
    else:
        report({"error": f"the test server did not listen within {ready_seconds} seconds", "returncode": None,
                "output": lines})


def ib_write_bw_argv(device):
    """The ``ib_write_bw`` arguments both ends use; ``serve`` adds the port, the client the port and address."""
    return ["ib_write_bw", "-b", "-d", device, "-x", str(GID_INDEX), "-s", str(MESSAGE_BYTES), "-D", str(SECONDS),
            "-F", "--report_gbits"]


def server_program(device):
    """``python3 -I -c`` argv that runs ``serve`` for RDMA device ``device``."""
    call = "serve(*" + repr((ib_write_bw_argv(device), list(PORTS), READY_SECONDS, SERVER_SECONDS)) + ")"
    return ["python3", "-I", "-c", inspect.getsource(serve) + "\n\n" + call + "\n"]


def client_argv(device, address, port):
    return ["timeout", "-k", "5", str(CLIENT_SECONDS), *ib_write_bw_argv(device), "-p", str(port), address]


def gid_program(devices):
    """``python3 -I -c`` argv that runs ``read_gids`` for ``devices``."""
    return ["python3", "-I", "-c", inspect.getsource(read_gids) + "\n\nread_gids(*"
            + repr((list(devices), GID_INDEX)) + ")\n"]


# Commands on the Sparks.

class Server:
    """A test server program started by ``Access.start``: its JSON lines and other output."""

    def __init__(self, process):
        self.process = process
        self.lines = queue.Queue()
        self.text = []
        self.stopped = False
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        try:
            for line in self.process.stdout:
                self.lines.put(line)
        finally:
            self.lines.put(None)

    def announcement(self, timeout):
        """The first JSON object the program prints; None when none arrives within ``timeout`` seconds."""
        deadline = time.monotonic() + timeout
        while (left := deadline - time.monotonic()) > 0:
            try:
                line = self.lines.get(timeout=left)
            except queue.Empty:
                return None
            if line is None:
                return None
            document = _json_object(line)
            if document is not None:
                return document
            self.text.append(line.strip())
        return None

    def stop(self):
        """End the program: close its input, wait ``STOP_SECONDS``, then kill it. Returns its other output."""
        if not self.stopped:
            self.stopped = True
            with contextlib.suppress(OSError):
                self.process.stdin.close()
            try:
                self.process.wait(STOP_SECONDS)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            while True:
                try:
                    line = self.lines.get(timeout=1)
                except queue.Empty:
                    break
                if line is None:
                    break
                if _json_object(line) is None and line.strip():
                    self.text.append(line.strip())
        return "\n".join(self.text)


class Access:
    """Commands on the recorded Sparks as root: Node A runs its own, workers over SSH with ``sudo -n``."""

    def __init__(self, plan, *, run=subprocess.run, popen=subprocess.Popen):
        self.hosts = plan["spec"]["hosts"]
        self._run, self._popen = run, popen

    def argv(self, rank, argv):
        if rank == 0:
            root = hasattr(os, "geteuid") and os.geteuid() == 0
            return list(argv) if root else ["sudo", "-n", *argv]
        return ["ssh", *SSH_OPTIONS, discovery.target(self.hosts[rank]["host"]), shlex.join(["sudo", "-n", *argv])]

    def run(self, rank, argv, *, timeout):
        """``(returncode, stdout, stderr)`` of argv on ``rank``; subprocess.TimeoutExpired after ``timeout`` s."""
        done = self._run(self.argv(rank, argv), stdin=subprocess.DEVNULL, capture_output=True, text=True,
                         encoding="utf-8", errors="replace", timeout=timeout)
        return done.returncode, done.stdout, done.stderr

    def start(self, rank, argv):
        """Start argv on ``rank`` with its input open and its output read line by line (``Server``)."""
        return Server(self._popen(self.argv(rank, argv), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace"))


def _json_object(line):
    try:
        value = json.loads(line)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _last_line(*texts):
    for text in texts:
        found = [line.strip() for line in str(text or "").splitlines() if line.strip()]
        if found:
            return found[-1]
    return ""


# Measurement.

def parse_bandwidth(text):
    """The BW average column, in Gb/s, of the result line of an ``ib_write_bw --report_gbits`` run; else None."""
    if "Gb/sec" not in str(text or ""):
        return None
    for match in RESULT.finditer(text):
        if int(match[1]) == MESSAGE_BYTES:
            return float(match[4])
    return None


def verdict(gbps):
    """``healthy`` at or above HEALTHY_GBPS, else ``degraded``."""
    return HEALTHY if gbps >= HEALTHY_GBPS else DEGRADED


def _failed(reason):
    return {"gbps": None, "verdict": FAILED, "reason": reason}


def _program_reason(returncode, output, hostname, limit):
    """Why the test program stopped on ``hostname`` without a result."""
    text = "\n".join(output) if isinstance(output, list) else str(output or "")
    if returncode == 127 or "failed to run command" in text:
        return f"ib_write_bw is not installed on {hostname} (Debian package perftest)"
    if returncode in (124, 137):
        return f"the test on {hostname} did not finish within {limit} seconds"
    last = _last_line(text)
    return f"{hostname}: {last}" if last else f"the test on {hostname} stopped with exit status {returncode}"


def gid_problem(end, entry):
    """Why RoCE GID index 3 of one end of a function does not carry its fabric address; None when it does.

    ``entry`` is ``[GID, GID type]`` as ``read_gids`` reports it.
    """
    gid, kind = entry if entry else (None, None)
    where = f"RoCE GID {GID_INDEX} of {end['netdev']} on {end['hostname']}"
    if gid is None:
        return f"{where} could not be read"
    try:
        mapped = ipaddress.IPv6Address(gid).ipv4_mapped
    except ValueError:
        mapped = None
    if mapped is None or int(mapped) == 0:
        return f"{where} holds no IPv4 address; the test needs its fabric address {end['address']} there"
    if str(mapped) != end["address"]:
        return f"{where} holds {mapped}; the test needs its fabric address {end['address']} there"
    if kind is not None and " ".join(kind.split()).lower() != "roce v2":
        return f"{where} is a {kind} entry; the test needs RoCE v2"
    return None


def read_entries(access, cables_, hostnames_):
    """``{rank: {device: [GID, type]}}`` of every measured function end, or ``{rank: error text}`` per unread Spark."""
    devices = {}
    for cable in cables_:
        for function in cable["functions"]:
            for end in (function["server"], function["client"]):
                devices.setdefault(end["rank"], []).append(end["rdma_device"])
    result = {}
    for rank, names in sorted(devices.items()):
        try:
            returncode, stdout, stderr = access.run(rank, gid_program(names), timeout=SSH_SECONDS + 10)
            value = _json_object(_last_line(stdout)) if returncode == 0 else None
            result[rank] = value if value is not None else (
                f"{hostnames_[rank]} could not be read: " + (_last_line(stderr, stdout) or f"exit status {returncode}"))
        except FAILURES as error:
            result[rank] = f"{hostnames_[rank]} could not be read: {error}"
    return result


def measure(access, function):
    """``{"gbps", "verdict", "reason"}`` of one bidirectional test between the two ends of one function.

    The server program starts on the server end and announces its port; the
    client then runs on the client end; the server program is stopped
    before this returns, whatever happened.
    """
    server_end, client_end = function["server"], function["client"]
    try:
        server = access.start(server_end["rank"], server_program(server_end["rdma_device"]))
    except FAILURES as error:
        return _failed(f"the test server did not start on {server_end['hostname']}: {error}")
    try:
        announced = server.announcement(READY_SECONDS + SSH_SECONDS)
        if announced is None or "port" not in announced:
            text = server.stop()
            if announced is None:
                last = _last_line(text)
                return _failed(f"{server_end['hostname']}: {last}" if last else
                               f"the test server on {server_end['hostname']} did not answer within "
                               f"{READY_SECONDS + SSH_SECONDS} seconds")
            if announced.get("returncode") is not None:
                return _failed(_program_reason(announced["returncode"], announced.get("output"),
                                               server_end["hostname"], SERVER_SECONDS))
            return _failed(f"{server_end['hostname']}: {announced.get('error')}")
        argv = client_argv(client_end["rdma_device"], server_end["address"], announced["port"])
        try:
            returncode, stdout, stderr = access.run(client_end["rank"], argv, timeout=CLIENT_SECONDS + SSH_SECONDS)
        except subprocess.TimeoutExpired:
            return _failed(f"the test on {client_end['hostname']} did not finish within "
                           f"{CLIENT_SECONDS + SSH_SECONDS} seconds")
        except FAILURES as error:
            return _failed(f"the test client did not start on {client_end['hostname']}: {error}")
    finally:
        server.stop()
    gbps = parse_bandwidth(stdout) if returncode == 0 else None
    if gbps is not None:
        return {"gbps": gbps, "verdict": verdict(gbps), "reason": None}
    if returncode == 0:
        return _failed(f"the test on {client_end['hostname']} printed no bandwidth result")
    return _failed(_program_reason(returncode, (stderr or "") + "\n" + (stdout or ""), client_end["hostname"],
                                   CLIENT_SECONDS))


def _cable_verdict(functions):
    found = {function["verdict"] for function in functions}
    return DEGRADED if DEGRADED in found else FAILED if FAILED in found else HEALTHY


def overall(cables_):
    """``degraded`` if any cable is, else ``failed`` if any is, else ``skipped`` if all were, else ``healthy``."""
    found = [cable["verdict"] for cable in cables_]
    if DEGRADED in found:
        return DEGRADED
    if FAILED in found:
        return FAILED
    if found and all(value == SKIPPED for value in found):
        return SKIPPED
    return HEALTHY


def check(cluster, *, access, busy=None, allow_serving=False, say=None, now=time.time):
    """Measure every recorded cable, one function at a time; a ``sparkring-fabric-bandwidth/v1`` document.

    ``busy`` is ``serving``'s map of Sparks a model serves on. A cable that
    touches one is skipped unless ``allow_serving``; then its result notes
    that the model's traffic can lower it. Before any test, RoCE GID index 3
    of every function end is read (``gid_problem``); a function whose ends
    fail that, or whose recorded addresses do not share a subnet, fails
    without a test. ``say`` receives one progress line per measured cable.

    The document holds ``measured_at`` (seconds since the epoch),
    ``cluster_id`` (the setup plan's ID), ``layout`` (``pair`` or ``ring``),
    ``threshold_gbps``, ``test`` (the ``ib_write_bw`` parameters),
    ``verdict`` (``overall``), ``notes`` and ``cables``. Each cable adds to
    ``cables``' fields its ``verdict`` (``healthy``, ``degraded``, ``failed``
    or ``skipped``), ``reason`` (why it was skipped), ``notes`` and
    ``repair`` (the repair lines of a degraded cable); each function adds
    ``gbps`` (the bidirectional BW average, or None), ``verdict`` and
    ``reason``.
    """
    say = say or (lambda line: None)
    busy = busy or {}
    plan = cluster["plan"]
    names = hostnames(plan)
    found = cables(plan)
    measured = []
    for cable in found:
        cable.update(verdict=None, reason=None, notes=[], repair=None)
        users, models = _users(cable, busy)
        if users and not allow_serving:
            cable.update(verdict=SKIPPED, reason=f"{users}, and the test would slow {'it' if models == 1 else 'them'}")
            for function in cable["functions"]:
                function.update(gbps=None, verdict=SKIPPED, reason=None)
            continue
        if users:
            cable["notes"].append(f"Measured while {users}; model traffic can lower the result.")
        measured.append(cable)
    entries = read_entries(access, measured, names) if measured else {}
    for cable in measured:
        say(f"Measuring {cable_text(cable)}")
        for function in cable["functions"]:
            problems = [function["problem"]] if function.get("problem") else []
            for end in (function["server"], function["client"]):
                read = entries.get(end["rank"])
                problem = read if isinstance(read, str) else gid_problem(end, (read or {}).get(end["rdma_device"]))
                if problem and problem not in problems:
                    problems.append(problem)
            function.update(measure(access, function) if not problems else _failed("; ".join(problems)))
        cable["verdict"] = _cable_verdict(cable["functions"])
        if cable["verdict"] == DEGRADED:
            cable["repair"] = repair_lines(cable)
    return {"schema": SCHEMA, "measured_at": now(), "cluster_id": plan.get("id"),
            "layout": "pair" if len(plan["spec"]["hosts"]) == 2 else "ring", "threshold_gbps": HEALTHY_GBPS,
            "test": {"program": "ib_write_bw", "bidirectional": True, "message_bytes": MESSAGE_BYTES,
                     "seconds": SECONDS, "gid_index": GID_INDEX},
            "verdict": overall(found), "notes": unmeasured(plan), "cables": found}


# Text.

def repair_lines(cable):
    """What to do about a degraded cable, naming both of its Sparks."""
    a, b = (end["hostname"] for end in cable["ends"])
    return [f"To repair: reboot both {a} and {b}, then run {COMMAND} again.",
            "Restarting the link or the network driver does not clear this.",
            "If the cable is still degraded after the reboot, reseat it at both ends."]


def _function_label(function):
    return f"{function['server']['netdev']} ↔ {function['client']['netdev']}"


def _function_value(function):
    if function["gbps"] is not None:
        return f"{function['gbps']:6.2f} Gb/s  {function['verdict']}"
    return f"failed: {function['reason']}"


def lines(document, *, repair=True):
    """Terminal lines: per cable its verdict, then each function's Gb/s and verdict, repair steps and notes.

    Without ``repair`` a degraded cable's repair lines are left out.
    """
    output = [f"Fabric bandwidth, both directions at once ({document['threshold_gbps']:g} Gb/s or more per link "
              "is healthy):"]
    for cable in document["cables"]:
        if cable["verdict"] == SKIPPED:
            output.append(f"{cable_text(cable)}: not measured; {cable['reason']} (--while-serving measures it anyway)")
            continue
        output.append(f"{cable_text(cable)}: {cable['verdict']}")
        width = max(len(_function_label(function)) for function in cable["functions"])
        for function in cable["functions"]:
            output.append(f"  {_function_label(function):<{width}}  {_function_value(function)}")
        if repair and cable.get("repair"):
            output += ["  " + line for line in cable["repair"]]
        output += ["  Note: " + note for note in cable.get("notes") or []]
    output += ["Note: " + note for note in document.get("notes") or []]
    return output


def warning_lines(document):
    """The setup warning for the degraded cables of ``document``: each cable, its rates and its repair lines."""
    output = []
    for cable in document["cables"]:
        if cable["verdict"] != DEGRADED:
            continue
        rates = " and ".join(f"{function['gbps']:.2f}" for function in cable["functions"]
                             if function["gbps"] is not None)
        output += [f"WARNING: the fabric cable {cable_text(cable)} is degraded ({rates} Gb/s; healthy is "
                   f"{document['threshold_gbps']:g} or more).",
                   "  Models run, but prompt processing over this cable is slower."]
        output += ["  " + line for line in cable["repair"]]
    return output


# The saved result.

def save(state_root, document):
    node.save(state_root, RECORD, document, mode=0o644)


def summary(state_root, *, now=time.time):
    """The saved result for ``sparkring status``: its document with ``state`` and ``age_seconds``; measures nothing.

    ``state`` is ``measured``, ``never-measured`` (no saved result) or
    ``unreadable`` (with ``error``).
    """
    path = Path(state_root) / RECORD
    if not path.exists():
        return {"state": "never-measured"}
    try:
        document = installer.read(path)
        if document.get("schema") != SCHEMA:
            raise ValueError(f"{path} is not a {SCHEMA} document")
        age = max(0, round(now() - float(document["measured_at"])))
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        return {"state": "unreadable", "error": str(error)}
    return {**document, "state": "measured", "age_seconds": age}


def status_lines(value, cluster_id=None):
    """``sparkring status`` lines for a ``summary``: its verdict and age, then every cable that is not healthy.

    A degraded cable's line carries its rates and its repair lines. A result
    saved for another setup plan than ``cluster_id`` counts as never measured.
    """
    again = f"{COMMAND} measures it"
    if value["state"] == "never-measured":
        return [f"Fabric bandwidth: never measured; {again}"]
    if value["state"] == "unreadable":
        return [f"Fabric bandwidth: the saved result cannot be read ({value['error']}); {again}"]
    if cluster_id is not None and value.get("cluster_id") not in (None, cluster_id):
        return [f"Fabric bandwidth: never measured on this setup (the saved result is from another setup of the "
                f"Sparks); {again}"]
    rows = value["cables"]
    age = node.age_text(value["age_seconds"])
    counts = {name: sum(cable["verdict"] == name for cable in rows) for name in (HEALTHY, DEGRADED, FAILED, SKIPPED)}
    total = len(rows)

    def count(name, text):
        return f"{counts[name]} of {total} {'cable' if total == 1 else 'cables'} {text}"

    if counts[DEGRADED]:
        head = count(DEGRADED, "degraded")
    elif counts[FAILED]:
        head = count(FAILED, "not measured")
    elif counts[SKIPPED] == total:
        head = "not measured"
    elif counts[SKIPPED]:
        head = f"healthy on {counts[HEALTHY]} of {total} cables"
    else:
        head = "healthy" + (f" on all {total} cables" if total > 1 else "")
    output = [f"Fabric bandwidth: {head}, measured {age}"]
    for cable in rows:
        if cable["verdict"] == DEGRADED:
            rates = " and ".join(f"{function['gbps']:.2f}" for function in cable["functions"]
                                 if function["gbps"] is not None)
            output.append(f"  {cable_text(cable)}: degraded, {rates} Gb/s (healthy is {value['threshold_gbps']:g} "
                          "or more)")
            output += ["    " + line for line in cable.get("repair") or repair_lines(cable)]
        elif cable["verdict"] == FAILED:
            reasons = [function["reason"] for function in cable["functions"] if function["verdict"] == FAILED]
            output.append(f"  {cable_text(cable)}: not measured: {reasons[0] if reasons else 'unknown reason'}")
        elif cable["verdict"] == SKIPPED:
            output.append(f"  {cable_text(cable)}: not measured; {cable['reason']}")
    return output


# Entry points.

def after_setup(state_root, cluster, *, access=None, say=print):
    """The last step of ``sparkring setup``: measure every cable, save the result and warn about degraded cables.

    Nothing is measured while a model serves on the cluster. A degraded
    cable is a warning with its repair lines; a check that cannot run is a
    warning too. Setup never fails here. Returns the document, or None when
    nothing was measured.
    """
    try:
        plan = cluster["plan"]
        busy = serving(state_root, len(plan["spec"]["hosts"]))
        if busy:
            say(f"Fabric bandwidth: not measured, because a model is serving; {COMMAND} measures it after the "
                "model stops.")
            return None
        say("Measure the bandwidth of each fabric cable (about 30 seconds per cable)")
        document = check(cluster, access=access or Access(plan), say=say)
        save(state_root, document)
    except FAILURES as error:
        say(f"Warning: the fabric bandwidth check could not run ({error}); {COMMAND} runs it again.")
        return None
    for line in lines(document, repair=False):
        say(line)
    for line in warning_lines(document):
        say(line)
    return document


def refusal(busy):
    """The error when serving models (``serving``) use every cable: each model, where it serves, its stop command."""
    models = []
    for row in busy.values():
        if row not in models:
            models.append(row)
    where = "; ".join(f"{row['profile']} serves on "
                      + ("every Spark" if row["placement"] is None else placement.text(row["placement"]))
                      for row in models)
    stops = " and ".join("sudo sparkring down " + (placement.flag(row["placement"]) + " " if row["placement"] else "")
                         + "--execute" for row in models)
    return (f"Every fabric cable is in use: {where}. The bandwidth test fills each cable for several seconds per "
            f"link and would slow {'the model' if len(models) == 1 else 'the models'}. Stop "
            f"{'it' if len(models) == 1 else 'them'} first ({stops}), or add --while-serving to measure anyway.")


def command(*, json_output=False, allow_serving=False, state_root=None, access=None):
    """``sudo sparkring cabling --bandwidth``: measure, save and print; the exit status.

    0 when every cable measured healthy, 1 when a cable is degraded, failed
    or was skipped. Raises ValueError when the check cannot run: no recorded
    cluster, another operation holds the installation lock, or a serving
    model uses every cable and ``allow_serving`` is false; nothing is saved
    then. The installation lock is held throughout, so no model starts and
    setup does not run while the check measures.
    """
    from runtime.common import process_lock
    from runtime.host import controller
    root = Path(state_root or controller.STATE)
    if not (root / "cluster.json").exists():
        raise ValueError(f"No pair or ring is recorded on this Spark; {COMMAND} runs on Node A after "
                         "sudo sparkring setup")
    cluster = installer.read(root / "cluster.json")
    size = len(cluster["plan"]["spec"]["hosts"])
    say = (lambda line: print(line, file=sys.stderr)) if json_output else print
    with process_lock.hold(root / "install.lock"):
        busy = serving(root, size)
        document = check(cluster, access=access or Access(cluster["plan"]), busy=busy, allow_serving=allow_serving,
                         say=say)
        if document["verdict"] == SKIPPED:
            raise ValueError(refusal(busy))
        save(root, document)
    print(json.dumps(document, indent=2) if json_output else "\n".join(lines(document)))
    return 0 if all(cable["verdict"] == HEALTHY for cable in document["cables"]) else 1
