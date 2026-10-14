"""Viewer-leased stat forwarding over the controller's existing SSH identity.

The host agent on Node A owns the SSH processes. The unprivileged dashboard
writes only a timestamp lease; destinations come exclusively from root-owned
controller records. One long-lived channel per managed Spark carries bounded snapshots.
"""
import json
from pathlib import Path
import select
import subprocess
import sys
import threading
import time

from runtime.host import discovery, telemetry, startup

LEASE = Path("/run/sparkring-dashboard/viewer")
OUTPUT = Path("/run/sparkring/dashboard-live.json")
INTERVAL = 2.0
TTL = 6.0
MAX_HEADS = 8
_started = False


def active(now, lease=LEASE, quiet=None, disabled=None):
    from runtime.host.dashboard import QUIET_PATH, quiet_state
    disabled = Path(disabled or "/etc/sparkring/dashboard.disabled")
    try:
        # No file body supplied by the web service is trusted by the root agent.
        age = now - lease.stat().st_mtime
        return (0 <= age <= TTL and not disabled.exists()
                and not quiet_state(Path(quiet or QUIET_PATH), now)["quiet"])
    except OSError:
        return False


def heads(state_root=Path("/var/lib/sparkring/controller"), pending=(), metrics=None):
    """Select managed Sparks from controller records, including pending installations."""
    from runtime.host.dashboard import SITE_PATH, read_json
    cluster = read_json(state_root / "cluster.json")
    hosts = cluster["plan"]["spec"]["hosts"]
    site = read_json(state_root.parent / "dashboard" / SITE_PATH.name)
    result = {}
    for row in [*site.get("deployments", []), *pending][:128]:
        positions = row.get("positions") or []
        deployment = row.get("deployment_id")
        if row.get("stopped") or not positions or not isinstance(deployment, str):
            continue
        for position in positions[:8]:
            if type(position) is not int or not 0 <= position < len(hosts):
                continue
            host = discovery.target(hosts[position]["host"])
            if metrics is not None and position == row.get("head_position", positions[0]):
                metrics[deployment] = host
            if host not in result and len(result) >= MAX_HEADS:
                continue
            result.setdefault(host, []).append(deployment)
    return {host: list(dict.fromkeys(ids))[:8] for host, ids in result.items()}, hosts[0]["host"]


def project(snapshots, ids, now):
    from runtime.host.dashboard import HISTORY_VALUES
    from runtime.host.dashboard_rates import number, plugin_fresh
    return {key: {k: value[k] for k in (*HISTORY_VALUES, "at", "monotonic") if number(value.get(k))}
            | {"schema": "sparkring-stats/v1", "run_id": value["run_id"],
               "step_timing": bool(value.get("step_timing"))}
            for key, value in snapshots.items() if key in ids and plugin_fresh(value, None, now)
            and now - value["at"] <= 12}


def stream(*, input_stream=None, output_stream=None, read=startup.read_live,
           wait=select.select, clock=time.monotonic):
    """Agent command: one bounded frame per renewal, EOF/6 s silence ends it.

    stdin contains only empty heartbeat lines. No paths, commands, settings or
    network destinations are accepted. Snapshots are public agent status fields.
    """
    source, sink = input_stream or sys.stdin, output_stream or sys.stdout
    next_sample = 0.0
    while wait([source], [], [], TTL)[0]:
        if source.readline(2) != "\n":
            return
        if clock() < next_sample:
            continue
        next_sample = clock() + 1.0  # tolerate channel jitter; never read faster than 1 Hz
        values = read()
        starting = startup.project(values.get("__startup__"), list(values.get("__startup__", {}))[:8], time.time())
        values = project(values, list(values)[:128], time.time())
        if starting:
            values["__startup__"] = starting
        try:
            sink.write(json.dumps(values, separators=(",", ":")) + "\n")
            sink.flush()
        except (OSError, BrokenPipeError):
            return


class Channel:
    def __init__(self, host, *, popen=subprocess.Popen, now=time.monotonic):
        self.now = now
        self.updated = now()
        self.values = {}
        self.lock = threading.Lock()
        # Share exactly the recovery client's argument construction and SSH config.
        argv = discovery.ssh_argv(host, ["/usr/bin/sparkring", "node", "telemetry"],
                                  options=("-o", "ServerAliveInterval=2", "-o", "ServerAliveCountMax=2"))
        self.process = popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, bufsize=0)
        self.reader = threading.Thread(target=self.receive, daemon=True)
        self.reader.start()

    def receive(self):
        try:
            while True:
                line = self.process.stdout.readline(telemetry.LIMIT * 128 + 1)
                if not line or len(line) > telemetry.LIMIT * 128:
                    break
                value = json.loads(line)
                if not isinstance(value, dict) or len(value) > 128:
                    break
                with self.lock:
                    self.values, self.updated = value, self.now()
        except (OSError, ValueError):
            pass
        finally:
            if self.process.poll() is None:
                self.process.terminate()

    def sample(self):
        if self.process.poll() is not None or self.now() - self.updated > TTL + 4:
            raise OSError("telemetry channel expired")
        self.process.stdin.write(b"\n")
        with self.lock:
            return dict(self.values)

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
        try:
            self.process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=1)
        self.process.stdin.close()
        self.reader.join(timeout=1)
        self.process.stdout.close()


class Forwarder:
    def __init__(self, *, channel=Channel, read=startup.read_live, output=OUTPUT):
        self.channel, self.read, self.output = channel, read, output
        self.channels, self.retry = {}, {}

    def cycle(self, wanted, local, now, operations=(), metrics=None):
        for host in list(self.channels):
            if host not in wanted:
                self.channels.pop(host).close()
        self.retry = {h: t for h, t in self.retry.items() if h in wanted}
        snapshots, starting = {}, {}
        for host, ids in list(wanted.items())[:MAX_HEADS]:
            try:
                if host == local:
                    values = self.read()
                else:
                    if now < self.retry.get(host, 0):
                        continue
                    if host not in self.channels:
                        self.channels[host] = self.channel(host)
                    values = self.channels[host].sample()
                metric_ids = ids if metrics is None else [key for key in ids if metrics.get(key) == host]
                snapshots.update(project(values, metric_ids, now))
                for key, ranks in startup.project(values.get("__startup__"), ids, now).items():
                    starting.setdefault(key, []).extend(ranks)
            except (OSError, ValueError, subprocess.SubprocessError):
                if host in self.channels:
                    self.channels.pop(host).close()
                self.retry[host] = now + 15
        if wanted or operations:
            temporary = self.output.with_suffix(".writing")
            temporary.write_text(json.dumps({"at": now, "telemetry": snapshots, "startup": starting, "operations": list(operations)}), encoding="utf-8")
            temporary.chmod(0o644)
            temporary.replace(self.output)
        else:
            self.output.unlink(missing_ok=True)


def run():
    forwarder = Forwarder()
    while True:
        now = time.time()
        try:
            watched = active(now)
            operations = startup.operations(now=now) if watched else []
            metrics = {}
            wanted, local = heads(pending=[op["deployment"] for op in operations], metrics=metrics) if watched else ({}, None)
            forwarder.cycle(wanted, local, now, operations, metrics)
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            try:
                forwarder.cycle({}, None, now)
            except OSError:
                pass  # retry after a temporary read-only or full runtime directory
        time.sleep(INTERVAL)


def start():
    global _started
    # Rechecked by the ordinary agent loop, including after first-time setup.
    if not _started and Path("/var/lib/sparkring/controller/cluster.json").is_file():
        _started = True
        threading.Thread(target=run, name="dashboard-telemetry", daemon=True).start()
