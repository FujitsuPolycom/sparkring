"""Placement and frequency of SparkRing's own periodic host work, switchable on a running ring.

Four periodic jobs run on Sparks that serve a model:

- the host agent on every Spark (``sparkring-agent.service``): a full observation every
  ``node.AGENT_INTERVAL`` seconds, which starts ``nvidia-smi``, ``docker``, ``ip``, ``nmcli``,
  ``devlink`` and other commands;
- the administration tunnel's check on every Spark (``sparkring-control-refresh.timer``);
- the recovery timer on Node A (``sparkring-recover.timer``): every 60 s while a model is active, it
  signs in to every other Spark of the deployment twice over SSH and reads Node A's own rank
  locally (``recovery.local_target``);
- the dashboard (``runtime/host/dashboard.py``), which places itself.

This module holds the settings that place and pace the first three. Without a settings file every
job keeps its built-in placement and period; a hardware A/B decides other defaults.

Settings (``/etc/sparkring/host-load.json``, ``sparkring-host-load/v1``, per Spark):

- ``placement``: ``default`` or ``efficiency``. ``efficiency`` writes a systemd drop-in
  (DROP_IN) for UNITS with ``CPUAffinity`` set to the CPUs outside the fastest class that
  SIRCL's classifier finds (the Cortex-A725 cores of a GB10), ``Nice=19``,
  ``CPUSchedulingPolicy=idle`` and ``IOSchedulingClass=idle``: the dashboard's own placement.
  SIRCL's serving and progress threads use the fastest class.
- ``agent_serving_interval``: seconds between the host agent's observations while a SparkRing
  model container runs on the Spark (``serving``); ``node.AGENT_INTERVAL`` otherwise. The agent
  reads it again every ``node.AGENT_INTERVAL`` seconds while it waits, so a change applies within
  that time without a restart.
  Readers mark a status document stale after ``stale_after`` seconds, which grows with it.
- ``control_refresh_interval``: seconds between the administration tunnel's periodic checks
  (``sparkring-control-refresh.timer``: ``sparkring node control-up`` repairs links, endpoints,
  firewall rules and DNS). 20 is the timer's own period. A longer one writes a timer drop-in
  (CONTROL_DROP_IN); link changes and ConnectX driver restarts still start a check at once
  (``sparkring-control-link`` for NetworkManager, ``90-sparkring-control.rules`` for udev), and
  while a peer is moving between paths, on a fallback path or behind a failed link each check
  schedules the next FOLLOW_UP_SECONDS later (``schedule_follow_up``), so path changes keep their
  20 s steps. Only a path that fails silently (carrier up, nothing received) waits longer: its
  handshake goes stale after ``control.HANDSHAKE_STALE`` seconds and the next periodic check
  moves the peer, at most HANDSHAKE_STALE + the interval after the failure.
- ``recovery_ssh_reuse``: ``true`` makes the recovery timer's SSH commands use one master
  connection per Spark that ``sparkring-ssh-mux.service`` keeps open (``mux_main``), instead of
  a new login per command. Each command still starts its own process on the Spark; the login
  (key exchange, authentication, PAM session) is what is saved. The master's keepalive
  (MUX_ALIVE_SECONDS × MUX_ALIVE_COUNT) bounds how long a command waits on a Spark that went
  away: at most that long beyond the 10 s connect timeout of a fresh login.

``sudo sparkring node host-load set KEY=VALUE ...`` changes one Spark;
``sudo sparkring host-load set KEY=VALUE ... --all`` on Node A changes every Spark of the fabric.
Placement takes effect with ``systemctl daemon-reload`` and a restart of the agent and the SSH
master service, which serve no model; the recovery timer's next run uses it.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

SCHEMA = "sparkring-host-load/v1"
SETTINGS = "/etc/sparkring/host-load.json"
DROP_IN = "50-sparkring-host-load.conf"
UNITS = ("sparkring-agent.service", "sparkring-recover.service", "sparkring-ssh-mux.service",
         "sparkring-control-refresh.service")
# Units whose processes run all the time; a placement change restarts them. The recovery
# service is a timer's oneshot and picks the drop-in up at its next run.
RESTARTED = ("sparkring-agent.service", "sparkring-ssh-mux.service")
AGENT_INTERVAL = 30          # node.AGENT_INTERVAL, repeated so this module imports nothing heavy
CONTROL_INTERVAL = 20       # sparkring-control-refresh.timer's OnUnitInactiveSec
DEFAULTS = {"placement": "default", "agent_serving_interval": AGENT_INTERVAL, "recovery_ssh_reuse": False,
            "control_refresh_interval": CONTROL_INTERVAL}
PLACEMENTS = ("default", "efficiency")
SERVING_INTERVALS = (AGENT_INTERVAL, 600)
CONTROL_INTERVALS = (CONTROL_INTERVAL, 900)
CONTROL_TIMER = "sparkring-control-refresh.timer"
CONTROL_SERVICE = "sparkring-control-refresh.service"
FOLLOW_UP_UNIT = "sparkring-control-follow-up"
FOLLOW_UP_SECONDS = CONTROL_INTERVAL
MUX_DIRECTORY = "/run/sparkring/ssh-mux"
MUX_ALIVE_SECONDS = 10
MUX_ALIVE_COUNT = 2
MUX_CHECK_SECONDS = 10
MUX_BACKOFF_SECONDS = (10, 60)
# A SparkRing model container: installer.container_name, "[sr-]<cluster>-<profile>-<6 hex>-r<rank>".
MODEL_CONTAINER = re.compile(r"(?:sr-)?[A-Za-z0-9][A-Za-z0-9_.-]*-[0-9a-f]{6}-r[0-9]+")


def _path(root, name):
    return Path(root) / name.lstrip("/")


def load(*, root="/") -> dict:
    """The settings, each missing or unusable value replaced by its default."""
    settings = dict(DEFAULTS)
    try:
        document = json.loads(_path(root, SETTINGS).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return settings
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        return settings
    if document.get("placement") in PLACEMENTS:
        settings["placement"] = document["placement"]
    interval = document.get("agent_serving_interval")
    if type(interval) is int and SERVING_INTERVALS[0] <= interval <= SERVING_INTERVALS[1]:
        settings["agent_serving_interval"] = interval
    if type(document.get("recovery_ssh_reuse")) is bool:
        settings["recovery_ssh_reuse"] = document["recovery_ssh_reuse"]
    interval = document.get("control_refresh_interval")
    if type(interval) is int and CONTROL_INTERVALS[0] <= interval <= CONTROL_INTERVALS[1]:
        settings["control_refresh_interval"] = interval
    return settings


def parse_assignments(items) -> dict:
    """``["placement=efficiency", "agent_serving_interval=120", "recovery_ssh_reuse=on"]`` -> changes;
    ValueError naming the first unusable item."""
    changes = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or key not in DEFAULTS:
            raise ValueError(f"{item!r}: use KEY=VALUE with KEY one of {', '.join(DEFAULTS)}")
        if key == "placement":
            if value not in PLACEMENTS:
                raise ValueError(f"placement is {' or '.join(PLACEMENTS)}, not {value!r}")
            changes[key] = value
        elif key in ("agent_serving_interval", "control_refresh_interval"):
            low, high = SERVING_INTERVALS if key == "agent_serving_interval" else CONTROL_INTERVALS
            if not value.isdigit() or not low <= int(value) <= high:
                raise ValueError(f"{key} is a number of seconds from {low} to {high}, not {value!r}")
            changes[key] = int(value)
        else:
            if value not in ("on", "off", "true", "false"):
                raise ValueError(f"recovery_ssh_reuse is on or off, not {value!r}")
            changes[key] = value in ("on", "true")
    return changes


def save(settings, *, root="/") -> None:
    path = _path(root, SETTINGS)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps({"schema": SCHEMA, **settings}, indent=2) + "\n", encoding="utf-8")
    temporary.chmod(0o644)
    temporary.replace(path)


# -- placement --------------------------------------------------------------------------------
def efficiency_cpus(root: Path | None = None, cpuinfo: Path | None = None) -> set | None:
    """The online CPUs outside the fastest class (on GB10 the ten Cortex-A725 cores), or None on a
    host whose cores are all of one class.

    The classes come from SIRCL's classifier (spark_transport/sircl/sparkring_sircl/cpus.py): Arm
    part numbers first, then cpu_capacity and cpuinfo_max_freq within a tolerance, because one
    GB10 X925 core reports a higher capacity than its siblings."""
    from spark_transport.sircl.sparkring_sircl import cpus
    options = {key: value for key, value in (("root", root), ("cpuinfo", cpuinfo)) if value is not None}
    online, fastest, source = cpus.classify(**options)
    rest = set(online) - set(fastest)
    return rest if source != "uniform" and rest else None


def drop_in_text(cpus) -> str:
    from spark_transport.sircl.sparkring_sircl import cpus as classifier
    lines = ["# Written by sparkring node host-load (placement efficiency): the CPUs outside the fastest",
             "# class, at idle priority. sudo sparkring node host-load set placement=default removes it.",
             "[Service]"]
    if cpus:
        lines.append("CPUAffinity=" + classifier.format_cpu_list(cpus))
    lines += ["Nice=19", "CPUSchedulingPolicy=idle", "IOSchedulingClass=idle"]
    return "\n".join(lines) + "\n"


def drop_in_path(unit, *, root="/") -> Path:
    return _path(root, f"/etc/systemd/system/{unit}.d/{DROP_IN}")


def apply_placement(settings, *, root="/", run=subprocess.run, cpus=None) -> dict:
    """Write or remove the drop-ins of UNITS for ``settings["placement"]``, reload systemd and
    restart the units in RESTARTED that are running; returns what it did."""
    if settings["placement"] == "efficiency":
        chosen = efficiency_cpus() if cpus is None else cpus
        text = drop_in_text(chosen)
        for unit in UNITS:
            path = drop_in_path(unit, root=root)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            path.chmod(0o644)
        done = {"placement": "efficiency", "cpus": sorted(chosen) if chosen else None}
    else:
        for unit in UNITS:
            drop_in_path(unit, root=root).unlink(missing_ok=True)
        done = {"placement": "default", "cpus": None}
    run(["systemctl", "daemon-reload"], check=True, capture_output=True, text=True, timeout=60)
    run(["systemctl", "try-restart", *RESTARTED], check=True, capture_output=True, text=True, timeout=120)
    done["restarted"] = list(RESTARTED)
    return done


# -- the administration tunnel's checks --------------------------------------------------------------
CONTROL_DROP_IN = "50-sparkring-host-load.conf"


def control_drop_in_path(*, root="/") -> Path:
    return _path(root, f"/etc/systemd/system/{CONTROL_TIMER}.d/{CONTROL_DROP_IN}")


def apply_control_interval(settings, *, root="/", run=subprocess.run) -> dict:
    """Write the timer drop-in for a longer ``control_refresh_interval`` (none for 20 s), reload
    systemd and restart the timer; the tunnel itself is not touched."""
    interval = settings["control_refresh_interval"]
    path = control_drop_in_path(root=root)
    if interval == CONTROL_INTERVAL:
        path.unlink(missing_ok=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Written by sparkring node host-load (control_refresh_interval): link events and\n"
                        "# driver restarts start the check at once; this is the periodic fallback.\n"
                        f"[Timer]\nOnUnitInactiveSec=\nOnUnitInactiveSec={interval}\n", encoding="utf-8")
        path.chmod(0o644)
    run(["systemctl", "daemon-reload"], check=True, capture_output=True, text=True, timeout=60)
    run(["systemctl", "try-restart", CONTROL_TIMER], check=True, capture_output=True, text=True, timeout=60)
    return {"control_refresh_interval": interval, "drop_in": path.exists()}


def schedule_follow_up(*, root="/", run=subprocess.run) -> bool:
    """With a control_refresh_interval longer than FOLLOW_UP_SECONDS, start one more check
    FOLLOW_UP_SECONDS from now (a transient timer); False when the interval is the default or
    a follow-up is already scheduled."""
    if load(root=root)["control_refresh_interval"] <= FOLLOW_UP_SECONDS:
        return False
    try:
        done = run(["systemd-run", "--quiet", "--collect", f"--unit={FOLLOW_UP_UNIT}",
                    f"--on-active={FOLLOW_UP_SECONDS}", "--timer-property=AccuracySec=1s",
                    "/usr/bin/systemctl", "start", "--no-block", CONTROL_SERVICE],
                   capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False   # the periodic timer still runs the check
    return done.returncode == 0


# -- the host agent's pace ----------------------------------------------------------------------
def serving(document) -> bool:
    """Whether a SparkRing model container runs on this Spark, from its node status document."""
    return any(isinstance(row, dict) and row.get("state") == "running"
               and MODEL_CONTAINER.fullmatch(str(row.get("name") or ""))
               for row in (document or {}).get("containers") or [])


def agent_interval(document, *, root="/") -> int:
    """Seconds until the host agent's next observation."""
    return load(root=root)["agent_serving_interval"] if serving(document) else AGENT_INTERVAL


def stale_after(document) -> int:
    """Seconds after which a status document is stale: 90, or more for an agent paced slower."""
    interval = (document or {}).get("interval_seconds")
    return max(90, 2 * interval + 30) if type(interval) is int and interval > 0 else 90


# -- the recovery timer's SSH ---------------------------------------------------------------------
def mux_options(directory=MUX_DIRECTORY) -> list:
    """ssh options that use an open master connection and fall back to a fresh login without one."""
    return ["-o", "ControlMaster=no", "-o", f"ControlPath={directory}/%C"]


def recovery_invoke(*, root="/", ssh=None):
    """The SSH function the recovery timer uses: discovery.ssh, through the master connections
    when ``recovery_ssh_reuse`` is on."""
    from runtime.host import discovery
    ssh = ssh or discovery.ssh
    if not load(root=root)["recovery_ssh_reuse"]:
        return ssh

    def through_master(node, args, **kwargs):
        return ssh(node, args, options=mux_options(), **kwargs)
    return through_master


def master_argv(target, directory=MUX_DIRECTORY) -> list:
    return ["ssh", "-M", "-N", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            "-o", f"ControlPath={directory}/%C", "-o", "ControlPersist=no",
            "-o", f"ServerAliveInterval={MUX_ALIVE_SECONDS}", "-o", f"ServerAliveCountMax={MUX_ALIVE_COUNT}",
            target]


def recorded_targets(state_root=Path("/var/lib/sparkring/controller")) -> list:
    """The SSH target of every Spark of the recorded cluster, as setup recorded them."""
    try:
        cluster = json.loads((Path(state_root) / "cluster.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [host["host"] for host in cluster["plan"]["spec"]["hosts"]]


class Masters:
    """One SSH master connection per target, started again after it exits (with a backoff)."""

    def __init__(self, *, popen=subprocess.Popen, clock=time.monotonic, directory=MUX_DIRECTORY):
        self.popen, self.clock, self.directory = popen, clock, directory
        self.running: dict = {}      # target -> process
        self.retry: dict = {}        # target -> (next start time, failures)

    def keep(self, targets) -> None:
        now = self.clock()
        for target in [t for t in self.running if t not in targets]:
            self.stop(target)
        for target in targets:
            process = self.running.get(target)
            if process is not None and process.poll() is None:
                continue
            if process is not None:
                del self.running[target]
                failures = self.retry.get(target, (0, 0))[1] + 1
                delay = min(MUX_BACKOFF_SECONDS[1], MUX_BACKOFF_SECONDS[0] * failures)
                self.retry[target] = (now + delay, failures)
            if now < self.retry.get(target, (0, 0))[0]:
                continue
            self.running[target] = self.popen(master_argv(target, self.directory), stdin=subprocess.DEVNULL,
                                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def stop(self, target=None) -> None:
        for name in ([target] if target else list(self.running)):
            process = self.running.pop(name, None)
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()


def mux_main(*, root="/", sleep=time.sleep, masters=None, rounds=None) -> int:
    """``sparkring node ssh-mux`` (sparkring-ssh-mux.service on Node A): while ``recovery_ssh_reuse``
    is on, keep one master connection to every recorded Spark; while it is off, keep none."""
    directory = _path(root, MUX_DIRECTORY)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    masters = masters or Masters(directory=str(directory))
    stopping = []
    signal.signal(signal.SIGTERM, lambda *_: stopping.append(True))
    try:
        while not stopping and (rounds is None or rounds > 0):
            masters.keep(recorded_targets() if load(root=root)["recovery_ssh_reuse"] else [])
            if rounds is not None:
                rounds -= 1
            sleep(MUX_CHECK_SECONDS)
    finally:
        masters.stop()
    return 0


# -- commands -------------------------------------------------------------------------------------
def show(*, root="/") -> dict:
    settings = load(root=root)
    return {"schema": SCHEMA, "settings": settings,
            "drop_ins": {unit: drop_in_path(unit, root=root).exists() for unit in UNITS},
            "file": str(_path(root, SETTINGS))}


MUX_UNIT = "sparkring-ssh-mux.service"
CONTROLLER_CLUSTER = "/var/lib/sparkring/controller/cluster.json"


def set_local(items, *, root="/", run=subprocess.run) -> dict:
    """Change this Spark's settings and apply them; returns the settings and what changed.

    A placement change rewrites the drop-ins and restarts the agent. On Node A (it holds the
    controller's cluster record) turning ``recovery_ssh_reuse`` on enables and starts MUX_UNIT,
    and turning it off disables and stops it, which closes the master connections."""
    before = load(root=root)
    settings = {**before, **parse_assignments(items)}
    save(settings, root=root)
    result = {"schema": SCHEMA, "settings": settings}
    if settings["placement"] != before["placement"]:
        result["placement"] = apply_placement(settings, root=root, run=run)
    if settings["control_refresh_interval"] != before["control_refresh_interval"]:
        result["control"] = apply_control_interval(settings, root=root, run=run)
    if settings["recovery_ssh_reuse"] != before["recovery_ssh_reuse"] and _path(root, CONTROLLER_CLUSTER).exists():
        verb = ["enable", "--now"] if settings["recovery_ssh_reuse"] else ["disable", "--now"]
        run(["systemctl", *verb, MUX_UNIT], check=True, capture_output=True, text=True, timeout=60)
        result["ssh_mux"] = "started" if settings["recovery_ssh_reuse"] else "stopped"
    return result


def node_main(argv) -> dict:
    """``sparkring node host-load show|set KEY=VALUE...|apply``."""
    parser = argparse.ArgumentParser(prog="sparkring node host-load")
    actions = parser.add_subparsers(dest="host_load_action", required=True)
    actions.add_parser("show", help="print this Spark's settings and drop-ins")
    change = actions.add_parser("set", help="change settings: placement=default|efficiency, "
                                            "agent_serving_interval=SECONDS, recovery_ssh_reuse=on|off, "
                                            "control_refresh_interval=SECONDS")
    change.add_argument("assignments", nargs="+")
    actions.add_parser("apply", help="write the placement drop-ins again, for example after a package upgrade")
    args = parser.parse_args(argv)
    if args.host_load_action == "show":
        return show()
    if args.host_load_action == "set":
        return set_local(args.assignments)
    settings = load()
    return {"schema": SCHEMA, "settings": settings, "placement": apply_placement(settings),
            "control": apply_control_interval(settings)}


def fabric_targets() -> list:
    return recorded_targets()


def main(argv=None) -> int:
    """``sudo sparkring host-load show|set KEY=VALUE... [--all]`` on Node A."""
    parser = argparse.ArgumentParser(prog="sparkring host-load", description=(
        "Show or change where and how often SparkRing's host agent and recovery timer run. Without --all, on "
        "this Spark only; with --all, on every Spark of the recorded fabric over the administration network."))
    parser.add_argument("action", choices=("show", "set"))
    parser.add_argument("assignments", nargs="*", help="KEY=VALUE: placement=default|efficiency, "
                                                         "agent_serving_interval=SECONDS, recovery_ssh_reuse=on|off, "
                                                         "control_refresh_interval=SECONDS")
    parser.add_argument("--all", action="store_true", help="every Spark of the recorded fabric (Node A, sudo)")
    args = parser.parse_args(argv)
    try:
        if args.action == "set":
            parse_assignments(args.assignments)  # refuse a bad value before any Spark changes
            if not args.assignments:
                raise ValueError("set needs at least one KEY=VALUE")
        if not args.all:
            if args.action == "set" and (not hasattr(os, "geteuid") or os.geteuid() != 0):
                raise ValueError("Run it with sudo")
            print(json.dumps(set_local(args.assignments) if args.action == "set" else show(), indent=2))
            return 0
        from runtime.host import discovery
        results, failed = {}, False
        for target in fabric_targets():
            command = ["/usr/bin/sparkring", "node", "host-load", args.action, *args.assignments]
            prefix = [] if target.startswith("root@") else ["sudo", "-n"]
            try:
                results[target] = json.loads(discovery.ssh(target, prefix + command))
            except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as error:
                results[target] = {"error": str(error).strip().splitlines()[-1] if str(error).strip() else "failed"}
                failed = True
        if not results:
            raise ValueError("No fabric is recorded on this Spark; run it on Node A after sudo sparkring setup")
        print(json.dumps(results, indent=2))
        return 1 if failed else 0
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print("SparkRing host-load: " + str(error), file=sys.stderr)
        return 2
