"""The ConnectX hairpin setting on one Spark: approval, application, status and the mesh start check.

Status: implemented. Standard library only; it also runs in early boot.

Four-Spark rings relay traffic between nonadjacent Sparks through ConnectX
hardware forwarding. Every ConnectX function needs the mlx5 devlink driverinit
parameters ``hairpin_queue_size`` 8192 and ``hairpin_num_queues`` 4 and hardware
TC offload. A changed driverinit value takes effect only through a *function
restart* (``devlink dev reload pci/<bdf> action driver_reinit``), which
recreates the function's netdev and RDMA device and removes the addresses,
routes, TC rules and per-interface sysctls on it. Every probe of a function,
including every boot, returns the values to the driver defaults.
``scripts/hairpin_setting.py`` holds the rule that decides whether a function's
setting is *in effect*.

Commands, dispatched by ``scripts/sparkring_node.py`` as ``sparkring node hairpin``:

- ``approve`` writes ``/etc/sparkring/hairpin.json``, this Spark's record of the
  operator's consent. It never enables, starts or restarts anything.
- ``apply`` runs only as ``sparkring-hairpin.service``. A *boot run* happens
  before NetworkManager has started in the current boot; any later run is a
  *live run*. It restarts the approved functions that need it, one at a time,
  with the function that carries this Spark's administration path to Node A
  last, and enables the service after a run in which every approved function
  ended in effect and no restart failed (the Spark is then *armed*).
  ``--dry-run [--boot]`` prints the plan and changes nothing.
- ``start --after <invocation>`` starts the unit without waiting, unless a run
  newer than ``<invocation>`` exists or one is in progress, so a repeated
  dispatch never restarts a run.
- ``status [--busy]`` prints ``sparkring-hairpin-status/v1``.
- ``require --unit <unit>`` is the mesh start check that the systemd generator
  ``sparkring-hairpin-mesh-check`` adds to every mesh unit.
- ``resume`` starts the enabled mesh units that the check refused in this boot.
  The ring procedure runs it on every Spark once every Spark's run succeeded,
  so no mesh starts while another Spark still restarts functions.
- ``revoke`` disables the service and removes the approval and the attempt
  journal. Values already in effect stay until the next probe.

Files on the Spark:

- ``/etc/sparkring/hairpin.json`` (0644), ``sparkring-hairpin-approval/v1``:
  ``node_id``, ``parameters``, ``hw_tc_offload`` and four ``functions`` rows
  (``role``, ``rdma_device``, ``pci_address``, ``netdev``, ``mac``).
- ``/var/lib/sparkring/hairpin-attempts.json`` (0600),
  ``sparkring-hairpin-attempts/v1``: the latest 64 restart attempts. A record
  is written ``started`` and synced to disk before any change, then updated to
  ``applied``, ``failed`` (class ``restart``) or ``check-failed`` (class
  ``check``), with the operator message (``error``) and its short ``cause``. A
  successful live run marks the ``started`` and ``failed`` records of earlier
  runs ``resolved``. While the latest record of an approved function is
  ``started`` or ``failed`` from an earlier boot, boot runs restart nothing.
- ``/run/sparkring-hairpin/state.json``, ``sparkring-hairpin-state/v1``: the
  last run of this boot, with its systemd invocation ID.
- ``/run/sparkring-hairpin-blocked/<unit>``: mesh units whose start the check
  refused in this boot; ``resume`` starts the enabled ones.
- ``/run/sparkring-hairpin.lock``: the ``flock`` taken by approve, apply,
  resume and revoke. ``/run`` exists before any unit starts.

Every subprocess has a time limit. A child that exceeds it is killed and
abandoned without being reaped, because a driver restart that hangs inside the
kernel cannot be reaped.
"""
import contextlib
import fnmatch
import ipaddress
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

from runtime.host import node
from scripts import hairpin_setting as rule

UNIT = "sparkring-hairpin.service"
APPROVAL = "/etc/sparkring/hairpin.json"
ATTEMPTS = "/var/lib/sparkring/hairpin-attempts.json"
STATE = "/run/sparkring-hairpin/state.json"
BLOCKED = "/run/sparkring-hairpin-blocked"
LOCK = "/run/sparkring-hairpin.lock"
FABRIC = "/etc/sparkring/fabric.json"
CONTROL = "/etc/sparkring/control.json"
NODE = "/etc/sparkring/node.json"
APPROVAL_SCHEMA = "sparkring-hairpin-approval/v1"
ATTEMPTS_SCHEMA = "sparkring-hairpin-attempts/v1"
STATE_SCHEMA = "sparkring-hairpin-state/v1"
STATUS_SCHEMA = "sparkring-hairpin-status/v1"
DRY_RUN_SCHEMA = "sparkring-hairpin-dry-run/v1"
ATTEMPT_LIMIT = 64
KERNEL_OPTION = "sparkring.hairpin=off"
PREFIX = "SparkRing hairpin: "
# Role order of scripts/deploy_network.py ROLES; restarts use it within each group.
ROLES = ("cw_primary", "cw_secondary", "ccw_primary", "ccw_secondary")
# Units whose presence makes a function restart unsafe, and the subset that
# the generator gives the start check.
BUSY_UNITS = ("sparkring-mesh.service", "sparkring-*-mesh.service",
              "sparkring-mesh-model.service", "sparkring-*-model.service")
MESH_UNITS = ("sparkring-mesh.service", "sparkring-*-mesh.service")
TUNNEL_INTERFACE = "sr-control"

# Time limits in seconds. A run first waits up to PRESENT for the approved
# functions (a boot run also up to UDEV for udev). BUDGET[mode] then covers the
# restarts: a restart starts only while RESERVE[mode] of the budget remains, and
# RESERVE is the sum of the waits of one restart: RELOAD + RETURN at boot, plus
# ADDRESSES, ADDRESSES_AFTER_UP and TUNNEL in a live run. Each command outside
# these waits (a read, param set, the endpoint refresh) is limited to READ and
# normally returns within a second. The steps after the last restart (offload,
# the follow-up requests, the final read, arming) have no budget of their own:
# sparkring-hairpin.service allows 330 s to start, which is PRESENT +
# BUDGET["live"] + 35 s for them. When commands hang up to their limits, the
# start timeout stops the run instead: the stop request (SIGTERM to this process
# only, KillMode=mixed) starts no further function, and the run exits once the
# function it is restarting has returned, within TimeoutStopSec=60 (RELOAD +
# RETURN).
PRESENT = 15
UDEV = 10
RELOAD = 30
RETURN = 20
ADDRESSES = 30
ADDRESSES_AFTER_UP = 10
TUNNEL = 30
BUDGET = {"boot": 120, "live": 280}
RESERVE = {"boot": RELOAD + RETURN, "live": RELOAD + RETURN + ADDRESSES + ADDRESSES_AFTER_UP + TUNNEL}
READ = 10
LOCK_WAIT = 10
POLL = 0.5

_BDF = re.compile(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]")
_NETDEV = re.compile(r"[A-Za-z0-9_.-]{1,15}")
_MAC = re.compile(r"(?:[0-9a-f]{2}:){5}[0-9a-f]{2}")
_RDMA = re.compile(r"[A-Za-z0-9_]{1,64}")
_UNIT = re.compile(r"[A-Za-z0-9@._:-]{1,240}\.service")


class HairpinError(ValueError):
    """A hairpin command stopped; the message is operator text.

    ``kind`` is ``restart`` or ``check`` for restart outcomes, otherwise the
    ID of the message in the design's message table (for example ``M3``).
    ``cause``, when given, is the short form that summaries and the ring
    procedure show instead of the whole message.
    """

    def __init__(self, message, kind=None, cause=None):
        super().__init__(message)
        self.kind = kind
        self.cause = cause


def execute(argv, *, timeout, input=None, **_):
    """``subprocess.run`` with captured text output whose timeout abandons the child.

    ``subprocess.run`` reaps a child after killing it, which blocks forever on
    a process stuck inside the kernel. This kills the child, closes the pipes
    and raises ``subprocess.TimeoutExpired`` without waiting.
    """
    process = subprocess.Popen(argv, stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        stdout, stderr = process.communicate(input, timeout=timeout)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(OSError):
            process.kill()
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                with contextlib.suppress(OSError):
                    stream.close()
        raise subprocess.TimeoutExpired(argv, timeout) from None
    return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)


class Completed:
    """Outcome of one command; ``returncode`` is None when it timed out."""

    def __init__(self, returncode, stdout="", stderr="", timed_out=False):
        self.returncode, self.stdout, self.stderr, self.timed_out = returncode, stdout or "", stderr or "", timed_out

    @property
    def ok(self):
        return self.returncode == 0 and not self.timed_out

    def reason(self):
        """The command's error text on one line, so that each log entry stays one journal line."""
        if self.timed_out:
            return "timed out"
        text = self.stderr if self.stderr.strip() else self.stdout
        return "; ".join(line.strip() for line in text.splitlines() if line.strip()) or f"exit status {self.returncode}"


def _journal_stream(environ, output):
    """Whether log lines reach the systemd journal, which reads a leading ``<3>`` as error priority.

    systemd sets JOURNAL_STREAM to the device and inode of the stream that
    connects a service's output to the journal. For the default output
    (stdout) they must match, because the variable can be inherited by a
    process whose output goes elsewhere; a given ``output`` relies on the
    variable alone.
    """
    value = environ.get("JOURNAL_STREAM")
    if not value:
        return False
    if output is not None:
        return True
    try:
        device, inode = (int(part) for part in value.split(":", 1))
        status = os.fstat(sys.stdout.fileno())
    except (ValueError, OSError, AttributeError):
        return False
    return (status.st_dev, status.st_ino) == (device, inode)


class Host:
    """This Spark as the commands see it: files below ``root``, commands, clocks and the log.

    ``run`` has the ``subprocess.run`` signature (it receives ``capture_output``,
    ``text`` and ``timeout``). Log lines go to ``output``. Under systemd an
    error line starts with ``<3>``, which the journal files at error priority;
    in a terminal it does not.
    """

    def __init__(self, *, root="/", run=None, clock=time.monotonic, sleep=time.sleep, now=time.time,
                 environ=None, output=None, refresh_endpoint=None, hostname=None):
        self.root = str(root)
        self.run = run or execute
        self.clock, self.sleep, self.now = clock, sleep, now
        self.environ = os.environ if environ is None else environ
        self.journal = _journal_stream(self.environ, output)
        self.output = output or (lambda line: print(line, flush=True))
        self._refresh_endpoint = refresh_endpoint
        self.hostname = hostname or socket.gethostname()
        self.stopping = False

    def stop(self, *_):
        """Request that a run starts no further function restart (SIGTERM)."""
        self.stopping = True

    def path(self, name):
        return Path(self.root) / str(name).lstrip("/")

    def command(self, argv, limit):
        try:
            result = self.run(list(argv), capture_output=True, text=True, timeout=limit)
        except subprocess.TimeoutExpired:
            return Completed(None, stderr=f"timed out after {limit} s", timed_out=True)
        except OSError as error:
            return Completed(127, stderr=str(error))
        return Completed(result.returncode, result.stdout, result.stderr)

    def text(self, name):
        try:
            return self.path(name).read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            return None

    def entries(self, name):
        try:
            return sorted(entry.name for entry in self.path(name).iterdir())
        except OSError:
            return None

    def log(self, line):
        self.output(line)

    def error(self, line):
        self.output(("<3>" if self.journal else "") + line)

    def refresh_endpoint(self, netdev):
        """Set the administration peer endpoint on netdev again; each command is limited to READ."""
        if self._refresh_endpoint is not None:
            return self._refresh_endpoint(netdev)
        from runtime.host import control_node

        def run(argv, **options):
            return self.run(argv, **{**options, "timeout": min(options.get("timeout") or READ, READ)})
        return control_node.refresh_endpoint(netdev, root=self.root, run=run)


def _json(result):
    if not result.ok:
        return None
    try:
        return json.loads(result.stdout or "null")
    except ValueError:
        return None


def boot_id(host):
    return host.text("/proc/sys/kernel/random/boot_id")


def _function_name(function):
    return f"{function['netdev']} (pci/{function['pci_address']})"


def _values_text(row):
    values = row.get("values") or {}
    return (f"hairpin_queue_size {values.get('hairpin_queue_size')}, "
            f"hairpin_num_queues {values.get('hairpin_num_queues')}, "
            f"driver restarts since boot {row.get('driver_reinit')}, hw-tc-offload {row.get('offload')}")


def _utc(value):
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(value)) if isinstance(value, (int, float)) else "an unknown time"


def _clause(text):
    """Text to embed before further words: one line, without a final period."""
    return " ".join(str(text).split()).rstrip(". ")


def _first_sentence(text):
    """The first sentence of an operator message, without the command prefix."""
    return _clause(str(text).removeprefix(PREFIX).removeprefix("<3>").split(". ", 1)[0])


# Messages. The IDs are those of the design's message table. Restart failures
# carry a short cause (for example "driver restart failed: <kernel text>"),
# which the attempt journal and state.json keep beside the full message.

RETRY = " On Node A, sudo sparkring hairpin retries it."
POWER_CYCLE = " If this Spark cannot be reached, power-cycle it: it then starts without restarting any function."


def cut_off_advice(function, *, head):
    """Remedy after a failed restart of a function that carries the administration tunnel to other Sparks.

    Those Sparks cannot be reached until the function works again, and sudo
    sparkring hairpin needs every Spark. After a failed or unfinished restart
    the next boot of this Spark restarts no function, so a reboot brings them
    back.
    """
    netdev = function["netdev"]
    if head:
        return (f" The workers behind {netdev} are cut off from the administration network, and sudo sparkring "
                "hairpin needs them: reboot Node A (power-cycle it if the reboot hangs), then run sudo sparkring "
                "hairpin.")
    return (f" The Sparks behind {netdev} are cut off from the administration network, and sudo sparkring hairpin "
            "needs them: reboot this Spark (power-cycle it if the reboot hangs), then run sudo sparkring hairpin "
            "on Node A.")


def message_restart(function, cause, advice=RETRY):
    """M1 and M2: a restart failed or did not finish; later functions stay untouched."""
    return (PREFIX + f"{_function_name(function)}: {_clause(cause)}. Later functions were not restarted; four-Spark "
            "forwarding stays off on this Spark and its mesh does not start. Boot runs on this Spark restart "
            "nothing until a live retry succeeds." + advice)


def message_m1(function, text, advice=RETRY):
    return message_restart(function, "driver restart failed: " + _clause(text), advice)


def message_m3(function):
    return (PREFIX + f"pci/{function['pci_address']} ({function['rdma_device']}) is not present after "
            f"{PRESENT} s; nothing was restarted. Check that port's cable and transceiver, then run sudo "
            "sparkring hairpin on Node A.")


def message_m4(hostname, findings):
    return (PREFIX + f"not restarting ConnectX drivers on {hostname}: "
            + "; ".join(item["detail"] for item in findings) + ". Nothing was changed.")


def message_m11(approved):
    return (PREFIX + f"{APPROVAL} approves hairpin_queue_size {approved}, but this package requires "
            f"{rule.HAIRPIN_QUEUE_SIZE}. Nothing was changed. On Node A, sudo sparkring hairpin asks to "
            "approve the required value.")


def record_cause(record):
    """The short cause of an attempt record: its ``cause``, else its message without the prefix, else None."""
    cause = record.get("cause") or record.get("error")
    return _clause(str(cause).removeprefix("<3>").removeprefix(PREFIX)) if cause else None


def message_m12(record):
    outcome = "did not finish" if record.get("state") == "started" else "failed"
    cause = record_cause(record)
    return (PREFIX + f"boot restarts are suspended on this Spark because the restart of {record.get('netdev')} "
            f"(pci/{record.get('pci_address')}) {outcome} during an earlier boot ({_utc(record.get('time'))})"
            + (f": {cause}" if cause else "") + ". Nothing was restarted; four-Spark forwarding stays off and its "
            "mesh does not start. On Node A, sudo sparkring hairpin retries it live.")


def suspension_text(record):
    """What suspends boot runs, for status warnings and the ring's consent lines."""
    name = record.get("netdev") or f"pci/{record.get('pci_address')}"
    if record.get("state") == "started":
        return f"the restart of {name} at {_utc(record.get('time'))} did not finish"
    cause = record_cause(record)
    return f"{name} failed to restart at {_utc(record.get('time'))}" + (f" ({cause})" if cause else "")


def message_m13(differences):
    return ("This Spark already approves the hairpin setting for other ConnectX functions ("
            + "; ".join(differences) + "). If its card was replaced, run sudo sparkring node hairpin revoke on "
            "it, then repeat.")


def message_m14(function, found):
    return (PREFIX + f"pci/{function['pci_address']} now carries {found} instead of the approved "
            f"{function['netdev']}, MAC {function['mac']}, {function['rdma_device']}; nothing was restarted. If "
            "the card was replaced, run sudo sparkring node hairpin revoke on this Spark, then sudo sparkring "
            "hairpin on Node A.")


def message_m16(hostname, row):
    if row.get("driver_reinit") is None or row.get("reload_failed") is None:
        missing = "devlink reload statistics are"
    elif None in (row.get("values") or {}).values() or not row.get("values"):
        missing = "the devlink hairpin values are"
    else:
        missing = "the hw-tc-offload setting is"
    return (f"{hostname}: {missing} unavailable for {row['netdev']}; SparkRing cannot tell whether the "
            "hairpin setting is in effect and restarts nothing.")


def message_m17(size):
    return (PREFIX + f"this Spark's fabric record describes a {size}-Spark setup; the hairpin setting applies "
            "to four-Spark rings only. Remove the approval with sudo sparkring node hairpin revoke.")


def check_cause(what, seconds):
    """The cause of a check failure after a restart that succeeded (M18)."""
    return f"restarted, but {what} did not return within {seconds} s"


def message_m18(function, what, seconds):
    return (PREFIX + f"{_function_name(function)}: {check_cause(what, seconds)}; later functions were not "
            "restarted.")


def message_m20(function):
    return (PREFIX + f"hardware TC offload is off and fixed on {function['netdev']}; four-Spark forwarding "
            "cannot use this function.")


# Files.

def _read_optional(host, name):
    """(document, error): (None, None) when the file does not exist."""
    try:
        path = node.location(host.root, name)
        if not path.exists():
            return None, None
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, ValueError) as error:
        return None, f"{name}: {error}"


def _write_synced(host, name, value, mode):
    """Replace a file atomically and sync the file and its directory to disk."""
    path = node.location(host.root, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = node.location(host.root, name + ".writing")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temporary, mode)
    os.replace(temporary, path)
    if os.name == "posix":
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


@contextlib.contextmanager
def locked(host):
    """Hold the hairpin lock; wait up to LOCK_WAIT seconds for another holder."""
    import fcntl
    path = host.path(LOCK)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        deadline = host.clock() + LOCK_WAIT
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if host.clock() >= deadline:
                    raise HairpinError(PREFIX + "another hairpin operation is active on this Spark; "
                                       "repeat when it has finished", "lock") from None
                host.sleep(0.2)
        yield
    finally:
        os.close(descriptor)


def fabric_size(host):
    """The size in /etc/sparkring/fabric.json, or None when the file does not exist."""
    document, error = _read_optional(host, FABRIC)
    if error:
        raise HairpinError(PREFIX + "cannot read the fabric record " + error, "fabric")
    return document.get("size") if isinstance(document, dict) else None


def _node_id(host):
    return node.read(host.root, NODE)["node_id"]


def validate_approval(record, node_id):
    """Return the approval record, or raise HairpinError naming what is wrong."""
    def invalid(reason):
        return HairpinError(PREFIX + f"{APPROVAL} is not a valid approval ({reason}). Nothing was changed. "
                            "Run sudo sparkring node hairpin revoke on this Spark, then sudo sparkring hairpin "
                            "on Node A.", "approval")
    if not isinstance(record, dict) or record.get("schema") != APPROVAL_SCHEMA:
        raise invalid("unknown schema")
    parameters = record.get("parameters")
    if parameters != rule.PARAMETERS:
        approved = parameters.get("hairpin_queue_size") if isinstance(parameters, dict) else parameters
        raise HairpinError(message_m11(approved), "M11")
    if record.get("hw_tc_offload") is not True:
        raise invalid("hw_tc_offload is not true")
    if record.get("node_id") != node_id:
        raise invalid("it belongs to another node identity")
    functions = record.get("functions")
    if not isinstance(functions, list) or sorted(f.get("role") for f in functions if isinstance(f, dict)) != sorted(ROLES):
        raise invalid("it must list the four ConnectX functions by role")
    for function in functions:
        if (not isinstance(function.get("pci_address"), str) or not _BDF.fullmatch(function["pci_address"])
                or not isinstance(function.get("netdev"), str) or not _NETDEV.fullmatch(function["netdev"])
                or not isinstance(function.get("mac"), str) or not _MAC.fullmatch(function["mac"])
                or not isinstance(function.get("rdma_device"), str) or not _RDMA.fullmatch(function["rdma_device"])):
            raise invalid("function " + str(function.get("role")) + " has an invalid identity")
    if len({f["pci_address"] for f in functions}) != 4 or len({f["netdev"] for f in functions}) != 4:
        raise invalid("functions repeat")
    return record


def load_approval(host):
    """The validated approval; stops with M17 on a Spark whose fabric record is not four-Spark."""
    document, error = _read_optional(host, APPROVAL)
    if error:
        raise HairpinError(PREFIX + "cannot read the approval " + error + ". Nothing was changed.", "approval")
    if document is None:
        raise HairpinError(PREFIX + f"{APPROVAL} does not exist; nothing was changed. On Node A, sudo sparkring "
                           "hairpin asks to approve the setting.", "approval")
    try:
        identity = _node_id(host)
    except (OSError, ValueError, KeyError, TypeError) as failure:
        raise HairpinError(PREFIX + f"cannot read this Spark's identity {NODE}: {failure}. Nothing was changed.",
                           "approval") from None
    record = validate_approval(document, identity)
    size = fabric_size(host)
    if size is not None and size != 4:
        raise HairpinError(message_m17(size), "M17")
    return record


def _identity(function):
    return {key: function[key] for key in ("role", "rdma_device", "pci_address", "netdev", "mac")}


def _sorted_functions(functions):
    return sorted(functions, key=lambda f: ROLES.index(f["role"]))


# sysfs.

def pci_address(host, rdma_device):
    """The PCI address of an RDMA device from /sys/class/infiniband/<rdma>/device, or None."""
    device = host.path(f"/sys/class/infiniband/{rdma_device}/device")
    try:
        name = device.resolve(strict=True).name.lower()
    except OSError:
        return None
    if _BDF.fullmatch(name):
        return name
    uevent = dict(line.split("=", 1) for line in (host.text(f"/sys/class/infiniband/{rdma_device}/device/uevent") or "").splitlines()
                  if "=" in line)
    value = uevent.get("PCI_SLOT_NAME", "").lower()
    return value if _BDF.fullmatch(value) else None


def _driver(host, rdma_device):
    link = host.path(f"/sys/class/infiniband/{rdma_device}/device/driver")
    if link.is_symlink():
        with contextlib.suppress(OSError):
            return link.resolve(strict=True).name
    uevent = dict(line.split("=", 1) for line in (host.text(f"/sys/class/infiniband/{rdma_device}/device/uevent") or "").splitlines()
                  if "=" in line)
    return uevent.get("DRIVER")


def _mac(host, pci, netdev):
    value = host.text(f"/sys/bus/pci/devices/{pci}/net/{netdev}/address")
    return value.lower() if value else None


def derive_functions(host):
    """The four function rows of this Spark from the fixed RDMA device names and sysfs."""
    from runtime.host.topology import DEVICES
    rows, problems = [], []
    for role in ROLES:
        rdma = DEVICES[role]
        pci = pci_address(host, rdma)
        if pci is None:
            problems.append(f"{rdma}: RDMA device or its PCI address is unavailable")
            continue
        if _driver(host, rdma) != "mlx5_core":
            problems.append(f"{rdma} (pci/{pci}): driver is not mlx5_core")
            continue
        netdevs = host.entries(f"/sys/bus/pci/devices/{pci}/net") or []
        if len(netdevs) != 1:
            problems.append(f"{rdma} (pci/{pci}): expected one netdev, found {len(netdevs)}")
            continue
        mac = _mac(host, pci, netdevs[0])
        if not mac or not _MAC.fullmatch(mac):
            problems.append(f"{netdevs[0]} (pci/{pci}): MAC address unavailable")
            continue
        rows.append({"role": role, "rdma_device": rdma, "pci_address": pci, "netdev": netdevs[0], "mac": mac})
    if problems:
        raise HairpinError(PREFIX + "cannot identify this Spark's four ConnectX functions: " + "; ".join(problems))
    return rows


def identity_state(host, function):
    """("present", None), ("absent", None) or ("changed", what the PCI function carries now)."""
    pci = function["pci_address"]
    if not host.command(["devlink", "-j", "dev", "show", "pci/" + pci], READ).ok:
        return "absent", None
    netdevs = host.entries(f"/sys/bus/pci/devices/{pci}/net")
    devices = host.entries(f"/sys/bus/pci/devices/{pci}/infiniband")
    if not netdevs or not devices:
        return "absent", None
    mac = _mac(host, pci, netdevs[0])
    if netdevs != [function["netdev"]] or mac != function["mac"] or devices != [function["rdma_device"]]:
        return "changed", f"{', '.join(netdevs)}, MAC {mac}, {', '.join(devices)}"
    return "present", None


def read_state(host, function):
    """One function's status row by the in-effect rule, read with devlink and ethtool."""
    device = "pci/" + function["pci_address"]
    values = {}
    for name in rule.PARAMETERS:
        document = _json(host.command(["devlink", "-j", "dev", "param", "show", device, "name", name], READ))
        values[name] = rule.driverinit_value(document, device, name)
    document = _json(host.command(["devlink", "-s", "-j", "dev", "show", device], READ))
    statistics = rule.reload_statistics(document, device)
    result = host.command(["ethtool", "-k", function["netdev"]], READ)
    offload = rule.ethtool_offload(result.stdout) if result.ok else None
    return {**_identity(function), **rule.evaluate(values, statistics, offload)}


def facts_state(function, facts):
    """One function's status row computed from inventory facts, without running devlink."""
    rdma = next((row for row in facts.get("rdma") or [] if row.get("device") == function["rdma_device"]), {})
    nic = next((row for row in facts.get("interfaces") or [] if row.get("name") == function["netdev"]), {})
    link = rdma.get("devlink") or {}
    parameters = link.get("parameters") or {}
    row = rule.evaluate({name: (parameters.get(name) or {}).get("value") for name in rule.PARAMETERS},
                        link.get("reload"), rule.offload_setting(nic.get("hw_tc_offload"), nic.get("hw_tc_offload_fixed")))
    return {**_identity(function), **row}


# The attempt journal.

def load_attempts(host):
    """(records, error). A missing journal has no records."""
    document, error = _read_optional(host, ATTEMPTS)
    if error:
        return [], error
    if document is None:
        return [], None
    if not isinstance(document, dict) or document.get("schema") != ATTEMPTS_SCHEMA or not isinstance(document.get("records"), list):
        return [], f"{ATTEMPTS}: unknown schema"
    return [record for record in document["records"] if isinstance(record, dict)], None


def save_attempts(host, records):
    _write_synced(host, ATTEMPTS, {"schema": ATTEMPTS_SCHEMA, "records": records[-ATTEMPT_LIMIT:]}, 0o600)


def suspending_record(records, functions, *, other_than_boot=None):
    """The unresolved record that suspends boot runs, or None.

    Only the latest record of each approved function counts, and only a
    ``started`` or ``failed`` one; ``check-failed`` records never suspend.
    With other_than_boot, records of that boot are ignored, which is the view
    of a boot run in that boot.
    """
    addresses = {function["pci_address"] for function in functions}
    latest = {}
    for record in records:
        if record.get("pci_address") in addresses:
            latest[record["pci_address"]] = record
    candidates = [record for record in latest.values() if record.get("state") in ("started", "failed")
                  and (other_than_boot is None or record.get("boot_id") != other_than_boot)]
    return max(candidates, key=lambda record: record.get("time") or 0) if candidates else None


# Order, busy check and mode.

def tunnel_roles(host):
    """Map each administration tunnel netdev to (``parent`` or ``child``, peer, ping address).

    Tunnel functions are the peers' netdevs in /etc/sparkring/control.json. On
    a worker the parent peer is the one whose allowed_ips contain head_address/32
    or 0.0.0.0/0; on Node A every peer is a child. A parent is checked with a
    ping of head_address, a child with a ping of its own address (its first
    allowed_ips entry).
    """
    document, error = _read_optional(host, CONTROL)
    if error:
        raise HairpinError(PREFIX + "cannot read the administration network record " + error
                           + "; nothing was changed.", "control")
    if document is None:
        return {}
    roles = {}
    head = document.get("head_address")
    for peer in document.get("peers") or []:
        allowed = peer.get("allowed_ips") or []
        parent = not document.get("head") and (f"{head}/32" in allowed or "0.0.0.0/0" in allowed)
        if parent:
            address = head
        else:
            address = str(ipaddress.ip_interface(allowed[0]).ip) if allowed else None
        roles[peer.get("netdev")] = ("parent" if parent else "child", peer, address)
    return roles


def control_head(host):
    """Whether /etc/sparkring/control.json makes this Spark Node A of the administration tree."""
    document, _ = _read_optional(host, CONTROL)
    return bool(isinstance(document, dict) and document.get("head"))


def restart_order(functions, tunnels):
    """Functions without a tunnel, then child-facing tunnel functions, then the parent-facing one."""
    group = {None: 0, "child": 1, "parent": 2}
    return sorted(functions, key=lambda f: (group[(tunnels.get(f["netdev"]) or (None,))[0]], ROLES.index(f["role"])))


def detect_mode(host):
    """``boot`` while NetworkManager has not started in this boot, otherwise ``live``.

    InactiveExitTimestampMonotonic stays 0 until the unit first leaves the
    inactive state, so the answer does not depend on NetworkManager's current
    state (restarting or failed).
    """
    result = host.command(["systemctl", "show", "-p", "InactiveExitTimestampMonotonic", "--value",
                           "NetworkManager.service"], READ)
    value = result.stdout.strip()
    if not result.ok or not value.isdigit():
        raise HairpinError(PREFIX + "cannot tell whether NetworkManager has started in this boot ("
                           + result.reason() + "); nothing was changed.", "mode")
    return "boot" if int(value) == 0 else "live"


def _show(host, units, properties):
    """``systemctl show`` properties for each unit, keyed by unit Id; None when systemctl fails."""
    result = host.command(["systemctl", "show", "-p", ",".join(("Id", *properties)), *units], READ)
    if not result.ok:
        return None
    shown, current = {}, {}
    for line in [*result.stdout.splitlines(), ""]:
        if not line.strip():
            if current.get("Id"):
                shown[current["Id"]] = current
            current = {}
            continue
        key, _, value = line.partition("=")
        current[key] = value
    return shown


def _loaded_units(host, patterns):
    """(names, error) of loaded units matching the patterns."""
    result = host.command(["systemctl", "list-units", "--all", "--plain", "--no-legend", "--no-pager",
                           "--type=service", *patterns], READ)
    if not result.ok:
        return [], result.reason()
    names = []
    for line in result.stdout.splitlines():
        words = line.split()
        if words and not _UNIT.fullmatch(words[0]):
            words = words[1:]
        if words and any(fnmatch.fnmatchcase(words[0], pattern) for pattern in patterns):
            names.append(words[0])
    return sorted(set(names)), None


def _unit_processes(host, group):
    if not group:
        return []
    text = host.text("/sys/fs/cgroup/" + group.strip("/") + "/cgroup.procs")
    return [line for line in (text or "").split() if line.strip()]


def _ipv4(host, netdev):
    rows = _json(host.command(["ip", "-j", "address", "show", "dev", netdev], READ)) or []
    row = rows[0] if isinstance(rows, list) and rows and isinstance(rows[0], dict) else {}
    info = [item for item in row.get("addr_info") or [] if isinstance(item, dict)]
    return {
        "operstate": row.get("operstate"),
        "ipv4": sorted(f"{item.get('local')}/{item.get('prefixlen')}" for item in info if item.get("family") == "inet"),
        "link_local": sorted(str(item.get("local")) for item in info
                             if item.get("family") == "inet6" and item.get("scope") == "link"),
    }


def _nmcli_values(text):
    r"""Lines of ``nmcli -g`` output with its ``\:`` and ``\\`` escapes removed."""
    return [line.replace("\\:", ":").replace("\\\\", "\\") for line in text.splitlines()]


def busy_findings(host, functions, mode, tunnels=None):
    """Findings that make a function restart unsafe now; each is {"kind", "detail", ...}.

    Units: mesh and model units that are not inactive or failed, or whose
    control group still lists processes. Forwarding: TC ingress filters on a
    fabric netdev. RDMA: queue pairs or protection domains with a PID on a
    fabric RDMA device. Live runs also check GPU compute processes and that
    NetworkManager will bring each addressed function back by itself.
    """
    findings = []
    units, error = _loaded_units(host, BUSY_UNITS)
    if error:
        findings.append({"kind": "unit", "detail": "cannot list mesh and model units: " + error})
    shown = _show(host, units, ("ActiveState", "ControlGroup")) if units else {}
    if shown is None:
        findings.append({"kind": "unit", "detail": "cannot read the state of " + ", ".join(units)})
        shown = {}
    for name in units:
        values = shown.get(name, {})
        state = values.get("ActiveState")
        processes = _unit_processes(host, values.get("ControlGroup"))
        if state not in ("inactive", "failed"):
            findings.append({"kind": "unit", "unit": name, "active_state": state, "processes": processes,
                             "detail": f"{name} is {state or 'in an unknown state'}"})
        elif processes:
            findings.append({"kind": "unit", "unit": name, "active_state": state, "processes": processes,
                             "detail": f"{name} is {state} but its processes {', '.join(processes)} still run"})
    for function in functions:
        result = host.command(["tc", "-j", "filter", "show", "dev", function["netdev"], "ingress"], READ)
        rules = _json(result)
        if not result.ok or (result.stdout.strip() and not isinstance(rules, list)):
            findings.append({"kind": "forwarding", "netdev": function["netdev"],
                             "detail": f"cannot read forwarding rules on {function['netdev']}: {result.reason()}"})
        elif rules:
            findings.append({"kind": "forwarding", "netdev": function["netdev"], "count": len(rules),
                             "detail": f"{len(rules)} forwarding rule(s) on {function['netdev']}"})
    devices = {function["rdma_device"] for function in functions}
    seen = set()
    for table in ("qp", "pd"):
        result = host.command(["rdma", "-j", "resource", "show", table], READ)
        rows = [] if result.ok and not result.stdout.strip() else _json(result)
        if not isinstance(rows, list):
            findings.append({"kind": "rdma", "detail": f"cannot read RDMA {table} resources: {result.reason()}"})
            continue
        for row in rows:
            if not isinstance(row, dict) or row.get("pid") is None:
                continue
            device = str(row.get("ifname") or "").split("/")[0]
            if device in devices and (device, row["pid"]) not in seen:
                seen.add((device, row["pid"]))
                findings.append({"kind": "rdma", "rdma_device": device, "pid": row["pid"], "command": row.get("comm"),
                                 "detail": f"RDMA user {row.get('comm') or 'unknown'} (PID {row['pid']}) on {device}"})
    if mode == "live":
        result = host.command(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], READ)
        if not result.ok:
            findings.append({"kind": "gpu", "detail": "cannot query GPU compute processes: " + result.reason()})
        else:
            for pid in [line.strip() for line in result.stdout.splitlines() if line.strip()]:
                findings.append({"kind": "gpu", "pid": pid, "detail": f"GPU compute process PID {pid}"})
        if tunnels is None:
            try:
                tunnels = tunnel_roles(host)
            except HairpinError as error:
                findings.append({"kind": "network", "detail": str(error)})
                tunnels = {}
        for function in functions:
            finding = _connection_finding(host, function, function["netdev"] in tunnels)
            if finding:
                findings.append(finding)
    return findings


def _connection_finding(host, function, tunnel):
    """A finding when NetworkManager would not bring an addressed function back after a restart."""
    netdev = function["netdev"]
    addresses = _ipv4(host, netdev)
    if not addresses["ipv4"] and not tunnel:
        return None
    result = host.command(["nmcli", "-g", "GENERAL.CON-UUID", "device", "show", netdev], READ)
    identifier = result.stdout.strip() if result.ok else ""
    if not identifier:
        return {"kind": "network", "netdev": netdev,
                "detail": f"{netdev} has addresses but no active NetworkManager connection"}
    result = host.command(["nmcli", "-g", "connection.id,connection.autoconnect,connection.interface-name,"
                           "802-3-ethernet.mac-address", "connection", "show", "uuid", identifier], READ)
    if not result.ok or not result.stdout.strip():
        return {"kind": "network", "netdev": netdev, "connection": identifier,
                "detail": f"cannot read NetworkManager connection {identifier} on {netdev}: {result.reason()}"}
    # nmcli prints one line per field; an empty value is an empty line.
    name, autoconnect, interface, mac = (_nmcli_values(result.stdout) + ["", "", "", ""])[:4]
    problems = []
    if autoconnect.strip() != "yes":
        problems.append("autoconnect is not yes")
    if interface.strip() != netdev and mac.strip().lower() != function["mac"]:
        problems.append("it is bound neither to the interface name nor to its MAC")
    if not problems:
        return None
    return {"kind": "network", "netdev": netdev, "connection": name,
            "detail": f"NetworkManager connection '{name}' on {netdev} would not come back after a restart: "
                      + " and ".join(problems)}


# apply.

class Run:
    """One run of ``sparkring node hairpin apply`` under its unit."""

    def __init__(self, host, invocation_id):
        self.host = host
        self.invocation_id = invocation_id
        self.started = host.clock()
        self.started_at = host.now()
        self.boot_id = boot_id(host)
        self.mode = None
        self.functions = []
        self.tunnels = {}
        self.head = False
        self.results = {}
        self.final = {}
        self.suspended = None
        self.restarted = []
        self.error = None
        self.kind = None
        self.cause = None
        self.failed_function = None
        self.cut_off = None
        self.in_effect = False
        self.armed = None
        self.attempts = []
        self.earlier = 0
        self.reported = set()
        self.deadline = None

    def fail(self, text, kind, *, cause=None, function=None, cut_off=None):
        """Log a failure; the first one names the run's error, its class and its short cause.

        ``cut_off`` is the netdev of a failed function that carries the
        administration tunnel to other Sparks, which the ring procedure's
        advice names.
        """
        self.host.error(text)
        if self.error is None:
            self.error, self.kind = text, kind
            self.cause = _clause(cause) if cause else _first_sentence(text)
            self.failed_function = _function_name(function) if function else None
            self.cut_off = cut_off

    def result(self, function):
        return self.results.setdefault(function["pci_address"],
                                       {"action": "none", "seconds": 0.0, "error": None, "cause": None})

    def execute(self):
        host = self.host
        approval = load_approval(host)
        self.functions = _sorted_functions(approval["functions"])
        self.mode = detect_mode(host)
        records, journal_error = load_attempts(host)
        self.attempts = records
        # A successful live run resolves the records of earlier runs only.
        self.earlier = len(records)
        if self.mode == "boot":
            if journal_error:
                # An unreadable journal cannot show that earlier restarts
                # finished; boot runs then restart nothing.
                self.suspended = {"state": "unreadable", "error": journal_error}
                self._settle()
                self.offload()
                raise HairpinError(PREFIX + "boot restarts are suspended on this Spark because the attempt "
                                   f"journal cannot be read ({journal_error}). Nothing was restarted. On Node A, "
                                   "sudo sparkring hairpin retries it live.", "M12")
            self.suspended = suspending_record(records, self.functions, other_than_boot=self.boot_id)
            if self.suspended:
                self._settle()
                self.offload()
                raise HairpinError(message_m12(self.suspended), "M12",
                                   cause="boot restarts suspended since " + suspension_text(self.suspended))
            self._settle()
        elif journal_error:
            host.error(PREFIX + "the attempt journal cannot be read (" + journal_error + "); this run starts a new one.")
        self.tunnels = tunnel_roles(host)
        self.head = control_head(host)
        self._wait_present()
        self.deadline = host.clock() + BUDGET[self.mode]
        before = {function["pci_address"]: read_state(host, function) for function in self.functions}
        unknown = [row for row in before.values() if row["state"] == rule.UNKNOWN]
        if unknown:
            raise HairpinError(PREFIX + message_m16(host.hostname, unknown[0]), "M16")
        needed = [function for function in restart_order(self.functions, self.tunnels)
                  if before[function["pci_address"]]["state"] in rule.RESTART_STATES]
        if needed:
            findings = busy_findings(host, self.functions, self.mode, self.tunnels)
            if findings:
                raise HairpinError(message_m4(host.hostname, findings), "M4")
        for function in needed:
            if host.stopping:
                self.fail(PREFIX + f"stopping on request; {function['netdev']} and later functions were not "
                          "restarted.", "signal")
                break
            remaining = self.deadline - host.clock()
            if remaining < RESERVE[self.mode]:
                self.fail(PREFIX + f"{function['netdev']} and later functions were not restarted: {remaining:.0f} s "
                          f"of the {BUDGET[self.mode]} s budget remain and one restart may need "
                          f"{RESERVE[self.mode]} s. On Node A, sudo sparkring hairpin retries it live.", "budget")
                break
            if not self.restart(function, before[function["pci_address"]]):
                break
        self.offload()
        if self.mode == "live" and self.restarted:
            # Also after a stop request: the restarts removed routes and
            # sysctls, and --no-block requests never delay the stop.
            self._follow_up()
        self.final = {function["pci_address"]: read_state(host, function) for function in self.functions}
        self.in_effect = all(row["state"] == rule.IN_EFFECT for row in self.final.values())
        self.log_functions()
        # A Spark is armed only after a run in which every restart succeeded. A
        # check failure after the last restart leaves every function in effect
        # and still arms it: its causes (addresses, the administration tunnel)
        # do not exist at boot.
        if self.in_effect and self.kind in (None, "check"):
            self._arm()

    def _settle(self):
        result = self.host.command(["udevadm", "settle", f"--timeout={UDEV}"], UDEV + 5)
        if not result.ok:
            self.host.log(PREFIX + "udevadm settle did not finish (" + result.reason() + "); continuing.")

    def _wait_present(self):
        host = self.host
        deadline = host.clock() + PRESENT
        while True:
            states = [(function, *identity_state(host, function)) for function in self.functions]
            if all(state == "present" for _, state, _ in states):
                return
            if host.clock() >= deadline:
                break
            host.sleep(POLL)
        for function, state, found in states:
            if state == "absent":
                raise HairpinError(message_m3(function), "M3")
        for function, state, found in states:
            if state == "changed":
                raise HairpinError(message_m14(function, found), "M14")

    def _journal(self, record, state, kind=None, error=None, cause=None):
        record.update(state=state, error=error, cause=cause, time=self.host.now(), **{"class": kind})
        save_attempts(self.host, self.attempts)

    def restart(self, function, before):
        """Restart one function; return True when its restart succeeded, so the next function may follow.

        When a stop request cuts the check after a successful restart short,
        the run records the stop as its error (class ``signal``).
        """
        host, pci = self.host, function["pci_address"]
        device = "pci/" + pci
        result = self.result(function)
        began = host.clock()
        result["action"] = "set-and-restart" if before["state"] == rule.DEFAULT else "restart"
        record = {"boot_id": self.boot_id, "mode": self.mode, "pci_address": pci, "netdev": function["netdev"],
                  "kernel": host.text("/proc/sys/kernel/osrelease"),
                  "firmware": host.text(f"/sys/class/infiniband/{function['rdma_device']}/fw_ver"),
                  "driver_reinit_before": before["driver_reinit"], "state": "started", "class": None,
                  "error": None, "cause": None, "time": host.now()}
        self.attempts.append(record)
        save_attempts(host, self.attempts)
        tunnel = self.tunnels.get(function["netdev"])
        # A failed child-facing function cuts the Sparks behind it off from Node A.
        cut_off = cut_off_advice(function, head=self.head) if tunnel and tunnel[0] == "child" else ""

        def end(text, state, kind, cause, cut=None):
            result.update(seconds=round(host.clock() - began, 1), error=text, cause=_clause(cause))
            self._journal(record, state, kind, text, _clause(cause))
            self.fail(text, kind, cause=cause, function=function, cut_off=cut)
            self.reported.add(pci)
            return False

        def failed(cause, advice=RETRY):
            return end(message_restart(function, cause, cut_off or advice), "failed", "restart", cause,
                       function["netdev"] if cut_off else None)

        snapshot = self._addresses(function) if self.mode == "live" else None
        changed = []
        for name, wanted in rule.PARAMETERS.items():
            if before["state"] != rule.DEFAULT or before["values"][name] == wanted:
                continue
            outcome = host.command(["devlink", "dev", "param", "set", device, "name", name, "value", str(wanted),
                                    "cmode", "driverinit"], READ)
            if not outcome.ok:
                self._set_back(function, before, changed)
                return failed(f"devlink param set {name} failed: {outcome.reason()}")
            changed.append(name)
        self.restarted.append(pci)
        outcome = host.command(["devlink", "dev", "reload", device, "action", "driver_reinit"], RELOAD)
        if outcome.timed_out:
            return failed(f"driver restart did not finish within {RELOAD} s", RETRY + POWER_CYCLE)
        if not outcome.ok:
            statistics = rule.reload_statistics(_json(host.command(["devlink", "-s", "-j", "dev", "show", device],
                                                                  READ)), device)
            counter = statistics["driver_reinit"]
            if statistics["failed"] is False and counter is not None and counter > before["driver_reinit"]:
                # The kernel finished the restart although devlink reported an
                # error, for example when a signal ended devlink after the
                # reload. The checks below decide.
                host.log(PREFIX + f"{_function_name(function)}: devlink reported {outcome.reason()}, but the driver "
                         f"restart completed (driver restarts since boot {counter})")
            else:
                if statistics["failed"] is False:
                    # The driver refused before promoting the pending values, or
                    # the counter cannot tell; set them back so that param show
                    # never reports values that are not in effect.
                    self._set_back(function, before, changed)
                return failed("driver restart failed: " + outcome.reason())
        if not self._wait(lambda: identity_state(host, function)[0] == "present", RETURN):
            return failed(f"not back within {RETURN} s after its driver restart", RETRY + POWER_CYCLE)
        unchecked = []
        if snapshot is not None:
            back = self._addresses_back(function, snapshot)
            if back is False:
                return end(message_m18(function, "its fabric addresses", ADDRESSES), "check-failed", "check",
                           check_cause("its fabric addresses", ADDRESSES))
            if back is None:
                unchecked.append("its fabric addresses")
        if self.mode == "live" and tunnel:
            problem, checked = self._tunnel_back(function, tunnel)
            if problem:
                return end(message_m18(function, problem, TUNNEL), "check-failed", "check",
                           check_cause(problem, TUNNEL))
            if not checked:
                unchecked.append(self._tunnel_name(tunnel))
        after = read_state(host, function)
        expected = (before["driver_reinit"] or 0) + 1
        if (after["values"] != rule.PARAMETERS or after["driver_reinit"] != expected
                or after["reload_failed"] is not False):
            return failed(f"not in effect after its driver restart ({_values_text(after)}, restart failure flag "
                          f"{after['reload_failed']}; expected driver restarts since boot {expected})")
        result["seconds"] = round(host.clock() - began, 1)
        self._journal(record, "applied")
        self._log_function(function, after)
        if unchecked:
            what = " and ".join(unchecked)
            self.fail(PREFIX + f"stopping on request; {function['netdev']} was restarted, but the stop ended the "
                      f"check of {what}. On Node A, sudo sparkring hairpin completes it.", "signal",
                      cause=f"stopping on request before the check of {what}", function=function)
        return True

    def _set_back(self, function, before, changed):
        device = "pci/" + function["pci_address"]
        for name in changed:
            prior = before["values"][name]
            outcome = self.host.command(["devlink", "dev", "param", "set", device, "name", name, "value", str(prior),
                                         "cmode", "driverinit"], READ)
            if not outcome.ok:
                self.host.error(PREFIX + f"{_function_name(function)}: could not set {name} back to {prior}: "
                                + outcome.reason())

    def _wait(self, condition, seconds, *, interruptible=False):
        """True once condition holds; False at the limit; None when a stop request ends an interruptible wait."""
        deadline = self.host.clock() + seconds
        while True:
            if condition():
                return True
            if interruptible and self.host.stopping:
                return None
            if self.host.clock() >= deadline:
                return False
            self.host.sleep(POLL)

    def _addresses(self, function):
        """What NetworkManager and the RDMA stack must restore after a live restart."""
        host, netdev, rdma = self.host, function["netdev"], function["rdma_device"]
        snapshot = _ipv4(host, netdev)
        result = host.command(["nmcli", "-g", "GENERAL.CON-UUID", "device", "show", netdev], READ)
        snapshot["connection"] = result.stdout.strip() if result.ok and result.stdout.strip() else None
        snapshot["rdma_active"] = "ACTIVE" in (host.text(f"/sys/class/infiniband/{rdma}/ports/1/state") or "")
        snapshot["gid"] = self._gid_maps(function, snapshot["ipv4"])
        return snapshot

    def _gid_maps(self, function, ipv4):
        if not ipv4:
            return False
        raw = self.host.text(f"/sys/class/infiniband/{function['rdma_device']}/ports/1/gids/3")
        try:
            mapped = ipaddress.IPv6Address(raw or "").ipv4_mapped
        except ValueError:
            return False
        return mapped is not None and str(mapped) == ipv4[0].split("/")[0]

    def _addresses_back(self, function, before):
        """True when what ``_addresses`` recorded is back; False at the limit; None when a stop request ends the wait."""
        host = self.host

        def returned():
            now = _ipv4(host, function["netdev"])
            if before["operstate"] == "UP" and now["operstate"] != "UP":
                return False
            if before["ipv4"] and now["ipv4"] != before["ipv4"]:
                return False
            if before["link_local"] and now["link_local"] != before["link_local"]:
                return False
            if before["rdma_active"] and "ACTIVE" not in (
                    host.text(f"/sys/class/infiniband/{function['rdma_device']}/ports/1/state") or ""):
                return False
            return not before["gid"] or self._gid_maps(function, before["ipv4"])

        back = self._wait(returned, ADDRESSES, interruptible=True)
        if back is not False or not before["connection"]:
            return back
        # The activation and the wait after it share ADDRESSES_AFTER_UP, which
        # keeps one live restart within RESERVE["live"].
        began = host.clock()
        outcome = host.command(["nmcli", "--wait", str(ADDRESSES_AFTER_UP), "connection", "up", "uuid",
                                before["connection"]], ADDRESSES_AFTER_UP)
        if not outcome.ok:
            host.log(PREFIX + f"nmcli connection up uuid {before['connection']}: {outcome.reason()}")
        return self._wait(returned, ADDRESSES_AFTER_UP - (host.clock() - began), interruptible=True)

    @staticmethod
    def _tunnel_name(tunnel):
        _, peer, address = tunnel
        return f"the administration tunnel to {address or peer.get('id') or 'its peer'}"

    def _tunnel_back(self, function, tunnel):
        """(problem, checked) for the administration tunnel over this function after its restart.

        The peer's endpoint is set again first, also after a stop request,
        because the restart made its interface index stale. ``problem`` names
        what did not return, or is None; ``checked`` is False when a stop
        request ended the check before the tunnel answered.
        """
        host = self.host
        address = tunnel[2]
        what = self._tunnel_name(tunnel)
        try:
            host.refresh_endpoint(function["netdev"])
        except (ValueError, KeyError, TypeError, OSError, RuntimeError, subprocess.SubprocessError) as error:
            host.error(PREFIX + f"{function['netdev']}: endpoint refresh failed: {error}")
            return (None, False) if host.stopping else (what, True)
        if not address:
            return what, True
        answered = self._wait(lambda: host.command(["ping", "-n", "-c", "1", "-W", "1", "-I", TUNNEL_INTERFACE,
                                                    address], 3).ok, TUNNEL, interruptible=True)
        if answered is None:
            return None, False
        return (None if answered else what), True

    def offload(self):
        """Turn hardware TC offload on where it is off and can be changed; M20 where it is fixed."""
        host = self.host
        for function in self.functions:
            result = host.command(["ethtool", "-k", function["netdev"]], READ)
            value = rule.ethtool_offload(result.stdout) if result.ok else None
            if value == rule.OFFLOAD_DISABLED:
                outcome = host.command(["ethtool", "-K", function["netdev"], "hw-tc-offload", "on"], READ)
                if outcome.ok:
                    entry = self.result(function)
                    entry["offload_enabled"] = True
                    host.log(PREFIX + f"{_function_name(function)}: hardware TC offload turned on")
                else:
                    self.fail(PREFIX + f"cannot turn on hardware TC offload on {function['netdev']}: "
                              + outcome.reason(), "offload", function=function)
            elif value == rule.OFFLOAD_FIXED:
                self.fail(message_m20(function), "M20", cause="hardware TC offload is off and fixed",
                          function=function)

    def _follow_up(self):
        """Restore routes and sysctls, which the restarts removed, and every administration endpoint."""
        host = self.host
        requests = [["systemctl", "--no-block", "try-restart", "sparkring-fabric.service"]]
        if node.location(host.root, CONTROL).exists():
            requests.append(["systemctl", "--no-block", "start", "sparkring-control-refresh.service"])
        for argv in requests:
            outcome = host.command(argv, READ)
            if not outcome.ok:
                host.error(PREFIX + " ".join(argv) + ": " + outcome.reason())

    def _arm(self):
        """Resolve the failures of earlier runs (live runs) and enable the unit for every boot."""
        host = self.host
        if self.mode == "live":
            earlier = [record for record in self.attempts[:self.earlier]
                       if record.get("state") in ("started", "failed")]
            for record in earlier:
                record["state"] = "resolved"
            if earlier:
                save_attempts(host, self.attempts)
        if host.command(["systemctl", "is-enabled", UNIT], READ).stdout.strip() == "enabled":
            self.armed = True
            return
        self.armed = False
        if host.stopping:
            # systemctl enable reloads systemd, which a stop (for example at
            # shutdown) does not wait for.
            self.fail(PREFIX + f"stopping on request; {UNIT} was not enabled, so boots do not apply the setting "
                      "yet. On Node A, sudo sparkring hairpin completes it.", "signal")
            return
        outcome = host.command(["systemctl", "enable", UNIT], 60)
        if not outcome.ok:
            self.fail(PREFIX + f"systemctl enable {UNIT}: {outcome.reason()}; boots do not apply the setting yet. "
                      "On Node A, sudo sparkring hairpin completes it.", "arm")
            return
        host.log(PREFIX + f"{UNIT} enabled; this Spark applies the setting at every boot")
        self.armed = True

    def document(self):
        rows = []
        for function in self.functions:
            entry = self.results.get(function["pci_address"],
                                     {"action": "none", "seconds": 0.0, "error": None, "cause": None})
            rows.append({**_identity(function), **entry, "after": self.final.get(function["pci_address"])})
        return {"schema": STATE_SCHEMA, "boot_id": self.boot_id, "invocation_id": self.invocation_id,
                "mode": self.mode, "started_at": self.started_at, "finished_at": self.host.now(),
                "seconds": round(self.host.clock() - self.started, 1), "in_effect": self.in_effect,
                "armed": self.armed, "suspended": self.suspended, "restarted": self.restarted,
                "stopped_on_request": self.host.stopping, "error": self.error, "class": self.kind,
                "cause": self.cause, "function": self.failed_function, "cut_off": self.cut_off, "functions": rows}

    def _log_function(self, function, row):
        entry = self.result(function)
        action = entry["action"]
        line = PREFIX + f"{_function_name(function)}: " + (f"{action} ({entry['seconds']} s)" if action != "none"
                                                           else "no restart")
        if row:
            line += "; " + _values_text(row) + "; " + row["state"]
        self.host.log(line)
        self.reported.add(function["pci_address"])

    def log_functions(self):
        """One line with its final state for each function that no restart line reported."""
        for function in self.functions:
            if function["pci_address"] not in self.reported:
                self._log_function(function, self.final.get(function["pci_address"]))

    def report(self):
        """The summary line; after a failure it names the first failure and has error priority."""
        host = self.host
        if self.final:
            self.log_functions()
        effective = sum(1 for row in self.final.values() if row["state"] == rule.IN_EFFECT)
        summary = (PREFIX + f"{self.mode or 'unstarted'} run finished in {round(host.clock() - self.started, 1)} s: "
                   f"{effective} of {len(self.functions) or 4} functions in effect, {len(self.restarted)} restarted")
        if self.error:
            host.error(summary + "; first failure: " + (f"{self.failed_function}: " if self.failed_function else "")
                       + self.cause)
        else:
            host.log(summary)


def apply(*, host=None):
    """``sparkring node hairpin apply`` as its unit runs it; returns the exit status.

    0 when every approved function is in effect and no step failed, 2
    otherwise. A check failure on the last function in the order leaves every
    function in effect (the Spark is armed) but still exits 2, so the ring
    procedure reports the missing addresses or tunnel. A stop request exits 2
    when it ended a check or kept the unit from being enabled.
    """
    host = host or Host()
    invocation = host.environ.get("INVOCATION_ID")
    shown = host.command(["systemctl", "show", "-p", "InvocationID", "--value", UNIT], READ)
    if not invocation or not shown.ok or shown.stdout.strip() != invocation:
        host.error(PREFIX + f"apply runs only as {UNIT}, which orders it before networking and the mesh. On Node "
                   "A, sudo sparkring hairpin runs it on every Spark; sudo sparkring node hairpin apply --dry-run "
                   "shows what it would do.")
        return 2
    run = Run(host, invocation)
    try:
        with locked(host):
            try:
                run.execute()
            except HairpinError as stop:
                run.fail(str(stop), stop.kind, cause=stop.cause)
            finally:
                run.report()
                node.save(host.root, STATE, run.document())
    except HairpinError as stop:
        host.error(str(stop))
        return 2
    return 0 if run.in_effect and run.error is None else 2


def apply_unit():
    """Entry point for the unit: SIGTERM finishes the current function and starts no other."""
    host = Host()
    previous = signal.signal(signal.SIGTERM, host.stop)
    try:
        return apply(host=host)
    finally:
        signal.signal(signal.SIGTERM, previous)


def preview(*, boot=False, host=None):
    """``apply --dry-run [--boot]``: the mode, suspension, order and planned actions; changes nothing.

    With ``boot``, every approved function is taken to be at its probe
    default, and every unresolved journal record counts, as at the next boot.
    A boot runs the unit only when it is enabled (``armed``), so without that
    every planned boot action is none.
    """
    host = host or Host()
    approval = load_approval(host)
    functions = _sorted_functions(approval["functions"])
    mode = "boot" if boot else detect_mode(host)
    armed = host.command(["systemctl", "is-enabled", UNIT], READ).stdout.strip() == "enabled"
    records, journal_error = load_attempts(host)
    suspended = None
    if mode == "boot":
        suspended = ({"state": "unreadable", "error": journal_error} if journal_error else
                     suspending_record(records, functions, other_than_boot=None if boot else boot_id(host)))
    tunnels = tunnel_roles(host)
    ordered = restart_order(functions, tunnels)
    rows = []
    for function in ordered:
        if boot:
            row = {**_identity(function), "state": rule.DEFAULT}
        else:
            row = read_state(host, function)
        state = row["state"]
        if boot and not armed:
            action = f"none ({UNIT} is not enabled)"
        elif suspended:
            action = "none (boot restarts suspended)"
        elif state == rule.DEFAULT:
            action = "set-and-restart"
        elif state in rule.RESTART_STATES:
            action = "restart"
        elif state == rule.OFFLOAD_OFF:
            action = "enable-offload" if row.get("offload") == rule.OFFLOAD_DISABLED else "none (hw-tc-offload fixed off)"
        elif state == rule.UNKNOWN:
            action = "none (state unreadable; the run stops)"
        else:
            action = "none"
        tunnel = tunnels.get(function["netdev"])
        rows.append({**row, "tunnel": tunnel[0] if tunnel else None, "action": action})
    document = {"schema": DRY_RUN_SCHEMA, "mode": mode, "armed": armed, "suspended": suspended,
                "order": [row["netdev"] for row in rows], "functions": rows,
                "restart_needed": not suspended and not (boot and not armed)
                and any(r["state"] in rule.RESTART_STATES for r in rows)}
    if mode == "boot":
        document["parameters"] = rule.PARAMETERS
    if mode == "live" and document["restart_needed"]:
        document["busy"] = busy_findings(host, functions, mode, tunnels)
    return document


# approve, revoke, status and require.

def approve(*, host=None):
    """Record this Spark's approval of the setting. Never enables, starts or restarts anything."""
    host = host or Host()
    with locked(host):
        size = fabric_size(host)
        if size is not None and size != 4:
            raise HairpinError(message_m17(size), "M17")
        record = {"schema": APPROVAL_SCHEMA, "node_id": _node_id(host), "parameters": dict(rule.PARAMETERS),
                  "hw_tc_offload": True, "functions": derive_functions(host)}
        existing, error = _read_optional(host, APPROVAL)
        if error:
            raise HairpinError(PREFIX + "cannot read the existing approval " + error
                               + ". Run sudo sparkring node hairpin revoke on this Spark, then repeat.", "M13")
        if existing is not None:
            differences = _differences(existing, record)
            if differences:
                raise HairpinError(message_m13(differences), "M13")
            if existing == record:
                return record
        node.save(host.root, APPROVAL, record, mode=0o644)
        return record


def _differences(existing, record):
    """What distinguishes an existing approval's node and functions from this Spark's."""
    if not isinstance(existing, dict):
        return ["the record is not an object"]
    differences = []
    if existing.get("node_id") != record["node_id"]:
        differences.append(f"node {existing.get('node_id')} instead of {record['node_id']}")
    rows = {f.get("role"): f for f in existing.get("functions") or [] if isinstance(f, dict)}
    for function in record["functions"]:
        other = rows.get(function["role"]) or {}
        changed = [f"{key} {other.get(key)} instead of {function[key]}"
                   for key in ("rdma_device", "pci_address", "netdev", "mac") if other.get(key) != function[key]]
        if changed:
            differences.append(function["role"] + ": " + ", ".join(changed))
    return differences


def revoke(*, host=None):
    """Disable the service and remove the approval and the attempt journal."""
    host = host or Host()
    with locked(host):
        outcome = host.command(["systemctl", "disable", UNIT], 60)
        if not outcome.ok:
            raise HairpinError(PREFIX + f"systemctl disable {UNIT}: {outcome.reason()}")
        removed = []
        for name in (APPROVAL, ATTEMPTS):
            path = node.location(host.root, name)
            if path.exists():
                path.unlink()
                removed.append(name)
    return {"revoked": True, "armed": False, "removed": removed,
            "note": "Values already in effect stay until the next driver probe, for example the next boot."}


def _status_functions(host, approval, fabric, facts):
    """(functions, source) for status: the approval's, the fabric record's, or this Spark's fixed devices."""
    if approval is not None:
        return _sorted_functions(approval["functions"]), "approval"
    interfaces = fabric.get("interfaces") if isinstance(fabric, dict) and fabric.get("size") == 4 else None
    rows = []
    if interfaces:
        for port in interfaces:
            pci = None
            if facts is not None:
                pci = next((row.get("pci_address") for row in facts.get("rdma") or []
                            if row.get("device") == port.get("rdma_device")), None)
            if pci is None:
                pci = pci_address(host, port.get("rdma_device", ""))
            rows.append({"role": port.get("role"), "rdma_device": port.get("rdma_device"), "pci_address": pci,
                         "netdev": port.get("netdev"), "mac": str(port.get("mac") or "").lower()})
        return _sorted_functions(rows), "fabric"
    if facts is not None:
        from runtime.host.topology import DEVICES
        for role in ROLES:
            rdma = next((row for row in facts.get("rdma") or [] if row.get("device") == DEVICES[role]), None)
            if rdma:
                nic = next((row for row in facts.get("interfaces") or [] if row.get("name") == rdma.get("netdev")), {})
                rows.append({"role": role, "rdma_device": DEVICES[role], "pci_address": rdma.get("pci_address"),
                             "netdev": rdma.get("netdev"), "mac": str(nic.get("mac") or "").lower()})
        return rows, "inventory"
    return derive_functions(host), "sysfs"


def _unknown_row(function, reason):
    return {**_identity(function), "values": {name: None for name in rule.PARAMETERS}, "driver_reinit": None,
            "reload_failed": None, "offload": None, "state": rule.UNKNOWN, "error": reason}


def status(*, facts=None, busy=False, root="/", run=None, host=None):
    """The ``sparkring-hairpin-status/v1`` document of this Spark. Never changes anything.

    With ``facts`` (a ``deploy_inventory`` collection), function states come
    from the facts without devlink calls, and the mesh unit start-check warning
    is left to the caller (``node.snapshot`` reports it). ``busy`` adds the
    findings that block a live restart; reading them needs root.
    """
    host = host or Host(root=root, run=run)
    read_errors = []
    approval_document, error = _read_optional(host, APPROVAL)
    approval_info = {"present": approval_document is not None or error is not None, "valid": False, "error": error}
    approval = None
    if approval_document is not None:
        try:
            approval = validate_approval(approval_document, _node_id(host))
            approval_info["valid"] = True
        except (HairpinError, OSError, ValueError, KeyError, TypeError) as failure:
            approval_info["error"] = str(failure)
    fabric, error = _read_optional(host, FABRIC)
    if error:
        read_errors.append(error)
    size = fabric.get("size") if isinstance(fabric, dict) else None
    try:
        functions, source = _status_functions(host, approval, fabric, facts)
    except (ValueError, KeyError, TypeError, AttributeError, OSError) as failure:
        functions, source = [], None
        read_errors.append("ConnectX functions: " + str(failure))
    rows = []
    for function in functions:
        if not function.get("pci_address"):
            rows.append(_unknown_row(function, "PCI address unavailable"))
        elif facts is not None:
            rows.append(facts_state(function, facts))
        else:
            rows.append(read_state(host, function))
    records, error = load_attempts(host)
    if error:
        read_errors.append(error)
    suspended = suspending_record(records, functions) if functions else None
    enabled = host.command(["systemctl", "is-enabled", UNIT], READ)
    armed = enabled.stdout.strip() == "enabled"
    shown = (_show(host, [UNIT], ("ActiveState", "Result", "InvocationID")) or {}).get(UNIT)
    unit = {key: (shown or {}).get(key) for key in ("ActiveState", "Result", "InvocationID")}
    last_run, error = _read_optional(host, STATE)
    if error:
        read_errors.append(error)
    in_effect = len(rows) == 4 and all(row["state"] == rule.IN_EFFECT for row in rows)
    kernel_off = KERNEL_OPTION in (host.text("/proc/cmdline") or "").split()
    warnings = []
    if size == 4 and in_effect and not (approval and armed):
        warnings.append("ConnectX hairpin setting is in effect but not applied at boot; on Node A: sudo sparkring hairpin")
    if suspended:
        warnings.append(f"boot restarts suspended since {suspension_text(suspended)}; on Node A: sudo sparkring "
                        "hairpin retries it live")
    if kernel_off and (size == 4 or approval_info["present"]):
        warnings.append(f"{UNIT} does not run in this boot ({KERNEL_OPTION} on the kernel command line); reboot "
                        "without the option to apply the ConnectX hairpin setting")
    if approval_info["present"] and size is not None and size != 4:
        warnings.append("hairpin approval on a Spark that is not in a four-Spark ring: sudo sparkring node hairpin revoke")
    if size == 4 and facts is None:
        warnings += node.mesh_units_without_start_check(root=host.root, run=host.run)
    try:
        from runtime.common import distribution
        record = distribution.installed(node.ROOT, verify=False)
        revision = record["revision"] if record else None
    except (OSError, ValueError, KeyError, TypeError) as failure:
        revision = None
        read_errors.append("installed distribution: " + str(failure))
    document = {"schema": STATUS_SCHEMA, "hostname": host.hostname, "revision": revision, "approval": approval_info,
                "armed": armed, "boot_disabled_by_kernel_command_line": kernel_off,
                "suspended": suspended, "unit": unit, "last_run": last_run, "function_source": source,
                "functions": rows, "in_effect": in_effect, "blocked_units": host.entries(BLOCKED) or [],
                "warnings": warnings, "read_errors": read_errors, "parameters": dict(rule.PARAMETERS)}
    if busy:
        document["busy"] = busy_findings(host, functions, "live")
    return document


def _block(host, unit, rows):
    directory = node.location(host.root, BLOCKED)
    directory.mkdir(parents=True, exist_ok=True)
    os.chmod(directory, 0o755)
    node.save(host.root, BLOCKED + "/" + unit, {"unit": unit, "boot_id": boot_id(host), "time": host.now(),
                                                "functions": [row.get("netdev") for row in rows]}, mode=0o644)


def require(unit, *, host=None):
    """The mesh start check. Returns 0 to let the unit start, 2 to refuse it.

    Hosts without /etc/sparkring/fabric.json, or whose record is a pair, pass.
    Otherwise every fabric function of the record must have the setting in
    effect by the package's values, whether or not this Spark has an approval.
    A refusal writes /run/sparkring-hairpin-blocked/<unit>.
    """
    host = host or Host()
    if not isinstance(unit, str) or not _UNIT.fullmatch(unit):
        raise HairpinError(PREFIX + "require needs a systemd service unit name")
    document, error = _read_optional(host, FABRIC)
    if document is None and error is None:
        return 0
    rows, reason = [], None
    if error or not isinstance(document, dict):
        reason = "its fabric record cannot be read (" + (error or "not an object") + ")"
    elif document.get("size") == 2:
        _unblock(host, unit)
        return 0
    else:
        for port in document.get("interfaces") or []:
            function = {"role": port.get("role"), "rdma_device": port.get("rdma_device"),
                        "netdev": port.get("netdev"), "mac": str(port.get("mac") or "").lower()}
            function["pci_address"] = pci_address(host, str(function["rdma_device"]))
            if function["pci_address"] is None:
                rows.append(_unknown_row(function, "RDMA device or its PCI address is unavailable"))
            else:
                rows.append(read_state(host, function))
        if len(rows) != 4:
            reason = f"its fabric record lists {len(rows)} functions instead of 4"
    failing = [row for row in rows if row["state"] != rule.IN_EFFECT]
    if not failing and reason is None:
        _unblock(host, unit)
        host.log(PREFIX + f"{unit}: the ConnectX hairpin setting is in effect on all 4 functions")
        return 0
    details = reason or rule.grouped(failing, name=lambda row: f"{row['netdev']} (pci/{row.get('pci_address')})",
                                     describe=lambda row: row.get("error") or rule.shortfall(row))
    _block(host, unit, failing)
    host.error(PREFIX + f"{unit} not started: the ConnectX hairpin setting is not in effect on {details}."
               + _refusal_cause(host))
    return 2


def _unblock(host, unit):
    with contextlib.suppress(OSError, ValueError):
        marker = node.location(host.root, BLOCKED + "/" + unit)
        if marker.exists():
            marker.unlink()


REFUSAL_REMEDY = (" On Node A, sudo sparkring hairpin applies it after asking; this unit then starts by itself if it "
                  "is enabled.")


def _refusal_cause(host):
    """The end of M5: why the setting is not in effect, when known, and the remedy."""
    if KERNEL_OPTION in (host.text("/proc/cmdline") or "").split():
        # The ring procedure cannot run the unit in this boot either.
        return f" This boot was started with {KERNEL_OPTION}, so {UNIT} does not run in it; reboot without it."
    approval, error = _read_optional(host, APPROVAL)
    if approval is None and error is None:
        return " This Spark has no SparkRing approval for the setting." + REFUSAL_REMEDY
    functions = approval.get("functions") if isinstance(approval, dict) else None
    records, _ = load_attempts(host)
    if isinstance(functions, list) and suspending_record(
            records, [f for f in functions if isinstance(f, dict) and "pci_address" in f],
            other_than_boot=boot_id(host)):
        return " Boot restarts are suspended after a failed restart in an earlier boot." + REFUSAL_REMEDY
    last_run, _ = _read_optional(host, STATE)
    shown = (_show(host, [UNIT], ("Result",)) or {}).get(UNIT, {})
    if (isinstance(last_run, dict) and last_run.get("boot_id") == boot_id(host) and not last_run.get("in_effect")) \
            or shown.get("Result") not in (None, "", "success"):
        return " Its boot run failed; see journalctl -b -u sparkring-hairpin.service." + REFUSAL_REMEDY
    return REFUSAL_REMEDY


# resume, start and reboot advice.

def resume(*, host=None):
    """Start the enabled mesh units whose start check refused them in this boot; remove every marker.

    The ring procedure runs this on every Spark only after every Spark's run
    succeeded, so no mesh starts while another Spark restarts functions. A
    unit starts when it is enabled and inactive or failed; its start check
    runs again before it starts. Returns ``{"started": [...], "skipped": [...]}``.
    """
    host = host or Host()
    started, skipped = [], []
    with locked(host):
        for unit in host.entries(BLOCKED) or []:
            if _UNIT.fullmatch(unit) and any(fnmatch.fnmatchcase(unit, pattern) for pattern in MESH_UNITS):
                enabled = host.command(["systemctl", "is-enabled", unit], READ).stdout.strip() == "enabled"
                state = (_show(host, [unit], ("ActiveState",)) or {}).get(unit, {}).get("ActiveState")
                if not enabled or state not in ("inactive", "failed"):
                    skipped.append({"unit": unit, "reason": "not enabled" if not enabled else f"{state or 'unknown'}"})
                else:
                    outcome = host.command(["systemctl", "--no-block", "start", unit], READ)
                    if outcome.ok:
                        started.append(unit)
                        host.log(PREFIX + f"started {unit}, which its start check refused earlier in this boot")
                    else:
                        skipped.append({"unit": unit, "reason": outcome.reason()})
                        host.error(PREFIX + f"systemctl start {unit}: {outcome.reason()}")
            with contextlib.suppress(OSError, ValueError):
                node.location(host.root, BLOCKED + "/" + unit).unlink()
    return {"started": started, "skipped": skipped}


def start(after, *, host=None):
    """Start sparkring-hairpin.service without waiting, unless a run newer than ``after`` exists.

    ``after`` is the unit's InvocationID read before the dispatch, or an empty
    string when the unit has not run in this boot. The unit is restarted only
    while its InvocationID still equals ``after`` and it is not activating,
    so a repeated dispatch never restarts a run in progress. Returns
    ``{"started", "invocation_id", "active_state"}``.
    """
    host = host or Host()
    shown = (_show(host, [UNIT], ("ActiveState", "InvocationID")) or {}).get(UNIT)
    if shown is None:
        raise HairpinError(PREFIX + f"cannot read the state of {UNIT}", "start")
    current, state = shown.get("InvocationID") or "", shown.get("ActiveState")
    if state == "activating" or current != (after or ""):
        return {"started": False, "invocation_id": current, "active_state": state}
    outcome = host.command(["systemctl", "restart", "--no-block", UNIT], READ)
    if not outcome.ok:
        raise HairpinError(PREFIX + f"systemctl restart --no-block {UNIT}: {outcome.reason()}", "start")
    return {"started": True, "invocation_id": current, "active_state": state}


def reboot_advice(*, root="/"):
    """The next action when a restart that failed in this boot cut other Sparks off from Node A, else None.

    That holds when the latest attempt record of a function is ``failed`` in
    this boot and the function carries a child-facing administration tunnel.
    Those Sparks cannot be reached, so sudo sparkring hairpin cannot run until
    this Spark reboots; its next boot restarts no function. Reads files only
    and never raises.
    """
    try:
        host = Host(root=root, hostname="-")
        records, error = load_attempts(host)
        roles = tunnel_roles(host)
        current = boot_id(host)
    except (HairpinError, OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
    if error or not current:
        return None
    latest = {}
    for record in records:
        latest[record.get("pci_address")] = record
    for record in latest.values():
        if (record.get("state") == "failed" and record.get("boot_id") == current
                and (roles.get(record.get("netdev")) or (None,))[0] == "child"):
            if control_head(host):
                return (f"reboot Node A ({record.get('netdev')} failed to restart and cut the workers behind it "
                        "off; the next boot restarts no ConnectX function), then sudo sparkring hairpin")
            return (f"reboot this Spark ({record.get('netdev')} failed to restart and cut the Sparks behind it off; "
                    "the next boot restarts no ConnectX function), then on Node A: sudo sparkring hairpin")
    return None
