"""Where a model's API listens on its Spark, and the address SparkRing shows for it.

vLLM serves the API on the deployment's first rank, the API rank: Node A for
a pair, a four-Spark model and a model on Sparks 0 and 1 of a ring, Spark 2
for a model on Sparks 2 and 3 (``runtime.host.placement``). Without options
the API listens on every address of that Spark at the profile's port, and
SparkRing shows it at the address setup recorded (``installer.connection``).

- ``--api-port N`` and ``--api-bind ADDRESS`` are serving settings
  (``runtime/common/serving.py``): part of the deployment's identity, written
  into the API rank's command and health check. Before such a deployment is
  planned, ``check`` reads the API Spark's IPv4 addresses and listening TCP
  ports (``inspect``) and refuses a listen address that is not the Spark's, a
  loopback address that ``--allow-loopback-bind`` did not allow, and a port
  that another program holds there. A port held by a deployment that the
  installation replaces or stops, or by the deployment itself, is free for it.
- ``--api-address ADDRESS`` names the address that status, the dashboard
  link, the "Model ready" card and the printed endpoint show: a DNS name, a
  Tailscale address or another NIC's address. It is shown only: SparkRing's
  own readiness, smoke and recovery checks keep using the listen address, or
  the automatic address, which Node A can reach. It is therefore not part of
  the deployment's identity: each installation records it in the deployment's
  directory (ADDRESS_FILE), and an installation without it records none.
- In a terminal, ``sparkring install`` without ``--yes`` and without these
  options asks for the listen address and the port (``ask``).
"""
import inspect as source_code
import ipaddress
import json
from pathlib import Path
import re
import urllib.parse

from runtime.common import installer, serving
from runtime.host import control, discovery, node
from runtime.host import placement as placements
from runtime.host.install_errors import NeedsInput

# The deployment directory's record of the address that SparkRing shows.
ADDRESS_FILE = "api-address.json"
ADDRESS_SCHEMA = "sparkring-api-address/v1"
# The fabric when the cluster records no fabric CIDR: the benchmarking range
# from which setup takes every fabric's addresses.
FABRIC_DEFAULT = "198.18.0.0/15"
# Listener addresses that hold a port on every address.
WILDCARDS = ("*", "0.0.0.0", "::")
PROBE_TIMEOUT = 60


def _probe():
    """The IPv4 addresses, the listening TCP ports and the administration subnet of this Spark.

    Runs as root on the API Spark: ``ss -p`` names other users' processes only
    for root. A listening process inside a container names the container, its
    name and its ``io.sparkring.deployment`` label.
    """
    import json
    from pathlib import Path
    import re
    import subprocess

    def run(argv):
        return subprocess.run(argv, capture_output=True, text=True, timeout=30, check=True).stdout

    addresses = []
    for row in json.loads(run(["ip", "-j", "-4", "address", "show"])):
        for item in row.get("addr_info", []):
            if item.get("family") == "inet" and item.get("local"):
                addresses.append({"interface": row.get("ifname"), "address": item["local"],
                                  "prefix": item.get("prefixlen"), "state": row.get("operstate")})
    listeners, owners = [], {}
    for line in run(["ss", "-H", "-ltnp"]).splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        host, _, port = fields[3].rpartition(":")
        if not port.isdigit():
            continue
        host = host.strip("[]").split("%")[0]
        if host.startswith("::ffff:") and "." in host:
            host = host[7:]
        processes = []
        for name, pid in re.findall(r'\("([^"]*)",pid=(\d+)', line):
            try:
                found = re.search(r"[0-9a-f]{64}", Path("/proc", pid, "cgroup").read_text())
            except OSError:
                found = None
            container = found.group(0) if found else None
            if container:
                owners[container] = {}
            processes.append({"name": name, "pid": int(pid), "container": container})
        listeners.append({"address": host, "port": int(port), "processes": processes})
    if owners:
        result = subprocess.run(["docker", "--context", "default", "inspect", *owners], capture_output=True, text=True,
                                timeout=30)
        try:
            for info in json.loads(result.stdout or "[]"):
                owners[info["Id"]] = {"name": info.get("Name", "").lstrip("/"),
                                      "deployment": (info.get("Config", {}).get("Labels") or {}).get(
                                          "io.sparkring.deployment")}
        except (ValueError, KeyError, TypeError):
            pass
    for listener in listeners:
        for process in listener["processes"]:
            if process["container"]:
                process["container"] = {"id": process["container"], **owners.get(process["container"], {})}
    try:
        subnet = json.loads(Path("/etc/sparkring/control.json").read_text()).get("subnet")
    except (OSError, ValueError, AttributeError):
        subnet = None
    return {"addresses": addresses, "listeners": listeners, "control_subnet": subnet}


PROBE = source_code.getsource(_probe) + "\nimport json\nprint(json.dumps(_probe()))\n"
PROBE_COMMAND = ["sudo", "-n", "python3", "-I", "-B", "-c", PROBE]


def api_rank(placement):
    """The ring rank of the Spark that serves the API: the placement's first rank, else Node A."""
    return placement[0] if placement else 0


def spark_name(cluster, rank):
    """``Node A``, or ``Spark 2 (spark-c)`` with the host name setup recorded."""
    if rank == 0:
        return "Node A"
    nodes = cluster["plan"].get("nodes") or []
    hostname = nodes[rank].get("hostname") if rank < len(nodes) and isinstance(nodes[rank], dict) else None
    return f"Spark {rank}" + (f" ({hostname})" if hostname else "")


def automatic_address(cluster, placement):
    """The address SparkRing shows and checks for a model without ``--api-bind``, as the deployment's site records it."""
    address = cluster.get("api_address")
    if placement is not None and placement[0] != 0:
        address, _ = placements.api_address(cluster, placement)
    return address or cluster["plan"]["spec"]["hosts"][api_rank(placement)]["management_address"]


def inspect(cluster, placement, *, invoke=None):
    """The ``_probe`` document of the API Spark of ``placement``."""
    invoke = invoke or discovery.ssh
    host = cluster["plan"]["spec"]["hosts"][api_rank(placement)]["host"]
    document = json.loads(invoke(host, list(PROBE_COMMAND), timeout=PROBE_TIMEOUT))
    if not isinstance(document, dict) or not isinstance(document.get("addresses"), list) \
            or not isinstance(document.get("listeners"), list):
        raise ValueError("invalid address and port report")
    return document


def _networks(cluster, document):
    """The fabric and administration networks, whose addresses the API is not offered."""
    found = []
    for text in (cluster["plan"].get("fabric_cidr") or FABRIC_DEFAULT, document.get("control_subnet")):
        try:
            found.append(ipaddress.IPv4Network(text, strict=False))
        except (TypeError, ValueError):
            continue
    return found


def offered(cluster, placement, document):
    """The API Spark's addresses that ``ask`` lists: ``[{"address", "interface"}]``.

    Left out are loopback and link-local addresses, addresses on an interface
    without a link (state DOWN), on the administration interface
    (``control.INTERFACE``) or on a fabric function, and addresses in the
    cluster's fabric CIDR (FABRIC_DEFAULT when it records none) or in the
    administration subnet of the Spark's ``/etc/sparkring/control.json``.
    """
    host = cluster["plan"]["spec"]["hosts"][api_rank(placement)]
    fabric = {port.get("netdev") for port in host.get("data_interfaces") or []}
    networks = _networks(cluster, document)
    rows = []
    for row in document["addresses"]:
        try:
            value = ipaddress.IPv4Address(row["address"])
        except (KeyError, TypeError, ValueError):
            continue
        if (value.is_loopback or value.is_link_local or row.get("state") == "DOWN"
                or row.get("interface") in fabric | {control.INTERFACE} or any(value in net for net in networks)):
            continue
        entry = {"address": str(value), "interface": row.get("interface")}
        if entry not in rows:
            rows.append(entry)
    return rows


def addresses_text(rows):
    """``198.51.100.10 (enP7s7), 198.51.100.11 (wlP9s9)``."""
    return ", ".join(f"{row['address']} ({row['interface']})" if row.get("interface") else row["address"]
                     for row in rows)


def _ipv4(text):
    try:
        ipaddress.IPv4Address(text)
        return True
    except ValueError:
        return False


def holders(document, address, port):
    """The listeners of ``document`` that keep an API from listening at ``address``:``port``.

    A listener on every address holds the port on every address; an API on
    every address (0.0.0.0) conflicts with any IPv4 listener on the port.
    """
    return [listener for listener in document["listeners"] if listener.get("port") == port and (
        listener.get("address") in WILDCARDS or listener.get("address") == address
        or (address == "0.0.0.0" and _ipv4(listener.get("address", ""))))]


def owners(directories):
    """``(deployment IDs, container names)`` of the deployments in ``directories`` whose locks can be read."""
    ids, names = set(), set()
    for directory in directories:
        try:
            lock = installer.read(Path(directory) / "deployment.lock.json")
            ids.add(lock["id"])
            names.update(row["name"] for row in installer.containers(lock))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return ids, names


def _owned(listener, ids, names):
    """Whether every process of ``listener`` runs in a container of one of the deployments ``ids`` or ``names``."""
    processes = listener.get("processes") or []
    return bool(processes) and all(
        (process.get("container") or {}).get("deployment") in ids or (process.get("container") or {}).get("name") in names
        for process in processes)


def _holder_text(listener):
    parts = []
    for process in listener.get("processes") or []:
        container = (process.get("container") or {}).get("name")
        parts.append(f"{process.get('name')} (pid {process.get('pid')}" + (f", container {container})" if container
                                                                           else ")"))
    return ", ".join(parts) or "another program"


def check(cluster, placement, settings, document, *, port, allowed=(), created=False, allow_loopback=False):
    """Refuse an API endpoint that the API Spark cannot serve; returns nothing.

    ``settings`` are the deployment's serving settings, ``port`` the API port
    they give, and ``document`` the API Spark's ``inspect`` report.
    ``allowed`` names the deployment directories whose model may hold the
    port: the deployment itself and those the installation replaces or
    stops. A loopback listen address needs ``allow_loopback`` when the
    installation ``created`` the deployment; Node A's checks cannot reach a
    loopback address of Spark 2, so a model on Sparks 2 and 3 never takes one.
    """
    rank = api_rank(placement)
    name = spark_name(cluster, rank)
    bind = settings.get("api_bind")
    if bind:
        if bind not in {row.get("address") for row in document["addresses"]}:
            listed = offered(cluster, placement, document)
            raise NeedsInput(f"--api-bind {bind} is not an address of {name}, which serves the model's API"
                             + (f". Its addresses: {addresses_text(listed)}" if listed else "")
                             + ". Nothing has been changed.", field="api_endpoint")
        if ipaddress.IPv4Address(bind).is_loopback:
            if rank != 0:
                raise NeedsInput(f"--api-bind {bind} is a loopback address of {name}; Node A, which checks the model "
                                 "through its API, could not reach it. Choose another address. Nothing has been "
                                 "changed.", field="api_endpoint")
            if created and not allow_loopback:
                raise NeedsInput(f"--api-bind {bind} is a loopback address: only programs on Node A could use the "
                                 "model. Add --allow-loopback-bind to serve it that way. Nothing has been changed.",
                                 field="api_endpoint")
    ids, names = owners(allowed)
    for listener in holders(document, bind or "0.0.0.0", port):
        if not _owned(listener, ids, names):
            raise NeedsInput(f"TCP port {port} on {name} is in use by {_holder_text(listener)} at "
                             f"{listener.get('address')}:{port}. Choose another port with --api-port, or stop that "
                             "program. Nothing has been changed.", field="api_endpoint")


def ask(cluster, placement, document, port, *, read=None, write=print):
    """The API endpoint settings chosen in a terminal: ``api_bind`` and ``api_port``, each only when changed.

    Lists the automatic choice first, then each ``offered`` address of the
    API Spark; Enter keeps the automatic address and the profile's port.
    ``read`` asks one question (default: ``input``).
    """
    read = read or input
    rank = api_rank(placement)
    rows = offered(cluster, placement, document)
    write("Model API (Enter keeps the first choice):")
    write(f"  1. Every address of {spark_name(cluster, rank)}, shown as "
          f"http://{automatic_address(cluster, placement)}:{port}/v1")
    for number, row in enumerate(rows, 2):
        write(f"  {number}. Only {row['address']}" + (f" ({row['interface']})" if row.get("interface") else ""))
    chosen = {}
    answer = read("API address number [1]: ").strip() if rows else ""
    if answer:
        if not answer.isdigit() or not 1 <= int(answer) <= len(rows) + 1:
            raise NeedsInput("Choose one of the listed API addresses. Nothing has been changed.", field="api_endpoint")
        if int(answer) > 1:
            chosen["api_bind"] = rows[int(answer) - 2]["address"]
    answer = read(f"API port [{port}]: ").strip()
    if answer and answer != str(port):
        value = int(answer) if answer.isdigit() else answer
        try:
            serving.normalized({"api_port": value})
        except ValueError as error:
            raise NeedsInput(f"{error}. Nothing has been changed.", field="api_endpoint") from None
        chosen["api_port"] = value
    return chosen


# Display address -------------------------------------------------------------

def shown_address(value):
    """``--api-address`` checked: a DNS name or an IP address, without a scheme, port or path; returns it.

    An IPv6 address is returned without brackets; ``url`` adds them.
    """
    text = str(value or "").strip()
    try:
        return str(ipaddress.ip_address(text.strip("[]")))
    except ValueError:
        pass
    labels = text.rstrip(".").split(".")
    if (not text or len(text) > 253
            or not all(re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", label) for label in labels)):
        raise ValueError(f"--api-address takes a host name or an IP address, such as llm.example.net or 192.0.2.10, "
                         f"without http:// or a port: {value}")
    return text


def url(api_url, address):
    """``api_url`` with ``address`` as its host; its port and path stay."""
    parts = urllib.parse.urlsplit(api_url)
    host = f"[{address}]" if ":" in address else address
    return urllib.parse.urlunsplit((parts.scheme, f"{host}:{parts.port}", parts.path, parts.query, parts.fragment))


def recorded(directory):
    """The address recorded for display in the deployment ``directory``, or None."""
    try:
        value = installer.read(Path(directory) / ADDRESS_FILE)
    except (OSError, ValueError):
        return None
    return value.get("address") if isinstance(value, dict) and value.get("schema") == ADDRESS_SCHEMA else None


def record(directory, address):
    """Record ``address`` as the deployment's display address; None removes the record."""
    path = Path(directory) / ADDRESS_FILE
    if address is None:
        path.unlink(missing_ok=True)
        return
    node.save(directory, ADDRESS_FILE, {"schema": ADDRESS_SCHEMA, "address": address}, mode=0o600)


def present(document, address):
    """``document`` with ``api_url`` showing ``address``, and ``check_url`` naming the URL SparkRing's checks use.

    ``document`` holds ``api_url`` (``installer.connection``). Without an
    address, or when it is already the URL's host, ``document`` is returned
    unchanged.
    """
    if not address or not document.get("api_url"):
        return document
    shown = url(document["api_url"], address)
    if shown == document["api_url"]:
        return document
    return {**document, "api_url": shown, "check_url": document["api_url"]}
