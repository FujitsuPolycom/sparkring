"""Read-only survey of the Sparks on this Spark's fabric cables.

The survey reaches each Spark without changing any host, in this order:

1. This Spark.
2. The other Sparks of this Spark's recorded cluster, over the existing
   administration access (root SSH with Node A's key, never a password).
3. Sparks named by what the reached Sparks see on their cables, at their LAN
   address when Node A's LAN has them: an LLDP neighbor's chassis ID is its
   LAN port's MAC, and a Spark's fabric MACs exceed its LAN MAC by one to
   eight (``lan_peers.match``). The LAN reaches a Spark even when its fabric
   ports have no IPv6 address.
4. Other Sparks over the cables' IPv6 link-local addresses, as setup's
   discovery does (``bootstrap.discover``), up to three hops from this Spark.

Steps 3 and 4 sign in as the operator's account; SSH asks for passwords and
for unknown host keys. On each Spark one program (``program``) runs, through
``sudo -n`` where that works and otherwise as the signed-in account: the
discovery inventory (``bootstrap.probe``), every interface MAC, the LLDP
neighbors (``lldpctl`` usually needs root) and, when asked, the SparkRing
state that re-forming a Spark handles. The program writes nothing.
"""
import inspect
import ipaddress
import json
import subprocess

from runtime.host import bootstrap, cabling, discovery, lan_peers

# The most Sparks and sign-in rounds a survey takes; a ring has four.
LIMIT = 8
ROUNDS = 6


def observe(state=False):
    """This Spark's inventory, interface MACs, LLDP document and, with ``state``, its SparkRing state.

    Runs on a Spark through ``python3 -I -c`` beside ``bootstrap.probe`` and,
    with ``state``, beside a ``prior_state()`` function (``program``); it reads only.
    """
    import json
    import os
    import subprocess

    def output(argv):
        done = subprocess.run(argv, capture_output=True, text=True, timeout=30)
        if done.returncode:
            raise RuntimeError((done.stderr or done.stdout).strip() or argv[0] + " failed")
        return done.stdout

    result = {"inventory": probe(), "root": os.geteuid() == 0,  # noqa: F821  (bootstrap.probe, sent beside this)
               "macs": [], "lldp": None, "lldp_error": None}
    try:
        result["macs"] = sorted({str(row["address"]).lower() for row in json.loads(output(["ip", "-j", "link", "show"]))
                                 if row.get("address")})
    except (RuntimeError, OSError, ValueError, KeyError, subprocess.SubprocessError):
        pass
    try:
        result["lldp"] = json.loads(output(["lldpctl", "-f", "json"]))
    except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as error:
        result["lldp_error"] = str(error)[:300]
    if state:
        result["state"] = prior_state()  # noqa: F821  (sent beside this function)
    return result


def program(state=None):
    """The Python source that ``observe`` runs as on a Spark; it prints one JSON document.

    ``state`` is a self-contained ``prior_state`` function whose result the
    document adds, or None.
    """
    parts = [bootstrap.fabric_identity, bootstrap.probe, observe] + ([state] if state else [])
    return ("\n".join(inspect.getsource(part) for part in parts)
            + f"\nimport json\nprint(json.dumps(observe({bool(state)!r})))\n")


def local(argv, data=None):
    """Run a command on this Spark; raise RuntimeError naming its error output when it fails."""
    done = subprocess.run(argv, input=data, capture_output=True, text=True, timeout=600)
    if done.returncode:
        raise RuntimeError((done.stderr or done.stdout).strip() or argv[0] + " failed")
    return done.stdout


class Reach:
    """How the survey reached one Spark, and how later steps run commands there.

    ``kind`` is ``local``, ``route`` (a ``bootstrap.SSH`` route of fabric or
    LAN hops) or ``target`` (an administration-network SSH target).
    """

    def __init__(self, kind, label, *, route=None, target=None):
        self.kind, self.label, self.route, self.target = kind, label, route or [], target

    def run(self, transport, argv, *, data=None, ssh=discovery.ssh):
        if self.kind == "local":
            return local(argv, data)
        if self.kind == "target":
            return ssh(self.target, argv, data=data)
        return transport.command(self.route, argv, data=data)

    def root(self):
        """Whether commands run as root here without sudo."""
        return self.kind != "route" or self.route[-1]["user"] == "root"

    def describe(self):
        return {"kind": self.kind, "label": self.label, "route": self.route, "target": self.target}


def run_program(reach, transport, *, state=None, ssh=discovery.ssh):
    """``observe``'s document from one Spark: through ``sudo -n`` when it works there, else unprivileged."""
    code = program(state)
    try:
        return json.loads(reach.run(transport, ["sudo", "-n", "python3", "-I", "-c", code], ssh=ssh))
    except (RuntimeError, ValueError):
        if reach.root():
            raise
    return json.loads(reach.run(transport, ["python3", "-I", "-c", code], ssh=ssh))


def _functions(data):
    return [f for f in data["inventory"].get("functions") or []]


def known_macs(found):
    """Every function and interface MAC of the Sparks found so far."""
    macs = set()
    for spark in found.values():
        macs |= {str(f.get("mac") or "").lower() for f in _functions(spark["data"])}
        macs |= set(spark["data"].get("macs") or [])
    return macs - {""}


def _hostname(found, key):
    return found[key]["data"]["inventory"].get("hostname") or key


def fabric_hops(found, *, user, port):
    """Link-local SSH routes to cable neighbors that are no found Spark's function, as discovery takes them."""
    known = known_macs(found)
    hops = []
    for key, spark in found.items():
        reach = spark["reach"]
        if reach.kind == "target" or len(reach.route) >= 3:
            continue
        local_functions = {f["netdev"]: f for f in _functions(spark["data"])}
        for neighbor in spark["data"]["inventory"].get("neighbors") or []:
            function = local_functions.get(neighbor.get("dev"))
            mac = str(neighbor.get("lladdr") or "").lower()
            if not function or not function.get("addresses") or not mac or mac in known or neighbor.get("answered") is not True:
                continue
            try:
                address = ipaddress.IPv6Address(str(neighbor.get("dst", "")).split("%")[0])
            except ValueError:
                continue
            if not address.is_link_local or str(address) in function["addresses"]:
                continue
            hop = {"user": user, "address": str(address), "interface": function["netdev"], "port": port}
            hops.append((Reach("route", f"a cable from {_hostname(found, key)} ({function['netdev']})",
                               route=[*reach.route, hop]), mac))
    return hops


def seen_macs(found):
    """(chassis MACs, fabric MACs) of cable neighbors that no found Spark owns, from LLDP and neighbor caches."""
    known = known_macs(found)
    chassis, fabric = set(), set()
    for spark in found.values():
        netdevs = {f["netdev"] for f in _functions(spark["data"])}
        for row in cabling.lldp_rows(spark["data"].get("lldp")):
            port = str(row["port"]).lower()
            if row["netdev"] not in netdevs or port in known or row["chassis"] in known:
                continue
            if cabling.MAC.fullmatch(row["chassis"]):
                chassis.add(row["chassis"])
            if cabling.MAC.fullmatch(port):
                fabric.add(port)
        for neighbor in spark["data"]["inventory"].get("neighbors") or []:
            mac = str(neighbor.get("lladdr") or "").lower()
            if neighbor.get("dev") in netdevs and cabling.MAC.fullmatch(mac) and mac not in known:
                fabric.add(mac)
    return chassis, fabric


def lan_hops(found, head, *, user, arp=lan_peers.arp_table, sweep=lan_peers.sweep):
    """SSH routes to unknown cable neighbors at their address on Node A's LAN, swept once when none is listed."""
    interface = found[head]["data"]["inventory"].get("uplink")
    chassis, fabric = seen_macs(found)
    if not interface or not chassis and not fabric:
        return []

    def candidates(table):
        rows = {mac: (table[mac], mac) for mac in chassis if mac in table}
        for lan_mac, (address, mac) in lan_peers.match(sorted(fabric), table).items():
            rows.setdefault(lan_mac, (address, mac))
        return rows

    table = arp(interface)
    rows = candidates(table)
    if not rows:
        sweep(interface)
        rows = candidates(arp(interface))
    taken = {spark["data"]["inventory"].get("api_address") for spark in found.values()}
    return [(Reach("route", f"the LAN at {address} as {user}",
                   route=[{"user": user, "address": address, "interface": None, "port": 22}]), mac)
            for lan_mac, (address, mac) in sorted(rows.items()) if address not in taken]


def survey(transport, *, recorded=(), user="root", port=22, sign_in=True, state=None, say=print,
           ssh=discovery.ssh, arp=lan_peers.arp_table, sweep=lan_peers.sweep, here=None):
    """Reach and observe the Sparks on this Spark's cables; change nothing.

    ``recorded`` are administration-network SSH targets of the recorded
    cluster's other Sparks. Without ``sign_in`` only this Spark and those are
    read; ``state`` is passed to ``program``. Returns ``{"head", "sparks":
    {id: {"reach", "data"}}, "notes"}``; ``notes`` name each Spark the survey
    could not read and why.
    """
    notes = []
    head_reach = here or Reach("local", "this Spark")
    data = run_program(head_reach, transport, state=state, ssh=ssh)
    head = data["inventory"]["id"]
    found = {head: {"reach": head_reach, "data": data}}

    def add(reach, data):
        ident = data["inventory"]["id"]
        if ident in found:
            return False
        found[ident] = {"reach": reach, "data": data}
        say(f"Read {data['inventory'].get('hostname') or ident} over {reach.label}")
        return True

    for target in recorded:
        reach = Reach("target", "the admin network at " + target.split("@")[-1], target=target)
        try:
            add(reach, run_program(reach, transport, state=state, ssh=ssh))
        except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as error:
            notes.append(f"{target.split('@')[-1]} (recorded cluster) did not answer over the admin network: "
                         + (str(error).strip().splitlines() or ["no answer"])[-1][:200])
    tried = set()
    for _ in range(ROUNDS if sign_in else 0):
        progress = False
        hops = lan_hops(found, head, user=user, arp=arp, sweep=sweep) + fabric_hops(found, user=user, port=port)
        for reach, mac in hops:
            key = json.dumps(reach.route, sort_keys=True)
            if key in tried or mac in known_macs(found) or len(found) >= LIMIT:
                continue
            tried.add(key)
            try:
                transport.login(reach.route)
                progress |= add(reach, run_program(reach, transport, state=state, ssh=ssh))
            except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as error:
                notes.append(f"Sign-in over {reach.label} failed: " + (str(error).strip().splitlines() or ["no answer"])[-1][:200])
        if not progress:
            break
    return {"head": head, "sparks": found, "notes": notes}


def records(result):
    """``cabling`` Spark records of a survey's Sparks."""
    return [cabling.from_probe(spark["data"]["inventory"], spark["data"].get("lldp"), macs=spark["data"].get("macs") or ())
            for spark in result["sparks"].values()]


def checked_lines(result):
    """One line per read Spark: its name, how it was reached and whether its LLDP was read."""
    rows = []
    for spark in result["sparks"].values():
        data = spark["data"]
        name = data["inventory"].get("hostname") or data["inventory"]["id"]
        lldp = "" if data.get("lldp") is not None else (
            " (LLDP not readable without sudo)" if not data.get("root") else " (LLDP not available)")
        rows.append(f"  {name}: {spark['reach'].label}{lldp}")
    return rows
