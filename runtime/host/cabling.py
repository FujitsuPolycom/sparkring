"""Diagnose the fabric cabling of a SparkRing pair or four-Spark ring from observations.

Each Spark has two ConnectX ports (QSFP cages), port 0 and port 1. Each port
appears as two network functions that share its cable (Socket Direct): port 0
is ``enp1s0f0np0`` and ``enP2p1s0f0np0``, port 1 is ``enp1s0f1np1`` and
``enP2p1s0f1np1``. SparkRing serves two layouts:

- A pair: two Sparks with a cable between their ports 0. Pair profiles use
  both functions of port 0 on both Sparks. A second cable between the two
  ports 1 is allowed; only the administration network's fallback path uses it.
- A four-Spark ring: one loop through four Sparks in which every cable joins
  port 0 of one Spark to port 1 of the next. Node A is rank 0, and rank r+1 is
  the Spark on rank r's port 0. The transport code in the serving images
  assumes this direction, so SparkRing never reassigns port roles to accept
  other cabling: the diagnosis names the physical change instead.

The diagnosis only reads observations. An observation is what one function of
a Spark sees across its cable: an LLDP neighbor, named by its chassis name and
port MAC or interface name, or an IPv6 neighbor-cache entry that answered an
echo and whose link-layer address belongs to another Spark's function. Both
functions of a port see the far port's two functions; each function also sees
its sibling on the same port, which is ignored. A Spark that sees its own
other port is cabled to itself.

Inputs are Spark records (``from_capture``, ``from_probe``, ``from_inspection``):

    {"key": unique identity, "name": hostname, "reached": bool,
     "functions": [{"netdev", "port", "mac", "carrier"}],
     "macs": [every interface MAC of the Spark],
     "lldp": [lldp_rows(...)] or None when LLDP was not read,
     "neighbors": [ip -6 neigh rows with "answered"] or None}

``diagnose`` returns a ``sparkring-cabling/v1`` document; ``lines`` and
``message`` render it.
"""
import re

SCHEMA = "sparkring-cabling/v1"
MAC = re.compile(r"(?:[0-9a-f]{2}:){5}[0-9a-f]{2}")
# The interface names of a Spark's ConnectX fabric functions.
FABRIC_NETDEV = re.compile(r"enP?\d*p\d+s\d+f[01]np[01]")
RULE = "In a ring, every cable runs from port 0 of one Spark to port 1 of the next."
PAIR_RULE = "Pair models use port 0 on both Sparks."
CHECK = "sparkring cabling shows the cables."
# The phrase that marks observations which can still arrive: LLDP announces a
# neighbor up to 30 seconds after its link comes up, as after a driver
# restart. hairpin_ring re-inspects while an error carries it.
MISSING = "Missing reciprocal LLDP cable evidence"


class CablingError(ValueError):
    """Setup stops because the cables do not form the layout it needs.

    ``result`` is the diagnosis. The message is one line (finding, fix,
    order); ``details`` carries the cables, problems and notes, which the
    setup command prints below it.
    """

    def __init__(self, result):
        super().__init__(message(result))
        self.result = result
        self.details = {"lines": lines(result, finding=False)}


def port_of(name):
    """The ConnectX port (0 or 1) of a fabric function's interface or RDMA device name, or None.

    ``enp1s0f1np1`` and ``rocep1s0f1`` are port 1: the PCI function number is the port.
    """
    match = re.search(r"f([01])(?:np[01])?$", str(name or ""))
    return int(match[1]) if match else None


def _mac(value):
    value = str(value or "").lower()
    return value if MAC.fullmatch(value) else None


def lldp_rows(document):
    """One row per LLDP neighbor in an ``lldpctl -f json`` document.

    Each row is ``{"netdev", "hostname", "chassis", "port", "port_type",
    "port_descr"}``: the local interface, the neighbor's chassis name and
    chassis ID, its port ID (a MAC or an interface name, per ``port_type``)
    and its port description, which Linux lldpd sets to the interface name.
    """
    interfaces = (document or {}).get("lldp", {}).get("interface", [])
    if isinstance(interfaces, dict):
        interfaces = [{name: value} for name, value in interfaces.items()]
    rows = []
    for entry in interfaces:
        for netdev, neighbors in entry.items():
            for neighbor in neighbors if isinstance(neighbors, list) else [neighbors]:
                chassis = neighbor.get("chassis", {})
                entries = [("", chassis)] if "id" in chassis else chassis.items()
                for name, details in entries:
                    if not isinstance(details, dict):
                        continue
                    port = neighbor.get("port", {}).get("id", {})
                    rows.append({"netdev": netdev, "hostname": details.get("name", name),
                                 "chassis": str(details.get("id", {}).get("value") or "").lower(),
                                 "port": str(port.get("value", "")), "port_type": port.get("type"),
                                 "port_descr": neighbor.get("port", {}).get("descr")})
    return rows


def from_capture(hostname, lldp, addresses, *, key=None):
    """A Spark record from raw captures: ``lldpctl -f json`` (None when unread) and ``ip -j address show``."""
    functions, macs = [], []
    for row in addresses:
        mac = _mac(row.get("address"))
        if mac:
            macs.append(mac)
        name = str(row.get("ifname") or "")
        if FABRIC_NETDEV.fullmatch(name):
            flags = row.get("flags") or []
            carrier = False if "NO-CARRIER" in flags else True if row.get("operstate") == "UP" else None
            functions.append({"netdev": name, "port": port_of(name), "mac": mac, "carrier": carrier})
    return {"key": key or hostname, "name": hostname, "reached": True, "functions": functions, "macs": sorted(set(macs)),
            "lldp": None if lldp is None else lldp_rows(lldp), "neighbors": None}


def from_probe(inventory, lldp=None, *, macs=()):
    """A Spark record from ``bootstrap.probe``'s inventory and, when it was read, its LLDP document.

    ``macs`` adds interface MACs the probe does not list, such as the LAN port's.
    """
    functions = []
    for function in inventory.get("functions") or []:
        port = port_of(function.get("device"))
        port = port_of(function.get("netdev")) if port is None else port
        if port is not None:
            functions.append({"netdev": function["netdev"], "port": port, "mac": _mac(function.get("mac")),
                              "carrier": function.get("carrier")})
    known = {f["mac"] for f in functions if f["mac"]} | {m for m in map(_mac, macs) if m}
    return {"key": inventory.get("id") or inventory.get("hostname"),
            "name": inventory.get("hostname") or inventory.get("id"), "reached": True, "functions": functions,
            "macs": sorted(known), "lldp": None if lldp is None else lldp_rows(lldp),
            "neighbors": inventory.get("neighbors")}


def from_inspection(node, ports):
    """A Spark record from a ``sparkring node inspect`` document and its endpoints (``topology.endpoints``)."""
    interfaces = {row.get("name"): row for row in node["facts"].get("interfaces") or []}
    functions = []
    for endpoint in ports.values():
        state = interfaces.get(endpoint["netdev"], {}).get("operstate")
        functions.append({"netdev": endpoint["netdev"], "port": endpoint["port"], "mac": endpoint["mac"],
                          "carrier": {"UP": True, "DOWN": False, "LOWERLAYERDOWN": False}.get(state)})
    macs = sorted({m for m in (_mac(row.get("mac")) for row in interfaces.values()) if m})
    return {"key": node["node_id"], "name": node["hostname"], "reached": True, "functions": functions,
            "macs": macs, "lldp": lldp_rows(node.get("lldp")), "neighbors": None}


def _names(spark):
    name = str(spark.get("name") or "").rstrip(".")
    return {name, name + ".local"} - {"", ".local"}


def _observe(sparks, *, neighbors):
    """Port-level observations of every reached Spark.

    Each is ``{"spark", "port", "peer", "peer_port", "source", "peer_name",
    "ambiguous"}``. ``peer`` is another record's key, or ``("unknown", ID)``
    for an LLDP neighbor that matches no record; ``peer_port`` is None when
    that neighbor's port cannot be told. Sibling observations are dropped.
    """
    owners = {}
    for spark in sparks:
        for function in spark["functions"]:
            if function["mac"]:
                owners.setdefault(function["mac"], []).append((spark["key"], function))
    result = []
    for spark in sparks:
        if not spark.get("reached", True):
            continue
        local = {f["netdev"]: f for f in spark["functions"]}
        own_functions = {f["mac"]: f for f in spark["functions"] if f["mac"]}
        own = set(own_functions) | set(spark.get("macs") or [])

        def add(function, peer, peer_port, source, peer_name=None, ambiguous=None):
            result.append({"spark": spark["key"], "port": function["port"], "peer": peer, "peer_port": peer_port,
                           "source": source, "peer_name": peer_name, "ambiguous": ambiguous})

        for row in spark.get("lldp") or []:
            function = local.get(row["netdev"])
            if function is None:
                continue
            port_mac = _mac(row["port"])
            named = str(row["hostname"] or "").rstrip(".")
            labels = {row["port"], row.get("port_descr")} - {None, ""}
            if port_mac in own_functions or row["chassis"] in own or named in _names(spark):
                target = own_functions.get(port_mac) or next((local[n] for n in sorted(labels) if n in local), None)
                if target is not None and target["port"] != function["port"]:
                    add(function, spark["key"], target["port"], "lldp")
                continue
            matches = set()
            for other in sparks:
                if other["key"] == spark["key"]:
                    continue
                identified = named in _names(other) or row["chassis"] in set(other.get("macs") or [])
                for candidate in other["functions"]:
                    if port_mac and port_mac == candidate["mac"] or identified and candidate["netdev"] in labels:
                        matches.add((other["key"], candidate["port"]))
            if len(matches) == 1:
                peer, peer_port = matches.pop()
                add(function, peer, peer_port, "lldp")
            elif matches:
                add(function, None, None, "lldp", named, sorted(m[0] for m in matches))
            else:
                guess = next((port_of(n) for n in sorted(labels) if port_of(n) is not None), None)
                add(function, ("unknown", row["chassis"] or named), guess, "lldp", named or row["chassis"])
        for neighbor in (spark.get("neighbors") or []) if neighbors else []:
            function = local.get(neighbor.get("dev"))
            mac = _mac(neighbor.get("lladdr"))
            if function is None or mac is None or neighbor.get("answered") is not True:
                continue
            if mac in own_functions:
                if own_functions[mac]["port"] != function["port"]:
                    add(function, spark["key"], own_functions[mac]["port"], "neighbor")
                continue
            found = {(key, candidate["port"]) for key, candidate in owners.get(mac, []) if key != spark["key"]}
            if len(found) == 1:
                peer, peer_port = found.pop()
                add(function, peer, peer_port, "neighbor")
    return result


def _ring_fix(order, ports, head):
    """Sparks whose two cables to swap so a loop follows the port rule, and the ring order afterwards.

    ``order`` walks the loop; ``ports[i]`` is ``(port at order[i], port at
    order[i+1])`` of the cable between them. Swapping a Spark's two cables
    flips the port at its end of both of its cables, so a cable becomes or
    stays mixed (port 0 to port 1) exactly when its two Sparks' swap choices
    differ by whether it joins equal ports. Each Spark has one port 0 and one
    port 1, so the loop has an even number of same-port cables, the choices
    close around the loop, and exactly two complementary swap sets exist: the
    smaller one wins, and on a tie the one that leaves Node A alone.
    """
    count = len(order)
    swaps = [0]
    for a, b in ports[:-1]:
        swaps.append(swaps[-1] ^ int(a == b))
    if swaps[-1] ^ int(ports[-1][0] == ports[-1][1]) != swaps[0]:
        raise AssertionError("a loop of Sparks with one port 0 and one port 1 each has an even number of same-port cables")
    other = [1 - x for x in swaps]
    chosen = min((swaps, other), key=lambda s: (sum(s), s[order.index(head)]))
    beyond = {}
    for i, (a, b) in enumerate(ports):
        j = (i + 1) % count
        beyond[(order[i], a ^ chosen[i])] = order[j]
        beyond[(order[j], b ^ chosen[j])] = order[i]
    ring = [head]
    while len(ring) < count:
        ring.append(beyond[(ring[-1], 0)])
    return sorted((order[i] for i in range(count) if chosen[i]), key=ring.index), ring


class _Diagnosis:
    """One diagnosis run; ``diagnose`` documents the inputs and the result."""

    def __init__(self, sparks, head, strict, whole, neighbors):
        self.sparks = {s["key"]: s for s in sparks}
        if len(self.sparks) != len(sparks):
            raise ValueError("Spark records must have distinct keys")
        self.names = {k: s.get("name") or str(k) for k, s in self.sparks.items()}
        self.reached = {k for k, s in self.sparks.items() if s.get("reached", True)}
        self.strict, self.whole = strict, whole
        self.head = head if head in self.sparks else min(self.reached, key=self.names.get, default=None)
        self.notes, self.problems = [], []
        self.observations = _observe(sparks, neighbors=neighbors and not strict)
        for row in self.observations:
            if isinstance(row["peer"], tuple):
                self.names.setdefault(row["peer"], row["peer_name"] or row["peer"][1])

    def end(self, end):
        return f"{self.names[end[0]]} port {'?' if end[1] is None else end[1]}"

    def cable(self, cable):
        return f"{self.end(cable[0])} ↔ {self.end(cable[1])}"

    def problem(self, text, ends=()):
        """Record an observation problem; ``ends`` are the (Spark, port) ends it concerns."""
        self.problems.append({"text": text, "ends": set(ends)})

    def merge(self):
        """Port-level cables from the observations: {cable: set of ends that saw it}."""
        seen = {}
        for row in self.observations:
            if row["ambiguous"]:
                self.problem(f"LLDP on {self.end((row['spark'], row['port']))} names a port that matches more than one "
                             "Spark: " + ", ".join(self.names[k] for k in row["ambiguous"]), [(row["spark"], row["port"])])
                continue
            seen.setdefault((row["spark"], row["port"]), {}).setdefault((row["peer"], row["peer_port"]), set()).add(row["source"])
        for ends in seen.values():
            # A neighbor whose port could not be told merges into a known port of the same neighbor.
            for vague in [e for e in ends if e[1] is None]:
                known = [e for e in ends if e[0] == vague[0] and e[1] is not None]
                if known:
                    ends[known[0]] |= ends.pop(vague)

        def order(end):
            return (self.names[end[0]], str(end[1]))
        conflicts = set()
        for local, ends in sorted(seen.items(), key=lambda item: order(item[0])):
            if len(ends) > 1:
                conflicts.add(local)
                self.problem(f"{self.end(local)} sees more than one far port: "
                             + ", ".join(self.end(e) for e in sorted(ends, key=order)), [local])
        cables = {}
        for local, ends in sorted(seen.items(), key=lambda item: order(item[0])):
            if local in conflicts:
                continue
            far = next(iter(ends))
            back = seen.get(far)
            if back is not None and far not in conflicts and local not in back:
                conflicts.update((local, far))
                self.problem(f"{self.end(local)} sees {self.end(far)}, but {self.end(far)} sees "
                             f"{self.end(next(iter(back)))}", [local, far])
                continue
            cables.setdefault(tuple(sorted((local, far), key=order)), set()).add(local)
        self.seen = seen
        return cables

    def port_state(self, key, port, port_cable):
        """``cabled``, ``no link`` (every function of the port reports no carrier) or ``unseen``."""
        if (key, port) in port_cable:
            return "cabled"
        carriers = [f.get("carrier") for f in self.sparks.get(key, {}).get("functions", []) if f["port"] == port]
        if key in self.reached and carriers and all(c is False for c in carriers):
            return "no link"
        return "unseen"

    def run(self):
        names, head = self.names, self.head
        cables = self.merge()
        unknown = {c for c in cables if any(isinstance(e[0], tuple) for e in c)}
        graph = {k: set() for k in names}
        for cable in cables:
            if self.strict and cable in unknown:
                continue
            a, b = cable[0][0], cable[1][0]
            if a != b:
                graph[a].add(b)
                graph[b].add(a)
        group = [head] if head is not None else []
        for key in group:
            group.extend(sorted(graph[key] - set(group), key=names.get))
        members = set(group)
        inner = {c: e for c, e in cables.items() if c not in unknown or not self.strict}
        inner = {c: e for c, e in inner.items() if c[0][0] in members and c[1][0] in members}
        port_cable = {end: cable for cable in inner for end in cable}
        # With a fixed set of two Sparks, the pair rules apply even before a cable joins them.
        pair = len(members) == 2 or self.whole and len(self.reached) == 2
        expected = (0,) if pair else (0, 1)
        for cable, ends in cables.items():
            if len(ends) == 2:
                continue
            local = next(iter(ends))
            far = cable[1] if cable[0] == local else cable[0]
            if far in self.seen:
                # The far end saw something else; merge() recorded that conflict.
                continue
            why = (f"{names[far[0]]} was not reached" if far[0] not in self.reached
                   else f"{names[far[0]]} reports nothing on port {far[1]}")
            text = f"{self.cable(cable)} was seen only from {names[local[0]]} ({why})"
            if self.strict and cable not in unknown:
                self.problem(f"{MISSING}: {text}; wait a minute for LLDP, or check that cable", cable)
            elif not self.strict:
                self.notes.append(text)
        if self.strict:
            for cable in sorted(unknown, key=str):
                ours = [e for e in cable if not isinstance(e[0], tuple)]
                self.problem(f"{self.end(ours[0])} is cabled to {names[next(e[0] for e in cable if e not in ours)]}, "
                             "which is not one of the Sparks setup signed in to", ours)
            for key in sorted(self.reached, key=names.get):
                if self.sparks[key].get("lldp") is None:
                    self.problem(f"LLDP of {names[key]} was not read", [(key, 0), (key, 1)])
                for port in expected:
                    # A port without carrier has no cable; the layout below names that.
                    if (key, port) not in self.seen and self.port_state(key, port, port_cable) == "unseen":
                        self.problem(f"{MISSING}: {names[key]} reports no LLDP neighbor on port {port} yet, though its "
                                     "link is up; wait a minute for LLDP, or check that cable", [(key, port)])
        if pair and self.strict:
            # Pair profiles never use port 1, so a finding about port 1 alone does not block a pair.
            for problem in list(self.problems):
                if problem["ends"] and all(end[1] == 1 for end in problem["ends"]):
                    self.problems.remove(problem)
                    self.notes.append(problem["text"])
        outside = sorted((k for k in self.reached if k not in members), key=names.get)
        if not self.whole:
            self.notes += [f"{names[k]} was reached but is not cabled to {names[head]}'s Sparks" for k in outside]
        self.result = {"schema": SCHEMA, "head": self.key(head), "layout": "unsupported", "ready": False,
                       "summary": None, "fix": [], "order": None, "order_names": None, "walk": [],
                       "cables": [{"ends": [{"spark": names[e[0]], "port": e[1]} for e in c],
                                   "seen_from": sorted(names[e[0]] for e in ends)}
                                  for c, ends in sorted(cables.items(), key=lambda item: self.cable(item[0]))],
                       "sparks": [{"name": names[k], "key": self.key(k), "reached": k in self.reached,
                                   "lldp": self.sparks.get(k, {}).get("lldp") is not None,
                                   "ports": {str(p): self.port_state(k, p, port_cable) for p in (0, 1)}}
                                  for k in sorted(members | self.reached, key=names.get)],
                       "notes": self.notes, "problems": [p["text"] for p in self.problems]}
        self.result["walk"] = [self.cable(c) for c in sorted(inner, key=self.cable)]
        return self.classify(group, members, inner, port_cable, graph, outside)

    @staticmethod
    def key(key):
        return None if isinstance(key, tuple) else key

    def finish(self, layout, summary, fix=(), order=None, ready=False):
        """Complete the result. Observation problems replace a finding that needs no physical change."""
        problems = self.result["problems"]
        if problems and not fix:
            summary = "SparkRing cannot confirm the cabling: " + problems[0].rstrip(".") + "."
            if layout in ("pair", "ring"):
                layout = "incomplete" if any(MISSING in p for p in problems) else "unsupported"
        self.result.update(layout=layout, summary=summary, fix=list(fix), ready=ready and not problems,
                           order=[self.key(k) for k in order] if order else None,
                           order_names=[self.names[k] for k in order] if order else None)
        return self.result

    def classify(self, group, members, inner, port_cable, graph, outside):
        names, head = self.names, self.head
        if head is None:
            return self.finish("incomplete", "No Spark was observed.")

        def state(key, port):
            return self.port_state(key, port, port_cable)

        def free(keys):
            return [self.end((k, p)) for k in keys for p in (0, 1) if state(k, p) != "cabled"]

        if len(port_cable) != 2 * len(inner):
            # One port end in two cables: observations disagree, and merge() recorded how.
            return self.finish("unsupported", "The cable observations disagree.")
        loops = sorted(c for c in inner if c[0][0] == c[1][0])
        if loops:
            return self.finish("unsupported", "; ".join(f"{names[c[0][0]]} is cabled to itself (port {c[0][1]} to "
                                                        f"port {c[1][1]})" for c in loops)
                               + ". Each cable must join two different Sparks.")
        if self.whole and outside:
            joined = {}
            for c in inner:
                joined.setdefault(frozenset((c[0][0], c[1][0])), []).append(c)
            double = next((k for k, v in joined.items() if len(v) == 2), None)
            others = " and ".join(names[k] for k in outside)
            if double is not None:
                a, b = sorted(double, key=names.get)
                return self.finish("unsupported", f"{names[a]} and {names[b]} are joined by two cables, so they "
                                   f"cannot also reach {others}. For a ring, each Spark needs one cable to each of two "
                                   f"different Sparks. {RULE}")
            return self.finish("unsupported", f"{others} {'is' if len(outside) == 1 else 'are'} not cabled to "
                               f"{names[head]}'s Sparks; setup needs one group of two or four cabled Sparks.")
        size = len(group)
        if size == 1:
            ports = ", ".join(f"port {p}: {state(head, p)}" for p in (0, 1))
            return self.finish("incomplete" if "unseen" in ports else "unsupported",
                               f"No cable from {names[head]} to another Spark was found ({ports}).")
        if size == 2:
            other = group[1]
            between = sorted({(c[0][1], c[1][1]) if c[0][0] == head else (c[1][1], c[0][1]) for c in inner})
            used = ", ".join(self.result["walk"])
            if (0, 0) in between:
                if len(between) == 2:
                    self.notes.append("The cable between the ports 1 carries only the admin network's fallback path.")
                return self.finish("pair", f"Pair: {used}. {PAIR_RULE}", order=[head, other], ready=True)
            if len(between) == 2:
                fix = [f"On {names[other]}, swap its two cables (port 0 ↔ port 1)."]
            else:
                ours, theirs = between[0]
                fix = [f"On {names[k]}, move the cable from port 1 to port 0."
                       for k, port in ((head, ours), (other, theirs)) if port == 1]
            return self.finish("pair", f"Pair: {used}, but no cable joins the two ports 0. {PAIR_RULE}", fix,
                               order=[head, other])
        if size == 3:
            loose = free(group)
            return self.finish("unsupported", "Three Sparks are cabled together (" + ", ".join(names[k] for k in group)
                               + "). SparkRing needs two Sparks (a pair) or four (a ring)."
                               + (" Free ports: " + ", ".join(loose) + "." if loose else ""))
        if size > 4:
            return self.finish("unsupported", f"{size} Sparks are cabled together. SparkRing needs two Sparks (a pair) "
                               "or four (a ring).")
        degree = {k: sum(e[0] == k for c in inner for e in c) for k in group}
        between = {frozenset((c[0][0], c[1][0])): c for c in inner}

        def ports_along(order, closing=None):
            rows = []
            for i, a in enumerate(order):
                b = order[(i + 1) % len(order)]
                cable = between.get(frozenset((a, b)))
                if cable is None:
                    rows.append(closing)
                else:
                    rows.append((next(e[1] for e in cable if e[0] == a), next(e[1] for e in cable if e[0] == b)))
            return rows

        if len(inner) == 3:
            ends = [k for k in group if degree[k] == 1]
            order = [ends[0]]
            while len(order) < 4:
                order.append(next(k for k in sorted(graph[order[-1]], key=names.get) if k in members and k not in order))
            gaps = [(k, next(p for p in (0, 1) if state(k, p) != "cabled")) for k in (order[-1], order[0])]
            states = [state(*gap) for gap in gaps]
            text = " and ".join(self.end(gap) for gap in gaps)
            if "unseen" in states:
                unseen = [gap[0] for gap, s in zip(gaps, states, strict=True) if s == "unseen"]
                why = ("not reached" if all(k not in self.reached for k in unseen)
                       else "reached, but the far end of a link there was not seen")
                return self.finish("incomplete", f"Four Sparks are cabled in a line; {text} have no cable seen. "
                                   + " and ".join(names[k] for k in unseen) + f" {'was' if len(unseen) == 1 else 'were'} "
                                   f"{why}, so the last cable is unknown.")
            swapped, ring = _ring_fix(order, ports_along(order, (gaps[0][1], gaps[1][1])), head)
            fix = [f"Connect a cable from {self.end(gaps[0])} to {self.end(gaps[1])}."]
            fix += [f"On {names[k]}, swap its two cables (port 0 ↔ port 1)." for k in swapped]
            return self.finish("ring", f"Four Sparks are cabled in a line: {text} have no cable (missing or loose). "
                               + RULE, fix, order=ring)
        if len(inner) == 4 and all(d == 2 for d in degree.values()):
            start = port_cable.get((head, 0)) or next(c for c in inner if any(e[0] == head for e in c))
            order = [head, next(e[0] for e in start if e[0] != head)]
            while len(order) < 4:
                order.append(next(k for k in graph[order[-1]] if k not in order))
            ports = ports_along(order)
            self.result["walk"] = [f"{self.end((order[i], a))} ↔ {self.end((order[(i + 1) % 4], b))}"
                                   for i, (a, b) in enumerate(ports)]
            swapped, ring = _ring_fix(order, ports, head)
            if not swapped:
                return self.finish("ring", "Four-Spark ring, cabled as SparkRing needs.", order=ring, ready=True)
            same = sum(a == b for a, b in ports)
            return self.finish("ring", f"The four Sparks form a loop, but {same} cables join the same port number at "
                               f"both ends. {RULE}", [f"On {names[k]}, swap its two cables (port 0 ↔ port 1)."
                                                       for k in swapped], order=ring)
        loose = free(group)
        return self.finish("unsupported", "Four Sparks are cabled together but do not form one loop."
                           + (" Free ports: " + ", ".join(loose) + "." if loose else "") + " " + RULE)


def diagnose(sparks, head=None, *, strict=False, whole=False, neighbors=True):
    """The ``sparkring-cabling/v1`` diagnosis of the Sparks' cables.

    ``head`` is Node A's key. Without ``whole`` the layout is that of Node A's
    group (the Sparks its cables reach) and other reached Sparks are noted;
    with ``whole`` the given Sparks are the intended set, and Sparks outside
    one cabled group make it unsupported. ``strict`` is setup's check: every
    cable must be confirmed from both ends by LLDP, a neighbor that matches no
    given Spark and conflicting observations are problems, and neighbor caches
    are not used. On a pair, findings that concern only port 1 are notes,
    because pair profiles do not use port 1.

    The result has ``layout`` (``pair``, ``ring``, ``unsupported`` or
    ``incomplete``), ``ready`` (cabled as SparkRing needs, nothing to change),
    ``summary``, ``fix`` (physical steps, in order), ``order`` and
    ``order_names`` (Node A first, after the fix), ``walk`` (the cables as
    text, around the loop when there is one), ``cables`` (each with the
    Sparks that saw it), ``sparks`` (each port ``cabled``, ``no link`` or
    ``unseen``), ``problems`` and ``notes``.
    """
    return _Diagnosis(sparks, head, strict, whole, neighbors).run()


def message(result):
    """One line for an error: the finding, the fix and the ring order afterwards."""
    text = "Fabric cabling: " + result["summary"].rstrip(".") + "."
    if result["fix"]:
        text += " To fix: " + " ".join(result["fix"])
    if result["order_names"] and result["layout"] == "ring":
        text += (" Then the ring order is " if result["fix"] else " Ring order: ") + " → ".join(result["order_names"]) + "."
    return text + " " + CHECK


def lines(result, *, finding=True):
    """Terminal lines of a diagnosis: the cables, the finding, the fix, the order, problems and notes.

    Without ``finding`` the summary, fix and order (which ``message`` carries) are left out.
    """
    output = []
    if result["walk"]:
        output += ["Cables:"] + ["  " + text for text in result["walk"]]
    if finding:
        output.append(result["summary"])
        if result["fix"]:
            output += ["To fix:"] + ["  " + step for step in result["fix"]]
        if result["order_names"] and result["layout"] == "ring":
            output.append(("Ring order after the fix: " if result["fix"] else "Ring order: ")
                          + " → ".join(result["order_names"]))
    output += ["Problem: " + p for p in result["problems"] if p.rstrip(".") not in result["summary"]]
    output += ["Note: " + n for n in result["notes"]]
    return output


def main(argv=None):
    """``sparkring cabling``: read the Sparks on this Spark's cables and print the diagnosis; change nothing.

    Exit status: 0 when the cables form a pair or ring as SparkRing needs, 1
    when they need a change or could not all be seen, 2 when the command
    could not run.
    """
    import argparse
    import json
    import os
    import subprocess
    import sys
    import tempfile
    from runtime.host import bootstrap, controller, single_uplink, survey

    parser = argparse.ArgumentParser(prog="sparkring cabling", description=(
        "Show how the Sparks on this Spark's fabric cables are cabled and what to change for a pair or a "
        "four-Spark ring. Reads only; changes nothing on any Spark."))
    parser.add_argument("--json", action="store_true", help="print one sparkring-cabling/v1 document")
    parser.add_argument("--ssh-user", default=os.environ.get("SUDO_USER") or "root",
                        help="account for signing in to the other Sparks (default: the account that ran sudo)")
    parser.add_argument("--no-sign-in", action="store_true",
                        help="read only this Spark and the Sparks of its recorded cluster; ask for no password")
    args = parser.parse_args(argv)
    say = (lambda line: print(line, file=sys.stderr)) if args.json else print
    try:
        if not hasattr(os, "geteuid") or os.geteuid() != 0:
            raise ValueError("Run sudo sparkring cabling: LLDP and the cluster record need root")
        recorded = (single_uplink.installed_targets(controller.STATE) or [])[1:]
        # Node A's setup key signs in where workers trust it; SSH asks for a
        # password elsewhere. Host keys and connections stay in a temporary
        # directory, so nothing persists.
        key = controller.STATE / "controller_ed25519"
        with tempfile.TemporaryDirectory(prefix="sparkring-cabling-") as directory:
            transport = bootstrap.SSH(directory, identity=key if key.is_file() else None)
            try:
                found = survey.survey(transport, recorded=recorded, user=args.ssh_user,
                                      sign_in=not args.no_sign_in, say=say)
            finally:
                transport.close()
        result = diagnose(survey.records(found), found["head"])
        result["read"] = [line.strip() for line in survey.checked_lines(found)]
        result["survey_notes"] = found["notes"]
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print("Sparks read:")
            for line in survey.checked_lines(found):
                print(line)
            for line in lines(result):
                print(line)
            for note in found["notes"]:
                print("Note: " + note)
        return 0 if result["ready"] else 1
    except (ValueError, KeyError, RuntimeError, OSError, subprocess.SubprocessError) as error:
        print("SparkRing: " + str(error), file=sys.stderr)
        return 2
