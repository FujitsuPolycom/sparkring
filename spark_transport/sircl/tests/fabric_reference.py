"""The universal relay table of an eight-Spark ring as a plain shell installer builds it.

This is the reference for the installer's ``ring8`` layout. Its rules, per
rank ``r`` and function (primary ``enp1s0f*``/``rocep1s0f*``, secondary
``enP2p1s0f*``/``roceP2p1s0f*``):

- routes: clockwise peers at distances 2 and 3 by their port-1 address of the
  function via ``r``'s port 0, counter-clockwise peers by their port-0 address
  via ``r``'s port 1, and the opposite peer (distance 4) by both of its port
  addresses, each the way it arrives; a /32 scope-link route with a permanent
  neighbour holding the adjacent Spark's MAC on that cable (no protocol mark);
- tags: one marker per RDMA device with one rule per routed destination,
  ``TAG(k) = 0x88b4 + k``, ``k`` = relays on the path;
- relays: on each of the four network devices, ``TAG(k) -> TAG(k - 1)`` for
  ``k = 1..3``, setting the next hop's MAC and redirecting out of the other
  port's device of the same function, at preference ``10 + k``, handle ``k``.

The addresses follow the simulator's ring plan
(:func:`sparkring_sircl.testing.relay_hosts.ring_netdevs`): port 0 of rank
``r`` is ``198.18.(2r + f).1`` and port 1 is ``198.18.(2(r - 1) + f).2``.
"""

from __future__ import annotations

N = 8
FUNCTIONS = {
    "primary": {"port0": "enp1s0f0np0", "port1": "enp1s0f1np1", "dev0": "rocep1s0f0", "dev1": "rocep1s0f1",
                "offset": 0},
    "secondary": {"port0": "enP2p1s0f0np0", "port1": "enP2p1s0f1np1", "dev0": "roceP2p1s0f0",
                  "dev1": "roceP2p1s0f1", "offset": 1},
}
MARKER = "/var/tmp/ring8-mesh-marker"
MAX_RELAYS = N // 2 - 1


def tag(k: int) -> int:
    return 0x0800 if k == 0 else 0x88B4 + k


def port0_ip(rank: int, function: str) -> str:
    return f"198.18.{2 * rank + FUNCTIONS[function]['offset']}.1"


def port1_ip(rank: int, function: str) -> str:
    return f"198.18.{2 * ((rank - 1) % N) + FUNCTIONS[function]['offset']}.2"


def plan(rank: int, mac) -> tuple[list[dict], list[dict]]:
    """Routes (with marker tags) and relay rules of one rank; ``mac(rank, netdev)`` gives MACs."""
    routes = []
    for function, fn in FUNCTIONS.items():
        for distance in range(2, N // 2 + 1):
            cw_peer, ccw_peer = (rank + distance) % N, (rank - distance) % N
            routes.append({"dest": port1_ip(cw_peer, function), "dev": fn["port0"], "rdma": fn["dev0"],
                           "src": port0_ip(rank, function), "via_mac": mac((rank + 1) % N, fn["port1"]),
                           "tag": tag(distance - 1), "peer": cw_peer, "hops": distance})
            routes.append({"dest": port0_ip(ccw_peer, function), "dev": fn["port1"], "rdma": fn["dev1"],
                           "src": port1_ip(rank, function), "via_mac": mac((rank - 1) % N, fn["port0"]),
                           "tag": tag(distance - 1), "peer": ccw_peer, "hops": distance})
    relays = []
    for function, fn in FUNCTIONS.items():
        for k in range(1, MAX_RELAYS + 1):
            relays.append({"in": fn["port1"], "out": fn["port0"], "next_mac": mac((rank + 1) % N, fn["port1"]), "k": k})
            relays.append({"in": fn["port0"], "out": fn["port1"], "next_mac": mac((rank - 1) % N, fn["port0"]), "k": k})
    return routes, relays


def up_script(rank: int, hostname: str, mac) -> str:
    """The shell script that installs rank ``rank``'s part of the table (its mutating commands)."""
    routes, relays = plan(rank, mac)
    lines = ["set -e", f'[ "$(hostname)" = "{hostname}" ]', "sudo -n pkill -x ring8-mesh-mark || true", "sleep 0.5"]
    netdevs = sorted({r["in"] for r in relays})
    for netdev in netdevs:
        lines.append(f"sudo -n tc qdisc del dev {netdev} ingress 2>/dev/null || true")
        lines.append(f"sudo -n tc qdisc add dev {netdev} ingress")
    for r in relays:
        lines.append(f"sudo -n tc filter add dev {r['in']} ingress protocol 0x{tag(r['k']):04x} pref {10 + r['k']} "
                     f"handle {r['k']} flower skip_sw action pedit ex munge eth dst set {r['next_mac']} pipe "
                     f"action pedit ex munge eth type set 0x{tag(r['k'] - 1):04x} pipe action mirred egress "
                     f"redirect dev {r['out']}")
    for r in routes:
        lines.append(f"sudo -n ip route replace {r['dest']}/32 dev {r['dev']} src {r['src']} scope link")
        lines.append(f"sudo -n ip neigh replace {r['dest']} lladdr {r['via_mac']} dev {r['dev']} nud permanent")
    by_device: dict[str, list[str]] = {}
    for r in routes:
        by_device.setdefault(r["rdma"], []).append(f"--rule {r['dest']}=0x{r['tag']:04x}")
    for device, rules in sorted(by_device.items()):
        lines.append(f"(sudo -n setsid {MARKER} --device {device} {' '.join(rules)} --managed "
                     f"< /dev/null > /tmp/ring8-mesh-marker-{device}.log 2>&1 &)")
    lines.append("sleep 1.5")
    return "\n".join(lines) + "\n"


def install(ring) -> None:
    """Install the table on every Spark of a :class:`SimRing` of eight with its own scripts."""
    def mac(rank: int, netdev: str) -> str:
        return ring.sparks[rank].netdevs[netdev].mac

    for rank, spark in enumerate(ring.sparks):
        result = spark.run(up_script(rank, spark.hostname, mac))
        assert result.returncode == 0, (spark.name, result.stderr)
