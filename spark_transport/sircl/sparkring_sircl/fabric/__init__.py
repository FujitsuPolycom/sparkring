"""Relay plan installer for SIRCL ring sessions on a switchless ring of DGX Sparks.

SIRCL sessions read host networking but never configure it: members of a
group that share no cable reach each other through ConnectX-7 relays, and the
relay plan is installed by site tooling before a session starts. This
subpackage is that tooling. It derives a layout's plan from SIRCL's own route
module (:mod:`sparkring_sircl.routes`), so the lanes SIRCL derives and the
relays installed cannot disagree, and it installs, compares and removes the
plan on the Sparks over SSH. SIRCL's session code never imports it.

Per Spark the plan holds:

- origin routes: a /32 route, scope link, over the lane's local network
  device, with a permanent neighbour entry holding the adjacent Spark's MAC;
- relay count tags: one marker process per RDMA device (``mesh_marker.c``)
  installs mlx5 RDMA-TX rules that rewrite the EtherType of RoCE packets to
  each routed destination into ``TAG(k) = 0x88b4 + k``, ``k`` relays left;
- relay filters: on the ingress of the device that receives a relayed lane, a
  hardware flower filter per tag that sets the next Spark's MAC, rewrites the
  tag to ``TAG(k - 1)`` (``TAG(0)`` is IPv4) and redirects out of the other
  port.

Layouts (:mod:`.layouts`): ``ring8`` (the universal relay table), ``2xTP4``,
``4xTP2`` and custom groups of consecutive Sparks, wrap-around included. With
several groups on one ring, routes and tags exist only for same-group
destinations on that group's members, and relay filters only where a member
lies inside a carried path of its own group.

Modules: :mod:`.layouts` (groups), :mod:`.plan` (derivation, isolation and
trace checks), :mod:`.commands` (shell commands and ownership marks),
:mod:`.state` (parsing what a Spark holds), :mod:`.diff` (comparison and
apply scripts), :mod:`.ops` (read, compare, apply), :mod:`.report` (text) and
:mod:`.cli` (``python -m sparkring_sircl.fabric``). The host-command
simulator for tests is :mod:`sparkring_sircl.testing.relay_hosts`.

Status: implemented; CPU-tested against the simulator. The ``ring8`` plan
reproduces the universal relay table object for object; the table's shell
installer is the tests' reference (``tests/fabric_reference.py``). The
installer has not run on a ring.
"""
