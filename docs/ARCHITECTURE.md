# SparkRing architecture

SparkRing separates cluster transport from serving profiles. The cluster layer
defines ranks, direct links, routes, collectives, and management traffic. A
profile composes that layer with a model runtime, checkpoint, memory geometry,
and scheduler settings.

## Networks

```text
management LAN ─┬─────────────┬─────────────┬─────────────┐
            ┌───┴───┐    ┌───┴───┐    ┌───┴───┐    ┌───┴───┐
     API ──>│ rank 0 ╞════╡ rank 1 ╞════╡ rank 2 ╞════╡ rank 3 │
            └───╤───┘    └────────┘    └────────┘    └───╤───┘
                ╚════════════════════════════════════════════════════════════════════════════════════════════╝

  ═══  one direct ConnectX-7 fabric link per edge
  ───  management LAN: SSH, rendezvous, and rank-0 API
```

The direct fabric and management network have distinct responsibilities. The
direct fabric carries distributed communication. The management network
carries cluster control and client traffic and must never be configured as a
fabric edge.

## Supported physical shapes

A two-rank pair uses one direct link and has no relayed fabric hop. A four-rank
cycle uses the edges `0-1`, `1-2`, `2-3`, and `3-0`. Each cycle rank has two
fabric neighbours and uses both links for communication.

Six-rank cycle discovery and diagnostics are implemented. Six-rank model
serving remains research-only until a named profile records live evidence.

## Routing

A switchless cycle has no shared broadcast domain. Reaching a non-adjacent
fabric subnet requires a route through a neighbouring rank. Every cycle rank
therefore needs:

- routes to non-adjacent fabric subnets;
- `net.ipv4.ip_forward=1`; and
- `DOCKER-USER` forwarding rules between its fabric interfaces.

[Ring Doctor](../scripts/ring_doctor.py) verifies these conditions and protects
the management path while applying an explicitly approved repair. The
[prerequisites](PREREQUISITES.md) define the complete host contract.

## Collective paths

SIRCL owns persistent RDMA sessions, registered arenas, and graph-replayable
command rings for its supported four-rank collective families. It decomposes
four-rank work into the physical cycle's two perfect matchings so every
transfer remains neighbour-to-neighbour.

Patched NCCL handles collective shapes, phases, and topologies outside SIRCL's
supported boundary. The selected deployment profile records the active path;
SparkRing does not infer transport eligibility from a model name.

See [SIRCL](SIRCL.md) and the [native transport
documentation](../spark_transport/README.md) for implementation details.

## Profile boundary

Profiles own model and serving choices, including:

- checkpoint and runtime identities;
- tensor and decode-context parallel dimensions;
- cache geometry and memory budgets;
- speculation and scheduler settings; and
- qualified topology and evidence.

The [profile registry](profiles/README.md) routes operators to those contracts.
Changing a profile does not redefine the cluster or transport architecture.
