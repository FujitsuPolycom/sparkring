# Inspect and repair ring networking

Use Ring Doctor to inspect routing, forwarding and firewall state on a configured
ring. Complete the [host setup](../GLM53_SPARK_MESH_HOST_SETUP.md) first.

## Routing and forwarding across the fabric

A switchless fabric has no shared broadcast domain: each node is directly
cabled only to its neighbours, so traffic to any other node is **relayed by a
neighbour**. Every node is therefore a router, and three conditions must hold
on every node before a launch:

- A kernel route to each fabric subnet the node is not directly attached to,
  via the neighbour that is.
- `net.ipv4.ip_forward=1`, without which the node accepts transit traffic and
  drops it.
- An unrestricted `DOCKER-USER` ACCEPT rule in both directions between the two
  fabric interfaces, reachable before any blocking rule. A `FORWARD` drop
  policy can leave non-adjacent nodes unreachable even when direct links work.

Ring Doctor checks rule order conservatively. It does not reorder existing
user firewall policy; an earlier blocking or unknown chain rule must be reviewed
before a later ACCEPT can establish unrestricted forwarding.

[`scripts/ring_doctor.py`](../../scripts/ring_doctor.py) checks all three, plus
addressing and reachability, and prints a repair plan. When a canonical site
file is available, Ring Doctor also reuses the preflight implementation for
negotiated link speed, expected MTU and address, active RDMA ports, Ethernet
link mode, the configured RoCEv2 GID, and a don't-fragment jumbo ping. Run it
read-only first:

```bash
python scripts/ring_doctor.py \
  --site scripts/config/site.yaml \
  --verify
```

Run Ring Doctor on rank 0, the head node. The command verifies local identity
against the configured rank management addresses and SSH hostnames before it
contacts the cluster. If rank 0 cannot run the tool, run it from a configured
worker with the explicit recovery flag:

```bash
python scripts/ring_doctor.py \
  --site scripts/config/site.yaml \
  --allow-worker-controller \
  --verify
```

The flag does not permit execution from a laptop or unknown control host; the
local machine must still identify as one configured worker rank. The report
records when worker recovery mode was used.

Require zero `ERROR` findings, a passing canonical fabric preflight, and a
reachability matrix in which every pair passes. `--apply` executes the printed
plan only after both the discovered cycle and canonical fabric checks pass. It
is idempotent and needs non-interactive `sudo` on each node. Before any repair
command runs, Ring Doctor executes `sudo -n true` on every node whose plan has
commands. If any node fails that check, Ring Doctor reports each failing node
and executes no route, forwarding, or firewall repair command on any node.

## Management safety during repair

Ring Doctor treats management reachability as a hard mutation invariant. It
does not change management addresses, links, routes, or NetworkManager
profiles. Before `--apply` or `--emit-unit`, every node must meet all of these
conditions:

- discovery reached the node directly, not through a fabric jump host;
- the canonical management interface exists and holds an IPv4 address; and
- the management interface is distinct from both fabric interfaces.

Each repair operation is restricted to observed fabric interfaces and fabric
subnets. Ring Doctor checks that the active SSH session terminates on a guarded
management address and that its return route uses a guarded management
interface immediately before and after every individual change. The remaining
plan stops on the first mismatch. Generated boot programs also verify the exact
recorded management addresses before and after every change. They run without
an SSH session and do not verify a route back to the controller. If a direct
`--node` invocation is used instead of `--site`, name the management interface
for every node with `--socket-interface`; otherwise all mutation is withheld.

The repairs are runtime state and do not survive a reboot. Do not use a cron
`@reboot` job to restore `DOCKER-USER` rules: cron can run before Docker creates
its firewall chains, and Docker can replace rules installed that early.

`--emit-unit DIR` writes a per-node program and systemd unit that revalidate the
addresses and reapply the complete route, forwarding, and `DOCKER-USER` plan at
boot. The unit orders itself after `network-online.target` and `docker.service`
when those units are active. A missing management address or firewall chain
fails closed, and systemd retries the program after ten seconds.

The generated files are not installed automatically. Install the program at a
path that exists **on the node**, and set the unit's `ExecStart` to that path:
the emitted unit names the directory the files were generated in, which is only
correct when they are generated on the node itself.
