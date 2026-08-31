# SparkRing prerequisites

Complete this checklist before deploying a profile. It defines the shared host,
network, storage, and safety conditions for SparkRing. The [profile
registry](profiles/README.md) adds model, topology, runtime, and capacity
requirements for each deployment.

Ring Doctor, canonical site validation, and fabric preflight support closed
four- and six-Spark cycles. Six-Spark serving profiles remain `research-only`
until their runtime and performance evidence are qualified.

## Hardware and topology

For a four-Spark cycle:

- Four NVIDIA DGX Sparks with both 200 Gb/s ConnectX-7 ports available.
- Four qualified 200 Gb/s DACs cabled exactly as `0-1-2-3-0`.
- Stable rank assignment: the same host is rank 0, 1, 2, or 3 throughout a
  deployment.

For a six-Spark cycle (`research-only` serving profiles):

- Six NVIDIA DGX Sparks with both 200 Gb/s ConnectX-7 ports available.
- Six qualified 200 Gb/s DACs cabled exactly as `0-1-2-3-4-5-0`.
- Stable rank assignment from rank 0 through rank 5 throughout a deployment.
- A canonical site file with six edges and six ranks; every rank still owns
  exactly two distinct fabric interfaces and two ring neighbours.

For a two-Spark pair:

- Two NVIDIA DGX Sparks with at least one 200 Gb/s ConnectX-7 port available on
  each.
- One qualified 200 Gb/s DAC joining them, cage 0 to cage 0, so that both ranks
  name the same interface.
- Stable rank assignment: the same host is rank 0 throughout a deployment, and
  serves the API.

All topologies also require a management LAN reachable by the operator and
every rank.

The direct cabling is the inference fabric. Do not use a fabric port as a
management interface.

## Operating system and storage

Each rank needs Linux ARM64, a Docker-compatible runtime with GPU access,
`/dev/infiniband`, and enough writable local storage for the selected image,
checkpoint, JIT cache, and optional persistent context cache. Paths mounted
into containers must exist on every rank at the locations named by the profile.

Checkpoint sizes, build scratch space, and cache budgets are profile
properties. Determine them from the selected profile before placing artifacts
on any rank.

## Network requirements

- RoCEv2 must be configured on every fabric port a rank uses.
- Every direct cable must pass link, address, and RDMA checks before a model
  launch.
- The management LAN must permit SSH between the operator and ranks.
- It must also permit a profile's rendezvous and control traffic when that
  profile configures management addresses for those channels.
- Rank 0 must expose the configured API port to intended clients.

### Routing and forwarding across the fabric

A switchless fabric has no shared broadcast domain: each node is directly
cabled only to its neighbours, so traffic to any other node is **relayed by a
neighbour**. Every node is therefore a router, and three conditions must hold
on every node before a launch:

- A kernel route to each fabric subnet the node is not directly attached to,
  via the neighbour that is.
- `net.ipv4.ip_forward=1`, without which the node accepts transit traffic and
  drops it.
- An unrestricted `DOCKER-USER` ACCEPT rule in both directions between the two
  fabric interfaces. Installing Docker sets the `FORWARD` chain policy to
  `DROP`, which silently blocks fabric transit. **This is the most commonly
  missed condition, and it presents exactly like a dead cable**: links are up,
  addresses are configured, neighbours ping, and every non-adjacent node is
  unreachable.

[`scripts/ring_doctor.py`](../scripts/ring_doctor.py) checks all three, plus
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
is idempotent and needs passwordless `sudo` on each node.

### Management safety during repair

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
recorded management addresses before and after every change. If a legacy
`--node` invocation is used instead of `--site`, name the management interface
for every node with `--socket-interface`; otherwise all mutation is withheld.

The repairs are runtime state and do not survive a reboot. `--emit-unit DIR`
writes a per-node program and systemd unit that revalidate the addresses and
reapply the plan at boot. Install the program at a path that exists **on the
node**, and set the unit's `ExecStart` to that path: the emitted unit names the
directory the files were generated in, which is only correct when they are
generated on the node itself.

## Local configuration and preflight

Copy a topology-compatible site template to the ignored canonical path,
complete its placeholders, and validate it:

```bash
cp <site-template> scripts/config/site.yaml
$EDITOR scripts/config/site.yaml
python scripts/sparkring_site.py scripts/config/site.yaml
python scripts/preflight.py --site scripts/config/site.yaml --print-plan
```

The final command is offline and prints the remote checks. Review it before
running the same command without `--print-plan`, which contacts configured
hosts without mutating them.

Profiles that use per-rank environment files document where to copy them and
which placeholders are site-specific. Keep resolved site files and environment
files outside version control. The [configuration index](../scripts/config/README.md)
routes each profile to its templates and validator.

## Safety boundary

A plan-only command is offline. Remote preflight is read-only. Starting or
replacing a serving stack mutates hosts and can stop serving; do not execute a
start command without explicit authorization for every named host in the
selected topology and the action.
