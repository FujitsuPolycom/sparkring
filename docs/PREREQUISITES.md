# SparkRing prerequisites

Complete this checklist before deploying any supported profile. It defines the
hardware and operator conditions required by the
[GLM](GLM52_35BPW_QUICKSTART.md) and
[DeepSeek](DEEPSEEK_V4_FLASH_QUICKSTART.md) quickstarts, and the
[Qwen3.8-27B four-Spark quickstart](QWEN38_27B_EXL3_K5K6_QUICKSTART.md). The
GLM, Qwen, and DeepSeek cycle configurations require four Sparks; the DeepSeek
pair requires two.

## Hardware and topology

For a four-Spark cycle:

- Four NVIDIA DGX Sparks with both 200 Gb/s ConnectX-7 ports available.
- Four qualified 200 Gb/s DACs cabled exactly as `0-1-2-3-0`.
- Stable rank assignment: the same host is rank 0, 1, 2, or 3 throughout a
  deployment.

For a two-Spark pair:

- Two NVIDIA DGX Sparks with at least one 200 Gb/s ConnectX-7 port available on
  each.
- One qualified 200 Gb/s DAC joining them, cage 0 to cage 0, so that both ranks
  name the same interface.
- Stable rank assignment: the same host is rank 0 throughout a deployment, and
  serves the API.

Both topologies also require a management LAN reachable by the operator and
every rank.

The direct cabling is the inference fabric. Do not use a fabric port as a
management interface.

## Operating system and storage

Each rank needs Linux ARM64, a Docker-compatible runtime with GPU access,
`/dev/infiniband`, and enough writable local storage for the selected image,
model checkpoint, and JIT cache. Model paths mounted into containers must exist
on every rank at the paths used by the launch command.

The GLM checkpoint index totals 346,218,639,128 bytes. The DeepSeek checkpoint
has 48 shards totaling about 167 GB. The Qwen checkpoint has three shards and
requires about 22 GB before runtime and JIT caches. Budget additional image and
cache headroom.

## Network requirements

- RoCEv2 must be configured on every fabric port a rank uses.
- Every direct cable must pass link, address, and RDMA checks before a model
  launch.
- The management LAN must permit SSH between the operator and ranks, and
  rendezvous traffic between ranks.

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
addressing and reachability, and prints a repair plan. Run it read-only first:

```bash
python scripts/ring_doctor.py   --node <user>@<host0> --node <user>@<host1>   --node <user>@<host2> --node <user>@<host3> --verify
```

Require zero `ERROR` findings and a reachability matrix in which every pair
passes. `--apply` executes the printed plan, which is idempotent and needs
passwordless `sudo` on each node.

The repairs are runtime state and do not survive a reboot. `--emit-unit DIR`
writes a per-node program and systemd unit that revalidate the addresses and
reapply the plan at boot. Install the program at a path that exists **on the
node**, and set the unit's `ExecStart` to that path: the emitted unit names the
directory the files were generated in, which is only correct when they are
generated on the node itself.
- Rank 0 must expose the configured API port to intended clients.

## Local configuration and preflight

The GLM quickstart uses an ignored site file. Copy, complete, and validate it:

```bash
cp scripts/config/exl3-r7-site.example.yaml scripts/config/site.yaml
$EDITOR scripts/config/site.yaml
python scripts/sparkring_site.py scripts/config/site.yaml
python scripts/preflight.py --site scripts/config/site.yaml --print-plan
```

The final command is offline and prints the remote checks. Review it before
running the same command without `--print-plan`, which contacts configured
hosts without mutating them.

The DeepSeek quickstart uses one local copy of
`scripts/config/deepseek-v4-flash-0731.env.example` per rank. Replace only its
network interface and fabric-address placeholders.

## Safety boundary

A plan-only command is offline. Remote preflight is read-only. Starting or
replacing a serving stack mutates hosts and can stop serving; do not execute a
start command without explicit authorization for the four named hosts and the
action.
