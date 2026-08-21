# SparkRing prerequisites

Complete this checklist before deploying any supported profile. It defines the
hardware and operator conditions required by the
[GLM](GLM52_35BPW_QUICKSTART.md) and
[DeepSeek](DEEPSEEK_V4_FLASH_QUICKSTART.md) quickstarts. The GLM quickstart and
the DeepSeek four-Spark cycle require four Sparks; the DeepSeek two-Spark pair
requires two.

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
has 48 shards totaling about 167 GB. Budget additional image and cache headroom.

## Network requirements

- RoCEv2 must be configured on every fabric port a rank uses.
- Every direct cable must pass link, address, and RDMA checks before a model
  launch.
- The management LAN must permit SSH between the operator and ranks, and
  rendezvous traffic between ranks.
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
