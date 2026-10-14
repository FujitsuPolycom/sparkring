# SparkRing architecture

SparkRing runs large language models on two to eight DGX Sparks that are cabled
directly to each other, with no switch. vLLM serves each model; every Spark of
a model runs one tensor-parallel rank in a container of the same image. Node A,
the Spark connected to your network, runs the installer and the controller.
The [README](../../README.md#profiles) lists the profiles that `sparkring install`
sets up; the [profile catalog](../../profiles/README.md) lists every profile.

## Topology

![A ring of eight DGX Sparks serving two four-Spark models on positions 0-3 and 4-7; Node A serves the API and dashboards, and traffic between Sparks that share no cable is forwarded in hardware by the ConnectX-7 of the Sparks between them](assets/sparkring-cluster.svg)

Setup forms one of three layouts ([cabling](../operations/install-reference.md#cabling)):

| Layout | Cabling |
|---|---|
| Pair | One cable from port 0 (p0) to p0 |
| Line of three to eight Sparks | Each Spark's p0 to the next Spark's p1, starting at Node A |
| Ring of three to eight Sparks | The same, plus a cable from the last Spark back to Node A |

Node A is position 0, and positions follow the cables. Each ConnectX-7 port
appears as two PCI functions, and each cable carries one fabric subnet per
function ([addresses](../operations/install-reference.md#addresses-on-any-layout)).

Sparks that share no cable reach each other through the Sparks between them,
whose ConnectX-7 cards forward the traffic in hardware. Setup installs one
[relay table](../operations/install-reference.md#the-relay-table) for the
whole fabric: routes and permanent neighbour entries toward every Spark not
cabled to this one, ConnectX ingress rules that pass a packet out of the
other port, and relay markers that tag RDMA packets with the number of relays
left. Every Spark between two cables needs the ConnectX
[hairpin setting](../operations/install-reference.md#the-hairpin-setting),
which setup applies and `sparkring-hairpin.service` repeats at every boot.

A SIRCL lane crosses at most three relays. One model on a line of six or
more Sparks is therefore unsupported: its end Sparks are four or more relays
apart ([models on part of the fabric](../operations/install-reference.md#models-on-part-of-the-fabric)).

A fabric serves one model on all its Sparks, or several models side by side
on groups of consecutive Sparks that share no Spark (`--on 0-3`, `--on 4-7`).
Each group's first Spark serves its API. Operations on disjoint groups can
run at the same time ([independent slices](independent-slices.md)).

Only Node A needs your network: it serves the API of a group that starts at
Node A, the ring dashboard, and downloads the image and checkpoint for the
other Sparks. Everything between Sparks runs over the cables: collectives,
vLLM's startup rendezvous, the administration network (WireGuard over the
cables' IPv6 link-local addresses, with other cables and the LAN as fallbacks)
that carries SSH and package copies, and image and checkpoint copies.

## Serving container

![One serving container per Spark: the API server, vLLM with B12X kernels, SparkRing's runtime-status plugin, SIRCL ring sessions as the default collective path and the prepared RoCEnante and NCCL transport for images without SIRCL, over the four ConnectX-7 RDMA functions](assets/sparkring-serving-stack.svg)

Every Spark of a model runs one container of the same image, one
tensor-parallel rank. Its kernels come from
[B12X](https://github.com/local-inference-lab/b12x), Local Inference Lab's
kernel library for GB10 (SM121) GPUs. Each profile's `profiles/<id>/config.json`
selects the model, the vLLM arguments and the transport limits; the
[install reference](../operations/install-reference.md) explains them.

The default installer image is the image of release 2026.10.4,
`dev-20261013-dcp2prefill-status038-sircl-tp6`. It carries the vLLM and B12X sources that read the CSF checkpoints,
with GLM-5.3-Flash's prefill admitted at TP2/DCP2, SIRCL 0.3.2, libsircl
0.6.0, three GLM-5.3 vLLM plugins and the runtime-status dashboard 0.3.8
([images](../operations/images.md#installer-image)). Its
[image lock](../../runtime/releases/dev-20261013-dcp2prefill-status038-sircl-tp6/installer-image.json)
lists the transports it carries and the 15 profiles admitted to it.

## Collective path

![One decode step on four ranks: two SIRCL all-reduces per layer, a SIRCL all-gather of the LM head's vocabulary slices, draft verification and the next draft; one SIRCL all-reduce stages its input in a pinned arena, writes stripes and flags over RDMA through hardware relays where needed, and sums in rank order](assets/sparkring-decode-step.svg)

Tensor-parallel ranks combine their results with collectives: all-reduce,
all-gather, reduce-scatter and all-to-all. `sparkring install` runs them on
one of these transports
([transport and receipts](../operations/install-reference.md#transport-and-receipts)):

| Transport | When it runs | Status |
|---|---|---|
| `sircl`: SIRCL ring sessions, NCCL off | The default wherever it can run: an image whose lock lists `sircl`, such as the default image, on a fabric that `sparkring setup` recorded with its relay table. Any installer profile on a pair, a line of three to five Sparks, a whole ring of three to eight, or a group within it | One-shot all-reduce and all-gather **qualified** on the eight-Spark ring; serving through vLLM **research-only** ([status table](../../spark_transport/sircl/STATUS.md#component-status)) |
| `prepared`: RoCEnante plus NCCL 2.32.3 | Images without the SIRCL layer, such as release 2026.10.1's, a fabric that does not carry `sircl`, or `--transport prepared`. A pair, a ring of four and that ring's halves only | The transport of images without the SIRCL layer |
| `nccl`: vLLM's PyNccl on NVIDIA NCCL | Only with `--transport nccl`, on a cabled pair or a whole ring; refused where ranks need relays | Opt-in |
| `libsircl`: vLLM's PyNccl on libsircl | Only with `--transport libsircl`, on an image whose lock lists `libsircl` | **research-only** |

**SIRCL**, the Switchless Inference RDMA Collective Layer, is SparkRing's
collective layer for 2 to 8 Sparks ([SIRCL](sircl.md)). vLLM's `sircl`
platform and general plugins put a SIRCL session in front of every collective
of the tensor-parallel and decode-context-parallel groups. Sessions reach
peers that share no cable through the relays, keep each relayed lane's
unacknowledged bytes within a forward window that the relay's hairpin queue
can hold, and choose their algorithms from a tuning table: the package's
default or one that `sudo sparkring fabric tune` measured on your fabric. NCCL runs only where the operator
opts in with `--nccl auto`. Each rank writes receipts that the installer
checks after the model answers.

The **prepared** transport, shown as `RoCEnante (b12x)`, splits collectives
by size. RoCEnante, B12X's RoCE all-reduce with SparkRing's adaptive peer
paths, takes all-reduces up to 2 MiB and all-gathers up to 16 MiB per shard,
the per-token traffic of decode. NCCL 2.32.3 from the image takes larger
collectives, mostly prefill; on a ring of four it runs ring algorithms along
the cable cycle.

**libsircl** implements NCCL's public C API on SIRCL's wire protocol, so a
program written against NCCL loads it in place of NCCL
([libsircl](libsircl.md)). The installer uses it only when named; every part
of its installer integration is research-only.

**net-sircl** is an NCCL network plugin: NVIDIA NCCL keeps its own
algorithms and CUDA-graph support, and the plugin supplies the transport
between directly cabled Sparks, including peers reached through the relays.
It is published separately at
[FujitsuPolycom/net-sircl](https://github.com/FujitsuPolycom/net-sircl);
version 0.1.0 is research-only and tested with NCCL 2.32.3. Its source is not
in this repository, and `sparkring install` does not use it.

## Profile composition

Catalog profiles that `sparkring install` does not set up keep their own
images, launchers and guides; each guide states its collective path. They use
one of these:

- **Patched NCCL alone**, with the environments in `scripts/config/`:
  DeepSeek-V4-Flash-0731 and the Qwen3.8-27B EXL3 K5/K6 pair and ring.
- **Four-rank native SIRCL sessions with patched NCCL**: the retired GLM-5.2
  EXL3 3.5-bpw profile, a TP4/DCP4 configuration that SparkRing does not
  support, uses SIRCL for its tensor-parallel all-reduce and vocabulary
  collectives. These sessions serve exactly four ranks on a four-Spark ring
  and are kept for retained images ([SIRCL](sircl.md#profile-use)).
- **The four-Spark mesh composition** of four-rank SIRCL, an adapted RoCEnante
  all-reduce and patched NCCL, with a per-deployment mesh service
  ([composition](sircl.md#composition-with-hardware-forwarded-mesh)).

On rings of four, these profiles need routes to the non-adjacent fabric
subnets, `net.ipv4.ip_forward=1` and an unrestricted `DOCKER-USER` forward
rule on every Spark. [Prerequisites](../operations/prerequisites.md) lists
them and [`scripts/ring_doctor.py`](../../scripts/ring_doctor.py) checks them.

Their guides have the setup and details:
[GLM-5.2 EXL3 3.5-bpw (retired)](../../profiles/glm52-exl3-r7-3.5bpw/README.md),
[DeepSeek-V4-Flash-0731](../operations/deepseek-0731.md),
[Qwen3.8-27B EXL3 K5/K6 pair](../../profiles/qwen38-27b-exl3-k5k6-pair/README.md) and
[Qwen3.8-27B EXL3 K5/K6 ring](../../profiles/qwen38-27b-exl3-k5k6/README.md).
