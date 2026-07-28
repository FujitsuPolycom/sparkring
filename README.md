# SparkRing

SparkRing serves GLM-5.2 (the 382 GiB `aidendle94/GLM-5.2-MXFP4-Experts-GPTQ`
checkpoint) across four NVIDIA DGX Spark desk units cabled directly to each
other: four ConnectX-7 200 Gb/s DAC cables in a ring, no network switch
anywhere. Stock NCCL cannot initialize on that topology, so SparkRing ships a
two-patch NCCL build plus an original fail-closed vLLM runtime overlay with
custom RDMA collectives: registered mapped-host arenas, GPU doorbells,
device-published command rings, a CUDA-graph-capturable two-matching ring
all-reduce, DCP4 decode context parallelism, and adaptive MTP speculative
decoding. Every number below was measured end-to-end on the real hardware.

## Headline results

| Measured result | Configuration (compressed; full claim labels in [docs/RESULTS.md](docs/RESULTS.md)) |
|---|---|
| **834 / 884 / 854 tok/s** uncached prefill at 8K / 16K / 32K | TP4/DCP4 switchless ring, adaptive MTP2/4, CUDA graphs; C1 single-sample prefill scouts per context, not prefix-cache hits |
| **63.60 tok/s** aggregate sustained decode at C8 (7.95 tok/s per user, 3.51x C1, still rising at the configured cap) | TP4/DCP1, Q40 graph plan, 15 s cells; **shared-prefix** 8K contexts, i.e. a concurrency baseline, not a unique-context capacity result |
| **20.83 / 19.28 / 21.43 tok/s** p50 single-stream decode at 8K / 16K / 32K | v40 window (2026-07-28): TP4/DCP4 with the full custom DCP collective trio captured inside FULL CUDA graphs, KV 4 GB/rank (500,224-token pool); sealed 30 s cells, zero JIT events, frozen transport counters |
| **27.2 tok/s median** (29.3 max) over five sequential real coding runs | TP4/DCP1 checkpoint, C1, 30 s windows; the most honest realistic-workload single-stream number in the repo |
| **375,040-token KV pool** (4x the DCP1 pool) while sustaining 56.70 tok/s aggregate C8 decode | 2026-07-27 DCP4 switchless window: TP4/DCP4, `nvfp4_ds_mla` KV at 3.0 GB/rank; **shared-prefix** 8K contexts, same baseline caveat as above. The current v40 window config (2026-07-28) allocates 4 GB/rank for a 500,224-token pool |

All rows are end-to-end serving through the OpenAI-compatible API with
per-cell pass/fail gates (zero request errors, graph census, transport
audits). C*N* means N concurrent streams. Full configuration labels, gates,
and machine-readable evidence JSON: [docs/RESULTS.md](docs/RESULTS.md).

## How it works

**Topology.** Four Sparks, four cables, ring 0-1-2-3-0, each edge one direct
ConnectX-7 200 Gb/s link with cage-matched ends. A 4-cycle decomposes into two
perfect matchings, so every four-rank collective runs as two rounds of
simultaneous pairwise exchanges (0/1 + 2/3, then 0/3 + 1/2) with both NIC
cages busy at once.

**Why stock NCCL fails here.** At init NCCL builds Tree/PAT channels between
all rank pairs, and the non-adjacent pairs (0-2, 1-3) have no L2 adjacency on
a switchless ring, so QP setup fails. Two small patches against NCCL 2.30.7
fix it: skip Tree/PAT connections entirely (ring-feasible neighbors only,
after Joseph Rose's switchless approach), and advertise both local RoCE GIDs
so the /24-subnet-aware connector picks the device on the correct cable. The
build is checksum-pinned, LD_PRELOADed read-only, and attested on every rank
at launch.

**Mapped-host arenas.** GB10 does not give this transport GPUDirect RDMA into
device memory, so the collectives run out of registered `cudaHostAllocMapped`
arenas that both the GPU and the NIC can address. Measured cost of that
detour: essentially zero. A GPU-produced 16 KB RC write lands in 4.53 us p50,
against 4.75 us for plain host memory over the same bare cable.

**GPU doorbells and command rings.** The GPU publishes collective work into a
64-slot device-published mapped command ring; a persistent lock-free host
progress thread drives the NIC. Because submission is one stream-ordered
kernel writing a doorbell, the whole collective captures into CUDA graphs.
Firing a captured four-rank all-reduce costs ~1.9 us of host time at ~39 us
device time per call, byte-audited through 10,000 replays with exact
published/consumed/completed sequence agreement on all ranks. Graph capture
censuses are window-scoped: the 2026-07-27 DCP4 switchless window recorded
6,744 custom all-reduce + 24 custom vocabulary captures with 2,904 attested
stock captures per rank; the 2026-07-28 v40 window recorded 5,464 custom
all-reduce + 24 custom vocabulary captures, plus 1,272 custom DCP query +
1,272 custom combine nodes per rank, with 360 stock owner-top-k nodes per
rank as the only stock captures.

**Fail-closed overlay, not a fork.** SparkRing does not fork vLLM. An overlay
on `PYTHONPATH` monkey-patches the installed vLLM at container start, and
every patch verifies the SHA-256 of the exact upstream source it expects
before installing; on mismatch it refuses to start rather than guess.
Unsupported shapes and operations fall back to the stock vLLM/NCCL path. The
orchestrator attests the NCCL binary hash, mounts, environment, and full
command line on every rank, and rolls back the previous container on any gate
failure.

**DCP4.** Decode context parallelism shards the KV cache across all four
ranks: 4x the KV pool at a ~4% single-stream 32K decode cost versus DCP1. It
runs on custom query all-gather and fused online-softmax combine collectives,
each validated byte-exact or within a stated numerical envelope before any
model traffic, and (since the v40 window) captured inside FULL CUDA graphs.
The 32K custom-trio decode number is ~12% above the earlier stock-trio
measurement, but that comparison spans separate windows and is indicative,
not a sealed A/B.

**Adaptive MTP.** The checkpoint's MTP draft head proposes 2 or 4 tokens per
round, switching depth on a 32-round acceptance window. In the best observed
structured-code window (a 10-second server-side peak of 43.0 aggregate tok/s
at C2), mean acceptance length was 4.30 with 82.5% draft acceptance.

**Transport floor.** The stack is built on 4.75 us p50 one-way RC writes at
16 KB, a 20.67 us p50 GPU-visible closed-loop round trip that includes
GPU-side payload verification, and a fused TP2 BF16 exchange-plus-add of a
real GLM hidden vector (12 KB) at 20.27 us p50. A separate one-million-
iteration 16 KB burn of the same TP2 primitive ran at 22.544 us p50 with
zero mismatched iterations ([docs/RESULTS.md](docs/RESULTS.md), row T3).

Full design detail: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Repository map

```text
spark_transport/
  src/, include/      transport core -> libspark_transport_capi.so: verbs endpoints,
                      TP4 two-matching schedule, GPU doorbells, DCP query/combine,
                      vocabulary all-gather, graph command ring
  app/                RDMA and collective probe binaries (transport, TP2/TP4,
                      graph replay, DCP, vocabulary)
  integrations/vllm/  the fail-closed runtime overlay: SHA-256-attested
                      monkey-patch adapters, flight recorder, shadow validation
  experiments/
    nccl_switchless_ring/   NCCL 2.29.7/2.30.7 patches + model-down four-rank
                            collective probes (run these before any model work)
    ...                     adaptive MTP controller, MoE round-floor, phase timing
  scripts/, tests/    per-edge cable qualification, CTest suite
APPROACH.md           phased bring-up rationale
docs/                 RESULTS.md, ARCHITECTURE.md, SETUP.md
```

## Getting started

1. Cable the ring cage-matched (four DAC cables), one dedicated /24 per link, MTU 9000, RoCEv2 GID index 3.
2. Qualify every edge with `spark_transport/scripts/qualify_direct_cable.py`: exactly 200 Gb/s on both ports, verified 12 KB and 16 KB RC writes in both directions, default 20 us p99 target ([spark_transport/CABLE_QUALIFICATION.md](spark_transport/CABLE_QUALIFICATION.md)).
3. Build the patched NCCL 2.30.7 and `libspark_transport_capi.so` for sm_121; record your own SHA-256s (the launcher pins them).
4. Download the checkpoint on one node, rsync it to the other three over the 200 G fabric.
5. Run the orchestrator preflight, then execute; it verifies artifacts, runs model-down probe gates, attests the runtime, and fails closed.

The full bring-up sequence, environment reference, and verification gates are
in [docs/SETUP.md](docs/SETUP.md).

## Status

This is a research pre-release. There is no stable API; environment flags,
ABIs, and module layouts change without notice.

The runtime the published numbers were measured on was built from a private
vLLM fork inside a community GB10 container image; that fork is not included
here. This is survivable by design: the overlay attests the exact upstream
source it patches by SHA-256 and fails closed on mismatch, so it will refuse
to run against sources it does not recognize rather than silently misbehave.

Honest caveat: a clean-room reproduction from this public snapshot has not
yet been verified end-to-end. The setup guide flags every reconstructed step,
and fresh builds of the patched NCCL and transport library will produce new
hashes that must be re-pinned everywhere the launcher enforces them.

## License

Apache-2.0. See [LICENSE](LICENSE).

The NCCL patch files contain portions of NVIDIA NCCL (Apache-2.0); the serve
scripts embed excerpts of vLLM (Apache-2.0); the switchless skip-Tree/PAT
approach originates with Joseph Rose's `nccl-spark-switchless`. SparkRing is
not a fork of vLLM or NCCL. Full attribution and relationship statements:
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
