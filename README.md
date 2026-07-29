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

## What is SIRCL?

**SIRCL** is our internal codename for the **S**witchless
**I**nference **R**DMA **C**ollective **L**ayer inside SparkRing. It is the
low-level direct-cable runtime:
persistent RDMA sessions, registered mapped-host arenas, sequence and
acknowledgement protocol, GPU copy/reduction operations, bounded asynchronous
submission, and CUDA-graph-replayable command rings.

SparkRing is the complete inference stack built around that layer. SparkCache
is separate: it persists rank-local KV state to NVMe and does not define the
live collective transport. See [docs/SIRCL.md](docs/SIRCL.md) for the exact
boundary, implemented surfaces, fallback contract, and generic-backend
direction.

## Start here: choose the capability you want

The measured system and the code this checkout can run are not the same lane.
As of 2026-07-29, **no end-to-end public-functional acceptance result exists**.
The recovered GLM-5.2 reference overlay is now published and wired into the
builder. A native ARM64 build, identical four-rank distribution, all public
startup gates, distributed initialization, complete model/MTP loading, B12X
prewarm, and KV allocation have passed. The corrected build has not yet served
and validated an API request, so do not interpret the published measurements
below as fresh-clone results.

| Goal | Minimum environment | Current status | Start here |
|---|---|---|---|
| Read, lint, or run Python contracts | Python 3.12, CPU | **Offline-validated:** 1,612 passed, 5 skipped from a clean tracked checkout | [Offline quickstart](#offline-contributor-quickstart) |
| Describe and inspect a four-Spark site | Python 3.10; SSH only for live preflight | Site schema validated; preflight is **read-only by construction** | [Site configuration](scripts/config/README.md) |
| Build native code or qualify cables | One Spark to build; two per cable; four for collective probes | Public source and clean-room probe evidence exist; hardware gates are manual | [Four-Spark transport bring-up](#four-spark-transport-bring-up) |
| Reproduce the headline serving numbers | Four Sparks plus the reference runtime | **Source available, reproduction pending:** 71 recovered vLLM changes are published and fail-closed; native build and partial four-rank bring-up passed, but no public-lane serving acceptance exists yet | [Runtime gaps](docs/RUNTIME_GAPS.md) |
| Try the public end-to-end serving lane | Four Sparks | **Candidate:** native build, identical four-rank distribution, preflight, model load, B12X prewarm, and KV allocation passed; corrected API/request acceptance remains | [Four-Spark quickstart](docs/QUICKSTART.md) |
| Study or port SparkCache | CPU for contracts; vLLM/DCP deployment for live use | Source published; live results were measured on the reference lane, not an accepted public runtime | [SparkCache](sparkcache/README.md) |

Machine-readable component status is in
[`docs/STATUS.json`](docs/STATUS.json). Coding agents should begin with
[`AGENTS.md`](AGENTS.md).

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
round, switching depth on a 32-round acceptance window. The public launch
enables true selected-depth drafting, so a K2 decision executes only two draft
steps rather than computing four and discarding the unused suffix. In the best observed
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
AGENTS.md             zero-context agent map, safety boundaries, task recipes
requirements-dev.txt  pinned GPU-free development and CI dependencies
sparkcache/           SparkCache: persistent NVMe context cache
                      (vLLM KV-Connector-V1 + DCP; topology-independent
                      design, 15-24x restore measured on the reference lane;
                      not LMCache). Self-contained
                      package, liftable without the transport ->
                      sparkcache/README.md
spark_transport/
  src/, include/      transport core -> libspark_transport_capi.so: verbs endpoints,
                      TP4 two-matching schedule, GPU doorbells, DCP query/combine,
                      vocabulary all-gather, graph command ring
  app/                RDMA and collective probe binaries (transport, TP2/TP4,
                      graph replay, DCP, vocabulary)
  integrations/vllm/  the fail-closed runtime overlay: SHA-256-attested
                      monkey-patch adapters, flight recorder, shadow validation
                      (SparkCache deploys beside these modules; it is
                      packaged at sparkcache/)
  experiments/
    nccl_switchless_ring/   NCCL 2.29.7/2.30.7 patches + model-down four-rank
                            collective probes (run these before any model work)
    ...                     adaptive MTP controller, MoE round-floor, phase timing
  scripts/, tests/    per-edge cable qualification, CTest suite
runtime/              candidate public runtime builder, lock, and public patches
scripts/              site schema, read-only preflight, fail-closed four-rank
                      launcher, acceptance gate, evidence collection, and tools
scripts/config/       sanitized site, launch, and acceptance-gate templates
docs/                 results, architecture, public-lane contract, setup
                      reconstruction, runtime gaps, and machine-readable status
```

## Fastest four-Spark path

You do **not** need to rebuild the complete ARM64 vLLM/CUDA stack just to try
SparkRing. The faststart lane uses the exact public GB10 image lineage from the
reference deployment, applies the recovered SparkRing changes fail-closed,
builds only patched NCCL plus the small native transport, and downloads the
pinned public checkpoint:

```bash
git clone https://github.com/FujitsuPolycom/sparkring.git
cd sparkring

OUTPUT_IMAGE=sparkring/glm52-faststart:trial \
  ./runtime/build-faststart.sh

./scripts/download-glm52.sh \
  /srv/models/GLM-5.2-MXFP4-Experts-GPTQ
```

The base is pinned by immutable ARM64 manifest digest:

```text
aidendle94/sparkrun-vllm-ds4-gb10@sha256:93824a946f1f0ad0867132a2c3809e0e7d8bec6ab38e7d0ef9fc3046e11bc8c7
```

Those are only the first two commands. Cabling, image/model fanout, exact
configuration fields, read-only preflight, eager first launch, logs, API test,
and the graph-enabled follow-up are all written out in the
**[four-Spark quickstart](docs/QUICKSTART.md)**.

The faststart path is a candidate awaiting a complete fresh-pull four-Spark
serving acceptance run. Its exact-source preimage gate is intentional: if the public base has
drifted or does not match the recovered source lineage, the build stops before
native compilation rather than producing a subtly wrong runtime.

## Offline contributor quickstart

**Safety: OFFLINE.** These commands do not contact a cluster or require a GPU.

```bash
python -m pip install -r requirements-dev.txt
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.11.0"
python -m pytest spark_transport sparkcache runtime scripts -q
ruff check --select E,F,W --ignore E501 --exclude runtime/patches .
```

## Site configuration and safe inspection

**Safety: OFFLINE, then READ-ONLY REMOTE.** The first two commands only parse
the sanitized example. The last command contacts the four hosts in your local
configuration but is guarded against remote mutation.

```bash
python scripts/sparkring_site.py scripts/config/site.example.yaml
python scripts/preflight.py --site scripts/config/site.example.yaml --print-plan
python scripts/sparkring_launcher.py \
  --site scripts/config/site.example.yaml \
  --launch-config scripts/config/launch.example.json plan

# After copying the example to the gitignored scripts/config/site.yaml,
# replacing every placeholder, and reviewing the printed plan:
python scripts/preflight.py --site scripts/config/site.yaml
```

For an actual cluster, start by copying only the two templates:

```bash
cp scripts/config/site.example.yaml scripts/config/site.yaml
cp scripts/config/launch.example.json scripts/config/launch.json
```

Edit the hosts, interfaces, addresses, artifact hashes, image digest and model
path in those two files, then use them with the same commands above. Planning
is connection-free and starting is still a dry run unless `start --execute`
is explicitly used. The validators identify the exact field needing attention;
users do not manually reproduce checksum, topology, model-layout, or capability
checks.

The launcher and acceptance gate are dry-run by default. The launcher emits
the exact per-rank `docker run` commands but makes no SSH connection unless
`--execute` is explicitly supplied. The acceptance gate validates the current
runtime lock, your site identity, and launcher commands, and reports any remaining
blocker without executing them. A successful plan is not an acceptance result.
Its `--execute` mode starts and stops the serving stack and is therefore
**STOPS SERVING**; never aim it at a production-serving cluster.

## Four-Spark transport bring-up

1. Cable the ring cage-matched (four DAC cables), one dedicated /24 per link, MTU 9000, RoCEv2 GID index 3.
2. Qualify every edge with the complete invocation in [the cable qualification guide](spark_transport/CABLE_QUALIFICATION.md#200g-roce-cable). The command requires both fabric IPs and both RDMA device names in addition to SSH targets and interfaces.
3. Run `runtime/build-faststart.sh` on one Spark. It applies the recovered
   Python layer, builds patched NCCL 2.30.7 and
   `libspark_transport_capi.so` for sm_121, then bakes and hashes them in one
   image.
4. Run `scripts/download-glm52.sh` on one node, then rsync the pinned
   checkpoint to the other three over the 200 G fabric.
5. Run the public read-only preflight and model-down probes. Copy the sanitized
   launch template, inspect `sparkring_launcher.py ... plan`, and use the
   acceptance gate for the eventual controlled model-down deployment. The
   historical reference orchestrator remains private; the clean-room public
   launcher does not depend on it.

**Safety:** cabling, host setup, network configuration, artifact staging, and
container lifecycle are **MUTATES HOST** operations. The full reference
deployment reconstruction and the subset runnable from this tree are in
[docs/SETUP.md](docs/SETUP.md).

## Status

This is a research pre-release. There is no stable API; environment flags,
ABIs, and module layouts change without notice.

There are two build lanes:

- **Faststart/reference-lineage lane (candidate):** the exact public ARM64
  community base used by the measured deployment plus the now-published,
  preimage-attested recovered overlay, patched NCCL, and SparkRing transport.
  It avoids the multi-hour vLLM/kernel rebuild. All inputs are public; a clean
  fresh-pull four-Spark acceptance is still pending.
- **Full source-reproducible lane (candidate):** what
  `runtime/build-runtime.sh` builds from pinned upstream source.
  The transport library, all probes, and the full native and Python test
  suites have been verified clean-room from this tree on DGX Spark hardware
  (stock CUDA devel image, no private artifacts: 36/36 targets and 20/20 native
  tests). The complete current GPU-free Python contract suite is 1,612 passed
  with 5 skipped. End-to-end serving from this lane is not yet
  performance-equivalent to the reference lane and is not supported until a
  full acceptance gate passes; fresh builds produce new artifact hashes that
  the fail-closed launcher must have re-pinned.

## License

Apache-2.0. See [LICENSE](LICENSE).

The NCCL patch files contain portions of NVIDIA NCCL (Apache-2.0); the serve
scripts embed excerpts of vLLM (Apache-2.0); the switchless skip-Tree/PAT
approach originates with Joseph Rose's `nccl-spark-switchless`. SparkRing is
not a fork of vLLM or NCCL. Full attribution and relationship statements:
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
