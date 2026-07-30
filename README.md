# SparkRing

SparkRing is a low-latency collective and inference-runtime stack for
switchless NVIDIA DGX Spark clusters.

It serves GLM-5.2 across four directly cabled Sparks using four ConnectX-7
200 Gb/s links in a ring. No Ethernet or InfiniBand switch is required for
the inference fabric.

SparkRing combines a ring-safe NCCL build, custom RDMA collectives,
CUDA-graph-replayable command rings, a fail-closed vLLM overlay, DCP4, and
adaptive MTP speculative decoding.

## Current configurations

| Lane | Configuration | Status |
|---|---|---|
| Documented reference | `aidendle94/GLM-5.2-MXFP4-Experts-GPTQ`, TP4/DCP4 | Published in the public quickstart |
| Active development | GLM-5.2 MXFP8/NVFP4/NF3 hybrid, TP4/DCP4 | Live candidate; public profile integration in progress |
| Transport | Four 200 Gb/s direct links, cycle `0-1-2-3-0` | Validated on four Sparks |
| API | OpenAI-compatible vLLM endpoint | Validated on the reference and NF3 lanes |

The project is a research pre-release. Pin a commit when deploying because
environment flags, source attestations, and integration ABIs can change.

## Measured results

### Active NF3-hybrid candidate

| Measurement | Result |
|---|---:|
| C1 warm coding sanity | **20.93 tok/s** |
| C2 warm coding sanity | **33.38 tok/s aggregate** |
| Reported KV capacity | **511,488 tokens** |
| KV allocation | 7,000,000,000 bytes/rank |
| CUDA workspace reserve | 805,306,368 bytes/rank |

These short sanity cells confirm the current live configuration. They do not
replace the complete context-by-concurrency benchmark matrix.

### Published reference measurements

| Measurement | Result |
|---|---:|
| Uncached prefill at 8K / 16K / 32K | **834 / 884 / 854 tok/s** |
| Shared-prefix sustained decode at C8 | **63.60 tok/s aggregate** |
| DCP4 C1 decode at 8K / 16K / 32K | **20.83 / 19.28 / 21.43 tok/s** |
| Five-run sequential coding median | **27.2 tok/s** |
| GPU-produced 16 KB RC write | **4.53 us p50** |
| Graph-replayed four-rank all-reduce | **about 39 us device time/call** |

Concurrency results are aggregate throughput. Shared-prefix measurements are
not unique-context capacity measurements.

See [docs/RESULTS.md](docs/RESULTS.md) for configurations, methodology,
evidence, and claim boundaries.

See [docs/TESTING_HISTORY.md](docs/TESTING_HISTORY.md) for experiments,
resolved failures, superseded configurations, and pending acceptance work.

## What SparkRing provides

- A model-agnostic direct-cable transport core.
- Two- and four-rank collective schedules.
- Registered mapped-host RDMA arenas.
- GPU doorbells and device-published command rings.
- CUDA-graph-compatible asynchronous submission.
- Custom all-reduce, all-gather, DCP query, DCP combine, and vocabulary paths.
- Numerical shadow validation and transport flight recording.
- A source-attested, fail-closed vLLM integration layer.
- Site configuration, topology discovery, cable qualification, and launch
  tooling.
- SparkCache for persistent rank-local KV state on NVMe.

Unsupported operations retain an explicit runtime fallback. Model-specific
features remain above the transport layer.

## Architecture

### Four-link topology

```text
            200 Gb/s
       S0 --------- S1
       |             |
200 Gb/s             200 Gb/s
       |             |
       S3 --------- S2
            200 Gb/s
```

A four-cycle decomposes into two perfect matchings:

```text
Round A: S0 <-> S1    S2 <-> S3
Round B: S0 <-> S3    S1 <-> S2
```

Both NIC cages can remain active while each collective completes in two
pairwise rounds.

### Ring-safe NCCL

Stock NCCL creates Tree and PAT connections between non-adjacent ranks. Those
pairs do not have Layer-2 adjacency on a switchless four-cycle.

SparkRing uses a checksum-pinned NCCL build that skips those connections and
advertises both local RoCE GIDs. Ring-feasible neighbors remain available.

### SIRCL

SIRCL is the **Switchless Inference RDMA Collective Layer** inside SparkRing.

It owns persistent RDMA sessions, mapped-host arenas, sequencing,
acknowledgements, GPU copy and reduction operations, bounded submission, and
CUDA-graph-replayable command rings.

SparkRing is the complete inference stack around SIRCL. SparkCache is a
separate persistent-context component.

See [docs/SIRCL.md](docs/SIRCL.md) and
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

### Fail-closed vLLM integration

SparkRing applies a thin overlay through `PYTHONPATH`. Each adapter verifies
the exact source SHA-256 and ABI it expects before installation.

A source mismatch stops startup. The orchestrator also attests native
libraries, mounts, launch arguments, and rank topology.

### DCP4 and adaptive MTP

DCP4 shards KV state across four ranks. Custom query all-gather and fused
online-softmax combine collectives serve the sparse-attention path.

Adaptive MTP selects two or four draft steps from a 32-round acceptance
window. A two-token decision executes two draft steps rather than computing
and discarding the longer suffix.

## Quickstart

The fastest public path builds a thin image on the pinned ARM64 GB10 base:

```bash
git clone https://github.com/FujitsuPolycom/sparkring.git
cd sparkring

OUTPUT_IMAGE=sparkring/glm52-faststart:trial \
  ./runtime/build-faststart.sh

./scripts/download-glm52.sh \
  /srv/models/GLM-5.2-MXFP4-Experts-GPTQ
```

The base image is pinned by ARM64 manifest digest:

```text
aidendle94/sparkrun-vllm-ds4-gb10@sha256:93824a946f1f0ad0867132a2c3809e0e7d8bec6ab38e7d0ef9fc3046e11bc8c7
```

Continue with the
[four-Spark quickstart](docs/QUICKSTART.md) for cabling, image and model
fanout, configuration, preflight, launch, logs, and API validation.

## Offline contributor quickstart

These commands do not contact a cluster or require a GPU:

```bash
python -m pip install -r requirements-dev.txt
python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  "torch==2.11.0"
python -m pytest spark_transport sparkcache runtime scripts -q
ruff check --select E,F,W --ignore E501 --exclude runtime/patches .
```

## Site configuration

Create local configuration from the sanitized templates:

```bash
cp scripts/config/site.example.yaml scripts/config/site.yaml
cp scripts/config/launch.example.json scripts/config/launch.json
```

Set hosts, interfaces, fabric addresses, hashes, image digest, and model path.
Then run the read-only preflight:

```bash
python scripts/sparkring_site.py scripts/config/site.yaml
python scripts/preflight.py --site scripts/config/site.yaml
python scripts/sparkring_launcher.py \
  --site scripts/config/site.yaml \
  --launch-config scripts/config/launch.json \
  plan
```

Planning is connection-free. Preflight is read-only. Starting containers
requires an explicit `start --execute`.

See [scripts/config/README.md](scripts/config/README.md) and
[docs/SETUP.md](docs/SETUP.md).

## Transport bring-up

1. Cable the four-cycle with cage-matched DACs.
2. Configure one dedicated subnet per link, MTU 9000, and RoCEv2.
3. Qualify every edge with the
   [cable guide](spark_transport/CABLE_QUALIFICATION.md).
4. Build and distribute one exact image.
5. Download and distribute one exact model checkpoint.
6. Run read-only preflight and model-down collective probes.
7. Launch through the fail-closed orchestrator.

Cabling, network configuration, artifact staging, and container lifecycle
mutate hosts. Review generated plans before execution.

## Repository map

```text
AGENTS.md              agent map and safety boundaries
runtime/               pinned runtime builders and public patches
scripts/               site, preflight, launch, and evidence tools
scripts/config/        sanitized site and launch templates
spark_transport/       SIRCL transport, probes, tests, and vLLM adapters
sparkcache/            persistent NVMe context cache
docs/ARCHITECTURE.md   transport and runtime design
docs/QUICKSTART.md     complete four-Spark bring-up
docs/RESULTS.md        measured results and claim boundaries
docs/TESTING_HISTORY.md experiment and regression chronology
docs/STATUS.json       machine-readable component status
```

Coding agents should begin with [AGENTS.md](AGENTS.md).

## Build lanes

The faststart lane applies source-attested SparkRing changes to a pinned
public ARM64 base. It avoids rebuilding the complete vLLM and CUDA stack.

The source-reproducible lane builds from pinned upstream source through
`runtime/build-runtime.sh`.

Machine-readable component status is in
[docs/STATUS.json](docs/STATUS.json). Runtime lineage and reproduction
boundaries are in [docs/RUNTIME_GAPS.md](docs/RUNTIME_GAPS.md).

## License

Apache-2.0. See [LICENSE](LICENSE).

The NCCL patch files include NVIDIA NCCL code under Apache-2.0. Serve scripts
include vLLM excerpts under Apache-2.0.

The switchless skip-Tree/PAT approach originates with Joseph Rose's
`nccl-spark-switchless`. Full attribution is in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
