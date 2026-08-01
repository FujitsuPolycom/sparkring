# SparkRing

SparkRing is a low-latency collective transport and inference runtime for switchless GB1X (NVIDIA DGX Spark) clusters.

Today it runs GLM-5.2 across four directly connected DGX Sparks. Four 200 Gb/s ConnectX-7 links form a physical ring, 
with no external Ethernet or InfiniBand switch in the inference fabric.

The stack combines SIRCL custom RDMA collectives, CUDA-graph-replayable command rings, a source-attested and fail-closed vLLM overlay, 
DCP4, support for fixed and adaptive MTP speculative decoding, and a patched ring-safe NCCL fallback for communication not yet handled by the custom path.

The long-term goal is a model-agnostic runtime for efficient, switchless multi-node inference on DGX Spark.

## Acknowledgements

SparkRing would not exist without the months of research, profiling, kernel development, runtime patching, model conversion, and operational testing shared by the RTX 6000 Pro inference community.
https://github.com/local-inference-lab/vllm

It builds heavily on work from the contributors behind B12X, SparkInfer, vLLM, the GLM model and quantization ecosystem, and the broader NVIDIA inference community. 

SparkRing’s contribution is to adapt, integrate, and extend those foundations for low-latency inference across switchless DGX Spark clusters.

Detailed project and contributor credits are maintained in the acknowledgements and provenance documentation.
## Current deployment

SparkRing's default public-functional target is:
[`madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid`](https://huggingface.co/madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid)
at immutable revision
`66f3623dd8fefb5ca8046706912d5d31c8d196af`.

| Item | Configuration | Status |
|---|---|---|
| Model | MXFP8/NVFP4/NF3 hybrid | Validated on four DGX Sparks |
| Parallelism | TP4/DCP4, adaptive MTP2/4 | Validated |
| Transport | Four 200 Gb/s direct links, cycle `0-1-2-3-0` | Validated |
| API | OpenAI-compatible vLLM endpoint | Validated |
| Public bootstrap | Pinned ARM64 base + thin local NF3 image | Clean-checkout four-Spark run validated |

The maintainer's later one-million-token NF3 operator profile is captured
separately in the
[live configuration audit](docs/NF3_LIVE_CONFIGURATION_20260731.md). It records
the effective command, environment, rank-local transport order, immutable
runtime hashes, 9 GB/rank NVFP4+FP8-RoPE KV allocation, and reported 1,125,632
token capacity. It is a live-observed configuration snapshot, not a replacement
for the smaller accepted public-bootstrap defaults above.

A second, non-default
[`willfalco/GLM-5.2-EXL3-TR3-3.25bpw`](https://huggingface.co/willfalco/GLM-5.2-EXL3-TR3-3.25bpw)
recipe now records the maintainer's live EXL3 configuration. Its exact model
hashes, source pins, environment, vLLM arguments, Q32/C8 graph contract, and
1M/9-GB KV profile are in
[`recipes/glm52-exl3-tr3-3.25bpw.json`](recipes/glm52-exl3-tr3-3.25bpw.json).
The configuration is live-validated. Its public, receipt-gated source bootstrap
is now offline-validated; the remaining gate is a clean-checkout four-Spark
run. See [the EXL3 recipe and bootstrap](docs/EXL3_RECIPE.md). NF3 remains the
default, fully live-validated quickstart.

The former Aiden MXFP4/GPTQ reference is preserved in
[the historical lane document](docs/history/AIDEN_MXFP4_GPTQ.md). It is not a
second supported deployment target.

The project is a research pre-release. Pin a commit when deploying because
environment flags, source attestations, and integration ABIs can change.

Before downloading the model or building an image, complete the
**[exhaustive prerequisites checklist](docs/PREREQUISITES.md)**. It separates
facts the operator must supply from hardware/network details a bot can discover
and provides the exact validation sequence.

## Measured results

### NF3-hybrid

Two KV layouts are available for the same NF3 checkpoint and launch:

| Bootstrap profile | KV capacity observed | Status |
|---|---:|---|
| `fp8` (default) | 511,488 tokens | public bootstrap profile |
| `nvfp4-rope8` | **875,520 tokens** | clean-checkout public bootstrap validated |

The optional profile stores the compressed latent KV in NVFP4 with per-token
scaling while retaining FP8 RoPE data. It changes neither the model download
nor the TP4/DCP4/MTP/C8/Q40 serving policy. Its public clean-checkout bootstrap
has now built, attested, distributed, launched, captured every configured CUDA
graph, and served a deterministic API request on four Sparks. See the
[validation receipt](docs/NF3_NVFP4_PUBLIC_VALIDATION.md).

| Measurement | Result |
|---|---:|
| C1 warm coding reference | **22 tok/s** |
| C2 warm coding sanity | **33.38 tok/s aggregate** |
| Reported KV capacity | **511,488 tokens** |
| KV dtype | FP8 |
| KV allocation | 7,000,000,000 bytes/rank |
| CUDA workspace reserve | 805,306,368 bytes/rank |

The live NF3 gate also completed all CUDA captures, served a 512-token decode
while an 18,562-token prefill was active, and returned both requests with
post-test health intact. These are stability and sanity measurements, not a
complete performance matrix.

The older coherent GPTQ matrix and its exact configuration are retained on
[the historical Aiden lane](docs/history/AIDEN_MXFP4_GPTQ.md).

### EXL3 3.25-bpw candidate

| Item | Current live recipe |
|---|---|
| Model | `willfalco/GLM-5.2-EXL3-TR3-3.25bpw@d7d79c2...` |
| Parallelism | TP4 / DCP4, fixed MTP3 |
| Batch/graph contract | C8, Q32, 4,096 batched tokens |
| Context/KV | 1,048,576 model limit; 9 GB/rank; 1,125,632 reported tokens |
| KV representation | NVFP4 latent plus FP8 RoPE |
| Public maturity | live configuration; public source bootstrap offline-validated |

Inspect either recipe without contacting the cluster:

```bash
python scripts/sparkring_recipe.py list
python scripts/sparkring_recipe.py plan --recipe glm52-exl3-tr3-3.25bpw
```

Transport-only results, historical DCP1 peaks, workload-specific coding
measurements, and superseded configurations remain documented separately in
[docs/RESULTS.md](docs/RESULTS.md) and
[docs/TESTING_HISTORY.md](docs/TESTING_HISTORY.md).

See [docs/RESULTS.md](docs/RESULTS.md) for configurations, methodology,
evidence, and claim boundaries.

See [docs/TESTING_HISTORY.md](docs/TESTING_HISTORY.md) for experiments,
resolved failures, superseded configurations, and pending acceptance work.

## What SparkRing provides

- A GLM/vLLM-oriented direct-cable transport implementation, with reusable
  low-level transport primitives.
- Two- and four-rank collective schedules used by the current GLM/vLLM paths.
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
                 operator or bot
                       |
          Wi-Fi / LAN / USB / Tailscale
             |      |      |      |
             S0     S1     S2     S3

                   200 Gb/s
              S0 ========== S1
              ||             ||
     200 Gb/s ||             || 200 Gb/s
              ||             ||
              S3 ========== S2
                   200 Gb/s

       Management: SSH, downloads, launch, API
       200 GbE ring: RDMA inference traffic only
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

Clone once on rank 0, fill in only your hosts/interfaces/paths, inspect the
plan, then execute:

```bash
git clone https://github.com/FujitsuPolycom/sparkring.git
cd sparkring
cp scripts/config/site.example.yaml scripts/config/site.yaml
$EDITOR scripts/config/site.yaml

python scripts/bootstrap_nf3.py plan \
  --site scripts/config/site.yaml

python scripts/bootstrap_nf3.py execute \
  --site scripts/config/site.yaml \
  --confirmation BOOTSTRAP-NF3-ALL-FOUR
```

Add `--profile nvfp4-rope8` to select the larger-capacity live candidate. The
same command reuses complete model/draft downloads and cached NF3 layers.

The bootstrap is resumable and fail-closed. It reuses complete model files and
existing exact images, otherwise it pulls the pinned public ARM64 base,
downloads and verifies the NF3 target plus MTP draft on all four ranks, fetches
the exact B12X and Spark-port commits, builds one thin derived image, fans that
exact image ID to the other ranks, verifies the generated receipt, runs
preflight, and launches the validated C8/Q40 profile with SparkCache disabled.
It does not rebuild Torch, vLLM, FlashInfer, or the base kernel stack.

See the [four-Spark quickstart](docs/QUICKSTART.md) for cabling, site fields,
tail commands, and acceptance checks. Start with
[PREREQUISITES.md](docs/PREREQUISITES.md) on a new cluster.

To prepare the non-default EXL3 profile instead, reuse the same completed
`site.yaml` and run its independent plan:

```bash
python scripts/bootstrap_exl3.py plan \
  --site scripts/config/site.yaml
```

The EXL3 bootstrap adopts or resumes the pinned 81-shard model, reconstructs
the exact public ExLlamaV3 and SparkInfer trees, builds one ARM64 derived image
on rank 0, and fans model/image bytes over the 200 GbE ring. Its execute path is
documented in [EXL3_RECIPE.md](docs/EXL3_RECIPE.md); it remains a candidate
until the clean-checkout four-Spark gate is recorded.

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
