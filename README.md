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

SparkRing's default, main advertised, and currently running public-functional
configuration is
[`willfalco/GLM-5.2-EXL3-TR3-3.25bpw`](https://huggingface.co/willfalco/GLM-5.2-EXL3-TR3-3.25bpw)
at immutable revision
`d7d79c2d14599dfce7a5d12b85f7ad73f40e623d`, with LMCache CS512.

| Item | Configuration | Status |
|---|---|---|
| Model | EXL3/Trellis 3.25 bpw, legacy `per_expert_v1` | Clean-checkout live-validated on four DGX Sparks |
| Parallelism | TP4/DCP4, fixed MTP2 | Validated |
| Transport | Four 200 Gb/s direct links, cycle `0-1-2-3-0` | Validated |
| API | OpenAI-compatible vLLM endpoint | Validated |
| Cache | native prefix cache + one LMCache CS512 server/rank; SparkCache disabled | Bounded live validation |
| Public bootstrap | Receipt-gated EXL3 derived image | Clean-checkout identical-image four-Spark run validated |

The exact image ID
`sha256:20c4099f2e7e3dd3c8ab64f7d7930bde4f372df1895aa3ffa593252ca04ae96f`
was deployed identically on all four ranks. The run passed 116/116 post-stop
preflight checks, started four engines and four LMCache servers with zero
restarts, captured 16/16 piecewise and 12/12 full graphs, and served a
524,288-token model limit with 562,688 reported KV tokens. Five consecutive
bounded gates passed; ten fixed-seed 128-token completions were byte-identical.
See the [EXL3 quickstart](docs/EXL3_QUICKSTART.md) and
[evidence-scoped recipe](docs/EXL3_RECIPE.md).

This makes EXL3+LMCache the public default, not a blanket correctness or
release-acceptance claim. NF3 remains an accepted deterministic alternative;
its quickstart is [here](docs/NF3_QUICKSTART.md).

The maintainer's later one-million-token NF3 operator profile is captured
separately in the
[live configuration audit](docs/NF3_LIVE_CONFIGURATION_20260731.md). It records
the effective command, environment, rank-local transport order, immutable
runtime hashes, 9 GB/rank NVFP4+FP8-RoPE KV allocation, and reported 1,125,632
token capacity. It is a live-observed configuration snapshot, not a replacement
for the smaller accepted public-bootstrap defaults above.

The dated
[DCP4 fixed-MTP2 live recipe](docs/NF3_FIXED_MTP2_RECIPE_20260801.md) records a
later operator variant and its recovered benchmark artifact: the effective
four-rank process state, exact delta from the full audit, sanitized rerun
command, 732-779 prefill tok/s, up to 60.4 aggregate decode tok/s, and the
capacity-limited C8 cells that must not be quoted as valid throughput. It is a
public-functional, live-validated operator snapshot, not a reference-lane or
current-checkout result.

The exact EXL3 model hashes, source pins, environment, vLLM arguments, and
Q4096/C8/Q32 contract are in
[`recipes/glm52-exl3-tr3-3.25bpw.json`](recipes/glm52-exl3-tr3-3.25bpw.json).

The [fixed-MTP2 EXL3 alternative](docs/EXL3_FIXED_MTP2_RECIPE_20260802.md)
records the exact four-Spark operator overlay and its duration-based
16K-128K/C1-C8 matrix. It is live-validated external evidence, not
clean-checkout public acceptance, correctness acceptance, or a reference-lane
result.

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

### EXL3 3.25-bpw + LMCache CS512

| Item | Current live recipe |
|---|---|
| Model | `willfalco/GLM-5.2-EXL3-TR3-3.25bpw@d7d79c2...` |
| Parallelism | TP4 / DCP4, fixed MTP2 |
| Batch/graph contract | C8, Q32, 4,096 batched tokens |
| Context/KV | 524,288 model limit; 4.5 GB/rank; 562,688 reported tokens |
| KV representation | NVFP4 latent plus FP8 RoPE |
| Cache | native prefix cache + LMCache CS512; SparkCache disabled |
| Public maturity | clean-checkout live-validated; main advertised/current-running configuration |

Inspect either recipe without contacting the cluster:

```bash
python scripts/sparkring_recipe.py list
python scripts/sparkring_recipe.py plan
```

For maintainers advancing EXL3 from bounded live validation toward full
public-functional acceptance, use the dry-run-first
[EXL3 + LMCache acceptance runbook](docs/EXL3_ACCEPTANCE_RUNBOOK.md). The
workflow is published and offline-validated; its remaining live gates are not
yet an acceptance claim.

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

python scripts/bootstrap_exl3.py plan \
  --site scripts/config/site.yaml

python scripts/bootstrap_exl3.py execute \
  --site scripts/config/site.yaml \
  --no-launch \
  --confirmation BOOTSTRAP-EXL3-ALL-FOUR
```

The bootstrap is resumable and fail-closed. It reuses complete model files and
existing exact images, adopts or resumes the pinned 81-shard model,
reconstructs the pinned ExLlamaV3, SparkInfer, and LMCache trees, builds one
ARM64 derived image on rank 0, and fans model/image bytes over the 200 GbE
ring. `--no-launch` leaves the generated contract ready for review before a
serving cutover.

See [QUICKSTART.md](docs/QUICKSTART.md) for the shortest setup path and
[EXL3_QUICKSTART.md](docs/EXL3_QUICKSTART.md) for the full receipt, cabling, site fields,
review, launch, tail, and bounded gate commands. Start with
[PREREQUISITES.md](docs/PREREQUISITES.md) on a new cluster. NF3 remains an
accepted deterministic alternative through the
[NF3 quickstart](docs/NF3_QUICKSTART.md).

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

Create local site configuration from the sanitized template:

```bash
cp scripts/config/site.example.yaml scripts/config/site.yaml
```

Set hosts, interfaces, fabric addresses, hashes, image digest, and model path.
Then follow the default [EXL3 quickstart](docs/QUICKSTART.md), which generates
the exact ignored EXL3 launch profile. The checked-in `launch.example.json`
and `gate.example.json` belong to the accepted NF3 alternative; they are not
EXL3 templates.

Run the read-only site preflight:

```bash
python scripts/sparkring_site.py scripts/config/site.yaml
python scripts/preflight.py --site scripts/config/site.yaml
```

Recipe and launcher planning are connection-free. Preflight is read-only.
Starting containers requires an explicit `--execute` operation.

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
docs/QUICKSTART.md     default EXL3 four-Spark bring-up
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
