# SparkRing

SparkRing is a low-latency collective transport and inference runtime for switchless GB1X (NVIDIA DGX Spark) clusters.

Today it runs GLM-5.2 across four directly connected DGX Sparks. Four 200 Gb/s ConnectX-7 links form a physical ring, 
with no external Ethernet or InfiniBand switch in the inference fabric.

The stack combines SIRCL custom RDMA collectives, CUDA-graph-replayable command rings, a source-attested and fail-closed vLLM overlay, 
DCP4, support for fixed and adaptive MTP speculative decoding, and a patched ring-safe NCCL fallback for communication not yet handled by the custom path.

The long-term goal is a model-agnostic runtime for efficient, switchless multi-node inference on DGX Spark.

Use the [documentation map](docs/README.md) to find the canonical specification,
runnable procedure, evidence record, or historical reference for a task.

## Acknowledgements

SparkRing would not exist without the months of research, profiling, kernel development, runtime patching, model conversion, and operational testing shared by the RTX 6000 Pro inference community.
https://github.com/local-inference-lab/

It builds heavily on work from the contributors behind B12X, SparkInfer, vLLM, the GLM model and quantization ecosystem, and the broader NVIDIA inference community. 

SparkRing’s contribution is to adapt, integrate, and extend those foundations for low-latency inference across switchless DGX Spark clusters.

Detailed project and contributor credits are maintained in the acknowledgements and provenance documentation.

## Choose a deployment profile

| Goal | Profile | Maturity and evidence scope | Start here |
|---|---|---|---|
| Use the operator-accepted 3.5-bpw configuration | EXL3 3.5-bpw fixed-MTP4 (`R7`) | Accepted on one four-Spark appliance; a clean rebuild still requires live qualification | [3.5-bpw quickstart](docs/EXL3_R7_QUICKSTART.md) |
| Use the reproducible public default | EXL3 3.25-bpw plus LMCache CS512 | Clean-checkout bounded live validation on four Sparks | [public-default quickstart](docs/QUICKSTART.md) |
| Use the deterministic alternative | NF3 | Accepted public-functional alternative | [NF3 quickstart](docs/NF3_QUICKSTART.md) |

## Featured model: EXL3 R7 3.5-bpw fixed-MTP4

The operator's accepted 3.5-bpw SparkRing profile uses
[`brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78`](https://huggingface.co/brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78)
at immutable revision `9ab9579774cc432df91567a36f6e9e863e0d4c9f`. Four directly
cabled DGX Sparks serve it with TP4/DCP4, fixed MTP4, 9.25 GB of KV memory per
rank, dynamic per-token NVFP4 latent KV plus FP8 RoPE, a 262,144-token request
limit, a 4,096-token prefill ceiling, and native SparkRing TP transport through
Q40. Eligible pure-prefill work uses the B12X transient full-CKV DCP gather.
The target-only exact-Q40 routed-MoE state uses capacity 40 and route block 8;
Q1-Q32, other prefill shapes, and the draft model retain their prior states.

| Prefill context | Prompt tokens | TTFT | C1 prefill | Samples |
|---|---:|---:|---:|---:|
| 8K | 8,194 | 12.06 s | **679 tok/s** | 2 |
| 16K | 16,386 | 24.36 s | **673 tok/s** | 1 |
| 32K | 32,770 | 49.17 s | **666 tok/s** | 1 |
| 64K | 65,538 | 99.72 s | **657 tok/s** | 1 |
| 128K | 131,074 | 203.09 s | **645 tok/s** | 1 |

| Aggregate decode tok/s | C1 | C2 | C4 | C8 |
|---|---:|---:|---:|---:|
| 4K context | 22.6 | 32.7 | 50.3 | **78.4** |
| 8K context | 22.0 | 35.3 | 51.9 | **71.3** |
| 16K context | 21.3 | 32.9 | 49.2 | **70.0** |
| 32K context | 20.4 | 32.3 | 45.6 | **65.5** |
| 64K context | 21.4 | 30.4 | 47.2 | **67.8** |

#### Coding benchmark (C1)

| Workload | Runs | Median tok/s | Mean tok/s | Maximum tok/s | CJK runs |
|---|---:|---:|---:|---:|---:|
| Coding peak | **5/5** | **27.3** | **27.3** | **28.8** | **0** |

Reported KV capacity is **1,156,864 tokens** across the four-rank serving profile.

The matched exact-Q40 decode bracket replayed the same eight unique 16K
payloads with full 8/8 residency and a 25-second measurement window. Its
73.208 tok/s candidate mean was 19.341% above the 61.344 tok/s control mean;
the slower candidate repeat exceeded the fastest control repeat by 14.93%.
All 75 target layers passed exact BF16 parity, deterministic 16K and 32K output
equality passed, and final graph, API, transport, and capacity gates passed.

The prefill and decode tables form the accepted operator performance matrix
snapshot. Prefill used 100% unique generated contexts. Several operator runs
produced similar throughput, but only the displayed prefill sample counts and
five coding-probe repeats are retained here; the decode cells are not presented
as repeat distributions. Server-side cached-token accounting was unavailable
for the prefill cells, so cache misses are not independently proven by the
benchmark artifact.
The predeclared exact-Q40 prefill reducer separately remains a machine failure:
its sole primary miss was 0.1215% at 64K. Operator acceptance treats that
bounded difference as measurement-neutral without relabelling the machine
result as a pass.

This configuration is the **accepted operator default for 3.5-bpw EXL3**. Its
acceptance scope is one four-Spark appliance; it is not the reproducible
public-functional default or an accepted public deployment matrix. Read the
[fixed-MTP4 specification](docs/EXL3_R7_FIXED_MTP4_PROFILE.md),
[optimization record](docs/EXL3_R7_OPTIMIZATION_20260811.md), and
[machine-readable operator acceptance](docs/configurations/glm52-exl3-r7-mtp4-q40-block8-20260812.json)
before reproducing or extending it.
The accepted performance matrix is preserved separately in
[machine-readable form](docs/configurations/glm52-exl3-r7-current-best-matrix-20260813.json).
The clean source/profile composition and local build commands are in
[the operator-profile reproduction guide](docs/EXL3_R7_OPERATOR_REPRODUCTION.md).
Use the [3.5-bpw quickstart](docs/EXL3_R7_QUICKSTART.md) for the executable
path and the [promotion checklist](docs/EXL3_R7_PROMOTION_CHECKLIST.md) before
transferring acceptance to a rebuilt image.

A live-validated LMCache NVMe candidate extension preserved the exact R7/Q40
serving contract while adding a lazy 512 MiB L1 and bounded 50 GiB O_DIRECT L2
per rank. One 32,506-token cold publication populated 63 NVMe chunks per rank
and measured 56.115 seconds of client TTFT. After LMCache-server and engine
restart cleared both volatile L1 and native vLLM prefix state, the identical
prompt measured 1.477 seconds with a 99.2% external-cache hit, 0.0% native-
prefix hit, and zero L1 data bytes. This single-pair result proves attributed
NVMe persistence, not a latency distribution or deterministic-output gate.
The extension remains a candidate and does not change the accepted operator
profile or public-functional default. See the
[R7 LMCache evidence](docs/configurations/glm52-exl3-r7-lmcache-nvme-20260813.json).

## Public default: EXL3 3.25-bpw with LMCache CS512

SparkRing's reproducible, main-advertised public-functional configuration is
[`willfalco/GLM-5.2-EXL3-TR3-3.25bpw`](https://huggingface.co/willfalco/GLM-5.2-EXL3-TR3-3.25bpw)
at immutable revision `d7d79c2d14599dfce7a5d12b85f7ad73f40e623d`.

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
release-acceptance claim.

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

### EXL3 3.25-bpw + LMCache CS512

| Item | Default public recipe |
|---|---|
| Model | `willfalco/GLM-5.2-EXL3-TR3-3.25bpw@d7d79c2...` |
| Parallelism | TP4 / DCP4, fixed MTP2 |
| Batch/graph contract | C8, Q32, 4,096 batched tokens |
| Context/KV | 524,288 model limit; 4.5 GB/rank; 562,688 reported tokens |
| KV representation | NVFP4 latent plus FP8 RoPE |
| Cache | native prefix cache + LMCache CS512; SparkCache disabled |
| Public maturity | clean-checkout live-validated; default and main advertised configuration |

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

### Archived and alternative configurations

NF3 remains documented as a deterministic public-functional alternative, but
it is no longer a featured README profile. Use the
[NF3 quickstart](docs/NF3_QUICKSTART.md),
[NVFP4 KV validation receipt](docs/NF3_NVFP4_PUBLIC_VALIDATION.md), and
[one-million-token operator audit](docs/NF3_LIVE_CONFIGURATION_20260731.md) for
its reproducible recipe and historical evidence. The former Aiden MXFP4/GPTQ
reference remains in the [historical lane](docs/history/AIDEN_MXFP4_GPTQ.md).

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
docs/GENERIC_RUNTIME.md  generic profile-driven runtime launcher (offline-validated)
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
