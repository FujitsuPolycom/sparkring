# SparkRing

SparkRing is a low-latency collective transport and inference runtime for switchless GB1X (NVIDIA DGX Spark) clusters.

Four 200 Gb/s ConnectX-7 links form a physical ring across four directly
connected DGX Sparks, with no external Ethernet or InfiniBand switch in the
inference fabric. Models are served over that ring as tensor-parallel
deployments; which models, and with what evidence, is recorded in the
[validated-profiles registry](docs/profiles/README.md).

The stack combines SIRCL custom RDMA collectives, CUDA-graph-replayable command rings, a source-attested and fail-closed vLLM overlay,
DCP4, support for fixed and adaptive MTP speculative decoding, and a patched ring-safe NCCL fallback for communication the custom path does not handle.
A pip-installable plugin (`sparkring_plugin/`) packages the vLLM adapters as a `vllm.general_plugins` entry point: fail-closed like the overlay, but feature-detected rather than source-attested, and offline-validated only.

Serving admission is model-agnostic: the transport carries no
model-specific policy, and models qualify against it through the
shadow-mode comparison windows recorded per profile (admission mechanics:
the [vLLM integration README](spark_transport/integrations/vllm/README.md)).

Use the [documentation map](docs/README.md) to find the canonical specification,
runnable procedure, evidence record, or historical reference for a task.

## Acknowledgements

SparkRing adapts, integrates, and extends foundations from the
[RTX 6000 Pro inference community](https://github.com/local-inference-lab/)
and the contributors behind B12X, SparkInfer, vLLM, the GLM model and
quantization ecosystem, and the broader NVIDIA inference community.
Detailed credits are in the acknowledgements and provenance documentation.

## Validated profiles

A profile is one model identity plus the serving configuration it was
validated with on the ring. The
[validated-profiles registry](docs/profiles/README.md) is the canonical
index; the rows here are the deployment entry points.

| Goal | Profile | Maturity and evidence scope | Start here |
|---|---|---|---|
| Use the operator-accepted 3.5-bpw configuration | GLM-5.2 EXL3 3.5-bpw fixed-MTP4 (`R7`) | Accepted on one four-Spark appliance; a clean rebuild requires live qualification | [3.5-bpw quickstart](docs/EXL3_R7_QUICKSTART.md) |
| Use the reproducible public default | GLM-5.2 EXL3 3.25-bpw plus LMCache CS512 | Clean-checkout bounded live validation on four Sparks | [public-default quickstart](docs/QUICKSTART.md) |
| Use the deterministic alternative | GLM-5.2 NF3 | Accepted public-functional alternative | [NF3 quickstart](docs/NF3_QUICKSTART.md) |
| Serve DeepSeek-V4-Flash-0731 | DeepSeek V4 Flash, official FP8, native DSpark speculation | Functional launch with operator-observed performance; not shadow-qualified | [DeepSeek quickstart](docs/DEEPSEEK_V4_FLASH_QUICKSTART.md) |
| Qualify a non-GLM width over the ring | Width-generic admission plus a per-model profile page | Evidence scope varies by profile; the small-model shadow set is the validation instrument | [profiles registry](docs/profiles/README.md) |

## Flagship profile: GLM-5.2 EXL3 R7 3.5-bpw fixed-MTP4

The operator's accepted 3.5-bpw profile serves
[`brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78`](https://huggingface.co/brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78)
at immutable revision `9ab9579774cc432df91567a36f6e9e863e0d4c9f` with TP4/DCP4,
fixed MTP4, and native SparkRing TP transport through Q40. Headline
operator-matrix numbers: 645-679 tok/s C1 prefill from 8K through 128K
contexts, 65-78 tok/s aggregate decode at C8, and a 1,156,864-token KV
capacity. Acceptance is scoped to one four-Spark appliance; it is not the
reproducible public default.

- Full performance matrix, matched Q40 comparison, and claim boundaries:
  [measured results](docs/RESULTS.md) and the
  [machine-readable matrix](docs/configurations/glm52-exl3-r7-current-best-matrix-20260813.json)
- Serving contract: [fixed-MTP4 specification](docs/EXL3_R7_FIXED_MTP4_PROFILE.md)
  and the [optimization record](docs/EXL3_R7_OPTIMIZATION_20260811.md)
- Executable path: [3.5-bpw quickstart](docs/EXL3_R7_QUICKSTART.md), then the
  [promotion checklist](docs/EXL3_R7_PROMOTION_CHECKLIST.md) before
  transferring acceptance to a rebuilt image
- LMCache NVMe persistence candidate (single-pair evidence, does not change
  the accepted profile): [R7 LMCache evidence](docs/configurations/glm52-exl3-r7-lmcache-nvme-20260813.json)

## Public default: EXL3 3.25-bpw with LMCache CS512

SparkRing's reproducible, main-advertised public-functional configuration is
[`willfalco/GLM-5.2-EXL3-TR3-3.25bpw`](https://huggingface.co/willfalco/GLM-5.2-EXL3-TR3-3.25bpw)
at immutable revision `d7d79c2d14599dfce7a5d12b85f7ad73f40e623d`. The `CS512`
label denotes the LMCache `chunk_size: 512` setting pinned in the recipe.

| Item | Configuration | Status |
|---|---|---|
| Model | EXL3/Trellis 3.25 bpw, `per_expert_v1` rotation layout | Clean-checkout live-validated on four DGX Sparks |
| Parallelism | TP4/DCP4, fixed MTP2 | Validated |
| Transport | Four 200 Gb/s direct links, cycle `0-1-2-3-0` | Validated |
| API | OpenAI-compatible vLLM endpoint | Validated |
| Cache | native prefix cache + one LMCache CS512 server/rank; SparkCache disabled | Bounded live validation |
| Public bootstrap | Receipt-gated EXL3 derived image | Clean-checkout identical-image four-Spark run validated |

Validated image identity:
`sha256:20c4099f2e7e3dd3c8ab64f7d7930bde4f372df1895aa3ffa593252ca04ae96f`,
served identically on all four ranks through the full bounded-gate
sequence; the run receipt (preflight, graph capture, KV accounting,
byte-identical fixed-seed outputs) is in the
[evidence-scoped recipe](docs/EXL3_RECIPE.md), with the executable path in
the [EXL3 quickstart](docs/EXL3_QUICKSTART.md).

This evidence qualifies EXL3+LMCache as the public default; it is not a
blanket correctness or release-acceptance claim.

The exact EXL3 model hashes, source pins, environment, vLLM arguments, and
Q4096/C8/Q32 contract are in
[`recipes/glm52-exl3-tr3-3.25bpw.json`](recipes/glm52-exl3-tr3-3.25bpw.json).

The [fixed-MTP2 EXL3 alternative](docs/EXL3_FIXED_MTP2_RECIPE_20260802.md)
records the exact four-Spark operator overlay and its duration-based
16K-128K/C1-C8 matrix. It is live-validated external evidence, not
clean-checkout public acceptance, correctness acceptance, or a reference-lane
result.

The Aiden MXFP4/GPTQ configuration is preserved as a historical reference in
[docs/history/AIDEN_MXFP4_GPTQ.md](docs/history/AIDEN_MXFP4_GPTQ.md). It is
not a supported deployment target.

The project is a research pre-release. Pin a commit when deploying because
environment flags, source attestations, and integration ABIs can change.

Before downloading the model or building an image, complete the
**[exhaustive prerequisites checklist](docs/PREREQUISITES.md)**. It separates
facts the operator must supply from hardware/network details a bot can discover
and provides the exact validation sequence.

## Measured results

Inspect either GLM recipe without contacting the cluster:

```bash
python scripts/sparkring_recipe.py list
python scripts/sparkring_recipe.py plan
```

[docs/RESULTS.md](docs/RESULTS.md) is the register of measured results,
methodology, evidence, and claim boundaries;
[docs/TESTING_HISTORY.md](docs/TESTING_HISTORY.md) preserves experiments,
resolved failures, superseded configurations, and pending acceptance work.
Maintainers advancing EXL3 toward full public-functional acceptance use the
dry-run-first
[EXL3 + LMCache acceptance runbook](docs/EXL3_ACCEPTANCE_RUNBOOK.md).

### Archived and alternative configurations

NF3 is documented as a deterministic public-functional alternative rather
than a featured README profile. Use the
[NF3 quickstart](docs/NF3_QUICKSTART.md),
[NVFP4 KV validation receipt](docs/NF3_NVFP4_PUBLIC_VALIDATION.md), and
[one-million-token operator audit](docs/NF3_LIVE_CONFIGURATION_20260731.md) for
its reproducible recipe and historical evidence. The Aiden MXFP4/GPTQ
historical reference is in
[docs/history/AIDEN_MXFP4_GPTQ.md](docs/history/AIDEN_MXFP4_GPTQ.md).

## What SparkRing provides

- A GLM/vLLM-oriented direct-cable transport implementation, with reusable
  low-level transport primitives and configurable graph-row geometry.
- Two- and four-rank collective schedules used by the GLM/vLLM paths.
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

The container deployment applies a thin overlay through `PYTHONPATH`. Each
adapter verifies the exact source SHA-256 and ABI it expects before
installation. A source mismatch stops startup. The orchestrator also attests
native libraries, mounts, launch arguments, and rank topology.

The pip-installable plugin (`sparkring_plugin/`) is the second integration
path: it registers through `vllm.general_plugins`, feature-detects the vLLM
integration point by signature instead of pinning a source hash, and
terminates before serving on any installation failure. It is fail-closed but
not source-attested at runtime; vendored-module parity with the source tree
is enforced in CI, not at serving install.

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
[PREREQUISITES.md](docs/PREREQUISITES.md) on a new cluster. NF3 is an
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
python -m pytest sparkring_plugin -q
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
sparkring_plugin/      pip-installable vLLM plugin packaging of the adapters
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
