# SparkRing documentation map

This page routes readers to the relevant document and canonical claim owner.
It is a navigation index, not an independent source of configuration,
maturity, or measurement truth. When documents disagree, use the
canonical-source table below and report the drift.

## Start here

| Goal | Document |
|---|---|
| Check hardware, operating-system, network, storage, and access requirements | [Prerequisites](PREREQUISITES.md) |
| Deploy the public-default EXL3 3.25-bpw plus LMCache CS512 profile | [Public-default quickstart](QUICKSTART.md) |
| Reproduce the operator-scoped EXL3 3.5-bpw profile; rebuilt images remain candidates until promotion | [EXL3 3.5-bpw quickstart](EXL3_R7_QUICKSTART.md) |
| Deploy the accepted deterministic NF3 alternative | [NF3 quickstart](NF3_QUICKSTART.md) |
| Serve DeepSeek-V4-Flash-0731 on the ring | [DeepSeek quickstart](DEEPSEEK_V4_FLASH_QUICKSTART.md) |
| Understand the supported public-functional matrix and acceptance requirements | [Public-functional lane definition](PUBLIC_FUNCTIONAL_TARGET.md) |
| Interpret measurements and evidence scope | [Measured results](RESULTS.md) |
| Understand the four-Spark runtime and transport design | [Architecture](ARCHITECTURE.md) |
| Find which models are validated against the transport, and with what evidence | [Validated-profiles registry](profiles/README.md) |

Commands that contact or modify a Spark retain the safety class stated in the
linked runbook. A document being listed here does not authorize remote action.

## Canonical sources

| Subject | Canonical source |
|---|---|
| Public-lane definition and open blockers | [Public-functional lane definition](PUBLIC_FUNCTIONAL_TARGET.md) |
| Public headless-startup ABI audit | [Public startup-shim audit](PUBLIC_STARTUP_SHIM_AUDIT.md) |
| Runtime input pins | [`runtime/runtime-lock.json`](../runtime/runtime-lock.json) |
| Site-config schema | [`scripts/sparkring_site.py`](../scripts/sparkring_site.py) |
| Repository prose standard | [Write Without Hidden Context](WRITING_STANDARD.md) |
| Model-profile validation status | [Validated-profiles registry](profiles/README.md), then the profile's linked evidence documents |
| Acceptance behavior and exit codes | [`scripts/acceptance_gate.py`](../scripts/acceptance_gate.py) |
| Measured claims and evidence labels | [Measured results](RESULTS.md) |
| Reference deployment reconstruction | [Reference bring-up reconstruction](SETUP.md) |
| Component maturity | [`STATUS.json`](STATUS.json), then the relevant component README |

## Runnable runbooks

| Document | Purpose and scope |
|---|---|
| [Prerequisites](PREREQUISITES.md) | Qualifies hardware, host software, management access, storage, and the direct-cable fabric before model work. |
| [Public-default quickstart](QUICKSTART.md) | Concise canonical entry point for the EXL3 3.25-bpw plus LMCache CS512 deployment. |
| [Detailed EXL3 procedure](EXL3_QUICKSTART.md) | Expands the public-default build, distribution, launch, attestation, and bounded gate procedure. |
| [EXL3 acceptance runbook](EXL3_ACCEPTANCE_RUNBOOK.md) | Runs the candidate workflow required to advance the public-default profile beyond bounded live validation. |
| [EXL3 3.5-bpw quickstart](EXL3_R7_QUICKSTART.md) | Builds or selects an image, derives and launches the profile, and provides rollback instructions; a rebuilt image remains a candidate until promotion. |
| [EXL3 3.5-bpw composition](EXL3_R7_OPERATOR_REPRODUCTION.md) | Reconstructs the public source and generated-layer composition used by the operator-accepted profile. |
| [EXL3 3.5-bpw promotion checklist](EXL3_R7_PROMOTION_CHECKLIST.md) | Qualifies an exact rebuilt image and defines the evidence required for a separate promotion decision. |
| [NF3 quickstart](NF3_QUICKSTART.md) | Prepares, launches, verifies, and troubleshoots the accepted deterministic alternative. |
| [DeepSeek quickstart](DEEPSEEK_V4_FLASH_QUICKSTART.md) | Launches DeepSeek-V4-Flash-0731 four-rank with its native speculative decoding, with the failure modes that cost time; functional launch, not shadow-qualified. |
| [CUDA-graph correctness gate](CUDAGRAPH_CORRECTNESS_GATE.md) | Runs the minimal live A/B and deterministic comparison required to diagnose graph replay correctness. |
| [SparkCache DCP2 dry-run plan](SPARKCACHE_DCP2_DRY_RUN_PLAN.md) | Regenerates and inspects a SparkCache plan; embedded operator state is historical and must be rediscovered. |
| [Eager width admission validation runbook](EAGER_WIDTH_VALIDATION_RUNBOOK.md) | Runs the three-leg validation for width-generic eager TP4 all-reduce admission; leg records are chronological evidence. |
| [SparkCache DCP2 live runbook](SPARKCACHE_DCP2_LIVE_RUNBOOK.md) | Defines preflight, authorized cutover, evidence collection, and rollback for the offline-validated integration. |

## Present-state specifications

| Document | Purpose and scope |
|---|---|
| [Architecture](ARCHITECTURE.md) | Defines the four-Spark topology, transport design, CUDA-graph mechanism, runtime integration, and fallback path. |
| [Public-functional lane definition](PUBLIC_FUNCTIONAL_TARGET.md) | Defines the supported matrix, unsupported scope, acceptance gates, evidence requirements, and change control. |
| [SIRCL transport](SIRCL.md) | Defines the internal transport name, implemented data path, supported payloads, and extension scope. |
| [Bulk striped transport proposal](../spark_transport/BULK_STRIPED_TRANSPORT.md) | Research-only design proposal for a dual-port bandwidth lane serving multi-MiB payloads; nothing implemented, adoption gated on staged measurements. |
| [Generic runtime launcher](GENERIC_RUNTIME.md) | Defines generic launcher profiles, schema dispatch, runtime bundles, safety behavior, and extension points. |
| [Evidence comparison checklist](EVIDENCE_COMPARISON_CHECKLIST.md) | Defines the validity contract for matched 16K sustained-decode comparisons. |
| [Runtime gaps](RUNTIME_GAPS.md) | Records the 2026-07-27 upstream comparison, the 2026-07-29 recovery status, and the remaining qualification gaps. |

## Profiles and evidence records

| Document | Purpose and evidence scope |
|---|---|
| [Measured results](RESULTS.md) | Register of separately labelled reference-lane measurements and public-functional receipts. |
| [EXL3 3.25-bpw recipe](EXL3_RECIPE.md) | Default-profile specification plus its clean-checkout receipt and bounded gate. |
| [EXL3 fixed-MTP2 record](EXL3_FIXED_MTP2_RECIPE_20260802.md) | Externally executed configuration, startup proof, measurements, and rollback conditions. |
| [EXL3 performance campaign](EXL3_AB_CAMPAIGN_20260802.md) | Dated matched measurements, rejected arms, and retained disposition. |
| [EXL3 LMCache campaign](EXL3_LMCACHE_CAMPAIGN_20260803.md) | Dated LMCache arm evidence, attribution limits, receipt, and remaining gates. |
| [EXL3 3.5-bpw fixed-MTP2 profile](EXL3_R7_FIXED_MTP2_PROFILE.md) | Live-validated intermediate weight, graph-stream, and speculative-decoding contract. |
| [EXL3 3.5-bpw fixed-MTP3 profile](EXL3_R7_FIXED_MTP3_PROFILE.md) | Live-validated intermediate depth-three profile and generated artifacts. |
| [EXL3 3.5-bpw fixed-MTP4 profile](EXL3_R7_FIXED_MTP4_PROFILE.md) | Operator-accepted profile, qualification evidence, acceptance scope, and limitations. |
| [EXL3 3.5-bpw qualification record](EXL3_R7_OPTIMIZATION_20260811.md) | Transport, DCP, MTP, KV-allocation, and CKV-gather qualification decisions. |
| [NF3 public validation](NF3_NVFP4_PUBLIC_VALIDATION.md) | Clean-bootstrap acceptance receipt, immutable inputs, effective configuration, and correctness request. |
| [NF3 serving snapshot](NF3_LIVE_CONFIGURATION_20260731.md) | Dated observed serving configuration and its drift from the default recipe. |
| [NF3 fixed-MTP2 record](NF3_FIXED_MTP2_RECIPE_20260801.md) | Dated live configuration, launch command, startup proof, and measurements. |
| [NF3 DCP2 record](NF3_DCP2_LIVE_CONFIGURATION_20260801.md) | Dated configuration delta, benchmark snapshot, result, and evidence boundary. |
| [Faststart validation](FASTSTART_VALIDATION.md) | One-Spark native build and partial four-Spark bring-up receipt. |
| [Public startup-shim audit](PUBLIC_STARTUP_SHIM_AUDIT.md) | Audit conclusion, validated follower behavior, and conditions that require a new audit. |
| [Dual-port striping probe record](DUAL_PORT_STRIPING_PROBE_20260818.md) | Model-free four-rank graph-collective measurements: the fused kernel's large-payload pathology, the kernel and schedule fixes with their controls, and a matched NCCL control; research-only, with limitations stated. |
| [SparkCache streaming snapshots](SPARKCACHE_STREAMING_SNAPSHOTS.md) | Records the streaming-snapshot live-evidence campaign, covering the end-of-prefill pipeline revision (v50) and its idle-progress fix (v51), and implementation invariants; the published TP4/DCP2 integration remains offline-validated and unpromoted. |

## History and reference reconstruction

| Document | Purpose and limitation |
|---|---|
| [Reference bring-up reconstruction](SETUP.md) | Reconstructs the historical reference deployment. Only stages explicitly marked runnable are supported from the public tree. |
| [Eager width admission review handoff](REVIEW_HANDOFF_EAGER_WIDTH_ADMISSION.md) | Component map, architecture, equivalence-evidence scope, and review questions for the width-generic admission work. |
| [Testing history](TESTING_HISTORY.md) | Preserves dated experiments, regressions, resolved failures, superseded configurations, and acceptance gaps. |
| [Historical Aiden MXFP4/GPTQ lane](history/AIDEN_MXFP4_GPTQ.md) | Preserves the Aiden MXFP4/GPTQ reference-lane configuration and its evidence scope. |
