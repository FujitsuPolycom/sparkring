# GLM-5.2 R7 3.5-bpw fixed-MTP4 operator default

## Status and evidence scope

This configuration is the **accepted operator default for 3.5-bpw EXL3** on
four directly cabled NVIDIA DGX Sparks / GB10 GPUs. It is live-validated in the
public-functional lane, but its acceptance scope is the operator's four-Spark
appliance. It is not the repository-wide public-functional default, an
accepted public deployment matrix, or a transferable result for other
hardware. The advertised public-functional default remains EXL3 3.25-bpw plus
LMCache CS512.

The operator profile serves
`brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78` at immutable revision
`9ab9579774cc432df91567a36f6e9e863e0d4c9f`. It combines TP4, DCP4,
fixed-depth-four MTP, dynamic per-token NVFP4 latent KV, FP8 RoPE storage, a
262,144-token request limit, a 4,096-token prefill ceiling, and a transient
full-CKV DCP gather for pure prefill. The exact four-rank startup, graph,
bounded correctness, speculative-decoding, transport, and matched
prefill/decode A/B gates described below passed on 2026-08-11. On 2026-08-12,
the operator accepted the target-only exact-Q40 block-8 policy described below
as the default extension to this profile.

The sanitized machine-readable result is
[glm52-exl3-r7-mtp4-nvfp4-ckv-gather-20260811.json](configurations/glm52-exl3-r7-mtp4-nvfp4-ckv-gather-20260811.json).
The raw endpoint, rank-status, and benchmark artifacts are maintainer-held and
identified by SHA-256 in that record. The earlier FP8, 65,536-token fixed-MTP4
qualification remains separately preserved in
[glm52-exl3-r7-mtp4-kv925-20260811.json](configurations/glm52-exl3-r7-mtp4-kv925-20260811.json).
The exact-Q40 acceptance result is
[glm52-exl3-r7-mtp4-q40-block8-20260812.json](configurations/glm52-exl3-r7-mtp4-q40-block8-20260812.json).
The operator-accepted current-best prefill snapshot is
[glm52-exl3-r7-current-best-prefill-20260813.json](configurations/glm52-exl3-r7-current-best-prefill-20260813.json).
The full prefill, decode, and coding snapshot is
[glm52-exl3-r7-current-best-matrix-20260813.json](configurations/glm52-exl3-r7-current-best-matrix-20260813.json).

## Serving contract

| Setting | Live-validated value |
|---|---|
| Hardware | Four directly cabled DGX Sparks / GB10 GPUs |
| Model | `brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78@9ab9579774cc432df91567a36f6e9e863e0d4c9f` |
| Image | `sha256:02881d5229d4f4d1cbba0cf40537492a2a505b9d4e43bbfe9a0b2a7bd0584513` |
| Parallelism | TP4 plus DCP4 `ag_rs`, interleave size one |
| Speculation | fixed MTP4, greedy draft sampling, adaptive depth disabled |
| Maximum sequences | 8 |
| Query-row contract | Q1 through Q40; `8 * (4 + 1) = 40` verification rows |
| Exact-Q40 routed-MoE policy | Target mixed-EXL3 layers use capacity 40 and route block 8 only at exactly 40 rows; Q1-Q32, other prefill shapes, and the draft model retain their prior states |
| KV representation | `nvfp4_ds_mla`, dynamic per-token scale, FP8 RoPE, 368-byte record |
| KV allocation | 9,250,000,000 bytes/rank; 37,000,000,000 bytes aggregate |
| Reported KV capacity | 1,156,864 tokens |
| Model limit | 262,144 tokens |
| Batching | 4,096 maximum batched tokens; chunked prefill; synchronous scheduler |
| Prefill gather | B12X transient full-CKV DCP gather, maximum 262,144 logical tokens |
| CKV workspace | 414.4 MiB/rank for two persistent execution lanes |
| Graphs | `FULL_AND_PIECEWISE`, Q1 through Q40, no eager model execution |
| TP transport | SparkRing native all-reduce and vocabulary paths with patched NCCL 2.30.7 NET/IB fallback |
| DCP, CKV, and indexer transport | stock NCCL-backed paths; custom DCP/indexer/CKV sessions disabled |
| Cache | native prefix caching enabled; LMCache and SparkCache disabled |

## LMCache NVMe candidate extension

The accepted operator profile above remains defined with LMCache disabled.
A separately scoped **public-functional, live-validated candidate extension**
attaches one LMCache MP server to each DCP rank without changing the model,
TP4/DCP4 topology, fixed-MTP4 policy, exact-Q40 state, KV representation,
9.25 GB/rank KV allocation, batch limits, or graph plan. This extension is not
an accepted operator default and has not been reproduced from a clean public
checkout.

Each rank uses LMCache CS512 with a lazy 512 MiB L1 initialized at zero bytes
and a bounded 50 GiB native-filesystem L2. The L2 uses O_DIRECT, two workers,
and LRU eviction with a 0.85 trigger watermark and 0.15 eviction ratio. The
engine connector uses the `kv_both` role and recomputes on a load failure.
At the post-validation health snapshot, the LMCache server container used
approximately 1.03 GiB/rank while the L1 data tier held zero bytes.

A 32,506-token cold request completed with HTTP 200 and 56.115 seconds of
client TTFT. It published 63 chunks and 257,854,464 L2 bytes on every rank.
The LMCache servers were then restarted, which emptied volatile L1 and released
CUDA IPC ownership while preserving the filesystem L2. After a clean engine
restart, the exact prompt completed with HTTP 200 and 1.477 seconds of client
TTFT. vLLM reported 0.0% native-prefix-cache hit rate and 99.2% external-prefix-
cache hit rate; LMCache L1 remained empty and all ranks retained the same 63
files and L2 byte count. This attributes the replay to the NVMe tier.

The supported restart procedure must recycle the LMCache servers before the
engines so stale CUDA IPC ownership is released. It must also remove only the
rank-local one-shot `q40-exact-state-serving-v1-rank{rank}.json` receipt while
preserving the warm compile cache and LMCache L2. The
[candidate launcher](../scripts/sparkring_exl3_r7_lmcache_canary.py) implements
the receipt cleanup. The running containers use Docker restart policy `no`, so
host reboot recovery still requires an explicit relaunch.

This evidence proves bounded functional publication and attributed NVMe reuse;
it is not a latency distribution, concurrent-load qualification, clean-checkout
reproduction, or deterministic-output gate. The cold and restart-replay
completion text differed. Exact values and scope are recorded in
[glm52-exl3-r7-lmcache-nvme-20260813.json](configurations/glm52-exl3-r7-lmcache-nvme-20260813.json).

The target model owns the online EXL3 K6 overlay for eligible BF16 weights.
The reused layer-78 draft retains its checkpoint EXL3 routed experts and
producer BF16 non-expert weights. The draft inherits the target compressed-KV
configuration. Every rank reported the `nvfp4_ds_mla` format, FP8 RoPE,
368-byte KV stride, and dynamic per-token scaling.

One process-and-device CUDA stream is shared across target, draft-prefill, and
draft-decode graph managers. Their graph-capture contexts and channel IDs stay
distinct. This preserves the Spark TP4 graph session's stable-caller-stream
invariant without merging graph ownership.

## Accepted exact-Q40 routed-MoE policy

The accepted operator profile adds one target-only EXL3 runtime state for
exactly 40 routed rows. That state uses capacity 40, route block 8, and the
existing prefill tiers and tile configuration. The insertion-only source delta
does not change Q1-Q32 decode dispatch, general prefill dispatch for row counts
other than 40, or the uniform draft-model path. The two unique Q40 arenas add
32,268,384 bytes per rank and leave the reported KV capacity unchanged at
1,156,864 tokens.

All four ranks attested all 75 routed target layers before graph capture. Each
layer produced exact BF16 equality between the new Q40 state and the deployed
general-prefill comparator. Deterministic 16K and 32K application requests also
matched the sealed control response and completion-token hashes, with finite
log probabilities and zero differences. Graph capture, API health, transport
sequence convergence, and the full 9.25 GB/rank KV allocation passed after the
measurements.

The matched warm C8 bracket replayed the same eight unique 16K request payloads
with a 25-second measurement window and full 8/8 residency:

| Arm | Aggregate tokens/s |
|---|---:|
| Baseline A1 | 59.656 |
| Baseline A2 | 61.468 |
| Baseline C2 | 62.907 |
| Baseline mean | 61.344 |
| Exact-Q40 candidate B2 | 74.119 |
| Exact-Q40 candidate B3 | 72.297 |
| Exact-Q40 candidate mean | **73.208** |
| Candidate change from baseline mean | **+19.341%** |

The slower candidate repeat exceeded the fastest baseline repeat by 14.93%.
A post-prefill durability replay measured 66.685 aggregate tokens/s with all
eight requests resident and no queue, request error, capacity limit, fatal
transport state, or overflow.

The predeclared prefill reducer remains recorded as a machine failure. Its sole
primary-gate miss was the 64K median: 618.246 tokens/s versus the lower sealed
baseline median of 618.998 tokens/s, or -0.1215%. The operator accepts that
difference as measurement-neutral rather than a material regression. The 16K
and 128K primary prefill gates passed; the 32K median was inside the baseline
envelope. This is an explicit engineering waiver of the literal threshold, not
a rewritten gate or a claim that the machine result passed.

Fixed-prompt non-Q40 checks measured C1 at 22.218 versus 22.362 tokens/s
(-0.65%) and C4 at 45.703 versus 46.578 tokens/s (-1.88%). These
source-identical paths are treated as bounded measurement noise, not as
optimization results.

## Current-best operator performance snapshot

The operator accepts the following 100%-unique-context C1 results as the best
measured prefill snapshot for this exact four-Spark profile:

| Context | Prompt tokens | Client TTFT | Client-timed prefill | Samples |
|---|---:|---:|---:|---:|
| 8K | 8,194 | 12.06 s | **679 tok/s** | 2 |
| 16K | 16,386 | 24.36 s | **673 tok/s** | 1 |
| 32K | 32,770 | 49.17 s | **666 tok/s** | 1 |
| 64K | 65,538 | 99.72 s | **657 tok/s** | 1 |
| 128K | 131,074 | 203.09 s | **645 tok/s** | 1 |

The displayed throughput changes by 5.0% across the 16x context span. Several
operator runs produced similar results, but the exact table is a bounded
snapshot with low visible sample counts rather than a throughput distribution.
Server-side cached-token accounting was unavailable for these cells; the
100%-unique request construction is recorded, but the captured result does not
independently prove cache misses.

The same benchmark display reported this aggregate decode matrix:

| Context | C1 | C2 | C4 | C8 |
|---|---:|---:|---:|---:|
| 4K | 22.6 | 32.7 | 50.3 | **78.4** |
| 8K | 22.0 | 35.3 | 51.9 | **71.3** |
| 16K | 21.3 | 32.9 | 49.2 | **70.0** |
| 32K | 20.4 | 32.3 | 45.6 | **65.5** |
| 64K | 21.4 | 30.4 | 47.2 | **67.8** |

Values are aggregate tok/s. The matrix display does not expose repeat counts,
so it is accepted as a bounded operating snapshot rather than a distribution.
The separately controlled exact-Q40 bracket remains the stronger 16K/C8
comparison because it replayed identical sealed payloads across repeated arms.

### Coding benchmark (C1)

| Workload | Runs | Median tok/s | Mean tok/s | Maximum tok/s | CJK runs |
|---|---:|---:|---:|---:|---:|
| Coding peak | **5/5** | **27.3** | **27.3** | **28.8** | **0** |

This is a distinct workload result, not a replacement for the standardized C1
matrix.

## Exact CKV-gather delta and rollback

The maintainer-held fail-closed generator derives the CKV-gather candidate
from the otherwise identical dynamic-NVFP4, FP8-RoPE, 4,096-token-prefill
control. Its SHA-256 is recorded in the sanitized machine-readable evidence.
The only effective runtime changes are:

```text
VLLM_B12X_MLA_CKV_GATHER:             unset -> 1
VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS:  unset -> 262144
```

Profile metadata also names the CKV-prefill contract. Model, image, TP/DCP
degree, MTP depth, KV representation and allocation, batch limits, graph plan,
transport settings, online-K6 scope, mounts, and source attestations remain
unchanged. The generated rollback profile and site are byte-identical to the
dynamic-NVFP4 control.

The candidate profile SHA-256 is
`9a44e093dfa2eab6c20e73ca2d6dc7494f576c021bda0361278fa0cb1b41e927`.
The rollback profile SHA-256 is
`76af60e3a07af9982ab3537dd39b3a18c4ad497b8fc735e02be549ca799870fa`.
Both use site SHA-256
`a1c62b5b42c98d75830a8a30ef71c33953fd8d28bd4dce28aecd0d133e81fe4c`.

`SPARK_TP4_ALLGATHER_ENABLE_CKV=0` remains intentional. It disables the
separate Spark custom-all-gather CKV signature; the qualified optimization is
the B12X transient full-CKV prefill path over the dedicated DCP communicator.

## Startup, correctness, and transport qualification

All four ranks completed target PIECEWISE 40/40 plus FULL 8/8 capture,
draft-prefill PIECEWISE 40/40 plus FULL 8/8 capture, and draft-decode FULL 8/8
capture. Each rank reported 7,704 native all-reduce graph nodes and 16 native
vocabulary graph nodes. After the benchmark, every rank was caught up at
59,302 published/consumed/completed all-reduce operations and 1,655
published/consumed/completed vocabulary operations. Fatal, overflow,
dropped-signature, and exact-shape Q1-Q40 stock TP/vocabulary fallback counts
were zero.

The first eligible prefill request produced exactly one
`Using transient full-CKV gather for B12X sparse MLA prefill` activation
record on every rank. No rank logged the corresponding ignore path. All four
containers remained running with restart count zero, `OOMKilled=false`, and
no traceback, runtime, CUDA-OOM, NCCL-warning, fatal, or non-finite record.

Three 128-token and three 256-token greedy completions were byte-identical to
the MTP-disabled control. A semantic chat probe passed, and all 96 requested
completion-logprob values were finite. MTP4 was active: 268 draft events
produced 1,072 draft tokens and accepted 1,036, with non-zero position counts
`[266, 263, 255, 252]` and a 96.642% aggregate acceptance rate.

## Matched CKV-gather performance A/B

The A/B changed only the two CKV-gather environment values above. Both arms
used dynamic NVFP4 latent KV, FP8 RoPE, fixed MTP4, TP4/DCP4, the 4,096-token
prefill ceiling, 9.25 GB KV/rank, and the 262,144-token model limit.

`llm_decode_bench.py` v0.4.31 sent one cold, fully unique scout at each prompt
size and then measured one temperature-zero, 25-second C8 decode cell with
eight unique 16K contexts. Prefill is a single sample per context and should
be treated as bounded diagnostic evidence rather than a distribution.

| Workload | NVFP4 control | CKV gather | Change |
|---|---:|---:|---:|
| 8K prefill | 435.19 tok/s | **499.82 tok/s** | **+14.85%** |
| 16K prefill | 437.16 tok/s | **608.33 tok/s** | **+39.16%** |
| 64K prefill | 434.20 tok/s | **563.21 tok/s** | **+29.71%** |
| 128K prefill | 424.60 tok/s | **551.25 tok/s** | **+29.83%** |
| C8 16K sustained decode | 45.4 tok/s | **47.85 tok/s** | +5.40% observed |

CKV gather is a prefill optimization and is not active during decode. The C8
result therefore establishes no measured decode regression; its positive
difference is not attributed causally to CKV gather. The candidate completed
eight requests with no request error, no scheduler preemption, no capacity
limit, and no queue after admission. The service returned to health HTTP 200,
zero running, zero waiting, and zero KV use.

### Bounded operator reruns

The following PowerShell command runs one cold, fully unique 16K scout and a
five-second C1 decode cell. It is a diagnostic smoke test, not acceptance:

```powershell
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$bench = "/path/to/llm_decode_bench.py"
$headAddress = "replace-with-head-management-address"
$output = "/path/to/evidence/ckv-smoke-$stamp.json"
python -u $bench `
  --host $headAddress `
  --port 8000 `
  --model glm-5.2-exl3-r7-3.5bpw `
  --contexts 16k `
  --prefill-contexts 16k `
  --concurrency 1 `
  --duration 5 `
  --max-tokens 256 `
  --temperature 0 `
  --unique-context-percent 100 `
  --dcp-size 4 `
  --kv-budget 1156864 `
  --decode-warmup-seconds 0 `
  --token-targeting estimate `
  --no-hw-monitor `
  --display-mode plain `
  --output $output
```

Adding `--prefill-contexts 16k,64k` gives a stronger short check. Reproducing
the complete matched workload uses `--contexts 16k`,
`--prefill-contexts 8k,64k,128k`, `--concurrency 8`, `--duration 25`, and
`--max-tokens 2048`. Operators must replace the host and output path with
their own values; site addresses and local paths are not repository defaults.

## Earlier fixed-MTP4 evidence

The earlier FP8, 65,536-token profile established the fixed-MTP4 depth and Q40
transport contract independently of this KV/prefill update. It measured
34.60, 51.44, 76.96, and 85.68 aggregate tokens/s at C1, C2, C4, and C8 in a
matched decode-only matrix; MTP4 improved C1-C4 and regressed C8 relative to
fixed MTP3. It also completed eight simultaneously resident 64K requests in a
675,840-token reported KV pool.

Those results remain evidence for the shared fixed-MTP4 and transport
contract, not for the exact dynamic-NVFP4/262K/CKV-gather profile. The larger
1,156,864-token pool has not yet repeated that near-capacity residency gate.

## Limitations

- This is the accepted operator default on one four-Spark appliance, not the
  repository-wide public-functional default or an accepted public deployment
  matrix.
- The exact-Q40 acceptance is an explicit engineering decision over a
  predeclared machine prefill failure at 64K of -0.1215%. The failure remains
  preserved in the machine-readable evidence.
- The +19.341% decode result applies to fixed-MTP4 Q40 at 16K with eight fully
  resident requests. It is not an all-shape or all-concurrency speed claim.
- The accepted exact-Q40 profile has bounded startup, output-equivalence,
  speculative-decoding, transport, and matched prefill/decode evidence. Its
  262,144-token request boundary and near-capacity concurrent residency remain
  unqualified.
- The accepted prefill table has one visible sample per context except for
  8K, which has two. Several operator runs were consistent, but a retained
  repeat distribution and longer soak test remain open.
- The control benchmark's JSON destination was invalid, so the harness printed
  the complete result but did not write its JSON artifact. The preserved
  console transcript is identified by SHA-256 in the sanitized evidence file.
- DCP, CKV, and indexer collectives remain on stock paths. Only the qualified
  TP all-reduce and vocabulary families use SparkRing native transport here.
- Fixed MTP5 is unsupported by this image. Q48 requires a Python contract
  extension and a rebuilt native vocabulary/DCP/indexer transport cap; it is
  not a profile-only change.

A rebuilt image must complete the
[promotion checklist](EXL3_R7_PROMOTION_CHECKLIST.md) before inheriting this
operator deployment's accepted maturity.
