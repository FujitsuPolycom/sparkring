# GLM-5.2 R7 3.5-bpw fixed-MTP4 dynamic-NVFP4 candidate

## Status and evidence scope

This configuration is the operator's **3.5-bpw quantization baseline** and a
**public-functional-lane, live-validated candidate** for four directly cabled
NVIDIA DGX Sparks / GB10 GPUs. Its baseline role identifies the comparison
configuration for 3.5-bpw research; it does not make the profile the repository
default, an accepted public-functional matrix, or a transferable result for
other hardware. The advertised default remains EXL3 3.25-bpw plus LMCache
CS512.

The candidate serves
`brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78` at immutable revision
`9ab9579774cc432df91567a36f6e9e863e0d4c9f`. It combines TP4, DCP4,
fixed-depth-four MTP, dynamic per-token NVFP4 latent KV, FP8 RoPE storage, a
262,144-token request limit, a 4,096-token prefill ceiling, and a transient
full-CKV DCP gather for pure prefill. The exact four-rank startup, graph,
bounded correctness, speculative-decoding, transport, and matched
prefill/decode A/B gates described below passed on 2026-08-11.

The sanitized machine-readable result is
[glm52-exl3-r7-mtp4-nvfp4-ckv-gather-20260811.json](configurations/glm52-exl3-r7-mtp4-nvfp4-ckv-gather-20260811.json).
The raw endpoint, rank-status, and benchmark artifacts are maintainer-held and
identified by SHA-256 in that record. The earlier FP8, 65,536-token fixed-MTP4
qualification remains separately preserved in
[glm52-exl3-r7-mtp4-kv925-20260811.json](configurations/glm52-exl3-r7-mtp4-kv925-20260811.json).

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

The target model owns the online EXL3 K6 overlay for eligible BF16 weights.
The reused layer-78 draft retains its checkpoint EXL3 routed experts and
producer BF16 non-expert weights. The draft inherits the target compressed-KV
configuration. Every rank reported the `nvfp4_ds_mla` format, FP8 RoPE,
368-byte KV stride, and dynamic per-token scaling.

One process-and-device CUDA stream is shared across target, draft-prefill, and
draft-decode graph managers. Their graph-capture contexts and channel IDs stay
distinct. This preserves the Spark TP4 graph session's stable-caller-stream
invariant without merging graph ownership.

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

- This is a live-validated candidate on one four-Spark appliance, not an
  accepted or default public-functional configuration.
- The current exact profile has bounded startup, output-equivalence,
  speculative-decoding, transport, and matched prefill/decode evidence. Its
  262,144-token request boundary and near-capacity concurrent residency remain
  unqualified.
- Each prefill row is one cold, fully unique request. Repeat distributions and
  longer soak testing remain open.
- The control benchmark's JSON destination was invalid, so the harness printed
  the complete result but did not write its JSON artifact. The preserved
  console transcript is identified by SHA-256 in the sanitized evidence file.
- DCP, CKV, and indexer collectives remain on stock paths. Only the qualified
  TP all-reduce and vocabulary families use SparkRing native transport here.
- Fixed MTP5 is unsupported by this image. Q48 requires a Python contract
  extension and a rebuilt native vocabulary/DCP/indexer transport cap; it is
  not a profile-only change.
