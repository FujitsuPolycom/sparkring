# GLM-5.2 R7 fixed-depth-four, 9.25 GB KV candidate

## Status and evidence scope

This configuration is a **public-functional-lane, live-validated candidate**
for four directly cabled NVIDIA DGX Sparks / GB10 GPUs. It is not the
repository default, an accepted public-functional matrix, or a transferable
result for other hardware. The advertised default remains EXL3 3.25-bpw plus
LMCache CS512.

The candidate serves
`brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78` at immutable revision
`9ab9579774cc432df91567a36f6e9e863e0d4c9f`. It uses TP4, DCP4, fixed-depth
four MTP, 9,250,000,000 KV-cache bytes per rank, `fp8_ds_mla`, and a 65,536
token model limit. The four-rank launch, graph, output-equivalence,
speculative-decoding, transport, sustained-decode, coding, long-context, and
recovery gates described below passed on 2026-08-11.

The sanitized machine-readable result is
[glm52-exl3-r7-mtp4-kv925-20260811.json](configurations/glm52-exl3-r7-mtp4-kv925-20260811.json).
The raw endpoint, rank-status, and telemetry artifacts are maintainer-held and
are identified by SHA-256 in that record.
The separately gated transport, DCP, MTP2, MTP3, and KV decisions are in the
[R7 optimization campaign record](EXL3_R7_OPTIMIZATION_20260811.md).

## Serving contract

| Setting | Qualified value |
|---|---|
| Hardware | Four directly cabled DGX Sparks / GB10 GPUs |
| Model | `brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78@9ab9579774cc432df91567a36f6e9e863e0d4c9f` |
| Parallelism | TP4 plus DCP4 `ag_rs`, interleave size one |
| Speculation | fixed MTP4, greedy draft sampling, adaptive depth disabled |
| Maximum sequences | 8 |
| Query-row contract | Q1 through Q40; `8 * (4 + 1) = 40` verification rows |
| KV representation | `fp8_ds_mla`, B12X block size 64 |
| KV allocation | 9,250,000,000 bytes/rank; 37,000,000,000 bytes aggregate |
| Reported KV capacity | 675,840 tokens / 2,640 DCP-effective blocks |
| Request-usable KV capacity | 675,584 tokens / 2,639 blocks after one permanent null block |
| Model limit | 65,536 tokens |
| Batching | 2,048 maximum batched tokens; chunked prefill; synchronous scheduler |
| Graphs | `FULL_AND_PIECEWISE`, Q1 through Q40, no eager model execution |
| TP transport | SparkRing native all-reduce and vocabulary paths with patched NCCL 2.30.7 NET/IB fallback |
| DCP and indexer transport | stock `ag_rs` DCP and stock indexer collectives |
| Cache | native prefix caching enabled; LMCache and SparkCache disabled |

The target model owns the online EXL3 K6 overlay for eligible BF16 weights.
The reused layer-78 draft retains its checkpoint EXL3 routed experts and
producer BF16 non-expert weights. The draft inherits the target
`fp8_ds_mla` cache configuration; an explicit draft KV dtype is unsupported
because it converts the default 16-token cache block into an explicitly pinned
value before B12X selects its required 64-token block.

One process-and-device CUDA stream is shared across target, draft-prefill, and
draft-decode graph managers. Their graph-capture contexts and channel IDs stay
distinct. This preserves the Spark TP4 graph session's stable-caller-stream
invariant without merging graph ownership.

## Derivation and rollback

`scripts/prepare_exl3_r7_mtp4.py` derives this profile from the qualified
fixed-MTP3, 9.25 GB KV profile. The only semantic changes are:

```text
profile and mode:              fixed-mtp3 -> fixed-mtp4
VLLM_SPARK_MTP_TOKENS:         3          -> 4
num_speculative_tokens:        3          -> 4
VLLM_SPARK_MAX_QUERY_ROWS:     32         -> 40
CUDA graph capture sizes:      Q1-Q32     -> Q1-Q40
maximum graph capture size:    32         -> 40
site serving.mtp_tokens:       3          -> 4
```

The generator rejects changes to the model, image, DCP degree, KV allocation,
KV dtype, online-K6 scope, transport choice, graph mode, shared-stream overlay,
or checkpoint attestations. Its rollback profile and site are byte-identical
to the qualified fixed-MTP3, 9.25 GB KV inputs.

## Functional and transport qualification

All four ranks completed target PIECEWISE 40/40 plus FULL 8/8 capture,
draft-prefill PIECEWISE 40/40 plus FULL 8/8 capture, and draft-decode FULL 8/8
capture. Each rank reported 7,704 native all-reduce graph nodes and 16 native
vocabulary graph nodes. Published, consumed, and completed sequences caught
up; fatal, overflow, and dropped-signature counters remained zero. Exact-shape
audits found no stock TP all-reduce `[Q,6144]` or vocabulary `[Q,38720]`
fallback for any Q1 through Q40 in eager or capture phases.

Stock DCP and indexer operation was intentional and separately observed. The
custom DCP and custom indexer session counts remained zero; their stock eager
counters advanced during qualification.

Three 128-token and three 256-token greedy completions were byte-identical to
the MTP-disabled control. A semantic chat probe passed, and all 96 requested
completion-logprob values were finite. MTP4 was active: 258 draft events
produced 1,032 draft tokens and accepted 1,001, with non-zero position counts
`[256, 254, 246, 245]`.

## Bounded performance evidence

The exact 1,024-prompt plus 128-output endpoint probe measured 414.74 prompt
tokens/s, 33.04 inter-token decode tokens/s, and 20.28 end-to-end output
tokens/s. Relative to the matched fixed-MTP3, 9.25 GB control, inter-token
decode improved 10.95% and end-to-end output improved 6.92%.

The matched 25-second decode-only matrix measured:

| Concurrency | Aggregate tok/s | Change from matched MTP3 |
|---:|---:|---:|
| C1 | 34.60 | +12.34% |
| C2 | 51.44 | +9.08% |
| C4 | 76.96 | +10.38% |
| C8 | 85.68 | -11.63% |

The low-concurrency geometric mean improved 10.69%; the all-cell geometric
mean improved 4.56%. The C8 regression is a material limitation and prevents
describing MTP4 as uniformly faster than MTP3.

An independent unique-16K-context decode suite, using temperature zero and a
25-second measurement window, measured 30.93, 41.30, and 46.71 aggregate
tokens/s at C2, C4, and C8. A separate five-run coding-peak workload omitted a
temperature field to preserve that harness's canonical methodology and
measured a 26.87 tokens/s median.

## Long-context and capacity qualification

Four sequential C1 requests used 16,384, 32,768, 49,152, and 65,280 prompt
tokens with 128 output tokens. Every request returned the exact requested
usage, finite logprobs, no preemption, and a healthy idle service afterward.

The capacity-residency arm submitted eight unique 64,000-token prompts with
1,408 output tokens per request. All eight completed with exact usage and
finite logprobs. One same-scrape sample observed eight running requests, zero
waiting requests, and 77.226% KV use, proving at least 512,000 logical tokens
resident simultaneously. The required threshold was 75.758%. Preemption,
swap, OOM, transport-fault, and thermal-violation counts were zero, and the
service returned to health HTTP 200, zero running, zero waiting, and zero KV
usage.

An earlier attempt was stopped by the sealed thermal guard before acceptance
when one host zone crossed its sustained limit. That attempt remains in the
evidence. After cooldown, the unchanged workload and guard passed; native
prefix-cache reuse shortened admission but did not replace the same-scrape
residency requirement.

## Limitations

- This is a live-validated candidate on one four-Spark appliance, not an
  accepted or default public-functional configuration.
- The complete raw evidence bundle and generated site/profile are local
  operator artifacts. The repository publishes their hashes, generators,
  contracts, and sanitized summary, not site addresses or host paths.
- MTP4 improves the measured C1-C4 cells but regresses the matched C8 cell.
- DCP and indexer collectives remain on the stock path. Only the qualified TP
  all-reduce and vocabulary families use the SparkRing native transport here.
- Fixed MTP5 is unsupported by this image. Q48 requires a Python contract
  extension and a rebuilt native vocabulary/DCP/indexer transport cap; it is
  not a profile-only change.
