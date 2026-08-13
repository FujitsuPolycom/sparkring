# SparkRing — Measured Results

> The detailed matrix below is the historical Aiden MXFP4/GPTQ evidence set.
> NF3 has a smaller accepted sanity set in [README.md](../README.md).
> EXL3+LMCache CS512 is the main advertised public-functional configuration and
> has the bounded clean-checkout gate below. The operator's running EXL3 R7
> fixed-MTP4 service is the separately scoped operator default.
> Results are not interchangeable between checkpoints or evidence origins.

## Current EXL3+LMCache public-path gate

On 2026-08-03, a clean checkout built the EXL3 derived image from source commit
`19523482c29860024c3a3cf51e793e8436e1c441`; launcher correction `cc9cc1e`
deployed it to four directly cabled DGX Sparks. Exact image
`sha256:20c4099f2e7e3dd3c8ab64f7d7930bde4f372df1895aa3ffa593252ca04ae96f`
was identical on all ranks. Post-stop preflight passed 116/116 checks, four
engines and four LMCache CS512 servers started with zero restarts, model load
was 84.43 GiB/rank, and 16/16 piecewise plus 12/12 full graphs captured.

Five consecutive bounded `exl3_live_gate.py` runs passed all configured
C1/C2/C8 regression floors and post-run health checks. Their ten fixed-seed
128-token completions were identical, with SHA-256
`a310b67d304b36f5dea88cbbcb18ba7be640001cc463590fe4e8cbb31042131c`.
Those bounded live-gate throughput samples varied and are not promoted as a
performance matrix; the standard matrix below is a separate run. This is
clean-checkout public-functional-lane bounded live
evidence, not reference-lane evidence, blanket correctness, persistence,
release promotion, or complete public-functional acceptance. See
[EXL3_RECIPE.md](EXL3_RECIPE.md).

The same clean-image deployment then completed the standard sustained-decode
16K matrix with `llm_decode_bench.py` v0.4.31: 25-second duration cells,
2,048-token maximum, temperature 0, duration-mode `ignore_eos`, 100% unique
contexts, DCP4, exact 562,688-token KV budget, three-second decode warmup,
prefill skipped, and a 300-second cell-warmup timeout.

| 16K concurrency | Aggregate tok/s | Effective concurrency | Errors |
|---:|---:|---:|---:|
| C1 | **18.33** | 1/1 | 0 |
| C2 | **27.61** | 2/2 | 0 |
| C4 | **45.11** | 4/4 | 0 |
| C8 | **59.40** | 8/8 | 0 |

The metric is `aggregate_tps/openai_continuous_usage`. Relative to the earlier
external CS512 quick matrix, the clean-image cells changed by -0.71%, -5.28%,
+3.96%, and -2.47% at C1/C2/C4/C8 respectively. This is a cross-run drift
comparison, not a sealed A/B. The first attempt used the harness's automatic
60-second readiness allowance: C1 was valid at 18.2 tok/s, while C2/C4/C8 were
correctly suppressed after warmup timeout. Those suppressions were readiness
timeouts, not KV-capacity failures. Unique 16K validation for this profile
therefore requires `--cell-warmup-timeout-seconds 300`.

### 2026-08-11 EXL3 R7 dynamic-NVFP4 CKV-gather candidate

The operator-running fixed-MTP4 profile uses dynamic per-token NVFP4 latent KV
plus FP8 RoPE, a 262,144-token request limit, a 4,096-token prefill ceiling,
and 9.25 GB of KV memory per rank. The runtime reported 1,156,864 KV tokens.
The matched CKV-gather A/B changed only these environment values:

```text
VLLM_B12X_MLA_CKV_GATHER:             unset -> 1
VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS:  unset -> 262144
```

The `llm_decode_bench.py` v0.4.31 workload used fully unique prompts,
temperature zero, and one cold scout at each prefill length. The decode row
used eight unique 16K contexts and a 25-second measurement window.

| Workload | NVFP4 control | CKV gather | Change |
|---|---:|---:|---:|
| 8K prefill | 435.19 tok/s | **499.82 tok/s** | **+14.85%** |
| 16K prefill | 437.16 tok/s | **608.33 tok/s** | **+39.16%** |
| 64K prefill | 434.20 tok/s | **563.21 tok/s** | **+29.71%** |
| 128K prefill | 424.60 tok/s | **551.25 tok/s** | **+29.83%** |
| C8 16K sustained decode | 45.4 tok/s | **47.85 tok/s** | +5.40% observed |

CKV gather is active during eligible pure prefill and not during decode. The
decode result therefore establishes no measured regression; its positive
difference is not attributed to CKV gather. Each prefill row is a single
sample, not a latency or throughput distribution.

Three 128-token and three 256-token greedy outputs remained byte-identical to
the MTP-disabled control, all 96 requested logprobs were finite, and fixed MTP4
accepted 1,036 of 1,072 draft tokens with position counts
`[266, 263, 255, 252]`. Every rank activated CKV gather, retained the exact
7,704/16 graph census, caught up transport sequences without fatal, overflow,
drop, or Q1-Q40 fallback, and returned to HTTP 200 with zero running, waiting,
KV use, and preemption.

The exact configuration, artifact hashes, control-artifact limitation, and
open capacity gates are in the
[fixed-MTP4 dynamic-NVFP4 candidate specification](EXL3_R7_FIXED_MTP4_PROFILE.md)
and
[sanitized machine-readable evidence](configurations/glm52-exl3-r7-mtp4-nvfp4-ckv-gather-20260811.json).

### 2026-08-12 EXL3 R7 exact-Q40 operator acceptance

The accepted operator 3.5-bpw profile adds a target-only routed-MoE state for
exactly 40 rows. It uses capacity 40 and route block 8 without changing Q1-Q32,
other prefill shapes, or the uniform draft path. The matched decode bracket
replayed the same eight unique 16K payloads with full 8/8 residency and a
25-second measurement window.

| Measurement | Control | Exact-Q40 profile | Change |
|---|---:|---:|---:|
| Warm C8 mean | 61.344 tok/s | **73.208 tok/s** | **+19.341%** |
| Slowest candidate versus fastest control | 62.907 tok/s | **72.297 tok/s** | **+14.93%** |
| Reported KV capacity | 1,156,864 tokens | **1,156,864 tokens** | unchanged |

The operator accepts the following client-timed C1 snapshot as the current
best measured prefill result for this configuration. The benchmark used 100%
unique generated contexts:

| Context | Client TTFT | Client-timed prefill | Samples |
|---|---:|---:|---:|
| 8K | 12.06 s | **679 tok/s** | 2 |
| 16K | 24.36 s | **673 tok/s** | 1 |
| 32K | 49.17 s | **666 tok/s** | 1 |
| 64K | 99.72 s | **657 tok/s** | 1 |
| 128K | 203.09 s | **645 tok/s** | 1 |

Throughput declines by only 5.0% from 8K to 128K while the context grows by
16x. The 8K, 16K, and 32K rows are respectively 1.6%, 2.1%, and 2.0% above the
earlier bounded snapshot. Several operator runs produced similar prefill
throughput, but the table reports only the visible samples. These are therefore
accepted best operating results, not throughput distributions. The benchmark
did not return server-side cached-token accounting for these cells, so cache
misses are not independently proven by the captured result. The transcription,
scope, and limitations are preserved in
[machine-readable form](configurations/glm52-exl3-r7-current-best-prefill-20260813.json).

The earlier bounded benchmark remains independently identified by complete
benchmark SHA-256
`feed67820caf37cc016473a38584b11b4205a628183f64e2b48b082a7bad2854`; it
measured 668, 659, and 653 tok/s at 8K, 16K, and 32K respectively.

The exact-Q40 prefill non-regression bracket measured medians of 526.449,
623.599, 615.930, 618.246, and 619.807 tok/s at 8K, 16K, 32K, 64K, and 128K.
The predeclared reducer returned failure only at 64K: 618.246 tok/s was 0.1215%
below the lower 618.998 tok/s baseline-envelope median. The operator accepted
that bounded difference as measurement-neutral without relabelling the machine
failure as a pass. All-rank runtime attestation, exact BF16 parity across all
75 target layers, deterministic 16K/32K output equality, graph capture,
transport convergence, API health, and final capacity passed.

This result is accepted only as the operator's four-Spark 3.5-bpw default. It
does not replace the clean-checkout 3.25-bpw public-functional default. The
complete contract and immutable receipt hashes are in the
[fixed-MTP4 specification](EXL3_R7_FIXED_MTP4_PROFILE.md) and
[operator-acceptance summary](configurations/glm52-exl3-r7-mtp4-q40-block8-20260812.json).

### 2026-08-11 EXL3 R7 fixed-MTP4 FP8 predecessor

On four directly cabled DGX Sparks, the public-functional R7 3.5-bpw research
checkpoint completed a bounded live qualification at TP4/DCP4, fixed MTP4,
Q1-Q40 CUDA graphs, `fp8_ds_mla`, and 9,250,000,000 KV-cache bytes per rank.
This is a live-validated candidate, not the advertised default or an accepted
public-functional matrix.

Three 128-token and three 256-token greedy completions matched the
MTP-disabled control byte-for-byte. A semantic probe passed, all 96 requested
logprob values were finite, and MTP4 accepted 1,001 of 1,032 draft tokens with
non-zero counts at all four draft positions. All ranks completed target,
draft-prefill, and draft-decode graph capture. Their transport census matched
at 7,704 native all-reduce nodes and 16 native vocabulary nodes, with zero
fatal, overflow, dropped-signature, or Q1-Q40 stock TP/vocabulary fallback
events. DCP and indexer collectives intentionally remained on the stock path.

The matched 25-second decode-only matrix used temperature zero:

| Concurrency | Aggregate tok/s | Change from matched fixed-MTP3 |
|---:|---:|---:|
| C1 | **34.60** | +12.34% |
| C2 | **51.44** | +9.08% |
| C4 | **76.96** | +10.38% |
| C8 | **85.68** | -11.63% |

The exact 1,024-prompt plus 128-output endpoint probe measured 33.04
inter-token decode tokens/s, 10.95% above the matched fixed-MTP3 control. A
separate unique-16K-context `llm_decode_bench.py` v0.4.31 matrix measured
30.93, 41.30, and 46.71 aggregate tokens/s at C2, C4, and C8. Its temperature
was zero. The canonical five-run coding-peak mode intentionally omitted a
temperature field and measured a 26.87 tokens/s median.

The 9.25 GB/rank pool reported 675,840 tokens; 675,584 were request-usable
after the runtime's permanent null block. Four C1 requests through 65,280
prompt tokens passed. The C8 residency arm then completed eight unique
64,000-prompt plus 1,408-output requests. A single scheduler scrape observed
eight running, zero waiting, and 77.226% KV use, proving at least 512,000
logical tokens resident simultaneously. All outputs had exact usage and finite
logprobs; preemption, OOM, transport-fault, and accepted-run thermal-violation
counts were zero. The service returned to HTTP 200 and zero scheduler/KV use.

The complete scope, limitations, configuration contract, and immutable raw
artifact hashes are in the
[fixed-MTP4 candidate specification](EXL3_R7_FIXED_MTP4_PROFILE.md) and
[sanitized machine-readable summary](configurations/glm52-exl3-r7-mtp4-kv925-20260811.json).
The separate promotion decisions for hybrid transport, custom indexer, custom
DCP, MTP2, MTP3, and KV growth are in the
[R7 optimization campaign record](EXL3_R7_OPTIMIZATION_20260811.md).

This document is the definitive record of SparkRing's measured performance. Every number here is a real measurement pulled from a dated deliverable, carries its full configuration label, and passed the verification gate stated on its row. Nothing in this document is a projection, an extrapolation, or a comparison against systems we did not measure ourselves.

If a number you have seen quoted about SparkRing does not appear here with a matching label, treat it as unofficial.

**A note on source citations.** The dated deliverables cited throughout this document — `deliverables/*.md`, `deliverables/evidence/*`, plus `CURRENT_STATUS.md`, `HANDOFF.md`, and `PUBLIC_RELEASE.md` — live in the maintainer's private evidence archive (paths below are archive-relative, not in this repository). They are retained here as the evidence trail behind each claim. Only `README.md` and the `spark_transport/` documents cited below are files in this repository; section 6 lists which is which.

---

## 1. Shared configuration

Unless a row says otherwise, every end-to-end serving result below was measured on this configuration:

- **Topology:** 4x NVIDIA DGX Spark in a **switchless, directly cabled ring**. Each ring edge is one direct ConnectX-7 200 Gbit/s link. There is no Ethernet switch on the data path, and NCCL Socket is not on the data path in the switchless-IB rows.
- **Checkpoint:** `aidendle94/GLM-5.2-MXFP4-Experts-GPTQ` — 382 GiB, loaded via the safetensors loader.
- **KV cache:** `nvfp4_ds_mla` with the per-token scale ABI, on all rows dated 2026-07-27 or later.

Where a row deviates (different DCP degree, eager vs. CUDA-graph execution, fallback transport), the deviation is stated in the row's claim label.

---

## 2. End-to-end serving results

These are end-to-end serving measurements: a real model, real requests, tokens delivered to a client. Each claim label states topology, parallelism config, execution mode, workload, concurrency, statistic, and the gate the run had to pass.

| # | Result | Full claim label | Source |
|---|---|---|---|
| 1 | **834 / 884 / 854 tok/s uncached prefill at 8K / 16K / 32K** | 4x Spark switchless ring; MXFP4-Experts-GPTQ; TP4/DCP4 (ag_rs), adaptive MTP2/4 window 32, FULL_AND_PIECEWISE CUDA graphs; custom SparkRing hot paths + checksum-pinned patched NCCL 2.30.7 NET/IB fallback; C1, one request at a time; end-to-end serving; **single-sample "integrated one-sample scout" per context, not prefix-cache-hit** (TTFT 9.828 / 18.359 / 37.844 s); gate: zero request errors, live graph census + 32-token live request gate passed, no NET/Socket data path | `deliverables/glm52-dcp4-switchless-result-20260727.md` ("C1 prefill and decode"); `CURRENT_STATUS.md`; `HANDOFF.md` |
| 2 | **63.60 tok/s aggregate sustained decode at C8** (7.95 tok/s per user; 3.51x C1; still rising at the configured C8 cap) | 4x Spark; MXFP4-Experts-GPTQ; TP4/**DCP1**, adaptive MTP2/4, Q40 capture plan (FULL+PIECEWISE), max sequences 8; fully **shared-prefix** 8K contexts; controlled **15-second sustained-decode cells**; aggregate (per-user also reported); end-to-end serving; single cell per concurrency; gate: zero errors/queueing at every C1–C8 step, 5,464 custom all-reduce + 24 custom vocabulary graph captures, **zero stock captures**, no Q1–Q40 stock fallback in decode | `deliverables/glm52-concurrency-fast-path-20260727.md` ("Passed C8 candidate"); `README.md` "Current result"; evidence: `deliverables/evidence/glm52-c8-baseline-20260727.json` |
| 3 | **20.83 / 19.28 / 21.43 tok/s p50 decode at 8K / 16K / 32K — sealed custom/custom CUDA-graph C1 cell (v40, 2026-07-28)** | 4x Spark; MXFP4-Experts-GPTQ; TP4/**DCP4**, adaptive MTP2/4 window 32, Q40 FULL_AND_PIECEWISE with the full custom DCP trio captured (1,272 query + 1,272 combine nodes/rank; only 360 stock `dcp_owner_topk_all_gather` nodes/rank, capture-unsupported by design); C1 single stream; 30 s sustained decode per context; per-user = aggregate; end-to-end serving; statistic p50 (inter-token p50 48.0 / 51.9 / 46.7 ms, TTFT p50 0.75 / 0.89 / 0.94 s); gate: sealed cell `custom-dcp-cell-20260728T023220Z`, cell-validation passed, zero JIT events, frozen transport counters, zero target-family stock fallback | `deliverables/dcp4-v40-window-result-20260728.md`; `README.md` "CUDA-graph checkpoint" |
| 4 | **27.246 median / 29.295 max tok/s, five-run coding-peak** | 4x Spark; MXFP4-Experts-GPTQ; TP4/DCP1 ("MIDWAY GOOD" checkpoint: adaptive MTP2/4 window 32, capture sizes 1/3/5, max seqs 3, NCCL Socket fallback in this config); C1; coding-peak prompts, 5 sequential runs, 30-second windows, max_tokens 2000; per-user = aggregate; end-to-end serving; statistic: median / mean 27.615 / max / min 26.690 over 5 runs; gate: all 5 completed, 0/5 CJK output; benchmark JSON SHA-256 pinned | `deliverables/glm52-midway-good-checkpoint-20260727.md` ("Five sequential coding-peak runs"); cross-referenced in `deliverables/glm52-dcp2-switchless-result-20260727.md` |
| 5 | **43.0 and 42.6 aggregate tok/s structured-code C2 peaks** | 4x Spark; MXFP4-Experts-GPTQ; TP4/DCP1, adaptive MTP2/4, Q10 C2 candidate (capture sizes 1–10); C2, two simultaneous real "write me a webpage" prompts via OpenWebUI; aggregate; end-to-end serving; statistic: two separate **ten-second server-side log windows** (not a controlled bench cell); gate: the 43.0 window had mean acceptance length 4.30 and 82.5% draft acceptance; explicitly labeled a workload-dependent peak, **not** the C2 baseline | `deliverables/glm52-concurrency-fast-path-20260727.md` ("Passed C2 candidate"); `README.md` "Current result" table |
| 6 | **Historical eager single-stream: 20.83 tok/s pinned / 19.88 tok/s novel-code median, with 708.1 tok/s uncached 32K prefill** | 4x Spark; MXFP4-Experts-GPTQ; TP4/DCP1, fixed-K4 MTP with B12X remap, **eager** (pre-graph) execution; exact-32K pinned context, C1; per-user = aggregate; end-to-end serving; statistics: 20.832311 median periodic pinned gate (G0d-c), 20.694064 median with proposal-scoped reuse (G0d-e — the README's "20.7 pinned"), 19.880764 median over **five runs** on the novel-code suffix (G0d-f); 708.081532 tok/s uncached 32K prefill (G0d-e); gate: coherent output + bounded numerical envelope; 100% P0–P3 acceptance on the pinned gate (pinned context makes MTP acceptance unusually easy — see caveats) | `deliverables/goal-ledger.md` rows G0d-c/e/f; `README.md` "Current result" |
| 7 | **375,040 KV tokens (4x the DCP1 pool) with 56.70 tok/s aggregate C8 decode** | 4x Spark; MXFP4-Experts-GPTQ; TP4/**DCP4** switchless (same config as row 1); KV allocation 3.0 GB/rank `nvfp4_ds_mla`; C8, fully **shared-prefix** 8K contexts, ~30-second cells; aggregate (7.09 tok/s per user); end-to-end serving; single cell; gate: zero errors/queueing, all-rings NET/IB, capacity 5.72x a full 65,536-token context, ~4.05 GiB limiting-rank MemAvailable after graph capture | `deliverables/glm52-dcp4-switchless-result-20260727.md`; `CURRENT_STATUS.md` |

### What each row means

1. The highest prefill throughput documented in this repository. DCP4 beat DCP1 prefill at every context length (+2.2% to +5.6%) while quadrupling KV capacity. This is an internal comparison only — this repository makes no claims about systems outside it.
2. The flagship aggregate number: 3.51x single-stream scaling on a fully verified zero-stock-capture custom-transport graph path, with throughput still rising at the configured concurrency cap of 8.
3. The newest sealed result (2026-07-28): the first time the full custom DCP query+combine trio ran inside FULL CUDA graphs. At 32K it lands roughly 12% above the earlier 19.14 tok/s stock-trio DCP4 measurement — **that comparison is indicative, not a sealed A/B** (the stock control was not rerun, and adaptive-MTP acceptance variance is real).
4. The best honest realistic-workload single-stream numbers in the repository: a real coding workload sustained ~27 tok/s median on four consumer-class desk units.
5. Proof the repaired C2 path can exceed 40 aggregate tok/s when MTP4 stays coherent — but it is a 10-second server-window observation on one workload, and must always be quoted with that label.
6. The historical single-stream mechanics checkpoint that the CUDA-graph work is measured against. The 708 tok/s prefill is a genuine uncached 32K prefill gate.
7. The capacity story: 4x the KV pool at a 4.4% single-stream 32K decode cost versus DCP1 (19.14 vs. 20.02 tok/s) — the repository's strongest capacity-vs-speed trade result.

---

## 3. Transport microbenchmarks

These are **transport-level** measurements: no model in the loop unless a row says otherwise. They establish the latency floor and correctness backbone that the serving results above are built on. Do not quote them as serving numbers.

| # | Result | Full claim label | Source |
|---|---|---|---|
| T1 | **4.752 us p50 / 5.536 us p99 RDMA RC write** (host memory); **4.528 us p50** GPU-produced `cudaHostAllocMapped` | 2 Sparks, one direct CX-7 200G link; 16 KB payload, 10,000 iterations; transport-only (write + local CQ completion; GPU produce/verify outside the timed loop for the host row); statistic p50/p99; gate: byte-correct | `spark_transport/README.md` "Results" tables |
| T2 | **20.67 us p50 / 20.99 us p99 GPU-visible closed-loop round trip, 16 KB** | 2 Sparks, direct CX-7 link; persistent GPU doorbell loop (host command → sender GPU produces → ordered RC payload+sequence writes → receiver GPU consumes with vectorized 1,024-thread verification → RC ack → sender observes); 10,000 iterations; transport-only; p50/p99; gate: all correct. Payload sweep 4 KB→64 KB: 19.62→26.91 us p50, 70,000 iterations all correct | `spark_transport/README.md` "Persistent GPU doorbell results" |
| T3 | **20.27 us p50 TP2 BF16 exchange + fused add of one 6,144-wide GLM hidden vector (12 KB)**; 16 KB one-million-iteration burn **22.544 us p50, zero mismatches** | 2 Sparks, direct CX-7 link; synthetic collective primitive (publish, RDMA-write, fused local+remote BF16 add, validate, ack); transport-only; p50/p99; gate: every result validated, 0 mismatched iterations in the 1M burn | `spark_transport/README.md` "TP2 BF16 exchange and fused add" |
| T4 | **1.91–1.94 us host graph submit, 39.06 us device time per call** (1,000 replays); 100 replays: 1.95–2.01 us / 39.14–39.15 us; 10,000 replays: 35.15 us host burst average incl. CUDA backpressure, 39.15 us device | 4 Sparks, direct-cable ring; device-published 64-slot mapped command ring, Q1 all-reduce CUDA-graph replay, graph submission and transport progress on separate CPUs (10/11); transport-only standalone gate; statistic: range over runs; gate: pass — identical staged-executable SHA-256 on all ranks, exact published/consumed/completed sequences, zero overflow/mismatch; Q1 slightly faster than the ~42 us eager path | `README.md` "CUDA-graph checkpoint" table; `spark_transport/GRAPH_NATIVE_TP4_Q1.md` |
| T5 | **159.20 us per collective, DCP query+combine 14-bucket graph probe** | 4 Sparks, direct-cable ring; model-down (no model loaded) v40 probe; 2,856 sequences; transport-only **mixed-bucket average — not a single-shape latency**; gate: 4/4 ranks pass, zero query byte mismatches, combine within numerical envelope | `deliverables/dcp4-v40-window-result-20260728.md` "Model-down probes" |
| T6 | **Native DCP primitives standalone: query all-gather 110.370 us @Q5 (byte exact); fused latent-512 online-softmax combine ~86.9 us @Q1 / 330.1 us @Q5 (bounded vs FP32 and stock); token-major vocabulary all-gather 399.442 us @Q5 (byte exact)** | 4 Sparks, direct-cable ring; fixed-K4 native probe set Q1–Q5; transport-only representative standalone times; gate: query and vocabulary passed byte-exact live shadow at Q1/Q3/Q5; combine live shadow Q5 had zero tolerance/non-finite failures, max output error 0.00012207, max LSE error 9.53674e-7 | `README.md` "Native direct-cable primitives" |
| T7 | **DCP4 eager collective probe p50 (rank 0): TP all-reduce `[1,6144]` BF16 55.74 us; query all-gather Q1 47.52 us / Q40 206.46 us; output reduce-scatter Q1 40.45 us** | 4 Sparks, direct-cable ring; pre-model four-rank communicator over the switchless patched NCCL-IB / custom stack; isolated primitives, explicitly **not an end-to-end critical-path sum**; statistic p50; gate: every eager and graph case passed value validation | `deliverables/glm52-dcp4-switchless-result-20260727.md` "Four-rank DCP4 collective probe" |

### What each transport row means

- **T1:** Sub-5-us one-way RDMA writes over a bare direct cable — the floor the whole stack is built on, with GPU-mapped memory costing essentially nothing extra.
- **T2:** The full GPU-to-GPU visible loop (produce, ship, consume, ack) in ~21 us at 16 KB. This is the "RDMA RTT ~20.67 us" number, and it includes GPU-side verification, not just NIC completion.
- **T3:** A real fused collective primitive — exchange+add of an actual GLM hidden vector — at the same ~20 us scale, validated for one million iterations without a single mismatch.
- **T4:** ~1.9 us host-side cost to fire a captured four-rank collective — the mechanism that makes CUDA-graph decode viable — with byte-audited rank-synchronous replay through 10,000 calls.
- **T5:** The v40 promotion evidence that the custom DCP query+combine graphs are fast and byte-correct before any model touches them. It is a mixed-bucket average; do not quote it as a single-op latency.
- **T6:** Each custom primitive individually validated byte-exact or within a stated numerical envelope — the correctness backbone behind every "zero stock capture" serving claim above.
- **T7:** The per-op price list for the DCP4 attention path on the fallback stack; useful context for why the custom-trio promotion (row 3 above) matters.

---

## 4. Methodology: the claim discipline

Use the [matched 16K sustained-decode checklist](EVIDENCE_COMPARISON_CHECKLIST.md)
and `scripts/compare_benchmark_evidence.py` for fail-closed offline comparison
of two `llm_decode_bench.py` v0.4.31 matrix documents.

The numbers above are only as good as the rules used to produce and report them. Those rules are part of the result, so they are stated here in full.

**Every number carries a full label.** A performance claim in this repository is not a bare number — it names the topology, the checkpoint, the parallelism configuration (TP/DCP degree), the execution mode (eager vs. CUDA-graph, custom vs. fallback transport), the workload and concurrency, the statistic (p50, median-of-N, single sample, log-window observation), and the gate the run had to pass. A number quoted without its label is considered misquoted.

**Sealed cells.** The strongest results come from sealed benchmark cells (e.g. `custom-dcp-cell-20260728T023220Z`): a cell is configured, run, and validated as a unit, with cell-validation checks (zero JIT events, frozen transport counters, zero target-family stock fallback) that must pass before the number is recorded. Machine-readable evidence JSONs are stored alongside the prose deliverables in the evidence archive, and benchmark artifacts are SHA-256 pinned where noted.

**Fail-closed transport validation.** Custom transport paths are never trusted by construction — they are validated fail-closed. Collectives run under live shadow comparison against reference implementations (byte-exact for data-movement collectives; bounded numerical envelope with recorded max error for arithmetic collectives), graph captures are audited by census (counting custom vs. stock nodes per rank), and executables are checked for identical SHA-256 across ranks. A serving row's "zero stock captures" gate means the capture census confirmed it, not that it was assumed.

**Measured vs. indicative labeling.** Only same-cell, same-config measurements are reported as sealed comparisons. Where two numbers come from different cells or configurations, any comparison between them is labeled *indicative* (see the v40-vs-stock-trio caveat below). Cross-configuration percentage deltas with mismatched cell durations are labeled *directional*.

**No projections, no external comparisons.** This document contains measured results only. Excluded by policy: capacity projections (e.g. larger per-rank KV allocations not yet run), planned-topology arithmetic, throughput goals, latency targets not yet achieved, rejected candidate configurations, and cache-hit observations that the repository itself excludes from headline reporting (e.g. an 8,005 tok/s cached-prefix observation, which is a prefix-cache artifact and not a prefill result). The repository makes internal comparisons only (e.g. "+3.0% vs DCP1 16K") and claims nothing relative to systems it has not measured.

---

## 5. Honest caveats

Read these before quoting anything above. They are part of the claims, not footnotes to them.

- **Concurrency rows are shared-prefix baselines.** The C1–C8 results (rows 2 and 7) use fully shared-prefix contexts. They are concurrency scaling baselines — *not* a unique-context capacity result, not a prefill result, and not a finite-request completion result.
- **Cross-DCP concurrency deltas are directional.** The DCP1 C8 cells were 15-second windows while the DCP4 C8 cells were ~30-second windows, so percentage comparisons between them (e.g. 63.60 vs. 56.70 aggregate) are directional, not sealed same-protocol comparisons.
- **The v40 32K "+12% vs. stock trio" is indicative, not a sealed A/B.** The stock-trio control was not rerun alongside the v40 cell, and adaptive-MTP acceptance variance is real. The 20.83/19.28/21.43 numbers themselves are sealed; the *improvement percentage* is not.
- **Prefill scouts are single-sample.** The 834/884/854 tok/s prefill figures (row 1) are single-sample integrated one-sample scouts per context length — genuine uncached prefill (not prefix-cache-hit), but one sample each, not a distribution.
- **The 43 tok/s C2 figure is a workload peak, not a baseline.** It comes from two ten-second server-side log windows on one structured-code workload, not a controlled benchmark cell. The C2 baseline is lower.
- **Pinned-context MTP acceptance is unusually easy.** The 100% P0–P3 acceptance on the historical pinned gate (row 6) is a property of the pinned-context setup; the novel-code median (19.88 tok/s) is the honest workload estimate for that checkpoint.
- **C8 was a configured cap, not a saturation point.** Aggregate throughput was still rising at C8 in row 2; the scaling curve beyond C8 is unmeasured.
- **T5 is a mixed-bucket average.** The 159.20 us DCP graph-probe figure averages 14 bucket shapes; it is not the latency of any single collective shape.
- **Transport rows are not serving rows.** T1–T7 have no model in the loop (T5 explicitly model-down) and are floors/price-lists, not end-to-end results. T7 in particular is a set of isolated primitive timings, not a critical-path sum.

---

## 6. Provenance

Primary sources in this repository:

- `README.md` — current-result and CUDA-graph checkpoint tables
- `spark_transport/README.md` and `spark_transport/GRAPH_NATIVE_TP4_Q1.md`

Primary sources in the maintainer's private evidence archive (archive-relative paths — see the note in the introduction):

- `CURRENT_STATUS.md`
- `HANDOFF.md`
- `PUBLIC_RELEASE.md` — the claim-discipline checklist this document follows
- `deliverables/glm52-dcp4-switchless-result-20260727.md`
- `deliverables/glm52-concurrency-fast-path-20260727.md`
- `deliverables/glm52-midway-good-checkpoint-20260727.md`
- `deliverables/dcp4-v40-window-result-20260728.md`
- `deliverables/dcp4-sequence67-resolution-20260727.md`
- `deliverables/goal-ledger.md`

Machine-readable evidence, in the archive:

- `deliverables/evidence/glm52-c8-baseline-20260727.json`
- `deliverables/evidence/glm52-dcp4-switchless-baseline-20260727.json`
- `deliverables/evidence/dcp4-custom-dcp-performance/custom-dcp-cell-20260728T023220Z/`
