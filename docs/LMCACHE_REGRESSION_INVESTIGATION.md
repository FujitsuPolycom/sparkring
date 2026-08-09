# LMCache throughput regression and EXL3 prefill limits: ranked hypotheses

Status: **offline investigation from repository code and configuration;
no live experiments were run. All hypotheses require bounded
one-variable-at-a-time live experiments to confirm or reject.**

This document investigates realistic causes of two observed issues
from the evidence already in the repository:

1. **LMCache throughput regression**: the clean-checkout CS512
   sustained-16K matrix showed -0.71% / -5.28% / +3.96% / -2.47%
   at C1/C2/C4/C8 versus the earlier external CS512 quick matrix
   (RESULTS.md). The C0-best → CS512 chunk-size change produced
   -0.71% C8 median regression. The campaign's NPC-off arm showed
   C8 fairness failures (lane collapse to 33-78% of median).

2. **EXL3 prefill limits**: prefill batch capacity is capped at
   `--max-num-batched-tokens 4096` with
   `VLLM_EXL3_PREFILL_BLOCK_M=64` for the Trellis prefill plan.
   `VLLM_EXL3_PREFILL_CHUNK=128` only controls the eager parity
   fallback path (m < `VLLM_EXL3_TRELLIS_MIN_M=1`); the full
   Trellis prefill plan handles m > `TRELLIS_MAX_M=32` up to the
   4096-token batch capacity. The Q6144 arm (4096 → 6144 batched
   tokens) was deferred. The NF3 lane sets
   `VLLM_SPARK_TP4_PREFILL_Q512=1` (transport/allreduce query
   capacity opt-in) but the EXL3 recipe sets it to `0`.

## Three cache layers — must not be conflated

| Cache | Implementation | Control | State in EXL3 recipe |
|---|---|---|---|
| **Native APC** | vLLM `--enable-prefix-caching` | In-engine | **Enabled** |
| **LMCache** | `lmcache.integration.vllm.lmcache_mp_connector` | External server process | **Enabled** (CS512, 1 GiB L1 lazy) |
| **SparkCache** | `sparkcache/` | `SPARK_CONTEXT_CACHE_ENABLE` | **Disabled** (=0) |

A throughput regression in the LMCache-on path could originate from
any of these layers or their interaction. Disabling one and measuring
the other two is the only way to isolate the source.

## LMCache throughput regression hypotheses (ranked by evidence)

### H1: LMCache connector overhead on the decode hot path (HIGH confidence)

**Evidence**: The recipe uses `transfer_mode: lmcache_driven` with
`max_gpu_workers: 1` and `max_cpu_workers: 2`. Every decoded chunk
triggers the LMCache connector to check, store, or retrieve KV shards
through the MP connector. The connector runs in the engine's decode
loop, so any per-step overhead directly reduces aggregate tok/s.

**Mechanism**: `lmcache_driven` mode means the engine's attention
backend calls the connector on every chunk boundary (512 tokens). At
C8 with 8 concurrent sequences, the connector is called 8 times per
decode step. If the connector's lookup or store path adds even ~0.5ms
per call, that is 4ms per step across 8 sequences — enough to explain
a 2-5% regression at C8.

**Proposed experiment**: Run the standard 16K matrix with LMCache
servers running but the connector disabled (if the integration
supports a passthrough/no-op mode), then with the connector enabled.
Same image, same configuration, same run session. One variable: connector
active vs inactive. Measure C1/C2/C4/C8 delta.

### H2: LMCache L1 lazy allocation and unified-memory contention (MEDIUM confidence)

**Evidence**: The recipe uses `l1_lazy: true` with `l1_init_size_gb: 0`
and `l1_size_gb: 1`. On DGX Spark (GB10), GPU memory and host memory
share a unified memory pool (~120 GiB). LMCache L1 is RAM-backed.
The first warm probe triggers lazy allocation of up to 1 GiB of
unified memory per rank. This allocation competes with the engine's
KV cache (4.5 GB/rank) and model weights (84.43 GiB/rank) for the
same physical memory pool.

**Mechanism**: When LMCache L1 lazily allocates during a warm request,
the allocator may cause page faults or memory pressure on the engine's
active KV cache, slowing decode. The L0 arm's CUDA OOM during
`REGISTER_KV_CACHE` at the full 9 GB/rank envelope with eager L1
confirms this memory pressure path is real.

**Proposed experiment**: Run the 16K matrix with `l1_size_gb: 0`
(LMCache enabled but L1 size zero — effectively connector-only, no
storage) versus `l1_size_gb: 1` (current recipe). Same image, same
run session. One variable: L1 size. Measure C1/C2/C4/C8 and
`free -h` during the run.

### H3: LMCache message-queue timeout and retry overhead (MEDIUM confidence)

**Evidence**: The recipe sets `mq_timeout_seconds: 10` and
`heartbeat_interval_seconds: 10`. If the MQ timeout fires during a
decode step (e.g. the server is briefly saturated), the connector
must retry or fall back, adding latency.

**Mechanism**: At C8 with 8 concurrent sequences hitting 4 LMCache
servers, the message queue can transiently saturate. A 10-second
timeout is generous, but if the server is briefly unresponsive (e.g.
during L1 eviction under LRU), the connector blocks. This would
manifest as lane collapse at C8 — exactly what the NPC-off arm
observed (minimum lane 33-78% of median).

**Proposed experiment**: Run the 16K C8 matrix with
`mq_timeout_seconds: 1` (aggressive) versus `10` (current). Same
image, same run session. One variable: MQ timeout. Measure C8 lane
fairness (minimum lane / median lane ratio).

### H4: Native APC + LMCache interaction (MEDIUM confidence)

**Evidence**: The recipe enables both native APC
(`--enable-prefix-caching`) and LMCache. The C0-best fresh prefix
probe showed a 67.97x warm/cold ratio attributed to "combined
NPC+LMCache warm-path benefit." But the interaction is bidirectional:
APC may serve a warm hit before LMCache is consulted, making the
LMCache connector overhead pure cost with no benefit on repeated
prefixes.

**Mechanism**: When APC hits, the engine skips the attention recompute
but the LMCache connector still runs its lookup (and potentially its
store) on every chunk. If APC already served the prefix, the LMCache
lookup is wasted overhead. This is consistent with the C1 regression
being small (-0.71%) but C2 being larger (-5.28%): at C2, more
concurrent streams mean more connector calls with diminishing APC hit
benefit.

**Proposed experiment**: Run the 16K matrix with four configurations
in one session: (A) APC on + LMCache on (current), (B) APC on +
LMCache off, (C) APC off + LMCache on, (D) APC off + LMCache off.
Same image, same run session. One variable: cache layer state.
Measure C1/C2/C4/C8 for each.

### H5: CS512 chunk-size 512 vs 256 decode overhead (LOW-MEDIUM confidence)

**Evidence**: CS512 changed `chunk_size` from 256 to 512. The C8
median regression was -0.71% (within noise), but the clean-checkout
matrix showed -2.47% at C8. A larger chunk size means fewer but
larger connector operations per decode step, which could reduce
per-step overhead but increase memory pressure per operation.

**Mechanism**: With chunk_size=512, each LMCache store/retrieve
handles 512 tokens worth of KV (vs 256). On GB10's unified memory,
a 512-token chunk may trigger a larger DMA or memcpy, potentially
stalling the decode loop. The prefill gain from CS512 (+8.64% at 16K)
comes from fewer, larger operations on the prefill path, but decode
has different latency sensitivity.

**Proposed experiment**: Run the 16K matrix with `chunk_size: 256`
versus `chunk_size: 512`. Same image, same run session. One variable:
chunk size. Measure C1/C2/C4/C8. This is a direct A/B of the
campaign's CS512 promotion delta.

### H6: `--no-async-scheduling` serializing decode steps (LOW confidence)

**Evidence**: The recipe uses `--no-async-scheduling`. With
asynchronous scheduling disabled, each decode step completes before
the next begins. LMCache connector calls are therefore synchronous
in the decode loop, adding their full latency to each step.

**Mechanism**: If async scheduling were enabled, the connector's
lookup/store could overlap with the next step's computation. With it
disabled, the connector overhead is pure serial latency. This setting
is required for the custom transport path (the recipe comment and
SETUP.md document this), so it cannot be changed independently.

**Proposed experiment**: This cannot be tested as a single variable
because `--no-async-scheduling` is required for the custom TP4
transport. Marking as low confidence because it is a fixed constraint,
not a tunable. Documented as context for why connector overhead
### P1: `VLLM_EXL3_PREFILL_CHUNK=128` only caps the eager parity fallback, not the main prefill path (MEDIUM confidence, downgraded from HIGH)

**Evidence**: From `runtime/exl3/overlay/vllm/model_executor/layers/quantization/exl3.py`:
`prefill_plan_enabled = prefill_trellis and max_batched_tokens > max_trellis_m`
(line 2005). When `prefill_plan_enabled` is true (which it is: `prefill_trellis=1`
and `max_batched_tokens=4096 > max_trellis_m=32`), the full Trellis
`prefill_plan` runs for all batches with m > `max_trellis_m=32`, planned
with `max_batched_tokens=4096` capacity and `prefill_block_m=64`.
`PREFILL_CHUNK=128` only sets `parity_rows = min(chunk, max_batched_tokens)`
(line 2007), which governs the eager parity fallback path for
m < `min_trellis_m=1` — a narrow window that is rarely hit in
practice since `min_trellis_m=1` means the Trellis plan handles
m >= 1.

**Mechanism [INFERENCE]**: The eager parity path (m < min_trellis_m) is
almost never reached because `min_trellis_m=1`. The Trellis prefill
plan handles all m >= 1 up to 4096 with block_m=64. Therefore
`PREFILL_CHUNK=128` has minimal effect on prefill throughput for the
common case. The prefill plan's block_m=64 and max_batched_tokens=4096
are the real determinants of prefill batch capacity.

The NF3 comparison is not about prefill graph capture.
`VLLM_SPARK_TP4_PREFILL_Q512` controls the native TP4 all-reduce
session's maximum query row capacity
(`_maximum_allreduce_query_rows` in `spark_tp4_backend.py:146-151`)
and graph arena bytes (`_graph_capacity_bytes`, line 154-157), not
CUDA graph bucket creation for prefill.

**Proposed experiment**: Run a single C1 16K prefill with
`VLLM_EXL3_PREFILL_CHUNK=256` versus `128`. Same image, same run
session. One variable: prefill chunk. Measure TTFT. If TTFT is
unchanged (as the code suggests), the chunk size is not a prefill
bottleneck. If it changes, investigate whether the parity path was
hit unexpectedly.

### P2: `VLLM_SPARK_TP4_PREFILL_Q512=0` limits transport/allreduce query capacity (LOW confidence, downgraded from MEDIUM)

**Evidence**: From `spark_transport/integrations/vllm/spark_tp4_backend.py`:
`_prefill_q512_enabled()` (line 137-143) controls
`_maximum_allreduce_query_rows()` (line 146-151) and
`_graph_capacity_bytes()` (line 154-157). When enabled, the native
contiguous BF16 TP all-reduce session admits PIECEWISE widths through
Q512 with a larger arena. This is a **transport/allreduce** capacity
setting, NOT a CUDA graph bucket creation setting for prefill. The
README confirms: "the native contiguous BF16 TP all-reduce session may
admit those PIECEWISE widths through Q512 when its arena was created
with matching capacity. Query, vocabulary, and DCP collectives remain
bounded at Q40."

The claim that Q512 "enables CUDA graph capture for prefill at Q512
sizes (48,72,144,224,288,352,432,512)" was **unsupported** —
those are PIECEWISE capture bucket sizes governed by
`VLLM_SPARK_PREFILL_PIECEWISE_CAPTURE_SIZES`, not by Q512.

**Mechanism [INFERENCE]**: With Q512=0, the all-reduce session's query
capacity is limited to `MAX_QUERY_ROWS` (Q40-bounded). This affects
the transport layer's ability to handle larger all-reduce payloads
during prefill, but does not directly control whether prefill CUDA
graphs are captured. The EXL3 recipe's prefill performance may be
limited by transport query capacity if prefill all-reduces exceed
Q40, but this is a transport bottleneck, not a graph-capture
bottleneck.

**Proposed experiment**: Attempt `VLLM_SPARK_TP4_PREFILL_Q512=1`
with the current EXL3 recipe and observe whether the transport session
initializes and whether prefill all-reduces are admitted. If it
initializes, run a C1 16K prefill and compare TTFT. If it fails to
initialize, document the failure mode.

### P3: `VLLM_SPARK_PREFILL_PIECEWISE_CAPTURE_SIZES=""` — empty in both EXL3 and NF3 (LOW confidence, downgraded from MEDIUM)

**Evidence**: The EXL3 recipe sets
`VLLM_SPARK_PREFILL_PIECEWISE_CAPTURE_SIZES=""` (empty string).
The NF3 lane also sets `VLLM_SPARK_PREFILL_PIECEWISE_CAPTURE_SIZES=""`
(see `docs/configurations/glm52-nf3-live-1m-20260731.json` line 375).
The previous claim that "the NF3 lane and the reference lane use
non-empty piecewise capture sizes for prefill" was **false** — both
lanes use an empty string. Piecewise capture sizes are a graph-bucket
contract setting (see `spark_cudagraph_bucket_contract.py:17-18`),
not a prefill-specific setting.

**Mechanism [INFERENCE]**: With an empty piecewise capture list, no
piecewise prefill graphs are captured. However, since NF3 also uses
an empty list and NF3 is accepted, this setting is not a
differentiator between the two lanes. The decode path uses
`VLLM_SPARK_GRAPH_CAPTURE_SIZES` (non-empty) for CUDA graph capture.

**Proposed experiment**: This is not a differentiating variable
between EXL3 and NF3. Populating
`VLLM_SPARK_PREFILL_PIECEWISE_CAPTURE_SIZES` with sizes like
"128,256,512" would violate the max padding 32 constraint
(`TRELLIS_MAX_M=32`) unless the bucket contract validates them
against the Trellis window. Any experiment must first verify that
proposed sizes pass the bucket contract validation in
`spark_cudagraph_bucket_contract.py` before attempting live
capture.

### P4: `--max-num-batched-tokens 4096` limits prefill admission (LOW-MEDIUM confidence)

**Evidence**: The recipe uses `--max-num-batched-tokens 4096`. The
Q6144 arm (4096 → 6144) was deferred because it required "at least
5% bracketed cold-prefill/admission gain and at most 3% decode
regression." The 4096 limit means a 16K prompt must be processed in
at least 4 chunks (16384/4096), each requiring a full attention pass.

**Mechanism**: Increasing `max-num-batched-tokens` would allow
larger prefill batches, reducing the number of forward passes for
long contexts. However, this also increases memory usage per batch
and could affect decode scheduling (the same budget is shared). The
Q6144 arm was deferred, not rejected, suggesting the gain is
plausible but unproven.

**Proposed experiment**: Run C1 16K prefill with
`--max-num-batched-tokens 4096` versus `6144`. Same image, same run
session. One variable: max batched tokens. Measure TTFT and decode
regression at C1/C2/C4/C8. This is the deferred Q6144 arm.

## Experimental protocol

All experiments must follow the evidence-comparison checklist
([EVIDENCE_COMPARISON_CHECKLIST.md](EVIDENCE_COMPARISON_CHECKLIST.md)):

1. One variable at a time; all other settings identical
2. Same image digest, same model revision, same hardware
3. Same run session (alternating order if possible)
4. Standard 16K sustained matrix: 25s cells, 1024 max tokens, temp 0,
   0% unique / 100% shared contexts, DCP4, KV budget 562688
   (auto-detected), 0s decode warmup, skip prefill, 600s cell-warmup
   timeout
5. Require exact effective concurrency and zero errors
6. Report deltas with claim labels, not bare numbers
7. Never compare bounded 128-token gate figures with sustained matrix

## Claims deliberately NOT made

- No live experiment was run. All hypotheses are from code/config analysis.
- No throughput number is claimed or projected.
- No hypothesis is promoted to confirmed without live evidence.
- The EXL3 prefill block sizes (BLOCK_M=64, CHUNK=128, TRELLIS_BLOCK_M=8)
  are required for EXL3 3.25-bpw correctness; changing them risks output
  divergence and is not recommended without a correctness gate.
- `--no-async-scheduling` is required for the custom TP4 transport
  and is not a tunable variable.
- SparkCache is disabled and is not implicated in any LMCache
  regression hypothesis.
- `VLLM_EXL3_PREFILL_CHUNK` only controls the eager parity fallback
  path (m < min_trellis_m), not the main Trellis prefill plan. The
  full prefill plan handles m > max_trellis_m up to max_batched_tokens.
- `VLLM_SPARK_TP4_PREFILL_Q512` controls transport/allreduce query
  capacity, not CUDA graph bucket creation for prefill.
- `VLLM_SPARK_PREFILL_PIECEWISE_CAPTURE_SIZES` is empty in both EXL3
  and NF3; it is not a differentiating variable between the two lanes.
- Proposed piecewise capture sizes must not violate the max padding 32
  constraint (`TRELLIS_MAX_M=32`) and must pass the bucket contract
  validation before live capture is attempted.
- Hypotheses labeled `[INFERENCE]` are derived from code reading, not
  from live measurement. They require live evidence to confirm or reject.
