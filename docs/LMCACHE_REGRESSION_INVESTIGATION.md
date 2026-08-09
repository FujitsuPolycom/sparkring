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

2. **EXL3 prefill limits**: prefill is capped at
   `--max-num-batched-tokens 4096` with
   `VLLM_EXL3_PREFILL_CHUNK=128` and `VLLM_EXL3_PREFILL_BLOCK_M=64`.
   The Q6144 arm (4096 → 6144 batched tokens) was deferred. The
   NF3 lane enables `VLLM_SPARK_TP4_PREFILL_Q512=1` (Q512 prefill
   opt-in) but the EXL3 recipe sets it to `0`.

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
matters more in this profile.

## EXL3 prefill limit hypotheses (ranked by evidence)

### P1: `VLLM_EXL3_PREFILL_CHUNK=128` caps prefill throughput (HIGH confidence)

**Evidence**: The recipe sets `VLLM_EXL3_PREFILL_CHUNK=128` and
`VLLM_EXL3_PREFILL_BLOCK_M=64`. The prefill chunk size determines
how many tokens are processed in one forward pass during chunked
prefill. With `--max-num-batched-tokens 4096`, the engine can
batch up to 4096 tokens, but the EXL3 prefill chunk limits each
sub-chunk to 128 tokens. This means a 16K prompt requires
16384/128 = 128 sub-chunks, each with its own kernel launch and
attention computation overhead.

**Mechanism**: The EXL3 Trellis quantization path processes tokens
in blocks of `PREFILL_BLOCK_M=64` within chunks of
`PREFILL_CHUNK=128`. The Trellis block size `VLLM_EXL3_TRELLIS_BLOCK_M=8`
with `TRELLIS_MAX_M=32` further constrains the expert routing
block structure. These small block sizes are required for the EXL3
3.25-bpw Trellis format's correctness, but they limit prefill
parallelism on the GB10 GPU.

The NF3 lane uses `VLLM_SPARK_TP4_PREFILL_Q512=1` (Q512 prefill
opt-in) which enables larger prefill graph capture sizes. The EXL3
recipe sets this to `0`, disabling the Q512 prefill path. This is
likely because the EXL3 Trellis path has not been validated with
Q512 prefill graphs.

**Proposed experiment**: Run a single C1 16K prefill with
`VLLM_EXL3_PREFILL_CHUNK=256` versus `128`. Same image, same run
session. One variable: prefill chunk. Measure TTFT. If TTFT improves
without output divergence, the chunk size is a prefill bottleneck.

### P2: `VLLM_SPARK_TP4_PREFILL_Q512=0` disables prefill graph capture (MEDIUM confidence)

**Evidence**: The NF3 lane sets `VLLM_SPARK_TP4_PREFILL_Q512=1`,
which enables CUDA graph capture for prefill at Q512 sizes
(48,72,144,224,288,352,432,512). The EXL3 recipe sets this to `0`,
meaning prefill runs in eager mode (no CUDA graph replay for prefill).
Eager prefill has higher per-step launch overhead.

**Mechanism**: Without Q512 prefill graphs, every prefill sub-chunk
incurs full kernel launch overhead. For a 16K prompt with 128-token
chunks, that is 128 eager kernel launches. With Q512 graphs, larger
prefill batches could be captured and replayed, reducing launch
overhead. However, the EXL3 Trellis path may not be compatible
with Q512 graph capture because the Trellis block structure
(BLOCK_M=8, MAX_M=32) may not have stable tensor shapes across
graph replay.

**Proposed experiment**: Attempt `VLLM_SPARK_TP4_PREFILL_Q512=1`
with the current EXL3 recipe and observe whether graph capture
succeeds. If it captures, run a C1 16K prefill and compare TTFT.
If it fails to capture, document the failure mode — this confirms
the Q512 path is incompatible with the EXL3 Trellis prefill
block structure.

### P3: `VLLM_SPARK_PREFILL_PIECEWISE_CAPTURE_SIZES=""` disables piecewise prefill graphs (MEDIUM confidence)

**Evidence**: The recipe sets
`VLLM_SPARK_PREFILL_PIECEWISE_CAPTURE_SIZES=""` (empty string),
which means no piecewise prefill graph capture sizes are configured.
The NF3 lane and the reference lane use non-empty piecewise capture
sizes for prefill. Piecewise graphs capture the prefill attention
computation in segments, reducing per-segment launch overhead.

**Mechanism**: Without piecewise prefill graphs, each prefill sub-chunk
runs eagerly. The decode path has full CUDA graph capture
(`VLLM_SPARK_DECODE_CAPTURE_SIZES` is non-empty with 16 sizes),
but prefill does not. This asymmetry means decode benefits from graph
replay but prefill does not, making prefill the throughput bottleneck.

**Proposed experiment**: Populate
`VLLM_SPARK_PREFILL_PIECEWISE_CAPTURE_SIZES` with sizes compatible
with `PREFILL_CHUNK=128` (e.g. "128,256,512"). Same image, same run
session. One variable: piecewise capture sizes. Measure C1 16K TTFT.
If graph capture fails, document the failure mode.

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
4. Standard 16K sustained matrix: 25s cells, 2048 max tokens, temp 0,
   100% unique contexts, DCP4, KV budget 562688, 3s decode warmup,
   skip prefill, 300s cell-warmup timeout
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
