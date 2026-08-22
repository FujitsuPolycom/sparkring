# DeepSeek-V4-Flash-0731 on Four DGX Sparks: SIRCL Findings

Status: **research-only; live-validated**. This page summarizes the four-Spark
DeepSeek-V4-Flash-0731 SIRCL investigation and its matched patched-NCCL
comparison. It does not promote SIRCL into the public DeepSeek quickstart.

## Executive summary

SIRCL works end to end on the four-Spark TP4/DCP1 DeepSeek deployment. It
captures and replays the width-4096 target and DSpark CUDA-graph collective path
on all four ranks, serves correct API output, and maintains zero overflow and
zero fatal state under sustained C32 load.

The current SIRCL path does not improve throughput. Prefill is effectively
unchanged because both arms use NCCL for prefill. Coding Peak is about 1.9%
lower by mean. At near-identical DSpark acceptance, matched live samples place
SIRCL approximately 2.4–3.0% below patched NCCL. Patched NCCL should remain the
DeepSeek default while the graph-transport overhead is investigated.

## Tested serving contract

| Setting | Value |
|---|---|
| Model | `deepseek-ai/DeepSeek-V4-Flash-0731` |
| Hardware | Four directly cabled NVIDIA DGX Sparks, direct cycle |
| Parallelism | TP4 / DCP1 |
| Runtime image | `ghcr.io/fujitsupolycom/gb10-vllm-serving@sha256:6fc26fdad81a18f0fff67ce0a05f6d90165625ea2e1cac8a6f39bfb462017028` |
| Weight dtype | BF16 |
| Model context | 1,048,576 tokens |
| Maximum sequences | 32 |
| Batch-token budget | 4,096 |
| KV reservation | 17,179,869,184 bytes/rank |
| KV format | `fp8_ds_mla` |
| KV block size | 256 |
| Scheduler | Async; full-input-length reservation |
| Prefix caching | Disabled |
| External KV cache | Disabled |
| Speculation | DSpark, 5 draft tokens, B12X, non-greedy draft |
| Public API model | `deepseek-v4-flash-0731` |

The A/B control preserved the same image, model, six runtime mounts, shared
CUDA-capture stream, scheduler, KV reservation, and DSpark settings. It changed
only transport activation:

| Arm | TP4 mode | Width-4096 admission |
|---|---|---:|
| SIRCL candidate | `custom` | `1` |
| Patched NCCL control | `disabled` | `0` |

## SIRCL execution evidence

| Gate | Observation |
|---|---|
| API health | HTTP 200 |
| Deterministic smoke | `SIRCL_OK_4096` returned |
| Captured graph nodes | 6,749 per rank |
| Final native sequence | 2,926,190 published/consumed/completed per rank |
| Overflow | 0 on every rank |
| Fatal state | None |
| Replay | Advanced and caught up on every rank |
| Final state | SIRCL containers stopped and preserved for later restart |

The active DeepSeek SIRCL session reported:

```text
graph kernel: tiered_64k
wire schedule: sequential
protocol: two_slot_deferred_ack
query capacity: 512 rows
width: 4096 elements
```

Five draft rows plus one target row require at most 192 verification rows at
32 sequences, inside the 512-row session capacity.

## Benchmark method

The sustained-decode matrix used separate invocations per concurrency, exact
token targeting, 100% unique contexts, temperature 1.0, EOS ignored, and a
90-second post-warmup window. The aligned transport comparison used the same
16K/C32 workload as the TP2 baseline:

| Parameter | Value |
|---|---:|
| Context | 16,384 tokens |
| Concurrency | 32 |
| Measurement | 240 seconds |
| Maximum output | 32,768 tokens |
| KV budget | 2,198,756 tokens |
| Readiness timeout | 600 seconds |
| Readiness gate | 32 running, 0 waiting, stable before measurement |
| Isolation | `--isolated-server` for the final frozen-harness pair |

The benchmark harness changed its client-accounting policy during the A/B/A
sequence. Earlier receipts retain server-counter observations; a frozen later
copy was used with explicit isolated-server authority for one additional run on
each arm. The client stream headline is rejected whenever it disagrees with the
server generation-token delta.

## Results

### Cold prefill

SIRCL is not expected to affect prefill. Both arms route prefill through NCCL.

| Exact context | SIRCL client tok/s | NCCL client tok/s | Difference |
|---:|---:|---:|---:|
| 8K | 2,445 | 2,395 | SIRCL +2.1% |
| 64K | 2,364 | 2,381 | NCCL +0.7% |
| 128K | 2,172 | 2,212 | NCCL +1.8% |

Server-side prefill validation tracked the client values within about 0.5%.
This is measurement noise, not a SIRCL effect.

### C1 decode, three temperature-1.0 repetitions

| Context | SIRCL repetitions | SIRCL mean | NCCL repetitions | NCCL mean |
|---:|---:|---:|---:|---:|
| 2K | 62.9, 97.1, 94.4 | 84.8 | 103.2, 103.2, 104.8 | 103.7 |
| 8K | 94.8, 99.3, 113.6 | 102.5 | 104.5, 54.3, 65.4 | 74.7 |

These short C1 runs are not a clean transport measurement. Temperature-1.0
DSpark acceptance varied enough to reverse the apparent winner between
contexts and repetitions.

### Coding Peak

Five sequential standard cc1 Coding Peak samples per arm:

| Metric | SIRCL | NCCL | SIRCL delta |
|---|---:|---:|---:|
| Median tok/s | 92.7 | 95.8 | -3.2% |
| Mean tok/s | 93.4 | 95.2 | -1.9% |
| Maximum tok/s | 96.8 | 100.0 | -3.2% |
| CJK-corrupted runs | 0 | 0 | — |

### Long C32/16K decode

Three 240-second server-counter observations per arm were retained. Raw rates
are strongly coupled to speculative acceptance:

| Arm | Round 1 | Round 2 | Round 3 | Mean | Mean acceptance |
|---|---:|---:|---:|---:|---:|
| SIRCL | 530.1 | 567.5 | 546.0 | 547.9 | 78.95% |
| NCCL | 550.6 | 577.8 | 472.9 | 533.8 | 69.70% |

The raw means are not a transport verdict because SIRCL had higher average
acceptance. The frozen isolated-server pair was also acceptance-confounded:

| Arm | Server tok/s | Acceptance |
|---|---:|---:|
| SIRCL | 598.8 | 84.06% |
| NCCL | 511.9 | 64.69% |

### Matched-acceptance live samples

The clearest transport signal came from nearby 10-second server-log samples
where mean accepted length and draft acceptance were effectively equal:

| Mean accepted length | Draft acceptance | SIRCL tok/s | NCCL tok/s | SIRCL delta |
|---:|---:|---:|---:|---:|
| 4.46 | 69.2% | 585.1 | 599.6 | -2.4% |
| ~4.85 | ~77.1% | 636.8 | 653.1 | -2.5% |

Other nearby matched windows placed SIRCL approximately 3% lower. These are
diagnostic windows rather than independent 240-second repetitions, but they
agree with the Coding Peak regression and isolate transport better than raw
temperature-1.0 whole-window averages.

## Tiered-64K versus striped transport

DeepSeek currently uses sequential `tiered_64k`. The separate dual-port striped
schedule has been probed, but it requires the fused graph kernel and cannot be
combined with tiered-64K in the current adapter.

Existing four-Spark transport probes found:

| Shape | Sequential tiered-64K | Dual-port striped/fused | Finding |
|---|---:|---:|---|
| Q40 p50 | 174.1 µs | 190.5–190.7 µs | Sequential faster |
| Q512 p50 | 1,223.4 µs | 1,228.5–1,234.6 µs | Essentially tied |

Striping did not show a median win, so it is not enabled for the DeepSeek
width-4096 path.

## Finding and next step

The result is not “SIRCL is broken.” SIRCL is correct and stable, but its
current width-4096 sequential graph transport adds a small decode cost relative
to patched NCCL. The public DeepSeek profile should remain on NCCL.

The next useful experiment is acceptance-controlled transport profiling:

1. Freeze one harness revision and one prompt set.
2. Use temperature zero or identical prompt/seed material to hold DSpark
   acceptance fixed.
3. Compare per-iteration collective timing and server generation counters.
4. Investigate the 2.4–3.0% graph transport gap before considering promotion.

The implementation and evidence are in draft [PR #109](https://github.com/FujitsuPolycom/sparkring/pull/109).
