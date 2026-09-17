# DSpark depth five versus seven on two Sparks

Status: **research-only**. Static depth seven started and completed bounded
requests on the identified TP2 runtime. **Five remains the profile default.**
These observations do not qualify the four-rank cycle requested in
[issue #191](https://github.com/FujitsuPolycom/sparkring/issues/191).

## Conditions and admission

Both arms used the same DeepSeek-V4-Flash-0731 checkpoint and per-rank local
image pair from source `9f5b1e619d037d1ae365f158135c3de42402cf4e`, identified in
the [API/IPC record](api-ipc-tp2-k5-20260917.md). TP2/DCP1 retained 16 GiB
`fp8_ds_mla` KV per rank, 32 sequences, a 4096-token batch budget and a 1 s IPC
spin interval. Serving environments, image IDs, mounts and HostConfig matched;
only the explicit DSpark speculative-depth argument changed.

The checkpoint's `dspark_block_size=5` is a lower bound, not a ceiling. A replay
of the pinned vLLM `e2666d9a…` block-size guard rejected four and admitted five,
six and seven. The [JSON record](dspark-depth-tp2-20260917.json) retains that
source hash and the completed serving observations. Guard admission alone does
not qualify every kernel or shape.

Depth-seven startup reported the sparse-indexer field `next_n=8`, a KV pool of
2,198,756 tokens and `max_num_scheduled_tokens=3904`. Depth five reported
`next_n=6`, the same pool, and 3968 scheduled tokens. The configured batch budget
stayed 4096; speculative slots changed the effective scheduler capacity.

## Measurements

Three batches at each concurrency used 1024 prompt tokens and 128 generated
tokens per request, temperature zero, streamed token IDs and usage checks.
All samples passed and were retained. Rates are total completion tokens divided
by concurrent batch wall time, **including prefill and queueing**.

| Concurrent requests | Median end-to-end tok/s, depth 5 | Depth 7 |
|---:|---:|---:|
| 1 | 55.82 | 61.38 |
| 4 | 108.56 | 122.19 |
| 8 | 138.98 | 144.93 |

The three depth-seven C8 samples were **100.24, 144.93 and 147.85 tok/s**. The
first took 10.216 seconds. Both workers logged a fused 4-bit MoE kernel
(`W4A16FusedMoeKernel`) disk-cache miss around `06:40:22Z`, inside that first
eight-request group's server-log interval. Other initialization events also
occurred. This correlation does not quantify how much initialization caused
the slow sample; no isolated compile duration or matched replay was measured.

## Limits and next check

Depth five received 36 additional calibration requests before benchmarking;
depth seven received none. Their warmup/cache histories are therefore
unmatched. Three repetitions cannot establish a repeatable speedup, significance
or latency-tail bound. The JSON preserves all three raw rates and batch times
per concurrency instead of discarding the slow first sample.

Acceptance is omitted: only cumulative post-run metrics were available, without
matching benchmark-window counter deltas. Collect per-window acceptance and
repeat with matched warmup before considering a default change. TP4 cycle,
full output-quality and soak qualification remain separate. No dynamic depth
ladder, profile default or published image identity was changed.
