# Snapshot arena/ring live matrix

Status: standalone measurement and grading contract. The native probe opens
CUDA only against throwaway source buffers; it does not load a model, touch
live KV state, contact a Spark, or change the serving runtime.

## Question

Choose the first live SparkCache streaming-snapshot configuration:

| Cell | Arena | Ring depth |
|---|---|---:|
| M2 | mapped host | 2 |
| M3 | mapped host | 3 |
| U2 | CUDA managed | 2 |
| U3 | CUDA managed | 3 |

Mapped depth 2 is the conservative prior. A third slot is useful only if it
materially reduces `WOULD_BLOCK`. Managed memory is useful only if its complete
GPU-write-to-CPU-consume path is measurably faster without inference
interference.

## Standalone probe versus live promotion

`spark_cache_snapshot_matrix_probe` now covers both the compact 2 MiB
correctness fixture and the production-sized `glm52` fixture with 101 sources
and 32 or 64 MiB slots. Its measured loop supports pipeline depth two or three,
writer holds, explicit saturation, CPU-read/GPU-fill overlap, and separate CPU
first-touch, warm-read, and gather-plus-consume timing.

A qualifying standalone matrix must use `--profile glm52`, exercise the full
configured pipeline depth, and pass the saturation, production-byte, memory,
and CPU-consumer gates below on all four ranks. This remains model-down
throwaway-buffer evidence: it cannot establish live vLLM stream ordering,
block-lease integration, CUDA-graph behavior, or serving interference. Only
the selected standalone candidate proceeds to the live serving A/B required
for promotion.

## Required run order

Do not tune against a single thermally lucky run.

1. Record a cache-off baseline immediately before the matrix.
2. Run one correctness-only 2 MiB ABI probe for each cell.
3. Run the model-down qualifying standalone matrix described below.
4. Select exactly one standalone candidate for a model reload.
5. Warm that cell outside the measured interval.
6. Run the live cell between two cache-off baselines.
7. Use the weaker baseline value for every throughput gate and the larger
   baseline latency for the latency comparison.

Run all four ranks in every standalone cell. Reject the entire cell if any rank
lacks a result. The original four-cell live run order remains useful only when
more than one standalone cell is indistinguishable; the normal path reloads
only the selected standalone candidate.

## Workloads

Each cell must contain:

- a standalone 2 MiB ABI gather and the pinned GLM-5.2 1,024-row production
  gather (exactly 32,743,424 bytes with the current registered layout) in the
  proposed 32 or 64 MiB slot;
- at least eight byte-exact production-payload CPU reads per rank;
- at least 10,000 ring submissions after warmup;
- at least 100 explicit saturation cycles per rank in which exactly `depth`
  tickets occupy distinct slots and submission `depth + 1` returns
  `WOULD_BLOCK`;
- forced writer holds long enough to exercise `WOULD_BLOCK`;
- at least 100 samples per rank where the CPU consumes one completed slot while
  the GPU fills a later slot;
- separate CPU first-touch, CPU warm-read, and gather-plus-CPU-consume latency;
- abandon while `GPU_FILLING`, `READY`, and `WRITING`;
- clean shutdown after every state drill;
- one live 32K prefill/decode shadow at C1 and C8;
- one large-context prefill long enough to emit at least 32 macro-batches.

The live shadow must use the same model, KV budget, CUDA-graph capture set,
adaptive-MTP policy, prompt set, and serving activity as the cache-off
baseline. No model or transport changes may share this A/B window.

## Timeouts

Timeouts are failure signals, not retry hints:

- native create/configure: 10 seconds;
- one CUDA gather completion: 2 seconds;
- writer claim/release drill: 10 seconds;
- standalone cell: 300 seconds;
- one live request: baseline observed duration plus 50%, with a 60-second
  minimum margin;
- one live cell: 15 minutes;
- full matrix: 90 minutes.

Any CUDA timeout, collective timeout, worker restart, request error, or
forced process kill rejects the cell.

## JSON output

`gate_snapshot_matrix.py` accepts one JSON document. Set `baseline` to `null`
and omit each cell's `memory` and `serving` sections for model-down standalone
selection. Supply them for final live promotion.

```json
{
  "schema": "sparkring-snapshot-matrix/v2",
  "live_candidate": {
    "arena_mode": "mapped",
    "ring_depth": 2
  },
  "baseline": {
    "prefill_tps": 800.0,
    "decode_c1_tps": 20.0,
    "decode_c8_tps": 50.0,
    "inter_token_p99_ms": 80.0,
    "kv_capacity_tokens": 458752
  },
  "cells": [
    {
      "arena_mode": "mapped",
      "ring_depth": 2,
      "slot_bytes": 67108864,
      "rank_count": 4,
      "safety": {
        "byte_mismatches": 0,
        "cuda_errors": 0,
        "stale_tickets": 0,
        "leaked_slots": 0,
        "leaked_leases": 0,
        "timeouts": 0,
        "request_errors": 0,
        "worker_restarts": 0,
        "shutdown_clean": true
      },
      "memory": {
        "peak_delta_bytes_per_rank": [150000000, 151000000, 149000000, 150000000],
        "minimum_free_bytes_per_rank": [3000000000, 3100000000, 3050000000, 2990000000],
        "kv_capacity_tokens": 458752
      },
      "ring": {
        "submissions": 10000,
        "would_block": 10,
        "gather_p95_ms": 3.0,
        "gather_p99_ms": 4.0,
        "submit_p99_us": 100.0,
        "completion_pause_p95_ms": 200.0,
        "managed_fault_events": 0
      },
      "standalone": {
        "production_payload_bytes": 32743424,
        "production_byte_checks_per_rank": [8, 8, 8, 8],
        "cpu_readback_bytes_per_rank": [
          261947392, 261947392, 261947392, 261947392
        ],
        "cpu_readback_mismatches": 0,
        "saturation_cycles_per_rank": [100, 100, 100, 100],
        "max_outstanding_per_rank": [2, 2, 2, 2],
        "distinct_slots_observed_per_rank": [2, 2, 2, 2],
        "depth_plus_one_would_block_per_rank": [true, true, true, true],
        "cpu_read_during_gpu_fill_samples_per_rank": [100, 100, 100, 100],
        "cpu_first_touch_p95_ms": 4.0,
        "cpu_warm_read_p95_ms": 1.0,
        "end_to_end_p95_ms": 8.0,
        "end_to_end_p99_ms": 10.0
      },
      "serving": {
        "prefill_tps": 792.0,
        "decode_c1_tps": 19.8,
        "decode_c8_tps": 49.5,
        "inter_token_p99_ms": 82.0
      }
    }
  ]
}
```

The standalone artifact must contain all four unique cells and use
`"baseline": null, "live_candidate": null`. A live artifact retains all four
standalone cells, names the one `live_candidate`, and adds `memory` and
`serving` only to that cell. The named live candidate must equal the grader's
standalone recommendation; this prevents silently swapping configurations
between the model-down and live gates.

## Hard promotion gates

Every promoted cell must satisfy:

- four-rank byte-exact correctness;
- a production payload of at least the pinned GLM-5.2 1,024-row size
  (32,743,424 bytes) and no more than 64 MiB that fits the configured 32 or
  64 MiB slot;
- at least eight complete CPU byte checks per rank, with every expected byte
  consumed and zero mismatches;
- actual simultaneous occupancy of every configured slot, distinct slot use,
  and a verified `depth + 1` `WOULD_BLOCK` on every rank;
- at least 100 CPU-read/GPU-fill overlap samples per rank;
- zero CUDA errors, stale tickets, leaked slots, leaked leases, timeouts,
  request errors, or worker restarts;
- clean checked shutdown;
- at least 10,000 measured submissions;
- `WOULD_BLOCK / submissions <= 0.5%`;
- gather p99 at most 50 ms and host submit p99 at most 500 us;
- completion pause p95 at most 500 ms;
- prefill, C1 decode, and C8 aggregate decode each at least 98% of cache-off;
- inter-token p99 no more than 5% above cache-off;
- measured memory delta no more than the larger of 384 MiB or
  `1.25 * slot_bytes * ring_depth`;
- at least 1 GiB free memory on every rank after allocation;
- unchanged configured KV capacity;
- gather-plus-CPU-consume p99 at most 100 ms.

Managed memory normally migrates on the first CPU access to GPU-written bytes.
Therefore a zero-fault counter is not a valid universal gate. Record migration
telemetry when available, but select on measured CPU first-touch and complete
gather-plus-CPU-consume latency. The CPU read must touch/hash all `used_bytes`;
reading only metadata or a sample does not qualify.

## Selection policy

Passing does not automatically justify extra memory.

1. Prefer depth 2.
2. Promote depth 3 only when its same-mode depth-2 cell exceeds the 0.5%
   `WOULD_BLOCK` gate and depth 3 reduces that rate by at least 50%, while
   preserving all interference gates.
3. Prefer mapped host.
4. Select managed memory only when its same-depth mapped cell also passes,
   managed gather-plus-CPU-consume p95 is at least 10% lower, and, for final
   promotion, its minimum serving ratio is no more than 0.5 percentage points
   worse.

Standalone selection is a reload shortlist, not production promotion. It can
decisively reject incorrect geometry, fake depth, insufficient capacity,
managed-memory migration losses, unsafe shutdown, and excessive backpressure.
Only the selected candidate proceeds to the live serving A/B. If no cell
passes, streaming snapshots remain disabled.

## Offline grading

```powershell
python sparkcache\native\experiments\snapshot_matrix\aggregate_snapshot_matrix.py `
  rank-results.jsonl `
  --output standalone-matrix.json

python sparkcache\native\experiments\snapshot_matrix\gate_snapshot_matrix.py `
  standalone-matrix.json `
  --output decision.json
```

`rank-results.jsonl` must contain exactly one passing
`sparkcache.snapshot_matrix.v1` object for every `(arena, depth, rank)` cell:
mapped/managed, depth 2/3, and ranks 0-3. The aggregator rejects missing or
duplicate cells/ranks, failed probes, schema/config/geometry disagreement,
partial readback, incomplete saturation, insufficient overlap, unsafe native
statistics, and invalid lifecycle-memory samples.

The aggregator's `aggregation.field_map` object records the complete raw-to-v2
mapping in the output. Counts and bytes are summed where they represent work
or errors; per-rank arrays preserve rank order; latency percentiles select the
worst rank rather than averaging percentiles; memory uses the lowest observed
free value and largest observed drop on each rank. Saturation's deliberate
depth-plus-one `WOULD_BLOCK` events are not mixed into the measured ring's
ordinary `would_block` count.

CPU consumption has two overlapping populations. Exact checks perform first
touch, warm read, and byte comparison; additional overlap passes perform first
touch while a later GPU fill remains in flight. Therefore first-touch and
end-to-end sample counts equal `consume_passes`, while warm-read samples equal
`checks`. The aggregator requires
`overlap_samples <= consume_passes <= checks + overlap_samples` and verifies
the exact, warm, and total-consumption byte counts against the full production
payload. It rejects older records that do not carry these population counters.

Exit code `0` means at least one configuration is promotable. Exit code `1`
means the standalone matrix is complete but no cell passes. Exit code `2`
means the input is malformed or incomplete. Exit code `3` means a standalone
candidate was selected but live serving evidence has not promoted it.
