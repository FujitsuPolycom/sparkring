# Snapshot gather byte-comparison gate

Status: GPU-free oracle implemented; live CUDA execution remains required.

The existing CUDA probe exercises one target-CKV source and one sparse-indexer
source. The portable C++ layout test includes MTP metadata but does not compare
record bytes. `python/snapshot_byte_comparison.py` closes the model-free gap:

- reference bytes come from the current `pack_record` codec;
- candidate bytes use the native kernel's layer-major layout and physical-slot
  indirection;
- target CKV, sparse indexer, and MTP draft KV are compared independently;
- source row stride may exceed useful per-token bytes;
- alignment padding is excluded because the CUDA kernel does not define it;
- four rank-specific scrambled slot tables and row counts 1, 63, 64, and 1024
  are covered.

Run:

```powershell
python -m pytest -q sparkcache/native/tests/test_snapshot_byte_comparison.py
python -m sparkcache.native.python.snapshot_byte_comparison
```

## Exact first live inputs

The first DGX Spark run remains a throwaway-buffer probe, never active model
KV. Begin with this deliberately small all-record-family fixture:

| source | kind | ordinal | rows | stride | useful bytes/token |
|---|---:|---:|---:|---:|---:|
| `model.layers.00.mla` | target CKV (0) | 0 | 2048 | 512 | 368 |
| `model.layers.01.mla` | target CKV (0) | 1 | 2048 | 512 | 368 |
| `model.layers.00.indexer` | sparse indexer (1) | 0 | 2048 | 256 | 132 |
| `model.layers.00.mtp` | MTP draft KV (2) | 0 | 2048 | 512 | 368 |

Useful source byte `(kind, layer, row, byte)` is:

```text
(kind * 71 + layer * 43 + row * 17 + byte * 29) & 0xff
```

Every stride-padding byte is `0xee`. For DCP rank `r`, physical row `i` is:

```text
(11 + r * 53 + i * 37) % 2039
```

Run all four rank values with:

- `row_count=64`: one 256-global-token DCP4 storage chunk;
- `row_count=1024`: one 4,096-global-token / 16-chunk macro batch;
- mapped-host and managed arena modes;
- a 2 MiB arena slot;
- nonblocking external CUDA stream;
- fixed context sequence and logical start recorded in the result.

This fixture exercises the mechanically supported separate-MTP layout. It is
not the production GLM-5.2 source inventory.

The live matrix executable is:

```bash
spark_cache_snapshot_matrix_probe \
  --arena mapped \
  --slots 2 \
  --rank 0 \
  --rows 64 \
  --iterations 100 \
  --compare-every 1 \
  --pipeline-depth 2 \
  --writer-hold-us 0
```

It emits exactly one `sparkcache.snapshot_matrix.v1` JSON object. Run the
Cartesian product of arena `{mapped,managed}`, slots `{2,3}`, rank
`{0,1,2,3}`, and rows `{64,1024}`. A successful process reports zero
mismatches, exact geometry and native counters, and p50/p95/p99 wall latency
for the host `try_submit` call, CUDA gather completion after submit returns,
and their total. `--compare-every N` bounds CPU byte-comparison work during a
long soak; geometry and native counters are still validated every iteration,
and the first and last iterations are always byte-compared. Use `1` for the
correctness matrix and a larger explicitly recorded cadence for a 10K soak.
`--pipeline-depth N` defaults to `1` and may be raised through the configured
slot count. The probe fills that many tickets before claiming the oldest
ticket, then refills the FIFO after every release. Each ticket retains its own
context sequence, logical start, iteration, submit timestamp, and completion
deadline; geometry and byte validation therefore prove that slot reuse did not
cross-wire transactions. Run depth `2` with two-slot arenas and depths `2` and
`3` with three-slot arenas.

`--writer-hold-us N` defaults to `0`. A nonzero value deliberately holds the
claimed `WRITING` view after validation and before release, simulating a slow
hash/write consumer while other tickets remain in flight. Use it for
backpressure and ownership tests, not headline gather latency.

Before measurement, the executable creates a separate runtime, fills every
configured slot without claiming, and submits once more. The extra submission
must return `WOULD_BLOCK`. This is repeated
`--saturation-cycles N` times (default 100), with distinct slot indices proven
inside every cycle. The probe abandons and drains each drill context, checks
the drill's native counters, destroys that runtime, and creates a fresh one
for latency measurement. JSON reports `would_block.intentional` separately
from `would_block.unexpected`; a successful default run has exactly `100`
intentional and `0` unexpected. This isolation keeps deliberate saturation
and abandonment out of measured latency and normal runtime counters.

The primary CPU consumer touches every `used_bytes` byte using aligned
64-bit additive lanes plus byte prefix/tail handling. This is intentionally a
compiler-vectorizable memory read, not a byte-serial cryptographic hash; its
checksum is retained so the reads cannot be discarded. End-to-end latency
ends after this one complete consume pass. This keeps the probe from charging
an artificial serial FNV loop to the arena/ring path.

Only sparse exact-check iterations perform a second complete warm-read pass
and then the source-by-source per-byte formula comparison. JSON separately
reports primary consume bytes/passes, warm-read bytes/passes, exact-check
bytes, mismatches, checksum, first-touch timing, warm-read timing, and
end-to-end timing. When a later FIFO ticket polls `NOT_READY` before CPU
consumption starts, the probe also records a CPU-read-during-GPU-fill overlap
sample. Use at least 100 overlap samples at depth 2/3 and at least eight
complete exact production-profile checks per rank.
The probe scans every later pending ticket: a `READY` older ticket does not
hide a `NOT_READY` newer fill at depth 3. `READY` continues the scan,
`NOT_READY` proves overlap, and any other status fails the run.

The overlap target is independent of exact-comparison cadence:
`--overlap-samples N` defaults to `100` and continues full CPU consumption
until it has observed that many later-ticket `NOT_READY` samples (or the run
ends). Thus a 10K soak may use a sparse `--compare-every` value without
silently reducing its overlap evidence.

The JSON `memory` object reports `cudaMemGetInfo` free/total bytes immediately
before the fresh measurement runtime is created, after create plus source
configuration, and after checked shutdown. It also reports
`nominal_arena_bytes = slot_bytes * slot_count`. The nominal value matters
because a mapped-host arena may consume pinned host memory without an
equivalent drop in CUDA-reported free device memory. Failure to collect any
requested CUDA memory sample is a probe failure.

Any argument, CUDA, geometry, byte, counter, timeout, release, or
checked-shutdown failure exits nonzero.

Expected metadata at 64 rows is:

```text
record_mask = 0b111
target:  offset=0      length=47,104
indexer: offset=47,104 length=8,448
MTP:     offset=55,552 length=23,552
used_bytes=79,104
```

Expected metadata at 1,024 rows is:

```text
record_mask = 0b111
target:  offset=0       length=753,664
indexer: offset=753,664 length=135,168
MTP:     offset=888,832 length=376,832
used_bytes=1,265,664
```

## Production-sized GLM-5.2 fixture

The attested live inventory is 101 registered sources per rank:

| family | source count | useful bytes/token | source row stride |
|---|---:|---:|---:|
| target CKV | 79 | 368 | 368 |
| sparse indexer | 22 | 132 | 132 |

Run the vectorized production fixture with:

```bash
spark_cache_snapshot_matrix_probe \
  --arena mapped \
  --slots 3 \
  --rank 0 \
  --rows 1024 \
  --iterations 100 \
  --compare-every 1 \
  --pipeline-depth 3 \
  --writer-hold-us 0 \
  --profile glm52 \
  --slot-mib 64
```

`--profile compact` is the default and retains the four-source fixture.
`--profile glm52` defaults to 64 MiB slots and permits an explicit
`--slot-mib 32` or `64`. The selected `slot_bytes` is recorded in JSON.

The 79 target sources are 78 transformer layers plus the colocated MTP
drafter. The runtime reports `policy=colocated_target`; there is no separately
named MTP source and therefore no MTP record in this fixture. Its expected
record mask is `0b011`. A separate `mtp_draft_kv` source would contradict the
live registration contract and double-count drafter state.

The 368- and 132-byte source strides are the tightly packed registered-tensor
contract used by native placement. The 512/256 strides in the small fixture
above are intentional padding stress, not claimed live strides. When the
source-table exporter is wired in, it must derive and attest
`tensor.stride(1) * tensor.element_size()` and fail closed unless it agrees
with the registered row width.

Expected metadata at 64 local rows (one 256-global-token DCP4 chunk) is:

```text
record_mask = 0b011
target:  offset=0         length=1,860,608
indexer: offset=1,860,608 length=185,856
MTP:     absent
used_bytes=2,046,464 (1.95166015625 MiB)
```

Expected metadata at 1,024 local rows (16 chunks) is:

```text
record_mask = 0b011
target:  offset=0          length=29,769,728
indexer: offset=29,769,728 length=2,973,696
MTP:     absent
used_bytes=32,743,424 (31.2265625 MiB)
```

A 32 MiB arena slot leaves 811,008 bytes (792 KiB) at 1,024 rows. This is the
smallest production-representative candidate in the requested 32-64 MiB
range. Test both 32 MiB and 64 MiB slots, but retain 64 MiB as the initial
operational default until live counters prove that 32 MiB has enough margin
for every admitted inventory. Slot size does not change the declared record
bytes.

The production profile preserves the same scrambled physical-slot and
byte-comparison gates as the small fixture and allocates all 101 tightly
packed source arrays. It does not approximate 101 layers by inflating one
source: per-kind ordinal and source-table traversal are part of the gate.

## Live-model follow-up inputs

After the throwaway fixture passes, export the registered live source table
from the connector and feed the same probe:

- exact sorted layer names;
- record kind and dense within-kind ordinal;
- device base address;
- source row count;
- source row stride;
- useful bytes per token;
- the exact 64 or 1,024 physical slots submitted;
- ready-view mask, offsets, lengths, `used_bytes`, row count, generation,
  context sequence, and logical start;
- SHA-256 and first mismatch for every declared record slice.

Convert a claimed native ready view without interpreting or repacking its
bytes:

```python
candidate = SnapshotPayloadView(
    row_count=int(ready.row_count),
    record_mask=int(ready.record_mask),
    record_offsets=tuple(int(v) for v in ready.record_offset_bytes),
    record_lengths=tuple(int(v) for v in ready.record_length_bytes),
    used_bytes=int(ready.used_bytes),
    payload=bytes(ready_memoryview(ready)[: ready.used_bytes]),
)
results = compare_ready_view(source_fixtures, physical_slots, candidate)
assert all(result.is_equal for result in results)
```

The table must cover every registered target, indexer, and separately named
MTP layer. For this GLM-5.2 runtime, `draft_kv_policy=colocated_target`, so the
source inventory contains exactly 79 target plus 22 indexer sources and mask
`0b011`.

## Pass criteria

Promotion requires all of the following:

1. Source registration and ready-view geometry match the GPU-free oracle.
2. Every declared record slice is byte-identical to the current Python gather
   and `pack_record` path; SHA-256 alone is diagnostic, not the comparison.
3. No stride-padding or physical-slot identifier appears in a record.
4. All four rank slot permutations pass at 64 and 1,024 rows in both arena
   modes.
5. Repeated submit/release and generation reuse remain byte exact.
6. One intentionally corrupted byte is reported in the correct record family
   and at the exact within-record offset.
7. Any missing source, width/ordinal mismatch, short payload, bad mask, or
   out-of-range slot fails closed before publication.

This gate proves layout and gather bytes only. It does not prove active-model
stream ordering, block-lease safety, CUDA-graph compatibility, interference,
or end-to-end manifest correctness.
