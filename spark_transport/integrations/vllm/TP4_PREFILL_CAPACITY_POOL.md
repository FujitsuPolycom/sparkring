# TP4 shared tiled-prefill engine research contract

## Status

The capacity selector in `spark_tp4_prefill_capacity_pool.py` is
**implemented** and **offline-validated**. Native dispatch is **unsupported**.
The serving adapter defaults `VLLM_SPARK_TP4_PREFILL_CAPACITY_POOL` to `0` and
fails before loading native code when an operator sets it to `1`.

This contract does not claim a live result. It defines the adapter surface and
evidence gates needed to replace exact-payload eager sessions with one physical
transport engine, one QP pair per edge, and one generation-tagged tile pool per
edge. Four logical capacity plans share that physical engine.

## Exact-payload scaling in the executable adapter

`_Backend.native_sessions` is indexed by payload bytes. Every newly observed
`[Q, 6144]` BF16 payload therefore creates a distinct native session. A native
session owns the transport resources behind the current C API, while
`active_port_reservations()` reserves one control-port pair for every Q that
the selected configuration could instantiate, including sessions that have
not yet been created lazily.

The stride-two port formula projects as follows with the default base pair
`11000/11001`:

| Maximum Q | Possible exact sessions/rank | Reserved or projected ports/rank | Last pair | Executable status |
|---:|---:|---:|---:|---|
| 40 | 40 | 80 | 11078/11079 | Admitted when the decode contract is configured through Q40 |
| 512 | 512 | 1,024 | 12022/12023 | Admitted only with `VLLM_SPARK_TP4_PREFILL_Q512=1` |
| 1,024 | 1,024 | 2,048 | 13046/13047 | Unsupported projection |
| 4,096 | 4,096 | 8,192 | 19190/19191 | Unsupported projection |

Q1024 and Q4096 are not eligible for the current adapter. Their rows quantify
what extending the exact-Q formula would cost; they are not executable claims.

## One transport engine with four logical capacity plans

The selector maps each operation to the smallest logical plan that contains
its active rows. Every plan returns the same durable transport key,
`prefill-tile-engine`, and the same proposed pair `12500/12501`:

| Active Q interval | Plan maximum Q | Maximum logical payload | Shared pair | 512-KiB payload tiles at the maximum |
|---:|---:|---:|---:|---:|
| 1-40 | 40 | 480 KiB | 12500/12501 | 1 |
| 41-512 | 512 | 6 MiB | 12500/12501 | 12 |
| 513-1,024 | 1,024 | 12 MiB | 12500/12501 | 24 |
| 1,025-4,096 | 4,096 | 48 MiB | 12500/12501 | 96 |

The plan maximum is not a registered-arena allocation. Eight 512-KiB tiles
provide 4 MiB of logical one-plane payload capacity per edge. That 4-MiB
number is not the registered footprint. Each slot contains two 256-KiB lanes;
each lane has distinct send and receive storage plus one 64-byte control. The
conservative registered storage is therefore:

```text
8 slots * 2 lanes * (256 KiB send + 256 KiB receive + 64 B control)
  = 8,389,632 bytes per edge
```

The figure excludes descriptor-ring storage and matches the source contract
in `tp4_tiled_session.hpp`. Larger operations stream through this bounded
pool. Q4096 is a 96-tile logical payload and therefore requires at least
twelve waves through an eight-tile pool.

The four plan maxima do not identify sessions, QPs, registered arenas, or port
pairs. They select descriptor bounds and kernel plans within the same physical
engine. A transitional four-session implementation is not the target.

## Required operation descriptor

A variable-size operation needs a descriptor equivalent to:

```text
OperationTicket {
    capacity_plan
    active_bytes
    first_tile_slot
    tile_count
    generation
}
```

Each edge maintains a monotonic `consumed_through_generation` watermark.
Acquisition can reuse a slot only after its prior generation is covered by the
watermark. Output readiness is separate from slot retirement. An unexpected
generation, invalid active byte count, or poisoned slot remains process-fatal.

The current eager all-reduce call carries no per-operation active byte count.
Its session configuration fixes `payload_bytes`, while the call receives only
the handle, input, output, and CUDA stream. Capacity dispatch cannot be
implemented safely by padding a smaller tensor or calling an invented symbol.
The Python adapter intentionally contains no tiled-engine native symbol lookup.

## Port coexistence

The proposed shared pair is disjoint from exact decode Q1-Q40 under both the
default exact base `11000/11001` and the canary base `11100/11101`. It cannot
coexist with an arbitrarily extended exact-Q prefill family:

- default exact base: the first projected collision is Q751;
- canary exact base: the first projected collision is Q701.

Capacity mode must replace exact-Q reservations above Q40. Before live use,
the shared namespace validator must reserve the one tiled-engine pair and prove
that it is free and globally unique on all four ranks. The Python selector only
proposes the pair; it does not bind or reserve it.

## Live prerequisites

All seven gates below are blocking:

1. **Native active-byte submission:** one engine submits any row-aligned
   `active_bytes` not greater than its plan maximum without recreation. Zero,
   misaligned, and oversized values fail before touching CUDA.
2. **Generation-tagged tile pool:** every edge uses fixed `(generation, slot)`
   tickets and operation descriptors carry `active_bytes`. Unexpected
   generations and poisoned slots terminate the worker.
3. **Cumulative credit retirement:** each edge publishes a monotonic
   consumed-through watermark; reciprocal slot retirement is not on the
   output-ready critical path.
4. **Capacity port namespace:** exactly one pair is reserved for the tiled
   engine, exact-Q prefill reservations above Q40 disappear, and global
   collision validation passes.
5. **Q4096 kernel capacity:** staging, reduction, and output kernels tile
   through Q4096 without a single-CTA whole-payload dependency or a 48-MiB
   registered arena per exact shape.
6. **Four-rank fixed-Q probe:** the bounded probe accepts Q40, Q512, Q1024, and
   Q4096 and reports plan, engine, tile, generation, credit, overflow, and
   poison counters on every rank.
7. **Serving engine receipt:** model evidence proves exactly one tiled engine
   per rank, zero exact-Q prefill sessions, the logical plan and active bytes
   selected for every observed shape, and zero overflow or poison. The receipt
   separately records 4,194,304 logical one-plane payload bytes and 8,389,632
   registered tile-storage bytes per edge; descriptor storage is accounted for
   separately.

## Qualification sequence

Exercise the offline selector and its plan contract without contacting a
host:

```powershell
python -m pytest spark_transport/integrations/vllm/test_spark_tp4_prefill_capacity_pool.py -q
```

The tracked tree does not provide a standalone planner command. The test
above validates the offline selector in
[`spark_tp4_prefill_capacity_pool.py`](spark_tp4_prefill_capacity_pool.py)
and the plan contract it returns; emitting a plan as JSON from a command
line is separate implementation work, and no such command may be quoted
here until it is tracked in this repository.

After all prerequisites are implemented, the live harness must run bracketed
baseline/candidate arms for Q40, Q512, Q1024, and Q4096. It must also exercise
Q1, Q39, Q40, Q41, Q511, Q512, Q513, Q1023, Q1024, Q1025, Q4095, and Q4096 to
prove boundary reuse rather than four exact-size special cases.

Acceptance requires all-rank exit zero, exact integer-valued oracle equality,
the declared random-BF16 association gate, fixed-seed token equality, matching
engine/tile/credit counters, and separate p50/p95 results for every fixed-Q
arm. Serving evidence must use shape tracing to prove which Q values actually
occur; prompt length alone is not evidence that a particular collective shape
was dispatched.
