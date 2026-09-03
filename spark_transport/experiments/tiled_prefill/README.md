# Tiled TP4 collective substrate

Status: **research-only**. This directory contains two related surfaces. The
generic tiled planner, correctness executor, CUDA bulk kernels, correctness
kernels, and stable edge adapters are exercised by the standalone
`spark_tp4_tiled_prefill_probe`; they do not support a serving-performance
claim. The bidirectional-ring executor and fused-prefill kernels and verbs
proxy are linked into `libspark_transport_capi.so` for the GLM-5.3 research
transport. The public C ABI and vLLM adapter select those serving components
through explicit shape and topology gates. Delayed-credit backpressure remains
unsupported by the generic correctness executor.

`substrate.py` models a bounded replacement for exact-shape transport
sessions. BF16 `[Q, 6144]` all-reduce widths from Q1 through Q4096 map onto
four stable capacity identities: latency through Q40, medium through Q512,
large through Q1024, and streaming through Q4096. Each identity uses the same eight-slot,
512-KiB-per-slot geometry. A Q4096 operation therefore describes 96 logical
tiles while retaining 4 MiB of logical payload-slot capacity per edge.

`gpu_harness.py`, `tiled_bulk_kernels.cuh`, and `tiled_bulk_kernels.cu` define
an offline, research-only GPU execution contract for the same Q1-Q4096 range.
Each 512-KiB payload tile uses a stable eight-CTA launch shape made from
64-KiB worker slices. The CUDA bulk kernels vectorize aligned 16-byte spans,
bound every load and store by `active_bytes`, and keep protocol polling out of
worker CTAs. The Python DAG names the required protocol dependencies, exact
prior-generation credit, GPU output readiness, final slot retirement, timing
receipt fields, and correctness counters consumed by the standalone executor.
The generic DAG and correctness executor are not used by the serving sessions
and have no qualified performance evidence. The DAG deliberately does not
assign CUDA streams. A performance executor must preserve every listed edge
while allowing independent tiles to overlap; placing all polling control nodes
and worker grids on one stream would serialize the pipeline.

`tiled_executor.hpp` and `tiled_executor.cpp` provide the standalone native
integration seam. One single-owner host state machine borrows two stable edge
ports with one distinct QP identity per edge; it never creates or destroys
those ports. The executor converts validated `(generation, slot,
active_bytes)` tiles directly into the ABI from `tiled_bulk_abi.hpp`, invokes
one bulk launch at a time, and routes the lower and upper tensor halves over
opposite edge orders. Its declared stream policy is
`single_stream_correctness_only`; the CPU receipt makes no latency or
bandwidth claim.

Every tile passes `Tp4TiledCreditWindow::try_acquire` before its first bulk
launch. The ninth logical tile cannot reuse slot zero until cumulative peer
watermarks on both edges cover ordinal zero. Unexpected or regressing
watermarks poison the state permanently. Cumulative outbound credit uses one
fixed staging cacheline per edge, so a newer watermark cannot overwrite it
until the preceding signaled credit write reports completion.
The mapped acknowledgement word is zero before any peer publication, so
credits use the one-based encoding `wire_credit = consumed_through + 1`.
Edge adapters must transmit `wire_credit`, ignore a raw zero word, and decode
nonzero observations as `consumed_through = wire_credit - 1`. Encoding the
maximum unsigned 64-bit ordinal fails closed instead of wrapping to zero.

`TiledExecutor::drain()` is the registered-storage teardown gate. A completed
output is not sufficient: `safe_to_release_registered_storage` becomes true
only after both final peer watermarks arrive and both final outbound credit
writes complete. The executor destructor does not drive progress or release
the externally owned QPs and arenas; their owner must obtain `kIdle` or
`kComplete` from `drain()` before destroying them. A poisoned result requires
process termination because safe recovery is not established.

A native correctness harness must compare every active element against the
counter-rotating BF16 association, place sentinels before and after active
input/output ranges, vary inputs across physical-slot reuse, and fail on any
unexpected generation or regressing credit. The existing graph probe checks
active values but does not guard inactive capacity bytes, so it is not by
itself a sufficient tiled-tail qualification.

The substrate's 4-MiB figure is the logical payload capacity of eight tiles,
not a proven registered-arena allocation. The conservative counter-rotating
storage contract keeps distinct send and receive regions for both tensor
halves on each endpoint. Eight slots therefore require 8,389,632 registered
tile-storage bytes per edge, including 64-byte controls but excluding any
descriptor ring. Reducing that footprint requires an explicit RDMA ownership
proof before send/receive storage may alias.

Every tile descriptor contains an exact `active_bytes` range and a
`(generation, slot)` ticket. Generation zero is invalid. Reusing ordinal `i`
requires the peer's inclusive consumed-through watermark to cover
`i - slots_per_edge`; output readiness is intentionally independent from this
retirement condition. Ticket generations are bounded to an unsigned 64-bit
wire value and fail closed at exhaustion.

The symbolic schedule verifier tracks contributor sets, BF16 association
trees, causal segment availability, and bytes sent by each rank on each TP4
endpoint. For a payload of `n` bytes per rank:

- The serialized XOR-1/XOR-3 all-reduce has a `2n` ideal bandwidth critical
  path. Counter-rotating halves retain `2n` total bytes but reduce the critical
  path to `n` when both endpoints progress concurrently.
- Counter rotation preserves the existing `((r0+r1)+(r2+r3))` association for
  the lower half but produces `((r0+r3)+(r1+r2))` for the upper half. Native
  qualification therefore needs an explicit numerical gate; contributor
  completeness alone cannot establish byte-identical BF16 output.
- Recursive-doubling all-gather has a `3n` critical path for a local shard of
  `n` bytes. Sending the shard to both neighbors and relaying disjoint halves
  to the opposite rank proves complete delivery with `1.5n` bytes on each
  endpoint and a `1.5n` critical path.

Run the CPU contract with:

```text
python -m pytest spark_transport/experiments/tiled_prefill -q
```

Inspect the Q512 and Q4096 GPU contracts with:

```text
python -m spark_transport.experiments.tiled_prefill.gpu_harness \
  --query-rows 512 4096
```

Inspect the standalone four-rank qualification matrix without contacting any
host with:

```text
powershell -ExecutionPolicy Bypass -File \
  spark_transport/scripts/run_tp4_tiled_prefill_qualification.ps1
```

The matrix covers isolated and steady receipts at Q40/Q512/Q1024/Q4096,
capacity-transition correctness at Q513/Q1025, single-edge credit-delay arms,
and symmetric poison injection. It requires exactly one JSON receipt from
each rank, 8,389,632 registered tile-storage bytes per edge, half-specific
BF16 association identifiers, inactive-capacity sentinels, and drained
teardown. The runner defaults to plan-only; remote execution requires the
explicit `-Execute` switch and still does not start a serving model.

The native correctness executor is constrained to
`single_stream_correctness_only`. Timing fields are useful instrumentation,
but cannot support a performance claim until a reviewed multi-stream mapping
preserves the dependency DAG and concrete GPU/edge ports bind every executor
protocol seam.

Compile and run the standalone native executor test without CUDA or verbs:

```text
g++ -std=c++17 -Wall -Wextra -Wpedantic -Werror \
  -Ispark_transport/include \
  -Ispark_transport/experiments/tiled_prefill \
  spark_transport/tests/tp4_tiled_executor_test.cpp \
  spark_transport/experiments/tiled_prefill/tiled_executor.cpp \
  -o tp4_tiled_executor_test
./tp4_tiled_executor_test
```
