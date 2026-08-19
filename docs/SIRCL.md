# SIRCL (internal transport codename)

**SIRCL** is SparkRing's internal codename for the
**S**witchless **I**nference **R**DMA **C**ollective **L**ayer.

The names describe different boundaries:

- **SIRCL** is the low-level direct-cable collective runtime.
- **SparkRing** is the complete inference stack: SIRCL, topology and launch
  tooling, fail-closed runtime adapters, model-specific optimizations,
  validation, and fallback paths.
- **SparkCache** is the independent persistent-context subsystem. It stores
  rank-local KV state on NVMe and does not define the live collective
  transport.

SIRCL is not a second daemon and it is not a renamed NCCL fork. It is the
SparkRing-authored machinery under `spark_transport/` that carries admitted
steady-state tensors over the four direct ConnectX-7 links.

## Implemented data path

For the validated four-Spark cycle, one collective is decomposed into two
perfect matchings:

```text
round 0: 0 <-> 1    2 <-> 3
round 1: 0 <-> 3    1 <-> 2
```

The implementation combines:

1. persistent RC queue pairs on each directly attached edge;
2. registered `cudaHostAllocMapped` arenas addressable by the NIC and GPU;
3. monotonically increasing sequence numbers, acknowledgements, and bounded
   in-flight slots;
4. GPU copy/reduction kernels around the registered arenas;
5. device-published command rings and system-scope fences for CUDA-graph
   replay; and
6. pinned host progress threads that submit and reap verbs work.

The operator-accepted EXL3 3.5-bpw profile (R7) uses the versioned
`two_slot_deferred_ack` graph protocol with the `tiered_64k` kernel selector.
Two generation-tagged payload slots let graph replay defer acknowledgement
without permitting premature reuse. The selector keeps small captured Q on
the fused kernel and uses split 64-KiB work for larger captured Q. Build,
attestation, qualification, and rollback instructions are in
[`spark_transport/TIERED_DEFERRED_GRAPH.md`](../spark_transport/TIERED_DEFERRED_GRAPH.md).

The endpoint exchange is explicitly versioned, and the deployed descriptor
layout is pinned by the attested runtime bundle. A slot cannot be reused
before its consumer acknowledges the sequence. Publication regression,
timeout after native enqueue, or protocol disagreement is process-fatal
because falling back after a CUDA stream has already been gated would expose
partial state.

## What it carries

The validated reference runtime has specializations for measured GLM-5.2
surfaces, including TP4 all-reduce, generic all-gather, vocabulary all-gather,
and DCP query/combine operations. CUDA-graph command rings cover the admitted
decode widths used by the published configurations. The public tree contains
the corresponding source and native probes, but its clean-reproduction lane
has not yet matched the reference lane end to end.

Unsupported data-plane operations run on the explicitly attested NCCL
fallback. Gloo is limited to bootstrap and control traffic outside SIRCL.
SIRCL does not own weight loading, arbitrary node counts, switched fabrics,
or every datatype/layout.

## Direction

The intended reusable boundary is a size-generic collective API for
contiguous tensors on fixed two-node and four-node direct-cable topologies,
with standard PyTorch integration and visible fallback counters. The
public snapshot is not yet a drop-in `torch.distributed` backend; model and
runtime adapters live beside the transport core.

The dual-port striped schedule and prefill-capacity pool published beside the
accepted selector are research-only and default off. They are not implied by
the accepted sequential tiered/deferred result.

The acronym is an internal technical name because the unrelated
[Sircl HTML extension library](https://www.getsircl.com/) already uses the
name. The
public project and repository are named **SparkRing**. This repository does
not ship a `sircl` package, a `backend="sircl"` ProcessGroup, or a stable
`sircl_*` API.

### Design note: ring sizes beyond four

The validated collective decomposes the four-Spark cycle into two perfect
matchings, exploiting a coincidence specific to N=4: two rounds of pairwise
sums complete a full all-reduce. This does not generalize.

- Even N > 4: the cycle still splits into two matchings, but two pairwise
  rounds no longer complete a reduction. A switchless N-ring needs the
  classic ring reduce-scatter/all-gather schedule (about 2(N-1) neighbor
  hops, latency growing linearly in N) or relay rounds for the exchanges
  the cycle does not provide directly.
- Odd N (5, 7, ...): an odd cycle admits no perfect matching, so the
  pairwise structure is unavailable outright; only ring schedules apply.
- Model admission gates independently of the transport: vLLM requires
  attention-head counts divisible by the TP degree, which makes TP6
  (24/48/96-head models) and TP8 plausible while TP5/7/10 are rarely
  satisfiable.

No implementation or evidence exists beyond N in {2, 4}; this note records
the design boundary only.

## Code and evidence map

- `spark_transport/src/`, `spark_transport/include/`: verbs endpoints,
  registered arenas, sessions, sequence protocol, GPU operations, and graph
  command rings.
- `spark_transport/integrations/vllm/`: fail-closed runtime adapters.
- `spark_transport/app/`: standalone transport and collective probes.
- `spark_transport/tests/`: protocol, numerical, graph, and integration
  tests.
- [`ARCHITECTURE.md`](ARCHITECTURE.md): full stack architecture.
- [`RESULTS.md`](RESULTS.md): measured claims and evidence labels.
