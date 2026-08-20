# Tiered graph all-reduce with deferred credits

## Status and scope

The sequential two-slot deferred-credit protocol plus the tiered 64-KiB graph
kernel is **live-validated** in the operator-accepted four-Spark EXL3 3.5-bpw
profile (R7). It serves TP4 BF16 all-reduce shapes from Q1 through Q40. The public
source is hardware-specific: it assumes four ranks, two directly attached
ConnectX-7 edges per rank, and CUDA architecture SM121.

Hidden width is a qualification scope, not a source assumption. Eager
admission is width-generic behind `VLLM_SPARK_TP4_EAGER_WIDTHS`, described in
the [vLLM integration README](integrations/vllm/README.md); the
DeepSeek-V4-Flash-0731 profile admits width 4096, and the shadow validation set
exercises widths 512 through 2048. What is qualified only at the GLM hidden
width of 6,144 is CUDA-graph capture: the graph paths in the accepted profiles
capture that width alone, and any other width is research-only.

The dual-port striped schedule and the prefill-capacity pool are
**research-only** and default off. They are included because the vLLM adapter
shares the same versioned ABI and port-namespace validator; they are not part
of the accepted operator profile.

This transport does not implement the measured exact-Q40 routed-MoE speedup.
That optimization is a separate target-only EXL3 source overlay. A complete
operator-profile reproduction needs both layers.

## Build the native library

Build on an ARM64 Spark host or in an ARM64 CUDA development container that
has CUDA 13, CMake, a C++17 compiler, and libibverbs development headers:

```bash
cmake -S spark_transport -B build/sircl-tiered \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=121
cmake --build build/sircl-tiered \
  --target spark_transport_capi \
  --parallel
ctest --test-dir build/sircl-tiered --output-on-failure
```

The serving artifact is:

```text
build/sircl-tiered/libspark_transport_capi.so
```

Record the library hash before distribution and verify the same bytes on all
four ranks:

```bash
sha256sum build/sircl-tiered/libspark_transport_capi.so
```

The build keeps the original unversioned `spark_tp4_create` ABI. The accepted
path uses the additive
`spark_tp4_create_with_protocol_and_graph_kernel` entry point, allowing an
old or mismatched native library to fail before graph capture instead of
silently selecting another protocol.

## Install the vLLM adapter files

Stage the following files together with the native library and verify their
hashes on every rank:

```text
spark_transport/integrations/vllm/spark_tp4_backend.py
spark_transport/integrations/vllm/spark_tp4_port_namespace.py
spark_transport/integrations/vllm/spark_tp4_prefill_capacity_pool.py
build/sircl-tiered/libspark_transport_capi.so
```

Mount the three Python files into a read-only adapter directory on
`PYTHONPATH`. Mount the native library read-only and point
`SPARK_TP4_LIBRARY` at its container path. The generic launcher can express
these as `extra_volumes`; its attestation hook should validate all four
SHA-256 values before vLLM starts.

## Accepted selector settings

The operator-accepted transport settings are:

```text
VLLM_SPARK_TP4_MODE=custom
VLLM_SPARK_TP4_GRAPH_Q1=1
VLLM_SPARK_TP4_GRAPH_ALLREDUCE_PROTOCOL=two_slot_deferred_ack
VLLM_SPARK_TP4_GRAPH_KERNEL_STRATEGY=tiered_64k
VLLM_SPARK_TP4_PREFILL_Q512=0
SPARK_TP4_MAX_INFLIGHT=64
SPARK_TP4_GRAPH_SUBMIT_CPU=10
SPARK_TP4_GRAPH_PROGRESS_CPU=11
```

The graph control ports must be a distinct validated pair for every active
session. The accepted profile also uses separate port pairs and progress CPUs
for vocabulary transport. Site-specific peer addresses, devices, GIDs, CPU
sets, and ports belong in the ignored site configuration; do not copy values
from another appliance without validating its topology.

`tiered_64k` selects the fused graph kernel for small Q and the split 64-KiB
kernel for larger captured Q. The choice is fixed for each captured node.
`two_slot_deferred_ack` gives each physical edge two generation-tagged payload
slots. A slot cannot be reused until its peer returns the exact prior
generation credit. Shutdown drains both slots before releasing registered
memory.

## Fail-closed qualification

Before selecting `custom` in serving, run the CPU/offline adapter contracts:

```bash
python -m pytest \
  spark_transport/integrations/vllm/test_spark_tp4_backend_dispatch.py \
  spark_transport/integrations/vllm/test_spark_tp4_port_namespace.py \
  spark_transport/tests/test_tp4_deferred_ack_source_contract.py \
  spark_transport/tests/test_tp4_split_graph_source_contract.py \
  -q
```

On four Sparks, require all of the following before accepting the process:

- the startup hook hashes the exact native library and adapter files;
- every rank reports the requested deferred-credit and tiered-kernel flags;
- every required Q1-Q40 graph node is captured;
- a real request advances published, consumed, and completed sequences;
- all ranks converge on the same sequence count;
- fatal, overflow, and cold-fallback counters remain zero;
- fixed-seed outputs and finite log probabilities match the control;
- the service returns healthy and idle after the bounded workload.

Captured-node count alone proves only graph definition. Sequence advancement
and convergence prove replay transport. The exact-Q40 EXL3 overlay adds its
own pre-graph all-layer numerical parity receipt and must be qualified
separately.

## Rollback

Keep the prior hash-pinned launch profile intact. A safe rollback stops the
candidate, removes only the candidate containers, starts the preserved
profile, waits for graph capture and HTTP health, and confirms the original KV
capacity. Never change protocol or kernel settings inside an already captured
process.
