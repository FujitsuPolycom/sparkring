# Spark Transport

`spark_transport` is a topology-aware transport prototype for direct-attached
DGX Sparks. Two Sparks are the smallest test cell; the design target is four or
more ranks with multiple independent edges per process.

Before model work, qualify every physical edge with the non-destructive,
machine-readable [direct-cable qualification runner](CABLE_QUALIFICATION.md).
It verifies exact link/IP/route/MTU/RoCE state, tests GLM-sized payloads in
both directions, and separates cable/PHY faults from software-path latency.

An early two-node-era milestone ran TP2 x PP2:

```text
vLLM global rank 0 = spark0
vLLM global rank 1 = spark1
vLLM global rank 2 = spark3
vLLM global rank 3 = spark2

TP: [0,1] and [2,3]
PP: [0,2] and [1,3]
```

With the second pipeline stage physically reversed, all four logical groups
mapped to direct 200G links. That milestone is historical; the production
topology is the four-node TP4/DCP4 switchless ring described in the
repository root `README.md` and `docs/SETUP.md`.

For the fail-closed, reboot-recoverable routed-QSFP NCCL Socket network
bootstrap, see
[`ROUTED_QSFP_NCCL_BOOTSTRAP.md`](ROUTED_QSFP_NCCL_BOOTSTRAP.md).

## Modules

- `ControlChannel`: versioned endpoint exchange and test coordination over TCP.
- `MemoryBuffer`: swappable host, CUDA-mapped, and CUDA write-combined storage.
- `VerbsEndpoint`: one RC QP and MR for one directed peer edge.
- `Topology`: an arbitrary directed graph; the four-Spark TP2 x PP2 graph is
  covered by a unit test.
- `Statistics`: deterministic latency summaries.
- `spark_transport_probe`: the two-node vertical-slice executable.

There is no global QP or global memory region. A four-rank process creates one
endpoint per local edge, allowing both ConnectX-7 ports to operate
independently. Future 10G backends can implement the same peer-edge contract.

## Build

The DGX host has CMake and verbs headers, while the GLM image has `nvcc`.

```bash
cmake -S /src -B /build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=121
cmake --build /build --parallel
ctest --test-dir /build --output-on-failure
```

The gate is the full CTest suite: all 20 test executables declared in
`CMakeLists.txt` must pass.

The operator-accepted four-rank graph all-reduce uses the additive tiered
kernel and deferred-credit ABI published in this directory. See
[TIERED_DEFERRED_GRAPH.md](TIERED_DEFERRED_GRAPH.md) for its build, adapter,
attestation, qualification, and rollback contract. The exact-Q40 routed-MoE
optimization is a separate EXL3 overlay; compiling the transport library does
not install it.

## Pair probe

Server on `spark1`:

```bash
spark_transport_probe \
  --server \
  --device rocep1s0f0 \
  --gid 3 \
  --control-port 9410 \
  --bytes 16384 \
  --memory host
```

Client on `spark0`:

```bash
spark_transport_probe \
  --client 192.0.2.2 \
  --device rocep1s0f0 \
  --gid 3 \
  --control-port 9410 \
  --bytes 16384 \
  --memory host
```

CUDA-mapped validation adds:

```text
--memory cuda-mapped
--gpu-producer   # client
--gpu-verifier   # server
```

## Results

Direct `spark0 -> spark1`, 16 KB, 10,000 iterations:

| Memory backend | Producer | Verifier | p50 | p99 | Correct |
|---|---|---|---:|---:|---|
| page-aligned host | CPU | CPU | 4.752 us | 5.536 us | yes |
| `cudaHostAllocMapped` | GPU | GPU | 4.528 us | 4.816 us | yes |
| mapped + write-combined | GPU | GPU | 4.688 us | 4.864 us | yes |

The CUDA result proves:

1. A `cudaHostAllocMapped` pointer can be registered with `ibv_reg_mr`.
2. The GPU can produce data in that registered allocation.
3. ConnectX-7 can RDMA-write into the peer's registered mapped allocation.
4. The peer GPU can read and verify the resulting data.
5. No explicit `cudaMemcpy` is required for this storage path.

The reported timed region measured the reused RDMA write plus local CQ
completion. GPU production occurred once before the timed loop, and GPU
verification once after it. The persistent doorbell mode below removes that
limitation.

## Persistent GPU doorbell results

`--gpu-roundtrip` measures this closed loop:

```text
host command
-> persistent sender GPU produces 16 KB
-> ordered RC payload and sequence writes
-> persistent receiver GPU observes and consumes
-> RC acknowledgement
-> sender GPU observes acknowledgement
```

At 16 KB, 10,000 measured iterations:

| Receiver operation | p50 | p99 | Correct |
|---|---:|---:|---|
| sequence observation only | 14.64 us | 19.06 us | yes |
| byte-wise verification, 256 threads | 42.02 us | 42.82 us | yes |
| vectorized verification, 1,024 threads | 20.67 us | 20.99 us | yes |

Vectorized full-consume sweep:

| Payload | p50 | p99 |
|---:|---:|---:|
| 4 KB | 19.62 us | 22.88 us |
| 8 KB | 20.00 us | 21.38 us |
| 12 KB | 20.30 us | 20.72 us |
| 16 KB | 20.67 us | 20.99 us |
| 24 KB | 21.94 us | 23.71 us |
| 32 KB | 22.45 us | 22.96 us |
| 64 KB | 26.91 us | 27.90 us |

All 70,000 measured sweep iterations were correct. Direct registration of
both `cudaMalloc` and `cudaMallocManaged` allocations was also tested and
failed at `ibv_reg_mr` on both Sparks. This makes CUDA-mapped ingress plus a
persistent vectorized GPU consumer the qualified path under these registration
constraints.

## TP2 BF16 exchange and fused add

`spark_tp2_probe` is the TP2 BF16 collective primitive. Both ranks
simultaneously:

1. publish a local BF16 tensor;
2. RDMA-write it into the peer's mapped ingress slot;
3. add local and remote BF16 pairs into ordinary `cudaMalloc` output memory;
4. validate every result;
5. acknowledge consumption before the slot is reused.

The packed BF16-pair implementation achieved:

| Payload | Meaning | p50 | p99 | Correct |
|---:|---|---:|---:|---|
| 4 KB | small decode/control tensor | 14.98 us | 15.34 us | yes |
| 8 KB | small hidden fragment | 17.76 us | 19.22 us | yes |
| 12 KB | one 6,144-wide GLM BF16 vector | 20.27 us | 20.75 us | yes |
| 16 KB | benchmark target | 22.83 us | 30.02 us | yes |
| 24 KB | two GLM hidden vectors | 27.81 us | 28.85 us | yes |
| 32 KB | larger batch | 32.77 us | 33.76 us | yes |
| 64 KB | larger batch/prefill | 52.51 us | 61.14 us | yes |

A separate one-million-iteration 16 KB burn used a lane-varying,
sequence-varying pattern and produced zero mismatched iterations on either
rank:

```text
p50 22.544 us
p99 23.072 us
p99.9 37.425 us
```

The uniform-value fast path reached 21.84 us p50, 20.8% below the
clean 27.58 us adjacent NCCL-IB p50. The stricter burn remained about 18%
below NCCL while providing materially stronger corruption detection.

The integration-faithful 12 KB path starts with alternating,
lane-varying tensors in ordinary `cudaMalloc` memory, copies them into mapped
send storage, exchanges them, and writes the fused sum to ordinary
`cudaMalloc` output memory:

| Scheduling | Iterations | p50 | p99 | Correct |
|---|---:|---:|---:|---|
| unpinned | 10,000 | 20.64 us | 29.98 us | yes |
| process pinned to CPU 10 | 100,000 | 21.09 us | 22.82 us | yes |

The ordinary-CUDA staging path therefore adds less than 1 us at p50. CPU
affinity is a deployment requirement: it removed roughly 7 us of p99
host-polling and acknowledgement jitter.

The measured shape curve supports hybrid dispatch:

- custom TP2 for latency-sensitive decode tensors around 4-16 KB;
- benchmark both paths near 24 KB;
- retain NCCL for larger tensors until pipelined multi-slot transport is
  implemented.

The reusable scripts are:

- `scripts/run_two_spark_sweep.sh` for the GPU-visible transport;
- `scripts/run_tp2_sweep.sh` for the fused BF16 operation.

## GLM checkpoint precision

All four Sparks have the same `config.json` SHA-256:

```text
ffd30e72ab8bb7e8ad560f2aaab03cc595f3106f0acf793ef96eedaf90f66d69
```

That checkpoint declares:

- model/output dtype BF16;
- routed experts MXFP4;
- attention and shared-expert weights/inputs FP8;
- runtime NVFP4 MLA KV cache.

Weight quantization therefore does not make packed NVFP4 a valid all-reduce
format. The TP partial-sum path remains BF16 unless the runtime explicitly
adds a quantize/dequantize collective. NVFP4 CKV gather is a separate raw-copy
transport operation with scale metadata, not a reduction.

## TP4 status and open work

The GLM trace confirmed that the hot decode collective is contiguous BF16
`[1, 6144]`, or 12,288 bytes, and that the runtime issues 128 of these
all-reduces per generated token.

The measured four-rank primitive uses two direct-cable perfect matchings:

```text
round 0 / cage 0: 0 <-> 1    2 <-> 3
round 1 / cage 1: 0 <-> 3    1 <-> 2
```

The final ordinary-CUDA-tensor implementation achieved 40.14-40.22 us p50
and 50.11-50.85 us p99 over 10,000 iterations, with zero mismatches on all
four ranks. See
`2026-07-25-glm52-tp4-direct-cable.md` (private archive).

After staging `spark_tp4_tensor_probe` as `/tmp/spark_tp4_tensor_probe` on all
four nodes, reproduce the ordinary-CUDA-tensor path from a Windows host with:

```powershell
.\spark_transport\scripts\run_tp4_tensor_probe.ps1 `
  -Warmup 1000 `
  -Iterations 10000
```

The runner uses the exact two-round direct-cable matching above, pins each
probe to CPU 10, enforces a bounded watchdog, prints only `TP4_TENSOR` result
lines on success, and removes only its rank-specific test containers.

After staging the C ABI library and vLLM integration directory, compare both
BF16 reduction orders against an FP32 ground-truth sum with:

```powershell
.\spark_transport\scripts\run_tp4_numerical_audit.ps1 `
  -Iterations 1000
```

This runner refuses to start while `glm52-trace` is running. It reports MAE,
RMSE, maximum absolute error, per-element wins, and exact agreement with the
correctly rounded FP32 sum for both the custom tree and NCCL.

Open work:

1. Complete the FP32-ground-truth promotion audit.
2. Benchmark custom-only GLM decode against the NCCL Socket baseline.
3. Add multi-slot ingress so consecutive decode collectives can overlap.
4. Retain NCCL for prefill and every non-matching tensor.
