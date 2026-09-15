# Qwen TP4 prefill compute bundle

Status: **implemented**, with bounded DGX4 measurements. This opt-in bundle
targets Qwen3.8-Flash-Next NVFP4 QAD revision
`629bc3218833a38b475b719f34aa571666f4a03e` on the pinned SparkRing R37 ARM64
runtime. It preserves the checkpoint, tensor-parallel layout and serving
capacity. It does not select a transport or change profile defaults.

## Compute changes

- HC up-projection/gating: a Triton kernel computes the four stream coordinates
  for each hidden coordinate together, avoiding the full gate-logit intermediate.
  It retains BF16 projection logits, BF16 sigmoid, BF16 products, sequential
  FP32 stream summation and a BF16 mean. The operator returns fresh output
  storage so Torch compilation does not copy a mutable shared workspace into
  and out of the operation. Fewer than 128 rows use the original two-stage
  computation inside the operator.
- MTP prefill feedback: for at least 128 tokens, two standard Torch GEMMs replace
  the dedicated CuTe projections. Each GEMM stores BF16 before the token-path
  addition, which also stores BF16. This is deliberately not `addmm`, whose
  rounding boundary differs. Normalization, scratch ownership and small-row
  CuTe kernels are retained.

The geometry is BF16, hidden width 2560, four HC streams and HC low rank 320.
Both hooks require TP4 agreement on their source identities. The MTP path also
requires warmup outside CUDA-graph capture. HC graph replay, nonaligned row
counts, sampled checkpoint weights and concurrent API responses have bounded
checks; the evidence is not a long-context, multimodal or stability qualification.

## Package and admit

From the repository root:

```bash
python integrations/vllm/qwen38_prefill/package_prefill.py --destination /tmp/qwen-prefill
python -m pytest integrations/vllm/qwen38_prefill -q
```

The destination must not exist. The command emits a manifest SHA-256 and
cache namespace. Apply the following additions to each rank of an already
verified R37 Qwen TP4 launch:

| Container setting | Value |
|---|---|
| Read-only bundle mount | Package directory → `/opt/sparkring/qwen38-prefill` |
| Read-only startup hook mount | Package `qwen_prefill.pth` → `/opt/venv/lib/python3.12/site-packages/qwen_prefill.pth` |
| `SPARKRING_QWEN_PREFILL_MANIFEST_SHA256` | Full emitted manifest SHA-256 |
| `VLLM_CACHE_ROOT` | `/cache/qwen-prefill-<first12 manifest characters>/vllm` |
| `TORCHINDUCTOR_CACHE_DIR` | `/cache/qwen-prefill-<first12 manifest characters>/inductor` |

The `/cache` mount must be writable and persistent. The bootstrap verifies
every packaged file and the exact image-side HC/MTP Python sources before
installing the hooks. A missing manifest, changed source, stale compiler-cache
namespace or late hook installation stops startup. Source-bound cache paths
are mandatory: importing an HC hook alone does not invalidate vLLM's saved
compiled model.

Keep the image, checkpoint, model arguments, fabric and transport configuration
unchanged when comparing this bundle. The measured setup uses TP4/DCP1,
262144 context, 16 sequences, 8192 batched tokens, 24 GiB KV per rank and MTP3.
The separate [collective selector](../qwen38_collectives/README.md) continues to
route small reductions through RoCEnante and larger reductions through NCCL.

Restart all ranks when enabling or disabling the bundle. To roll back, stop the
bundle's containers and restart the preserved baseline deployment with its
original mounts, environment and compiler caches. Do not reuse the bundle's
compiled model as baseline evidence. This package does not configure the host
fabric, alter weights or install persistent host services.

## Evidence and scope

The final bundle's median cold-prefill measurements were:

| Input tokens | Baseline | Bundle | Throughput improvement |
|---|---:|---:|---:|
| 16K | 4.812s | 4.282s | 12.4% |
| 32K | 10.000s | 8.953s | 11.7% |

Final warm decode medians were about 80 tok/s at C1 and 247 at C8, versus 83
and 247 in the fresh baseline. C1 varied across runs; this does not establish
a decode improvement. Single-request arithmetic, eight concurrent arithmetic
requests and a 13K-token retrieval check passed. GPU comparisons using actual
QAD weights were bit-identical in the tested cases; MTP tail-buffer and graph
replay checks also passed.

See the [measurement record](../../../performance/records/qwen38-flash-next/qad-tp4-prefill-compute.json)
for exact conditions, samples and source hashes. Repeated cold 16K/32K requests
use identical token fixtures, disjoint prefixes and one output token. C1/C8
decode uses the same [benchmark harness](../qwen38_collectives/benchmark.py)
and separates cold and warm-prefix passes.

Use an idle API and a fresh serving process for each arm:

```bash
python performance/harnesses/vllm/qwen_prefill.py \
  --base-url http://SERVER:8015 --model Qwen3.8-Flash-Next-NVFP4-QAD \
  --fixtures /tmp/qwen-prefill-fixtures.json --output /tmp/qwen-prefill-baseline.json
```

Reuse the fixture file and choose a distinct output filename after enabling
the bundle. The harness saves requests' token counts, timings and responses.
First-use compilation of a previously unseen tail geometry can affect short
request latency; keep cold-request and warm decode measurements separate.

The MTP optimization is a B12X backend-selection candidate. HC fusion would
need a B12X operation and a corresponding vLLM caller change for upstream
adoption. These local hooks are source-bound integration evidence, not an
upstream release. TP2 evidence available for comparison uses different PTQ
weights, so this record does not establish a matched TP2-to-TP4 scaling factor.
