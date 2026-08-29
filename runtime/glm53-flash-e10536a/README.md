# GLM-5.3 source runtime at vLLM e10536a

Status: **implemented**. This builder pins
`local-inference-lab/vllm@e10536aadf02a18fccddda7ec939c33147e8b0b3`,
B12X, InstantTensor, CUDA, and SparkRing's source-built NCCL transport. It
does not inherit the qualification of the vLLM
`da4d7be6c97434f6942292ed8abbf4b32dc44355` image recorded in
`runtime/glm53-flash/pins.json`.

The vLLM revision adds GLM-5.3 internal MTP5 and opt-in acceptance-length
adaptation. A static MTP configuration remains static unless the launch
explicitly provides `adaptive_speculative_tokens_window`.

The revision also supplies the fastsafetensors parallel weight loader. The
research-only TP4 profile sets `VLLM_FASTSAFETENSORS_QUEUE_SIZE=1`; TP4 forces
the loader's `nogds=True` path, so the profile measures pipelined shard loading
rather than GPU Direct Storage. Loader selection does not change the target,
speculator, KV format, or SparkCache identity.

Build on Linux ARM64 with Docker BuildKit, at least 250 GiB of free storage,
and enough CPU and memory that the build does not interfere with serving:

```bash
IMAGE='sparkring-glm53-runtime:e10536a-source-arm64' \
BUILD_RECEIPT="$PWD/glm53-e10536a-runtime-receipt.json" \
bash runtime/glm53-flash-e10536a/build-image.sh
```

The builder verifies source commits and Git trees, builds for SM121, verifies
the output labels and imports, and writes an implemented-status receipt.
Four-rank serving remains unqualified until one immutable image digest passes
startup, semantic generation, persistent restore, shared-prefix concurrency,
and fatal-log checks.
