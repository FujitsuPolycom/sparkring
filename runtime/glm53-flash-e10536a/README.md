# GLM-5.3 source runtime at vLLM e10536a

Status: **implemented**. This builder pins
`local-inference-lab/vllm@e10536aadf02a18fccddda7ec939c33147e8b0b3`,
B12X, InstantTensor, CUDA, and SparkRing's source-built NCCL transport. It
does not inherit the qualification of the older public GLM-5.3 OCI image.

The vLLM revision adds GLM-5.3 internal MTP5 and opt-in acceptance-length
adaptation. A static MTP configuration remains static unless the launch
explicitly provides `adaptive_speculative_tokens_window`.

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
