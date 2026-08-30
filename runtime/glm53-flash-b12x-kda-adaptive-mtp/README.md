# GLM-5.3 runtime with adaptive MTP and live-tensor B12X KDA

Status: **implemented**. The builder pins
`local-inference-lab/vllm@0b67266a0f37d6146a8403fb8482403c62f412d5`
and Git tree `ba9484ccb33aa56e90ff2f447f15ca9b9da97639`. Four-rank
TP4/DCP1 serving remains **unqualified** until an immutable image digest has a
live qualification receipt.

The pinned vLLM history contains the complete GLM-5.3 source runtime from
`da4d7be6c97434f6942292ed8abbf4b32dc44355` through acceptance-based
adaptive MTP at `e10536aadf02a18fccddda7ec939c33147e8b0b3`, followed by three
commits that bind B12X KDA metadata once and operate on live layer tensors.
`pins.json` records and verifies the exact first-parent sequence.

The runtime also pins B12X, InstantTensor, CUDA, and SparkRing's source-built
NCCL transport. The fastsafetensors TP4 profile uses loader queue size one.
The vLLM implementation selects `nogds=True` when tensor parallelism exceeds
one, so TP4 uses pipelined host I/O without GPU Direct Storage.

Build on Linux ARM64 with Docker BuildKit and at least 250 GiB of free local
storage:

```bash
IMAGE='sparkring-glm53-runtime:b12x-kda-adaptive-mtp-0b67266a-arm64' \
BUILD_RECEIPT="$PWD/glm53-b12x-kda-adaptive-mtp-runtime-receipt.json" \
bash runtime/glm53-flash-b12x-kda-adaptive-mtp/build-image.sh
```

The builder verifies commits, Git trees, the complete vLLM lineage, source
licenses, patched NCCL bytes, output labels, and required Python imports. Its
receipt proves image construction only. Startup, semantic generation,
SparkCache restore, shared-prefix concurrency, and fatal-log checks require a
separate four-rank receipt.

The matching SparkCache reconstructed-page placement source is commit
`5d571018de5b63a9a90e5c11e6d6e86bbff4a957`, Git tree
`e864ed9ad64f771188fdb59aa9738e348134d636`, with clean deployable-source
SHA-256 `f7c0565521fddeff7085e4cc08043cb8d1e2bde33abc67f83b8608a162d05b88`.
Its Linux-byte-exact vLLM contract is
`vllm-kv-block-lease-contract-glm53-b12x-kda-adaptive-mtp.json`. The
runtime-bound embedded-MTP identity prevents this profile from reusing e105
adaptive-MTP entries until byte-equivalence across the KDA revisions is proven.
