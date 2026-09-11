# GLM-5.3 runtime with adaptive MTP and live-tensor B12X KDA

Status: **implemented**. The builder pins
`local-inference-lab/vllm@0b67266a0f37d6146a8403fb8482403c62f412d5`
and Git tree `ba9484ccb33aa56e90ff2f447f15ca9b9da97639`. Four-rank
TP4/DCP1 serving remains **unqualified** until an immutable image digest has a
live qualification receipt.

The runtime implements acceptance-based adaptive multi-token prediction and
binds B12X KDA metadata once while operating on live layer tensors.
`pins.json` records the exact source revisions and verifies their first-parent
relationships.

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

The matching SparkCache overlay is pinned to
`FujitsuPolycom/sparkcache@20838ace3ebda570ca039cb7f1976c29da554b39`.
Its Linux-byte-exact vLLM contract is
`vllm-kv-block-lease-contract-glm53-b12x-kda-adaptive-mtp.json`. The
runtime-bound embedded-MTP identity separates its cache entries from the
[adaptive-MTP source runtime at vLLM revision e10536a](../glm53-flash-e10536a/README.md).
Cross-revision reuse requires evidence of equivalent cached state.
