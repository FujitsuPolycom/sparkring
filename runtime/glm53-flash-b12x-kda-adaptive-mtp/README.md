# GLM-5.3 runtime with adaptive MTP and live-tensor B12X KDA

Status: **implemented**. The builder pins
`local-inference-lab/vllm@0b67266a0f37d6146a8403fb8482403c62f412d5`
at Git tree `ba9484ccb33aa56e90ff2f447f15ca9b9da97639` and
`local-inference-lab/b12x@b1d541f9e71a35f030d45fae437630fff7507c2a`
at Git tree `c69cdec1c59a08e8e0e549f930fa8abcfb5134ae`. Four-rank TP4/DCP1
serving remains **unqualified** until an immutable image digest has a live
qualification receipt.

The pinned vLLM history contains the complete GLM-5.3 source runtime from
`da4d7be6c97434f6942292ed8abbf4b32dc44355` through acceptance-based
adaptive MTP at `e10536aadf02a18fccddda7ec939c33147e8b0b3`, followed by three
commits that bind B12X KDA metadata once and operate on live layer tensors.
`pins.json` records and verifies the exact first-parent sequence.

The B12X pin accepts `kda_metadata_validation="trusted"` and derives tensor
capacity from each live request's projection and recurrent-state metadata.
B12X commit `2fcf23a0ce269be27b2e03fece73d46e90e6aeea` is **unsupported** for this
runtime: its `Caps` interface has no trusted-metadata selection, and its
`bind_kda` interface requires plan-sized tensors. The source preparer rejects
that byte-exact implementation and any source that lacks the required call
contract.

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

The builder verifies commits, Git trees, the complete vLLM and B12X lineages,
the trusted-metadata request-sized KDA call contract, source licenses, patched
NCCL bytes, output labels, and required Python imports. Its receipt proves
image construction only. Startup, semantic generation, SparkCache restore,
shared-prefix concurrency, and fatal-log checks require a separate four-rank
receipt.

The matching SparkCache overlay is pinned to
`FujitsuPolycom/sparkcache@20838ace3ebda570ca039cb7f1976c29da554b39`.
Its Linux-byte-exact vLLM contract is
`vllm-kv-block-lease-contract-glm53-b12x-kda-adaptive-mtp.json`. The
embedded-MTP identity binds both the vLLM and B12X revisions. Entries produced
by another KDA implementation therefore miss and recompute instead of crossing
an unproven numerical boundary.
