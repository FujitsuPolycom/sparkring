# GLM-5.3 DFlash7 public-base Python overlay

Status: **implemented**, not qualified. The builder constructs and verifies an
ARM64 image but no image digest from this path has completed four-rank serving
qualification.

The image combines these exact roles:

- retained vLLM native extensions and wheel metadata from
  `da4d7be6c97434f6942292ed8abbf4b32dc44355`;
- the 31-file vLLM Python delta at
  `0b67266a0f37d6146a8403fb8482403c62f412d5`;
- B12X `b1d541f9e71a35f030d45fae437630fff7507c2a`;
- SparkCache reconstructed-page placement source
  `9e779c32b285e00577a7829a75192069d12685dc`, Git tree
  `4df3ea1435241a688e6d44345687414605131450`;
- external BF16 DFlash2 weights with SHA-256
  `b33c03475ba7322cf398828f2d8d1be376df30dc05c6b40c28c8ea8da23e410b`.

The image builder shares the byte allowlist, retained-native verifier, exact
SparkCache patch chain, and eleven-file lease contract with
`runtime/glm53-flash-adaptive-mtp-python-overlay/`. Prepared image metadata is
rendered for external DFlash7; it does not claim adaptive MTP.

Build on Linux ARM64:

```bash
IMAGE='sparkring-glm53-sparkcache:dflash7-vllm-python-0b67266-native-da4d7be-b12x-b1d541f-arm64' \
BUILD_RECEIPT="$PWD/glm53-dflash7-python-overlay-image-receipt.json" \
bash runtime/glm53-flash-dflash7-python-overlay/build-image.sh
```

The script does not push the image. Its receipt verifies mixed vLLM
provenance, B12X, target loader dependencies, NCCL, SparkCache CUDA placement,
the clean SparkCache source receipt, the vLLM lease contract, and removal of the
unused `deep_ep==2.0.0+local` distribution. The removal receipt has SHA-256
`65514f44829e7d176b0b2cacc9559ed22724e525b7041a8bcd4d2e02d1f372e3`.
The build accepts removal only when `deep_ep` has that one distribution owner,
then verifies that the module is absent. DFlash model files remain
operator-mounted and are verified by the runtime profile.

Two executable profiles use external DFlash at depth seven and TP4, FP8 target
KV, 256-token vLLM blocks, 32 sequences, and SparkCache page-tail copy-on-write
publication with CUDA restore. Both are implemented but unqualified on the
composed 0b image. The conservative profile uses global safetensors. The mixed
profile uses global fastsafetensors for the target and an exact
`draft_load_config` selecting safetensors for DFlash. The image applies and
verifies the draft-loader patch before installing SparkCache patches. See
`docs/GLM53_DFLASH7_PYTHON_OVERLAY_SPARKCACHE_TP4_QUICKSTART.md`.

Both profiles preserve B12X compute backends and the pinned PYNCCL/NCCL
library. They disable unsupported symmetric-memory and FlashInfer all-reduce
probes, disable the all-reduce RMS fusion, select language-model-only serving,
and leave Torch thread selection unset. ModelOpt and FP8 KV warnings remain
visible because they describe supported-runtime limitations rather than unused
optional backends.

## Recurrent replay-boundary hand-off

Status: **implemented** in the vLLM overlay; unsupported until the selected KV
connector explicitly advertises `supports_recurrent_boundary_blocks` and
consumes the interface.

`SchedulerOutput.recurrent_boundary_blocks` has this schema:

```text
dict[str, list[tuple[int, int, int]]] | None
request_id -> [(group_id, block_id, boundary_tokens), ...]
```

Each entry identifies a Mamba `align` block whose prefix-cache hash covers
exactly `boundary_tokens`. The replay boundary is the greatest 256-token hash
boundary below the prompt end. A full 2,304-token recurrent page is admitted
only when its stored hash token count and group ID both match that boundary.
The scheduler never substitutes a later running-state or DFlash speculative
slot. Existing partial-tail copy-on-write targets remain available through
`partial_tail_offloads` and are also included in the new field.

The scheduler pins an admitted block before worker execution. A connector that
opts in must finish its worker-side snapshot before request cleanup; overlapping
scheduler steps defer request-block recycling until their execution fence has
completed. Request cleanup releases the pin, including cancellation paths.
Connectors without the capability receive no aligned-boundary metadata and
retain no additional recurrent block. The interface does not change
SparkCache cache identities or on-disk namespaces.
