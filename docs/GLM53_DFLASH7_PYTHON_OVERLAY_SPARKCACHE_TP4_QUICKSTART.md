# Serve GLM-5.3 with external DFlash7 and the exact Python-overlay runtime

Status: **implemented**, not qualified. The image builder, profile resolver,
and four-rank dry-run contract pass without GPUs. No image digest from this
path has completed TP4/DCP1 model loading, semantic generation, SparkCache
store/restart/restore, or concurrency qualification.

## Runtime contract

| Role | Exact identity |
|---|---|
| vLLM native extensions and wheel metadata | `da4d7be6c97434f6942292ed8abbf4b32dc44355` |
| vLLM Python source | `0b67266a0f37d6146a8403fb8482403c62f412d5`, tree `ba9484ccb33aa56e90ff2f447f15ca9b9da97639` |
| B12X | `b1d541f9e71a35f030d45fae437630fff7507c2a`, tree `c69cdec1c59a08e8e0e549f930fa8abcfb5134ae` |
| SparkCache reconstructed-page placement | `5d571018de5b63a9a90e5c11e6d6e86bbff4a957`, tree `e864ed9ad64f771188fdb59aa9738e348134d636` |
| DFlash draft-loader separation | Patch SHA-256 `39b567013ee7aed79f63200ed460129587933dc77fb430decdf19f78178de279`, postimage SHA-256 `98acbae2b3bb4482d83f9637c163ce7c92707ccdf6561b7e431f23337f151cf4` |
| Unused DeepEP removal | Distribution `deep_ep==2.0.0+local`, removal receipt SHA-256 `65514f44829e7d176b0b2cacc9559ed22724e525b7041a8bcd4d2e02d1f372e3` |
| Target | `local-inference-lab/GLM-5.3-Flash-NVFP4@520de24eabf507659eaef7c70f14fd584527facc` |
| External draft | `incoai/GLM-5.3-Flash-DFlash2@dc77ff1c99eeb2df044ee3d4f0094eb033fee410`, BF16 weights SHA-256 `b33c03475ba7322cf398828f2d8d1be376df30dc05c6b40c28c8ea8da23e410b` |

The serving contract uses seven speculative tokens, draft TP4, target FP8 KV,
32 sequences, and 256-token vLLM blocks. SparkCache selects
`tail-cow-v1`, which maps opaque GLM pages to the `page-tail-cow-v1`
namespace, and uses the canonical CUDA restore keys.

## Build the image

Run on Linux ARM64 from a clean checkout containing this guide:

```bash
IMAGE='sparkring-glm53-sparkcache:dflash7-vllm-python-0b67266-native-da4d7be-b12x-b1d541f-arm64' \
BUILD_RECEIPT="$PWD/glm53-dflash7-python-overlay-image-receipt.json" \
bash runtime/glm53-flash-dflash7-python-overlay/build-image.sh
```

The builder verifies the public da4 image, the 31-file Python overlay, retained
native ELFs and dispatch operators, B12X, SparkCache clean source, the CUDA
placement library, four exact vLLM patches, and the eleven-file lease contract.
The base-image inspection must identify exactly one installed distribution,
`deep_ep==2.0.0+local`, as the owner of the `deep_ep` module. The derived image
uninstalls that exact distribution and verifies that `deep_ep` is absent. The
profiles use B12X kernels and PYNCCL collectives, so DeepEP is not a serving
dependency. The builder does not push the image.

Both profiles leave Torch thread selection to vLLM, select language-model-only
serving, and disable the unsupported symmetric-memory and FlashInfer all-reduce
candidates. PYNCCL remains bound to `/opt/sparkring/nccl/libnccl.so.2`, B12X
remains the attention, MoE, and linear backend, and the all-reduce RMS fusion is
disabled. ModelOpt experimental-quantization and FP8 KV accuracy warnings remain
visible because they describe real format and accuracy limitations; the profiles
do not suppress warnings.

## Choose the target loader

Two profiles share the same image and DFlash7 cache identity:

| Profile | Status | Loader behavior |
|---|---|---|
| `glm53-flash-dflash7-python-overlay-safetensors-sparkcache-tp4-dcp1.example.json` | **implemented**, not qualified | Uses global safetensors for target and draft. This follows the qualified-compatible loader shape but still requires live qualification on the composed 0b image. |
| `glm53-flash-dflash7-python-overlay-fastsafetensors-sparkcache-tp4-dcp1.example.json` | **implemented**, not qualified | Uses global fastsafetensors with queue size one for the target and `draft_load_config={"load_format":"safetensors"}` for DFlash. |

The image applies an exact-input vLLM patch that passes
`SpeculativeConfig.draft_load_config` to the DFlash model loader. The image
receipt verifies patch SHA-256
`39b567013ee7aed79f63200ed460129587933dc77fb430decdf19f78178de279` and
postimage SHA-256
`98acbae2b3bb4482d83f9637c163ce7c92707ccdf6561b7e431f23337f151cf4`.
Both profiles remain unqualified until live four-rank gates pass.

## Resolve the profile and inspect the plan

Copy `scripts/config/glm53-flash-tp4-site.example.yaml`
outside version control and replace every address, interface, SSH target,
device, host path, and image identity. Select one profile template:

```bash
receipt="$PWD/glm53-dflash7-python-overlay-image-receipt.json"
image='sparkring-glm53-sparkcache:dflash7-vllm-python-0b67266-native-da4d7be-b12x-b1d541f-arm64'
profile_template='scripts/config/glm53-flash-dflash7-python-overlay-safetensors-sparkcache-tp4-dcp1.example.json'

python scripts/prepare_glm53_dflash7_python_overlay_profile.py \
  --profile-template "$profile_template" \
  --site-template /path/to/resolved-glm53-site.yaml \
  --image "$image" \
  --image-id "$(jq -r .image_id "$receipt")" \
  --cuda-placement-library-sha256 "$(jq -r .artifacts.sparkcache_cuda_placement_sha256 "$receipt")" \
  --native-elf-manifest-sha256 "$(jq -r .runtime_contract.native_elf_manifest_sha256 "$receipt")" \
  --native-dispatch-manifest-sha256 "$(jq -r .runtime_contract.native_dispatch_manifest_sha256 "$receipt")" \
  --source-receipt-sha256 "$(jq -r .artifacts.source_receipt_sha256 "$receipt")" \
  --profile-output /path/to/glm53-dflash7-profile.json \
  --site-output /path/to/glm53-dflash7-site.yaml

python scripts/sparkring_generic_launcher.py \
  --site /path/to/glm53-dflash7-site.yaml \
  --profile /path/to/glm53-dflash7-profile.json \
  plan
```

`plan` is offline. Inspect every rank action before a lifecycle command.

## Cache namespace impact

The external DFlash weights SHA-256 is stored as
`spark_cache_draft_checkpoint_sha256`. It cannot share entries with embedded
MTP profiles. `tail-cow-v1` also separates these entries from snapshot-v1
manifests. The two target-loader profiles share a namespace because loader
choice does not change target or draft model state; each profile uses a
different cache root and one-shot clear token while qualification is pending.
