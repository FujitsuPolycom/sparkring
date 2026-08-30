# Serve GLM-5.3 with external DFlash7 and the exact Python-overlay runtime

Status: **implemented** for the builder and profile resolver. Image ID
`sha256:9faa36a9f37aee16d97ab9214ef3153b4d200121126e6b2dee5ebb63109fea18`
is **qualified** only for the bounded startup, health, semantic smoke, and
restore cases in the
[live-validation record](../performance/records/glm53-flash/dflash7-python-overlay-pr25-live-validation.md).
DFlash response quality and serving configurations outside that record are
**unsupported** by its evidence.

## Runtime contract

| Role | Exact identity |
|---|---|
| vLLM native extensions and wheel metadata | `da4d7be6c97434f6942292ed8abbf4b32dc44355` |
| vLLM Python source | `0b67266a0f37d6146a8403fb8482403c62f412d5`, tree `ba9484ccb33aa56e90ff2f447f15ca9b9da97639` |
| B12X | `b1d541f9e71a35f030d45fae437630fff7507c2a`, tree `c69cdec1c59a08e8e0e549f930fa8abcfb5134ae` |
| SparkCache reconstructed-page placement | `5d571018de5b63a9a90e5c11e6d6e86bbff4a957`, tree `e864ed9ad64f771188fdb59aa9738e348134d636` |
| DFlash draft-loader separation | Patch SHA-256 `39b567013ee7aed79f63200ed460129587933dc77fb430decdf19f78178de279`, postimage SHA-256 `98acbae2b3bb4482d83f9637c163ce7c92707ccdf6561b7e431f23337f151cf4` |
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
It does not push the image.

## Choose the target loader

Two profiles share the same image and DFlash7 cache identity:

| Profile | Status | Loader behavior |
|---|---|---|
| `glm53-flash-dflash7-python-overlay-safetensors-sparkcache-tp4-dcp1.example.json` | **implemented** | Uses global safetensors for target and draft. This profile has no live evidence on the composed image. |
| `glm53-flash-dflash7-python-overlay-fastsafetensors-sparkcache-tp4-dcp1.example.json` | **qualified** for the recorded image and bounded gates | Uses global fastsafetensors with queue size one for the target and `draft_load_config={"load_format":"safetensors"}` for DFlash. |

The image applies an exact-input vLLM patch that passes
`SpeculativeConfig.draft_load_config` to the DFlash model loader. The image
receipt verifies patch SHA-256
`39b567013ee7aed79f63200ed460129587933dc77fb430decdf19f78178de279` and
postimage SHA-256
`98acbae2b3bb4482d83f9637c163ce7c92707ccdf6561b7e431f23337f151cf4`.
The all-safetensors profile is implemented and not qualified; it has no live
four-rank evidence.
The fastsafetensors profile completed the bounded four-rank gates recorded for
image ID
`sha256:9faa36a9f37aee16d97ab9214ef3153b4d200121126e6b2dee5ebb63109fea18`.
That result does not qualify other image IDs or the all-safetensors profile.

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

## Launch the recorded image with SparkCache pull request 25 keys

The validated image ID
`sha256:9faa36a9f37aee16d97ab9214ef3153b4d200121126e6b2dee5ebb63109fea18`
contains SparkRing commit
`e2d92fdc7d0306d664d6fd9f296dc2adcaf0fe05` and
[SparkCache pull request 25](https://github.com/FujitsuPolycom/sparkcache/pull/25)
commit `5d571018de5b63a9a90e5c11e6d6e86bbff4a957`. That SparkCache commit accepts
the configuration names whose literals begin `spark_cache_native_`. The
profile templates expose the canonical CUDA names, so translate the resolved
profile only when launching this exact image:

| Canonical profile key | Key accepted by the recorded image |
|---|---|
| `spark_cache_cuda_restore` | `spark_cache_native_restore` |
| `spark_cache_cuda_placement_library` | `spark_cache_native_library` |
| `spark_cache_cuda_placement_library_sha256` | `spark_cache_native_library_sha256` |
| `spark_cache_cuda_placement_arena_bytes` | `spark_cache_native_arena_bytes` |
| `spark_cache_cuda_restore_io_workers` | `spark_cache_native_io_workers` |

Apply the translation after resolving the profile and before running
`plan`. The program fails if a source key is absent or a destination key is
already present:

```bash
python - /path/to/glm53-dflash7-profile.json <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
profile = json.loads(path.read_text())
args = profile["extra_vllm_args"]
index = args.index("--kv-transfer-config") + 1
config = json.loads(args[index])
extra = config["kv_connector_extra_config"]
mapping = {
    "spark_cache_cuda_restore": "spark_cache_native_restore",
    "spark_cache_cuda_placement_library": "spark_cache_native_library",
    "spark_cache_cuda_placement_library_sha256": "spark_cache_native_library_sha256",
    "spark_cache_cuda_placement_arena_bytes": "spark_cache_native_arena_bytes",
    "spark_cache_cuda_restore_io_workers": "spark_cache_native_io_workers",
}
for source, destination in mapping.items():
    if source not in extra or destination in extra:
        raise SystemExit(f"refusing ambiguous SparkCache key translation: {source} -> {destination}")
    extra[destination] = extra.pop(source)
args[index] = json.dumps(config, separators=(",", ":"))
path.write_text(json.dumps(profile, indent=2) + "\n")
PY

python scripts/sparkring_generic_launcher.py \
  --site /path/to/glm53-dflash7-site.yaml \
  --profile /path/to/glm53-dflash7-profile.json \
  plan
```

[SparkRing pull request #137](https://github.com/FujitsuPolycom/sparkring/pull/137)
changes the image contract to SparkCache pull request 26 and canonical CUDA
configuration names. The image specified by pull request #137 has not been
built or live-validated. Do not use the translation above for an
image whose receipt binds SparkCache pull request 26 or a later source
contract.

## Cache namespace impact

The external DFlash weights SHA-256 is stored as
`spark_cache_draft_checkpoint_sha256`. It cannot share entries with embedded
MTP profiles. `tail-cow-v1` also separates these entries from snapshot-v1
manifests. The two target-loader profiles could share a namespace because
loader choice does not change target or draft model state. Their templates use
different cache roots and one-shot clear tokens so observations from one
loader do not enter validation of the other loader.
