# Serve GLM-5.3 with external DFlash7 and the exact Python-overlay runtime

Use the [GLM-5.3 routing guide](GLM53_FLASH_QUICKSTARTS.md) to compare this
local-image path with the published BF16 DFlash2 composition, adaptive MTP,
and the source-built `e10536a` path.

Status: **implemented** for reproducible image construction, profile
resolution, and four-rank dry-run planning. The executable fastsafetensors
profile defaults to full `snapshot-v1` publication, one SparkCache load
thread, one pending restore, and an isolated snapshot-v1 cache root and clear
token.

Exact local image
`sparkring-glm53-sparkcache:dflash7-pr42-page-base-flight-singletonfix-arm64`,
image ID
`sha256:35b58a7bf414059c65b8f74e4e4b17ee6a81b7008e1bffbc9bd298b5e08c739e`,
is **qualified** only for this bounded snapshot case in the
[machine-readable validation receipt](../performance/receipts/glm53-flash/dflash7-snapshot-v1-safe/validation.json):

- GLM-5.3 DFlash7 on four DGX Sparks at TP4/DCP1;
- 20 GiB of FP8 GPU KV per rank;
- one 131,072-token full page snapshot, encoded as 813,068,464 bytes and 13
  authenticated macro objects per rank;
- exact context digest
  `b4161571df103395e2abae10372a90f35468561ec6c42bf4a7b7f0d0dfda5873`;
- all-rank CUDA restore in 1.552–1.700 seconds;
- exact prompt SHA-256
  `965acd85cb28f804ab59cdc160688b04efaee14341e0bd27b647673e652ab812`;
- expected and observed oracle `red` before publication and after restart,
  with response SHA-256
  `2c68d02422a6c4bdb42bd10221940894e746342bef6a56695fdbcb549074a355`.

That image binds SparkCache
`a1511d26a1fe2b17b24561bc52e376bf7f54b06a`, tree
`4d5b8eb8c5c13793ee7a1e67b2b34bd38fcf4ddb`, source SHA-256
`6651f2823c816fac93779cbca54a8f19c0ed262830953149f3a87d189d1f833b`.
The checked-in builder source below is a separate construction; rebuilding it
does not inherit this qualification.

### Research-only tail publication and concurrent restore

Opaque-page tail copy-on-write deltas, host-base read coalescing, and
different-root concurrent restore are **research-only**. A C2 delta-restore
attempt did not complete correctly; one request recomputed and returned its
correct oracle. That result does not qualify C2 delta restore. The prior
`ed60...` and `eef...` artifact records remain useful bounded observations,
but they do not replace the snapshot-v1 default or qualify multi-root
concurrency. Response quality and public OCI publication are **unsupported**.

## Runtime contract

| Role | Exact identity |
|---|---|
| vLLM native extensions and wheel metadata | `da4d7be6c97434f6942292ed8abbf4b32dc44355` |
| vLLM Python source | `0b67266a0f37d6146a8403fb8482403c62f412d5`, tree `ba9484ccb33aa56e90ff2f447f15ca9b9da97639` |
| B12X | `b1d541f9e71a35f030d45fae437630fff7507c2a`, tree `c69cdec1c59a08e8e0e549f930fa8abcfb5134ae` |
| SparkCache exact prefix restore, GPU-free row-descriptor coalescing, tail-only copy-on-write publication, canonical CUDA configuration, and bounded page-delta reads | `65b6642df1afc64366430d3aef9aca01f5c5e1c3`, tree `41ad0a119ba109fd28900a2dcc9f9b4d8c293809`, clean source SHA-256 `a2add45a9f97446f6c2a843355161da9a5499ff7501b4750d2163591785d7345` |
| Recurrent replay-boundary producer | Patch SHA-256 `5a6561a5bbab990dcd03bfd6a485ea26c3b5a578c2fd61b76305767b16dbfba0`; produces the four postimages accepted by SparkCache lease contract SHA-256 `8adbdfa3fd4b06b213c3aab45255a0b039f1c9940a4b1fad0efd004d263227c9` |
| DFlash draft-loader separation | Patch SHA-256 `39b567013ee7aed79f63200ed460129587933dc77fb430decdf19f78178de279`, postimage SHA-256 `98acbae2b3bb4482d83f9637c163ce7c92707ccdf6561b7e431f23337f151cf4` |
| Unused DeepEP removal | Distribution `deep_ep==2.0.0+local`, removal receipt SHA-256 `65514f44829e7d176b0b2cacc9559ed22724e525b7041a8bcd4d2e02d1f372e3` |
| Target | `local-inference-lab/GLM-5.3-Flash-NVFP4@520de24eabf507659eaef7c70f14fd584527facc` |
| External draft | `incoai/GLM-5.3-Flash-DFlash2@dc77ff1c99eeb2df044ee3d4f0094eb033fee410`, BF16 weights SHA-256 `b33c03475ba7322cf398828f2d8d1be376df30dc05c6b40c28c8ea8da23e410b` |

The serving contract uses seven speculative tokens, draft TP4, target FP8 KV,
32 sequences, and 256-token vLLM blocks. The executable fastsafetensors
profile selects `snapshot-v1`, which stores full opaque pages in authenticated
macro objects. It uses the canonical CUDA restore keys, one load thread, and
one pending restore. The all-safetensors tail-cow profile is isolated and
research-only.

Do not derive concurrency by dividing the reported 916,676-token capacity by
prompt length. A no-cache C6 × 128K observation admitted one request at a time
and serialized completions. C2 × 128K is the only observed safe candidate
pending live CUDA qualification. C8 × 64K and C16 × 32K are planned and
unqualified. C16 × 128K is unsupported unless GPU-resident trunk pages are
shared or KV capacity increases. SparkCache base-read coalescing alone does
not change allocator admission. See
[resident-token concurrency](GLM53_FLASH_QUICKSTARTS.md#plan-resident-token-concurrency).

## Build the image

Run on Linux ARM64 from a clean checkout containing this guide:

```bash
IMAGE='sparkring-glm53-sparkcache:dflash7-vllm-python-0b67266-native-da4d7be-b12x-b1d541f-arm64' \
BUILD_RECEIPT="$PWD/glm53-dflash7-python-overlay-image-receipt.json" \
bash runtime/glm53-flash-dflash7-python-overlay/build-image.sh
```

The builder verifies the public da4 image, the 31-file Python overlay, retained
native ELFs and dispatch operators, B12X, SparkCache clean source, the CUDA
placement library, four SparkCache vLLM patches, the recurrent-boundary
producer patch, and the twelve-file lease contract. The resulting profile also
requires the producer patch label before starting any container.
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
| `glm53-flash-dflash7-python-overlay-fastsafetensors-sparkcache-tp4-dcp1.example.json` | Its snapshot-v1 settings are **qualified** only for exact image `sha256:35b58a7bf414059c65b8f74e4e4b17ee6a81b7008e1bffbc9bd298b5e08c739e` and the bounded full-snapshot case above; rebuilds are not qualified | Uses global fastsafetensors with queue size one for the target and `draft_load_config={"load_format":"safetensors"}` for DFlash. |

The image applies an exact-input vLLM patch that passes
`SpeculativeConfig.draft_load_config` to the DFlash model loader. The image
receipt verifies patch SHA-256
`39b567013ee7aed79f63200ed460129587933dc77fb430decdf19f78178de279` and
postimage SHA-256
`98acbae2b3bb4482d83f9637c163ce7c92707ccdf6561b7e431f23337f151cf4`.
The all-safetensors tail-cow profile remains research-only and unqualified.
The fastsafetensors snapshot result belongs only to the image ID and case
named above; it does not transfer to a rebuild.

SparkCache commit `65b6642df1afc64366430d3aef9aca01f5c5e1c3`
accepts the canonical CUDA keys used by both profiles. It consumes the
hash-proven recurrent hand-off emitted by this exact vLLM overlay and cancels
publication when the metadata is absent, incomplete, or contradictory.
No legacy-key rewrite is part of this path.

## Resolve the profile and inspect the plan

Copy `scripts/config/glm53-flash-tp4-site.example.yaml`
outside version control and replace every address, interface, SSH target,
device, host path, and image identity. Select one profile template:

```bash
receipt="$PWD/glm53-dflash7-python-overlay-image-receipt.json"
image='sparkring-glm53-sparkcache:dflash7-vllm-python-0b67266-native-da4d7be-b12x-b1d541f-arm64'
profile_template='scripts/config/glm53-flash-dflash7-python-overlay-fastsafetensors-sparkcache-tp4-dcp1.example.json'

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

`plan` is offline. Inspect every rank action before a command that changes
containers.

## Start and observe the four-rank service

Run the strict placeholder check before starting containers:

```bash
python scripts/preflight.py \
  --site /path/to/glm53-dflash7-site.yaml \
  --strict-placeholders \
  --json /path/to/glm53-dflash7-preflight.json
```

Starting the profile replaces the named four-rank deployment:

```bash
python scripts/sparkring_generic_launcher.py \
  --site /path/to/glm53-dflash7-site.yaml \
  --profile /path/to/glm53-dflash7-profile.json \
  --execute \
  --confirmation START_GLM53_FLASH_DFLASH7_PYTHON_OVERLAY_FASTSAFETENSORS_TP4 \
  start
```

Tail rank zero from another terminal:

```bash
ssh operator@rank0.example.net \
  'docker logs --follow --tail 120 glm53-flash-dflash7-python-overlay-fastsafetensors-sparkcache-tp4-r0 2>&1'
```

Wait for health and send a bounded generation smoke request with the model name from
the selected profile:

```bash
api_endpoint='http://rank0.example.net:8015'
served_model='glm-5.3-flash-nvfp4-dflash7-python-overlay-0b67266-on-da4d7be-b12x-b1d541f-tp4'
until curl --fail --silent "${api_endpoint}/health" >/dev/null; do sleep 5; done
curl --fail --silent --show-error "${api_endpoint}/v1/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${served_model}\",\"prompt\":\"The capital of France is\",\"max_tokens\":16,\"temperature\":0}"
```

The configured `spark_cache_clear_once` value is a durable operation token,
not a permanent clear-on-start switch. SparkCache removes its owned cache
content, writes the token's completion marker only after successful removal,
and treats later starts with the same token as no-ops. Change the token only
when another intentional cache reset is required.

`--prefill-schedule-interval` is not part of the selected DFlash7 profiles.
Test interval `8` in a separate research-only profile so its mixed
prefill/decode tradeoff is measured independently.

## Cache namespace impact

The external DFlash weights SHA-256 is stored as
`spark_cache_draft_checkpoint_sha256`. It cannot share entries with embedded
MTP profiles. The executable fastsafetensors profile uses `snapshot-v1`; the
all-safetensors research profile uses `tail-cow-v1`. The formats, cache roots,
and one-shot clear tokens are distinct, so their entries cannot alias even
though loader choice does not change target or draft model state.

The pinned SparkCache source combines canonical CUDA configuration names,
authenticated restore, full-page snapshots, and research-only page-delta
paths. Cache identities, digest salts, 256-token chunk geometry, stored wire
bytes, and the CUDA placement ABI are unchanged. Compatible snapshot entries
remain eligible only under the snapshot-v1 profile, and compatible
`page-tail-cow-v1` entries remain eligible only under the research profile.
The vLLM lease-contract bytes do change to accept the recurrent-boundary
postimages. Missing or malformed boundary evidence is a cache miss and
recomputation, never an unverified publication.
