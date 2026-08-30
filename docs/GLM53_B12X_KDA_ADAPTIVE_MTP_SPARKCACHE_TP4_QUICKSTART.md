# Serve GLM-5.3 with adaptive MTP, live-tensor B12X KDA, and SparkCache

Status: **implemented, not qualified**. This guide retains vLLM native
extensions from `da4d7be6c97434f6942292ed8abbf4b32dc44355`, overlays Python
source `0b67266a0f37d6146a8403fb8482403c62f412d5`, and installs SparkCache
commit `bf7174e341e032d9b5cc970cca3d6c2985d364fc`, Git tree
`21a0598e0f7f05739d2e27478c484a345d565556`, for four DGX Spark systems at
TP4/DCP1. The adaptive-MTP composition has GPU-free contract coverage but no
four-rank persistent-restore or performance qualification.

The serving profile uses embedded MTP with maximum depth five, initial depth
three, and a 32-step acceptance window. Fastsafetensors uses queue size one.
TP4 makes the pinned vLLM loader select `nogds=True`, so model loading uses
pipelined host I/O without GPU Direct Storage.

The profile reserves 20 GiB of FP8 KV per rank and enables SparkCache CUDA
restore, tail-only copy-on-write publication, shared-segment restore, and
bounded shared GPU prefix leases. The vLLM overlay emits only hash-proven
recurrent replay boundaries; SparkCache keeps publication pending while a
request has no hand-off and rejects incomplete, contradictory, or changed
evidence. Image construction and distribution
do not require stopping an existing service. Do not run the launch command
until all four ranks have the same verified image ID.

## Build the exact Python-overlay image

Use Linux ARM64 with Docker BuildKit and enough space for the exact source
trees, B12X wheel, SparkCache CUDA placement library, and derived image:

```bash
git clone https://github.com/FujitsuPolycom/sparkring.git sparkring
git -C sparkring checkout --detach <revision-containing-this-guide>
cd sparkring
IMAGE='sparkring-glm53-sparkcache:vllm-python-0b67266-native-da4d7be-b12x-b1d541f-arm64' \
BUILD_RECEIPT="$PWD/glm53-public-python-overlay-image-receipt.json" \
bash runtime/glm53-flash-adaptive-mtp-python-overlay/build-image.sh
```

The builder verifies the 31-file Python overlay, retained native ELF and
dispatch manifests, B12X `b1d541f`, SparkCache clean source SHA-256
`155a06101524d4c2d2f55dbbd01576e35d5c729888e216fd2f3963e275949ba0`,
four SparkCache patches, recurrent-boundary producer patch SHA-256
`5a6561a5bbab990dcd03bfd6a485ea26c3b5a578c2fd61b76305767b16dbfba0`,
and lease contract SHA-256
`f36ed14eaf1f97a5dffa94bda8151b1e0fa182afc0d121b757b70bebc6a43811`.
The receipt binds these sources to the local image ID. The builder does not
push the image or contact serving hosts.

`runtime/glm53-flash-b12x-kda-adaptive-mtp/` remains an exact full-source
builder for its older SparkCache contract. It does not apply the recurrent
producer after SparkCache's vLLM patches, so it is unsupported with SparkCache
`bf7174e` and must not be substituted in this guide.

## Resolve the TP4 profile

```bash
receipt="$PWD/glm53-public-python-overlay-image-receipt.json"
image='sparkring-glm53-sparkcache:vllm-python-0b67266-native-da4d7be-b12x-b1d541f-arm64'
profile_template='scripts/config/glm53-flash-public-python-overlay-mtp5-adaptive-fastsafetensors-sparkcache-tp4-dcp1.example.json'
site_template='scripts/config/glm53-flash-b12x-kda-adaptive-mtp-tp4-site.example.yaml'
python scripts/prepare_glm53_public_python_overlay_profile.py \
  --profile-template "${profile_template}" \
  --site-template "${site_template}" \
  --image "${image}" \
  --image-id "$(jq -r .image_id "${receipt}")" \
  --cuda-placement-library-sha256 "$(jq -r .artifacts.sparkcache_cuda_placement_sha256 "${receipt}")" \
  --native-elf-manifest-sha256 "$(jq -r .runtime_contract.native_elf_manifest_sha256 "${receipt}")" \
  --native-dispatch-manifest-sha256 "$(jq -r .runtime_contract.native_dispatch_manifest_sha256 "${receipt}")" \
  --source-receipt-sha256 "$(jq -r .artifacts.source_receipt_sha256 "${receipt}")" \
  --profile-output profile.json \
  --site-output site.yaml
```

Replace the documentation-only addresses, interfaces, RDMA devices, model
path, and rank-local cache roots. Every rank must use a different local
SparkCache directory. Do not change MTP depths, the observation window,
loader queue, source identities, or attestation command.

The one-shot clear token is recorded only after a successful SparkCache-owned
cache removal. Restarting this unchanged profile does not clear again. The
resolved profile requires the recurrent-boundary patch label before any
container starts.

`--prefill-schedule-interval` is not part of this implemented profile. Test
interval `8` as a separate research-only profile so its mixed prefill/decode
tradeoff cannot be confused with adaptive-MTP or SparkCache results.

## Verify and launch

```bash
python scripts/preflight.py \
  --site site.yaml --strict-placeholders --json preflight.json
python scripts/sparkring_generic_launcher.py \
  --site site.yaml --profile profile.json plan > start-plan.json
python scripts/sparkring_generic_launcher.py \
  --site site.yaml --profile profile.json \
  --execute \
  --confirmation START_GLM53_FLASH_PUBLIC_PYTHON_OVERLAY_MTP5_ADAPTIVE_FASTSAFETENSORS_TP4 \
  start
```

The final command changes the four-rank serving deployment. Tail rank zero:

```bash
ssh operator@rank0.example.net \
  'docker logs --follow --tail 120 glm53-flash-public-python-overlay-mtp5-adaptive-fastsafetensors-sparkcache-tp4-r0 2>&1'
```

Wait for health, then run the exact semantic request:

```bash
api_endpoint='http://rank0.example.net:8015'
served_model='glm-5.3-flash-nvfp4-python-overlay-0b67266-on-da4d7be-b12x-b1d541f-mtp5-adaptive-tp4'
until curl --fail --silent "${api_endpoint}/health" >/dev/null; do sleep 5; done
curl --fail --silent --show-error "${api_endpoint}/v1/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${served_model}\",\"prompt\":\"The capital of France is\",\"max_tokens\":16,\"temperature\":0}" \
  > semantic.json
```

Construction support does not prove four-rank serving. Qualification requires
health, exact visible output, adaptive-MTP activity, successful model loading,
persistent restore, shared-prefix C2/C8/C16 checks, and no engine exit, OOM,
NCCL error, or traceback.

## Cache namespace impact

The overlay does not change SparkCache wire fields, digest salts, 256-token
geometry, or stored object schemas. Its embedded-MTP digest is SHA-256 over
`glm53-embedded-mtp-composed-runtime-v1`, the target identity, the overlaid
vLLM Python commit, retained native vLLM commit, B12X commit, maximum depth
five, and `adaptive:3:32`, separated by zero bytes.

Including both retained-native and overlaid-Python revisions gives this runtime
a distinct draft-state cache identity from the e105 adaptive-MTP and older
full-source profiles. Stored entries therefore recompute instead of crossing a
runtime boundary without byte-equivalence evidence. SparkCache `49c517e` does
not change wire fields, digest salts, or 256-token geometry; the new lease
contract accepts the producer postimages needed to prove recurrent publication.
