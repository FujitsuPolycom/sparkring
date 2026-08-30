# Serve GLM-5.3 with adaptive MTP, live-tensor B12X KDA, and SparkCache

Status: **implemented, not qualified**. This guide builds vLLM commit
`0b67266a0f37d6146a8403fb8482403c62f412d5` and the SparkCache overlay from
commit `5d571018de5b63a9a90e5c11e6d6e86bbff4a957`, Git tree
`e864ed9ad64f771188fdb59aa9738e348134d636`, for four DGX Spark systems at
TP4/DCP1.

The serving profile uses embedded MTP with maximum depth five, initial depth
three, and a 32-step acceptance window. Fastsafetensors uses queue size one.
TP4 makes the pinned vLLM loader select `nogds=True`, so model loading uses
pipelined host I/O without GPU Direct Storage.

The profile reserves 20 GiB of FP8 KV per rank and enables SparkCache native
restore, tail-only publication, shared restore trunks, and bounded shared GPU
prefix leases. Image construction and distribution do not require stopping an
existing service. Do not run the launch command until all four ranks have the
same verified image ID.

## Build the runtime and SparkCache overlay

Use Linux ARM64 with Docker BuildKit and at least 250 GiB of free local
storage. Clone both repositories beside each other:

```bash
git clone https://github.com/FujitsuPolycom/sparkring.git sparkring
git -C sparkring checkout --detach <revision-containing-this-guide>
git clone https://github.com/FujitsuPolycom/sparkcache.git sparkcache
git -C sparkcache checkout --detach 5d571018de5b63a9a90e5c11e6d6e86bbff4a957

IMAGE='sparkring-glm53-runtime:b12x-kda-adaptive-mtp-0b67266a-arm64' \
BUILD_RECEIPT="$PWD/glm53-b12x-kda-adaptive-mtp-runtime-receipt.json" \
bash sparkring/runtime/glm53-flash-b12x-kda-adaptive-mtp/build-image.sh

runtime_image='sparkring-glm53-runtime:b12x-kda-adaptive-mtp-0b67266a-arm64'
runtime_image_id="$(docker image inspect --format '{{.Id}}' "${runtime_image}")"
python sparkcache/deploy/glm53_flash/build_image.py \
  --repository "$PWD/sparkcache" \
  --containerfile deploy/glm53_flash/Containerfile.b12x-kda-adaptive-mtp \
  --base-image "${runtime_image}" \
  --base-image-id "${runtime_image_id}" \
  --source-sha256 f7c0565521fddeff7085e4cc08043cb8d1e2bde33abc67f83b8608a162d05b88 \
  --sparkcache-revision 5d571018de5b63a9a90e5c11e6d6e86bbff4a957 \
  --output-image sparkring-glm53-sparkcache:b12x-kda-adaptive-mtp-0b67266a-arm64
```

Record immutable local identities:

```bash
sparkcache_image='sparkring-glm53-sparkcache:b12x-kda-adaptive-mtp-0b67266a-arm64'
sparkcache_image_id="$(docker image inspect --format '{{.Id}}' "${sparkcache_image}")"
native_sha256="$(docker run --rm --entrypoint sha256sum "${sparkcache_image}" \
  /opt/sparkcache-src/sparkcache/native/build-cuda/libspark_cache_placement.so \
  | cut -d ' ' -f1)"
test "${#native_sha256}" -eq 64
```

The runtime builder verifies the complete first-parent vLLM history from
`da4d7be` through adaptive MTP and the three live-tensor B12X KDA commits. The
SparkCache build verifies LF Linux preimages, four exact patches, and eleven
postimage source files.

## Resolve the TP4 profile

```bash
profile_template='sparkring/scripts/config/glm53-flash-b12x-kda-mtp5-adaptive-fastsafetensors-sparkcache-tp4-dcp1.example.json'
site_template='sparkring/scripts/config/glm53-flash-b12x-kda-adaptive-mtp-tp4-site.example.yaml'
python sparkring/scripts/prepare_glm53_b12x_kda_adaptive_mtp_profile.py \
  --profile-template "${profile_template}" \
  --site-template "${site_template}" \
  --image "${sparkcache_image}" \
  --image-id "${sparkcache_image_id}" \
  --parent-image "${runtime_image}" \
  --parent-image-id "${runtime_image_id}" \
  --native-library-sha256 "${native_sha256}" \
  --profile-output profile.json \
  --site-output site.yaml
```

Replace the documentation-only addresses, interfaces, RDMA devices, model
path, and rank-local cache roots. Every rank must use a different local
SparkCache directory. Do not change MTP depths, the observation window,
loader queue, source identities, or attestation command.

The one-shot clear token is recorded only after a successful SparkCache-owned
cache removal. Restarting this unchanged profile does not clear again.

## Verify and launch

```bash
python sparkring/scripts/preflight.py \
  --site site.yaml --strict-placeholders --json preflight.json
python sparkring/scripts/sparkring_generic_launcher.py \
  --site site.yaml --profile profile.json plan > start-plan.json
python sparkring/scripts/sparkring_generic_launcher.py \
  --site site.yaml --profile profile.json \
  --execute \
  --confirmation START_GLM53_FLASH_MTP5_ADAPTIVE_FASTSAFETENSORS_TP4 \
  start
```

The final command changes the four-rank serving deployment. Tail rank zero:

```bash
ssh operator@rank0.example.net \
  'docker logs --follow --tail 120 glm53-flash-b12x-kda-mtp5-adaptive-fastsafetensors-sparkcache-tp4-r0 2>&1'
```

Wait for health, then run the exact semantic request:

```bash
api_endpoint='http://rank0.example.net:8015'
served_model='glm-5.3-flash-nvfp4-b12x-kda-mtp5-adaptive-fastsafetensors-0b67266a-tp4'
until curl --fail --silent "${api_endpoint}/health" >/dev/null; do sleep 5; done
python sparkcache/deploy/glm53_flash/qualification_request.py \
  --endpoint "${api_endpoint}" --model "${served_model}" \
  --kind semantic --output semantic.json
```

Construction support does not prove four-rank serving. Qualification requires
health, exact visible output, adaptive-MTP activity, successful model loading,
persistent restore, shared-prefix C2/C8/C16 checks, and no engine exit, OOM,
NCCL error, or traceback.

## Cache namespace impact

The overlay does not change SparkCache wire fields, digest salts, 256-token
geometry, or stored object schemas. Its embedded-MTP digest is SHA-256 over
`glm53-embedded-mtp-runtime-v1`, the target identity, the full vLLM commit,
maximum depth five, and `adaptive:3:32`, separated by zero bytes.

Including the vLLM revision gives this runtime a distinct draft-state cache
identity from the e105 adaptive-MTP profile. Stored entries therefore
recompute instead of crossing the KDA source boundary without byte-equivalence
evidence.
