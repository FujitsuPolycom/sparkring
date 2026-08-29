# Serve source-built GLM-5.3 e10536a with SparkCache on four DGX Sparks

Status: **implemented, not qualified**. This guide builds an exact vLLM
`e10536aadf02a18fccddda7ec939c33147e8b0b3` runtime and a SparkCache overlay.
It does not replace or extend the qualification of the older public 8K OCI
image.

The first cutover profile retains external BF16 DFlash2 at depth five. Two
separate profiles exercise embedded static MTP5 and opt-in adaptive MTP5
with initial depth three and a 32-step acceptance window. Do not combine the
runtime upgrade and speculator change in the first live cutover.

All profiles use TP4/DCP1, 20 GiB FP8 KV per rank, a 524,288-token request
limit, 32 sequences, native direct restore, two restore lanes, eight native
I/O workers, two 256 MiB arenas, shared GPU-prefix leases, and one-shot cache
clearing.

## Build immutable inputs

Use a Linux ARM64 build host with Docker BuildKit and at least 250 GiB free.
Building on a serving rank competes for CPU, memory, storage, and network
bandwidth even though it does not stop the running service.

```bash
git clone https://github.com/FujitsuPolycom/sparkring.git sparkring
git -C sparkring checkout --detach <revision-containing-this-guide>
git clone https://github.com/FujitsuPolycom/sparkcache.git sparkcache
git -C sparkcache checkout --detach 2993e18355a505148bbda1cbc81c8c556826a4c2

IMAGE='sparkring-glm53-runtime:e10536a-source-arm64' \
BUILD_RECEIPT="$PWD/glm53-e10536a-runtime-receipt.json" \
bash sparkring/runtime/glm53-flash-e10536a/build-image.sh

runtime_image='sparkring-glm53-runtime:e10536a-source-arm64'
runtime_image_id="$(docker image inspect --format '{{.Id}}' "${runtime_image}")"
sparkcache_source_sha256='1bac4577b6a83c7d494a83f68f1262127a716e7ef4aab01aa58f128e69bf3e30'
python sparkcache/deploy/glm53_flash/build_image.py \
  --repository "$PWD/sparkcache" \
  --containerfile deploy/glm53_flash/Containerfile.e10536a \
  --base-image "${runtime_image}" \
  --base-image-id "${runtime_image_id}" \
  --source-sha256 "${sparkcache_source_sha256}" \
  --sparkcache-revision 2993e18355a505148bbda1cbc81c8c556826a4c2 \
  --output-image sparkring-glm53-sparkcache:e10536a-source-arm64
```

Record the exact output identities:

```bash
sparkcache_image='sparkring-glm53-sparkcache:e10536a-source-arm64'
sparkcache_image_id="$(docker image inspect --format '{{.Id}}' "${sparkcache_image}")"
native_sha256="$(docker run --rm --entrypoint sha256sum "${sparkcache_image}" \
  /opt/sparkcache-src/sparkcache/native/build-cuda/libspark_cache_placement.so \
  | cut -d ' ' -f1)"
test "${#native_sha256}" -eq 64
```

## Resolve a serving profile

Copy the e10536a site template and select exactly one profile:

| Candidate | Profile template | Purpose |
|---|---|---|
| Runtime-only cutover | `glm53-flash-e10536a-dflash2-bf16-sparkcache-tp4-dcp1.example.json` | e10536a with external DFlash2 depth five |
| Embedded MTP | `glm53-flash-e10536a-mtp5-sparkcache-tp4-dcp1.example.json` | static internal MTP5 |
| Adaptive embedded MTP | `glm53-flash-e10536a-mtp5-adaptive-sparkcache-tp4-dcp1.example.json` | MTP5 maximum, initial depth three, 32-step window |

The commands below prepare the runtime-only cutover. Change only
`profile_template` for a later MTP experiment.

```bash
profile_template='sparkring/scripts/config/glm53-flash-e10536a-dflash2-bf16-sparkcache-tp4-dcp1.example.json'
site_template='sparkring/scripts/config/glm53-flash-e10536a-tp4-site.example.yaml'
python sparkring/scripts/prepare_glm53_e105_profile.py \
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

Edit only host-specific values in `site.yaml` and the target, DFlash, and
rank-local cache paths in `profile.json`. The MTP profiles have no external
draft-model mount. Every rank must use a different local cache directory.

The one-shot clear token is stored after successful removal. Restarting an
unchanged profile does not clear again. Each speculator profile has a distinct
token and cache identity, so a deliberate profile switch clears once and then
uses its own namespace.

## Validate and launch

```bash
python sparkring/scripts/preflight.py \
  --site site.yaml --strict-placeholders --json preflight.json
python sparkring/scripts/sparkring_generic_launcher.py \
  --site site.yaml --profile profile.json plan > start-plan.json
python sparkring/scripts/sparkring_generic_launcher.py \
  --site site.yaml --profile profile.json \
  --execute --confirmation START_GLM53_FLASH_DFLASH2_TP4 start
```

Use the confirmation value from the selected profile. Tail rank zero:

```bash
ssh operator@rank0.example.net \
  'docker logs --follow --tail 120 glm53-flash-e10536a-dflash2-bf16-sparkcache-tp4-r0 2>&1'
```

Wait for health, then run the exact semantic canary:

```bash
api_endpoint='http://rank0.example.net:8015'
served_model='glm-5.3-flash-nvfp4-dflash5-e10536a-tp4'
until curl --fail --silent "${api_endpoint}/health" >/dev/null; do sleep 5; done
python sparkcache/deploy/glm53_flash/qualification_request.py \
  --endpoint "${api_endpoint}" --model "${served_model}" \
  --kind semantic --output semantic.json
```

The first cutover requires health on all four ranks, an exact semantic match,
the e10536a and SparkCache source labels, the ten-file lease contract, the
native-library digest, and no engine exit, OOM, NCCL error, or traceback. Run
persistent restore and C2/C8/C16 shared-prefix checks before calling the image
qualified.

## Cache namespace impact

The e10536a overlay does not change SparkCache wire fields, digest salts,
256-token geometry, or stored object formats. External DFlash, static MTP5,
and adaptive MTP5 deliberately use different draft identity digests. A
speculator-policy change therefore recomputes instead of reusing an entry
published under another policy.

Embedded MTP identities are SHA-256 over
`glm53-embedded-mtp-v1`, the target cache identity, maximum depth five, and
either policy string `static` or `adaptive:3:32`, separated by zero bytes.
The checked-in CPU contract recomputes both values instead of accepting
unexplained constants.
