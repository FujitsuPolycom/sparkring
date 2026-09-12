# Compare GLM-5.3 speculation policies with SparkCache on four DGX Sparks

Status: **Development**; four-rank serving is not validated. This guide builds vLLM
`e10536aadf02a18fccddda7ec939c33147e8b0b3` runtime and a SparkCache overlay.
The [DFlash2 validation record](../performance/records/glm53-flash/sparkcache-dflash2-bf16-tp4-20260828.md)
uses vLLM `da4d7be6c97434f6942292ed8abbf4b32dc44355`; its result does not
validate the runtime built here.

The external BF16 DFlash2 profile uses depth five. It is not a source-only
comparison with the linked validated profile, which uses depth seven and a
different KV memory budget. Separate profiles exercise embedded static MTP5 and
acceptance-based adaptive MTP5 with initial depth three and a 32-step
observation window. Qualify one profile at a time so each result names one
runtime and speculation policy.

An Experimental adaptive-MTP5 profile selects the fastsafetensors parallel
loader and sets `VLLM_FASTSAFETENSORS_QUEUE_SIZE=1`. The TP4 process group
forces `nogds=True` in vLLM revision `e10536a`, so this profile evaluates
pipelined shard loading without GPU Direct Storage. The queue can retain one
additional shard-sized device buffer during model loading.

The profile templates below select TP4/DCP1, 20 GiB FP8 KV per rank, a 524,288-token request
limit, 32 sequences, native direct restore, two restore lanes, eight native
I/O workers, two 256 MiB arenas, shared GPU-prefix leases, and one-shot cache
clearing.

## Build immutable inputs

Use a Linux ARM64 build host with Docker BuildKit and at least 250 GiB free.
The operator host needs Python 3.11 or newer with PyYAML. Use a reviewed
40-character SparkRing commit in place of the checkout placeholder below.
Building on a serving rank competes for CPU, memory, storage, and network
bandwidth even though it does not stop the running service.

```bash
git clone https://github.com/FujitsuPolycom/sparkring.git sparkring
git -C sparkring checkout --detach '<revision-containing-this-guide>'
git clone https://github.com/FujitsuPolycom/sparkcache.git sparkcache
git -C sparkcache checkout --detach eb3690c1aac2b9e86be8d513799dbb64afa53f25

IMAGE='sparkring-glm53-runtime:e10536a-source-arm64' \
BUILD_RECEIPT="$PWD/glm53-e10536a-runtime-receipt.json" \
bash sparkring/runtime/glm53-flash-e10536a/build-image.sh

runtime_image='sparkring-glm53-runtime:e10536a-source-arm64'
runtime_image_id="$(docker image inspect --format '{{.Id}}' "${runtime_image}")"
sparkcache_source_sha256='34108fb22ba95b457bf4b357407b176dcbf3a6db6227227b21ecee045502a16f'
python sparkcache/deploy/glm53_flash/build_image.py \
  --repository "$PWD/sparkcache" \
  --containerfile deploy/glm53_flash/Containerfile.e10536a \
  --base-image "${runtime_image}" \
  --base-image-id "${runtime_image_id}" \
  --source-sha256 "${sparkcache_source_sha256}" \
  --sparkcache-revision eb3690c1aac2b9e86be8d513799dbb64afa53f25 \
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

The source-built runtime at vLLM revision `e10536a` supports the following
speculation and loader comparisons. Copy its site template and select exactly
one profile:

| Status and runtime role | Profile template | Purpose |
|---|---|---|
| Development, external-speculator control | `glm53-flash-e10536a-dflash2-bf16-sparkcache-tp4-dcp1.example.json` | e10536a with external DFlash2 depth five |
| Development, embedded MTP | `glm53-flash-e10536a-mtp5-sparkcache-tp4-dcp1.example.json` | static internal MTP5 |
| Development, adaptive embedded MTP | `glm53-flash-e10536a-mtp5-adaptive-sparkcache-tp4-dcp1.example.json` | MTP5 maximum, initial depth three, 32-step window |
| Experimental, adaptive MTP loader comparison | `glm53-flash-e10536a-mtp5-adaptive-fastsafetensors-sparkcache-tp4-dcp1.example.json` | adaptive MTP5 with the fastsafetensors parallel loader, queue size one, and no GDS under TP4 |

The commands below prepare the external-speculator control profile. Select a
different table entry only when the named speculation or loader behavior is
the intended test variable.

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

Edit host-specific values in `site.yaml` and the target-model and rank-local
cache paths in `profile.json`. For the DFlash2 profile, also set its external
draft-model path; the MTP profiles have no external draft-model mount.
Each rank needs its own local cache storage; identical path strings on separate
hosts are acceptable. Do not use a shared cache directory.

Distribute the built SparkCache image to all four hosts before launch. Use the
[image archive fanout procedure](DIRECT_FABRIC_IMAGE_ARCHIVE_FANOUT.md) with
the resolved site file, or copy a `docker image save` archive and load it on
each host. Require the same exact image ID and selected tag on every rank;
the profile resolver does not distribute images.

The one-shot clear token is stored after successful removal. Restarting an
unchanged profile does not clear again. Every profile has a distinct clear
token. External DFlash2, static MTP5, and adaptive MTP5 have distinct cache
identities. The two adaptive-MTP5 profiles share a cache identity. Their distinct
clear tokens mean switching profiles can clear the selected cache root on first
use; shared identity does not prevent that clear operation.

## Validate and launch

```bash
python sparkring/scripts/preflight.py \
  --site site.yaml --strict-placeholders --json preflight.json
python sparkring/scripts/sparkring_generic_launcher.py \
  --site site.yaml --profile profile.json plan > start-plan.json
confirmation=$(python -c 'import json; print(json.load(open("profile.json", encoding="utf-8"))["confirmation"])')
python sparkring/scripts/sparkring_generic_launcher.py \
  --site site.yaml --profile profile.json \
  --execute --confirmation "${confirmation}" start
```

Tail the rank-zero container named by the selected profile. Replace the SSH
target and API endpoint below with rank zero's management address and the
configured API port:

```bash
container_name=$(python -c 'import json; print(json.load(open("profile.json", encoding="utf-8"))["container_name"] + "-r0")')
ssh operator@rank0.example.net \
  "docker logs --follow --tail 120 '${container_name}' 2>&1"
```

Wait for health, then run the exact semantic canary:

```bash
api_endpoint='http://rank0.example.net:8015'
served_model=$(python -c 'import json; d=json.load(open("profile.json", encoding="utf-8")); a=d["extra_vllm_args"]; print(a[a.index("--served-model-name") + 1])')
startup_timeout=$(python -c 'import json; print(json.load(open("profile.json", encoding="utf-8"))["startup_timeout_seconds"])')
timeout "$startup_timeout" bash -c 'until curl --fail --silent --show-error --max-time 10 "$1/health" >/dev/null; do sleep 5; done' _ "$api_endpoint" || {
  echo "Readiness failed or timed out; inspect all rank logs before retrying" >&2; exit 1;
}
python sparkcache/deploy/glm53_flash/qualification_request.py \
  --endpoint "${api_endpoint}" --model "${served_model}" \
  --kind semantic --output semantic.json
```

Qualification requires health on all four ranks, an exact semantic match, the
e10536a and SparkCache source labels, the ten-file lease contract, the
native-library digest, and no engine exit, OOM, NCCL error, or traceback. Run
persistent restore and shared-prefix checks with 2, 8 and 16 concurrent requests
before labeling the image Validated. The fastsafetensors profile also requires a model-load result that
records elapsed startup time and peak device memory; it remains Experimental
without that evidence.

## Cache namespace impact

The e10536a overlay does not change SparkCache wire fields, digest salts,
256-token geometry, or stored object formats. External DFlash, static MTP5,
and adaptive MTP5 deliberately use different draft identity digests. A
speculator-policy change therefore recomputes instead of reusing an entry
published under another policy. Standard safetensors and fastsafetensors
loading use the same adaptive-MTP5 identity on the assumption that they
materialize equivalent pinned tensors without changing the KV representation.
The CPU identity checks do not establish live cross-loader cache equivalence;
include a cross-loader restore check before relying on that reuse.

Embedded MTP identities are SHA-256 over
`glm53-embedded-mtp-v1`, the target cache identity, maximum depth five, and
either policy string `static` or `adaptive:3:32`, separated by zero bytes.
The [CPU contract](../scripts/test_prepare_glm53_e105_profile.py) recomputes both values instead of accepting
unexplained constants.

[Profile validation: performance, accuracy, and restart checks](PROFILE_VALIDATION.md).
