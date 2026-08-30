# GLM-5.3 public-base Python overlay

Status: **implemented** for image construction and offline verification. The
image is unsupported for serving until an immutable digest passes four-rank
TP4/DCP1 model loading, semantic generation, SparkCache restore,
failure-recovery, and concurrency checks.

This builder retains the compiled vLLM extensions and wheel metadata from the
qualified public runtime
`ghcr.io/fujitsupolycom/sparkring-glm53-runtime@sha256:864adfe68f458223e186a19844ac80c7adc7365e5db1f25e109b85fc19850dcd`.
It replaces the 31 Python files changed between vLLM
`da4d7be6c97434f6942292ed8abbf4b32dc44355` and
`0b67266a0f37d6146a8403fb8482403c62f412d5`.

The image also installs B12X
`b1d541f9e71a35f030d45fae437630fff7507c2a`. That revision provides the
trusted-metadata and request-sized live-tensor KDA interface required by the
vLLM Python sources. B12X
`2fcf23a0ce269be27b2e03fece73d46e90e6aeea` does not provide that interface
and is rejected.

The builder records every retained vLLM ELF file, the native dispatch operator
set, generated version metadata, and wheel metadata before replacing Python
files. Construction fails if any retained artifact changes. The output labels
identify the retained native commit and overlaid Python commit separately; the
image is not described as a source-built vLLM 0b67266 wheel.

The SparkCache source that routes reconstructed opaque pages through the SM121
placement library is commit
`5d571018de5b63a9a90e5c11e6d6e86bbff4a957`, Git tree
`e864ed9ad64f771188fdb59aa9738e348134d636`. The builder verifies clean
deployable-source SHA-256
`f7c0565521fddeff7085e4cc08043cb8d1e2bde33abc67f83b8608a162d05b88`
before generating the SparkCache CUDA placement library. It applies the VMM exemption,
load-failure recovery, shared-prefix retention, and follower-attachment
patches in order, then runs the eleven-file lease-contract verifier. The test
profile selects `spark_cache_publication_schema=tail-cow-v1`, which maps opaque
GLM page storage to the distinct `page-tail-cow-v1` cache namespace.

## Build

Run on a Linux ARM64 host with Docker BuildKit:

```bash
cd /path/to/sparkring
IMAGE='sparkring-glm53-sparkcache:vllm-python-0b67266-native-da4d7be-b12x-b1d541f-arm64' \
BUILD_RECEIPT="$PWD/glm53-public-python-overlay-image-receipt.json" \
bash runtime/glm53-flash-adaptive-mtp-python-overlay/build-image.sh
```

The script pulls and verifies the immutable public base, fetches exact source
commits into a temporary build context, builds a pure B12X wheel and the
SparkCache CUDA placement library, creates the composed image, and writes a local
receipt. It does not push an image or contact serving hosts.

## Resolve and inspect the four-rank plan

Copy the sanitized site template outside version control and replace its
addresses, SSH targets, interfaces, devices, paths, and image identity. Resolve
the runtime profile from values in the image receipt:

```bash
receipt="$PWD/glm53-public-python-overlay-image-receipt.json"
image='sparkring-glm53-sparkcache:vllm-python-0b67266-native-da4d7be-b12x-b1d541f-arm64'

python scripts/prepare_glm53_public_python_overlay_profile.py \
  --profile-template scripts/config/glm53-flash-public-python-overlay-mtp5-adaptive-fastsafetensors-sparkcache-tp4-dcp1.example.json \
  --site-template /path/to/resolved-glm53-site.yaml \
  --image "$image" \
  --image-id "$(jq -r .image_id "$receipt")" \
  --cuda-placement-library-sha256 "$(jq -r .artifacts.sparkcache_cuda_placement_sha256 "$receipt")" \
  --native-elf-manifest-sha256 "$(jq -r .runtime_contract.native_elf_manifest_sha256 "$receipt")" \
  --native-dispatch-manifest-sha256 "$(jq -r .runtime_contract.native_dispatch_manifest_sha256 "$receipt")" \
  --source-receipt-sha256 "$(jq -r .artifacts.source_receipt_sha256 "$receipt")" \
  --profile-output /path/to/glm53-public-python-overlay-profile.json \
  --site-output /path/to/glm53-public-python-overlay-site.yaml

python scripts/sparkring_generic_launcher.py \
  --site /path/to/glm53-public-python-overlay-site.yaml \
  --profile /path/to/glm53-public-python-overlay-profile.json \
  plan
```

`plan` is offline and makes no SSH connection. Inspect all four rank actions
before any lifecycle command. The profile uses separate container, served
model, JIT cache, SparkCache root, MTP identity, and one-shot clear token names
so it cannot silently reuse the source-built runtime's test state.
