# GLM-5.3 DFlash7 SparkCache PR42 image

Status: **implemented construction; research-only serving evidence**.

This isolated image contract retains the vLLM, B12X, DFlash, recurrent
publication, lease, and CUDA placement identities from SparkRing pull request
#146 while installing SparkCache commit
`a1511d26a1fe2b17b24561bc52e376bf7f54b06a`. Its build receipt records the
`sparkcache-page-base-restore-flight/v1` research contract. The codeword-oracle
harness under `performance/harnesses/glm53_page_base_flight/` records mechanism
and semantic evidence without emitting a qualification verdict.

Build on Linux ARM64:

```bash
IMAGE='sparkring-glm53-sparkcache:dflash7-pr42-page-base-flight-singletonfix-arm64' \
BUILD_RECEIPT="$PWD/glm53-pr42-page-base-flight-image-receipt.json" \
bash runtime/glm53-flash-dflash7-pr42-page-base-flight/build-image.sh
```

The builder contacts no serving host. A verified local image and receipt are
construction evidence only.

The offline-verified ARM64 artifact is
`sparkring-glm53-sparkcache:dflash7-pr42-page-base-flight-singletonfix-arm64`
with image ID
`sha256:35b58a7bf414059c65b8f74e4e4b17ee6a81b7008e1bffbc9bd298b5e08c739e`.
Its build receipt SHA-256 is
`ec51c5b99227fe14709977df026e25e3e60f220b81ae252155d048556e8ea90a`.
This construction evidence does not establish four-rank serving behavior.

Live research established that a 16-request, 131,072-token cohort performed
one base read and avoided 15 reads on every rank, but exceeded the 20 GiB GLM
hybrid-cache residency and failed its codeword checks. A resident-safe
two-request page-delta run also produced one wrong admitted response with both
Python and SparkCache CUDA placement, while recomputation returned the correct
response. Reconstructed page-delta admission is unsupported by this evidence.

The verified fallback is a 131,072-token flat snapshot stored as 13 macro
objects. Its restore completed in 1.55–1.70 seconds and returned the exact
`red` codeword. These observations are **research-only** and do not qualify the
image or runtime.
