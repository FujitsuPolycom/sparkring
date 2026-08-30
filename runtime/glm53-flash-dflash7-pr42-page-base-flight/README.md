# GLM-5.3 DFlash7 SparkCache PR42 image

Status: **implemented, not qualified**.

This isolated image contract retains the vLLM, B12X, DFlash, recurrent
publication, lease, and CUDA placement identities from SparkRing pull request
#146 while installing SparkCache commit
`5a6613e473a713695948e69e0027fd67530028f8`. Its build receipt records the
`sparkcache-page-base-restore-flight/v1` qualification contract. Live serving
claims require the deterministic harness under
`performance/harnesses/glm53_page_base_flight/`.

Build on Linux ARM64:

```bash
IMAGE='sparkring-glm53-sparkcache:dflash7-pr42-page-base-flight-sourcebytesfix-arm64' \
BUILD_RECEIPT="$PWD/glm53-pr42-page-base-flight-image-receipt.json" \
bash runtime/glm53-flash-dflash7-pr42-page-base-flight/build-image.sh
```

The builder contacts no serving host. A verified local image and receipt are
construction evidence only; the runtime remains unqualified until the
four-rank procedure is separately authorized and completed.
