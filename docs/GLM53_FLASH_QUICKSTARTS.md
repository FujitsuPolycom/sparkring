# Choose a GLM-5.3 Flash quickstart

SparkRing has one published BF16 DFlash2 composition and three source-built
runtime paths. Select a row by speculator and evidence status before copying a
profile. A result recorded for one image ID does not qualify another build.

| Runtime path | Status | Image availability | Procedure |
|---|---|---|---|
| BF16 DFlash2 with SparkCache | **qualified** for the bounded cases in its guide | Published immutable OCI digest | [SparkCache quickstart](GLM53_FLASH_DFLASH2_BF16_SPARKCACHE_TP4_QUICKSTART.md) |
| BF16 DFlash2 without an external KV cache | **qualified** for the bounded cases in its guide | Uses the same published image | [Cache-disabled quickstart](GLM53_FLASH_DFLASH2_BF16_TP4_QUICKSTART.md) |
| External DFlash7, fastsafetensors target | **qualified** only for local image ID `sha256:ed60be066d6d9eadea267bc4597a0687869f3ddb95a3e5c6f86649893a838eb8` and its recorded cases | Local image; no published OCI digest | [DFlash7 Python-overlay quickstart](GLM53_DFLASH7_PYTHON_OVERLAY_SPARKCACHE_TP4_QUICKSTART.md) |
| External DFlash7, all-safetensors target | **implemented**, not qualified | Build locally | [DFlash7 Python-overlay quickstart](GLM53_DFLASH7_PYTHON_OVERLAY_SPARKCACHE_TP4_QUICKSTART.md) |
| Adaptive embedded MTP with live-tensor B12X KDA | **implemented**, not qualified | Build locally | [Adaptive-MTP quickstart](GLM53_B12X_KDA_ADAPTIVE_MTP_SPARKCACHE_TP4_QUICKSTART.md) |
| Source-built vLLM `e10536a` profiles | **implemented**, not qualified | Build locally | [e10536a source-build quickstart](GLM53_E10536A_SPARKCACHE_TP4_QUICKSTART.md) |

All rows use the target checkpoint
`local-inference-lab/GLM-5.3-Flash-NVFP4@520de24eabf507659eaef7c70f14fd584527facc`.
The external DFlash checkpoints and embedded-MTP policies have distinct cache
identities. Do not point two speculator policies at the same writable cache
root.

## vLLM source relationship

The source-built `e10536a` runtime and the Python-overlay runtimes are separate
image constructions. Commit
`e10536aadf02a18fccddda7ec939c33147e8b0b3` is nine commits after
`da4d7be6c97434f6942292ed8abbf4b32dc44355`; commit
`0b67266a0f37d6146a8403fb8482403c62f412d5` is three commits after
`e10536a`.

The Python-overlay images retain compiled extensions, wheel metadata, version
metadata, and the console entry point from `da4d7be6`. They overlay exactly 31
production `vllm/**` Python files from `0b67266a`. The `da4d7be6..0b67266a`
production-source comparison contains no C, C++, CUDA, or build-source change,
so the overlay includes the Python developments through `e10536a` and the
three following commits without claiming a source-built `0b67266a` wheel.

## Put one exact image on every rank

Use one of these paths. Do not mix a local image ID with a registry digest in
one evidence record.

For the published BF16 DFlash2 image, copy the immutable reference from its
quickstart, inspect the offline plan, and then run the confirmed pull:

```bash
python scripts/pull_glm53_image_cluster.py \
  --site site.yaml \
  --image 'ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:cd4045bba2a0f3dc55361560f8c3a3f171939854db28d48dfdae58eed9c44943'

python scripts/pull_glm53_image_cluster.py \
  --site site.yaml \
  --image 'ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:cd4045bba2a0f3dc55361560f8c3a3f171939854db28d48dfdae58eed9c44943' \
  --execute --confirmation PULL_GLM53_IMAGE \
  --output cluster-image.json
```

For a locally built image, keep the builder receipt beside a checksum-bound
archive. Record the image ID before saving it:

```bash
image='REPLACE_WITH_LOCAL_IMAGE_TAG'
archive="$PWD/REPLACE_WITH_ARCHIVE_NAME.oci.tar"
image_id="$(docker image inspect --format '{{.Id}}' "${image}")"
docker image save "${image_id}" --output "${archive}"
sha256sum "${archive}" > "${archive}.sha256"
```

Use the [direct-fabric archive guide](DIRECT_FABRIC_IMAGE_ARCHIVE_FANOUT.md)
to plan, verify, distribute, and optionally import that exact archive. Its
command requires the archive SHA-256, byte count, expected local image ID, and
an explicit confirmation for host changes. Archive parity proves image
placement only; it does not qualify serving.

`runtime/glm53-flash/publish_image.py` publishes only the source-built parent
runtime described by `runtime/glm53-flash/PUBLISHING.md`. It requires that
builder's receipt schema, an SPDX JSON SBOM, and a destination under
`ghcr.io/fujitsupolycom/sparkring-glm53-runtime`. It is not a publisher for a
derived SparkCache or DFlash7 image.

## Resolve, inspect, and start

Follow the selected quickstart to resolve `site.yaml` and `profile.json`.
These common commands are safe until the confirmed `start` command:

```bash
python scripts/preflight.py \
  --site site.yaml --strict-placeholders --json preflight.json
python scripts/sparkring_generic_launcher.py \
  --site site.yaml --profile profile.json validate
python scripts/sparkring_generic_launcher.py \
  --site site.yaml --profile profile.json plan > start-plan.json
python scripts/sparkring_generic_launcher.py \
  --site site.yaml --profile profile.json --execute verify-image \
  > verify-image.json
```

Review `preflight.json`, `start-plan.json`, and `verify-image.json`. Start only
with the confirmation value in the selected profile and explicit authority to
replace or interrupt the named service:

```bash
confirmation='REPLACE_WITH_PROFILE_CONFIRMATION'
python scripts/sparkring_generic_launcher.py \
  --site site.yaml --profile profile.json \
  --execute --confirmation "${confirmation}" start
```

The selected quickstart defines health, semantic, persistent-restore, and
concurrency evidence. Offline verification and image parity do not replace
those checks.

## Record one exact artifact

Keep construction, placement, and live evidence separate:

1. The runtime `pins.json` names the source commit, Git tree, clean-source
   SHA-256, patch checksums, local image tag, image ID, and builder-receipt
   SHA-256.
2. An archive receipt names the image ID, archive path, archive SHA-256, and
   byte count. A registry publication receipt instead names the immutable
   repository digest and SPDX SBOM SHA-256.
3. A placement receipt names every rank and proves that the same image ID or
   immutable registry digest is present on all four ranks.
4. Machine-readable live evidence belongs under
   `performance/receipts/glm53-flash/<artifact>/`. The corresponding bounded
   interpretation belongs under `performance/records/glm53-flash/`, with a
   GPU-free regression test beside it.
5. The selected quickstart may state **qualified** only for the image ID or
   registry digest and cases named by that evidence. Rebuilds remain
   **implemented** until they have their own records.

Do not replace a local image ID with a registry digest unless an exact
publication receipt connects them. Do not use an archive or fanout receipt as
serving evidence.
