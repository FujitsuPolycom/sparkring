# Choose a GLM-5.3 Flash quickstart

The published JJ r7-compatible image pair is the operator start for GLM-5.3 on four GB10
systems. Select `base` or `sparkcache` in one environment file. Historical
artifact procedures remain linked below for reproducing their exact evidence.

## Upstream runtime and model artifacts

Local Inference Lab's
[Jovian Judgement vLLM branch](https://github.com/local-inference-lab/vllm/tree/dev/jovian-judgement)
is the primary source of GLM-5.3 runtime performance and correctness work.
[B12X](https://github.com/local-inference-lab/b12x) supplies the Blackwell
kernels and backend integration. Local Inference Lab publishes the
[NVFP4 target](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4)
and a separate
[MXFP8 DFlash2 checkpoint](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-DFlash2-MXFP8).

The published images use
[`GLM-5.3-Flash-NVFP4@520de24e`](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4/tree/520de24eabf507659eaef7c70f14fd584527facc).
Their external draft is
[`incoai/GLM-5.3-Flash-DFlash2@dc77ff1c`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2/tree/dc77ff1c99eeb2df044ee3d4f0094eb033fee410),
which is BF16 and is not the MXFP8 checkpoint.

| Published variant | Status | Immutable image | Procedure |
|---|---|---|---|
| Cache-disabled base | **implemented and TP4 smoke-verified**, not generally qualified | `ghcr.io/fujitsupolycom/sparkring-glm53-runtime@sha256:11922064b342de1fc98f0ef85e6648843c8fa7eb3e4f4353c6ad82d6e457dde0` | [Public-image quickstart](GLM53_JJ_R7_GB10_TP4_QUICKSTART.md) |
| SparkCache composition | **implemented and TP4 smoke-verified**, not generally qualified | `ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:f012dd915c0fff0be384820c2d72cd015b83b9b33c3f980445dd718a807cd0c5` | [Public-image quickstart](GLM53_JJ_R7_GB10_TP4_QUICKSTART.md) |

The images bind Jovian Judgement community r7 plus vLLM composition
`331573d20bd47e78327ed8d8b4d2e6d350bbb1ab`, B12X
`6255090a03b12c3f7d552102a02fac0b542fb8c9`, and the patched NCCL library
SHA-256 `5f1c3f10d5ace66d4ba584415bbfe42b6ac1a0a9116a3b81dcbe50516ad924b3`.

## Historical artifact and source-development procedures

| Purpose | Evidence status | Procedure |
|---|---|---|
| Published BF16 DFlash2 artifact | Qualified only for its named historical digest and cases | [SparkCache](GLM53_FLASH_DFLASH2_BF16_SPARKCACHE_TP4_QUICKSTART.md), [cache-disabled](GLM53_FLASH_DFLASH2_BF16_TP4_QUICKSTART.md) |
| Full-snapshot local artifact | Qualified only for its named 131,072-token case | [Artifact procedure](GLM53_DFLASH7_PYTHON_OVERLAY_SPARKCACHE_TP4_QUICKSTART.md#shortest-qualified-start) |
| Split-page local artifact | Qualified only for its named C8 × 16K case | [Artifact procedure](GLM53_SPLIT_PAGE_SPARKCACHE_TP4_QUICKSTART.md) |
| Adaptive embedded MTP source composition | Implemented; live SparkCache serving unqualified | [Source procedure](GLM53_B12X_KDA_ADAPTIVE_MTP_SPARKCACHE_TP4_QUICKSTART.md) |
| Embedded-MTP source composition | Implemented; live SparkCache serving unqualified | [Source procedure](GLM53_E10536A_SPARKCACHE_TP4_QUICKSTART.md) |

Historical procedures retain their own source ancestry, cache identities,
image IDs, and limitations. Do not substitute those identities into the public
r2 environment.


## Plan resident-token concurrency

The GPU KV pool limits request admission, but the GLM hybrid allocator does
not convert its reported token capacity into linear long-context concurrency.
The 20 GiB FP8 configuration reported approximately 916,676 tokens. In a
no-cache C6 × 128K observation, vLLM admitted one request at a time, reported
approximately 39–41% GPU KV use for each admitted request, kept four through
zero requests waiting as the cohort drained, and completed requests serially
in 61–313 seconds.

Do not estimate GLM hybrid concurrency by dividing 916,676 by prompt length.
Use these bounded statuses until each shape has direct live evidence:

| Workload | Status |
|---|---|
| C2 × 128K | Only observed safe candidate; live CUDA qualification is pending |
| C6 × 128K | Does not provide six-way concurrency under the observed allocator behavior; requests serialized |
| C8 × 64K | Planned and **unqualified** until measured |
| C16 × 32K | Planned and **unqualified** until measured |
| C16 × 128K | **Unsupported** at the recorded 20 GiB capacity unless GPU trunk pages are shared or KV capacity increases |

C16 × 128K is unsupported at the recorded capacity unless GPU trunk pages
are shared and verified or KV capacity increases.

External-cache read coalescing can avoid repeated NVMe reads without sharing
the restored GPU pages among requests. Generated tokens also consume KV
capacity. Treat the server's admission, waiting, preemption, and GPU KV
metrics as authoritative for each workload shape.

## Put one exact image on every rank

Use one of these paths. Do not mix a local image ID with a registry digest in
one evidence record.

For the published JJ r7-compatible SparkCache image, copy the immutable reference from its
quickstart, inspect the offline plan, and then run the confirmed pull. Replace
the reference with the base digest when selecting `IMAGE_VARIANT=base`:

```bash
python scripts/pull_glm53_image_cluster.py \
  --site site.yaml \
  --image 'ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:f012dd915c0fff0be384820c2d72cd015b83b9b33c3f980445dd718a807cd0c5'

python scripts/pull_glm53_image_cluster.py \
  --site site.yaml \
  --image 'ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:f012dd915c0fff0be384820c2d72cd015b83b9b33c3f980445dd718a807cd0c5' \
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
