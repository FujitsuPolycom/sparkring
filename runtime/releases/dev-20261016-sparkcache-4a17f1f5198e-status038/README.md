# Release 2026.10.5: the SparkCache installer image

SparkRing's profiles run three models created by Local Inference Lab, Inc. and
licensed under the Local Inference Lab License, Version 1.0:
[GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD)
(`glm53-flash-tp2`, `glm53-flash-tp4`, `glm53-flash-nvfp4-spark-tp2-dcp2`,
`glm53-flash-nvfp4-sparkcache-crc32-tp2`, `glm53-flash-nvfp4-sparkcache-crc32-tp4`),
[GLM-5.3-NVFP4-CSF](https://huggingface.co/local-inference-lab/GLM-5.3-NVFP4-CSF)
(`glm53-nvfp4-csf-tp6`) and
[Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD)
(`qwen38-flash-next-tp2`, `qwen38-flash-next-csf-sparkcache-tp2`).

GitHub release `2026.10.5` publishes installer image release
`dev-20261016-sparkcache-4a17f1f5198e-status038`. It is release 2026.10.4's
image with one layer added: SparkCache, the persistent KV cache, for
GLM-5.3-Flash, Qwen3.8-Flash-Next and DeepSeek-V4.1-Flash, with the
connector logging that the dashboard and the installer read. Its lock,
`sparkring-installer-image/v3`, admits 20 profiles: release 2026.10.4's 15
and five SparkCache variants. This page is the release record.
[RELEASE_NOTES.md](RELEASE_NOTES.md) is the summary for users, and
[status.json](status.json) states the same status for programs.

Status: **implemented**; release maturity **candidate**. The image is in the
registry as
`ghcr.io/fujitsupolycom/sparkring@sha256:ac313f040fe3e04238ad8ff152cf7fb427a050f40e0848c2e326c5b6b8717dfd`
([publication](#publication)). Two SparkCache profiles were installed and
served on this image, and three more on the same composition's recheck
build; the others were admitted by the installer's image checks and not
started on it ([evidence](#evidence)).

## Identity

| Item | Value |
|---|---|
| GitHub release tag | `2026.10.5`; `2026.10.4` precedes it and publishes the parent image |
| Image release name | `dev-20261016-sparkcache-4a17f1f5198e-status038`, the lock's `name` |
| Image configuration ID | `sha256:f625053225d4b57bfd7d0bb2e8348fea38e69472b10cef87c4701928abc7c4a8` |
| Registry reference | `ghcr.io/fujitsupolycom/sparkring@sha256:ac313f040fe3e04238ad8ff152cf7fb427a050f40e0848c2e326c5b6b8717dfd`, tagged `ghcr.io/fujitsupolycom/sparkring:dev-20261016-sparkcache-4a17f1f5198e-status038`: a single-image `linux/arm64` manifest whose configuration is the image ID, checked without credentials |
| Size | 31,898,421,029 bytes (29.7 GiB) unpacked; 15,283,745,437 bytes (14.2 GiB) to download in 58 layers. The first 57 layers are release 2026.10.4's, digest for digest, so a Spark that holds 2026.10.4's image downloads the one layer this image adds, 708,862 bytes (692 KiB) |
| Recorded lock | Written by `derived_layer.py` when the image was built, as the composition's `built-image.json`, SHA-256 `dfe3d6610b6a6c16fbe74ca85523d8192873a24aeaa00d875d9fd66e8c3b0704`. [installer-image.json](installer-image.json) is that lock with the registry reference as `image_reference` and the registry's compressed size as `download_bytes` |
| Build sources | The composition [installer-sparkcache-4a17f1f5198e](../../images/compositions/installer-sparkcache-4a17f1f5198e/README.md) of this tree: its [sources.json](../../images/compositions/installer-sparkcache-4a17f1f5198e/sources.json) pins every installed file by SHA-256 from the vendored SparkCache snapshot `integrations/sparkcache/4a17f1f5198e` (SparkCache commit `4a17f1f5198eb2402bcd6669425c62857937d39f`), and its [build.sh](../../images/compositions/installer-sparkcache-4a17f1f5198e/build.sh) ran on a Spark holding the parent image |
| Parent and rollback image | `2026.10.4`: `dev-20261013-dcp2prefill-status038-sircl-tp6`, `ghcr.io/fujitsupolycom/sparkring@sha256:e21ba936d4fd6bfeb6e7a651e54a03ac461bf4e1e6f5721c1a34845c80acd76f`; `sudo sparkring install --profile PROFILE --image 2026.10.4` selects it |

## Layers

Each row is one build step; its parent is the row above.

| Layer | Builder | Adds |
|---|---|---|
| Parent | `2026.10.4`, image `sha256:3cc9a81fd344…` | The TP2/DCP2 prefill admission, runtime-status 0.3.8 and the SIRCL 0.3.2 layer whose ring sessions admit GLM-5.3 on six Sparks, over release 2026.10.2's image ([release record](../dev-20261013-dcp2prefill-status038-sircl-tp6/README.md#layers)) |
| SparkCache 4a17f1f5198e | `installer-sparkcache-4a17f1f5198e`, [prepare_sparkcache.py](../../images/prepare_sparkcache.py) preparing the context that [derived_layer.py](../../images/derived_layer.py) builds and records | The `sparkcache` package, its runtime patches and the eleven-file KV-block lease contract in site-packages, for the `deepseek-v41-b12x-hma`, `qwen38-flash-next-hybrid` and GLM-5.3-Flash profile adapters; the change-only connector logging transform of `vllm/distributed/kv_transfer/kv_connector/v1/metrics.py`, applied after the parent's file is authenticated against its receipt; and the SparkCache profiles' settings. The parent's native libraries `libspark_cache_placement.so` and `libspark_cache_snapshot.so` keep their pinned bytes |

Runtime-status 0.3.8, SIRCL 0.3.2 (native ABI 9, tuning key unchanged),
libsircl 0.6.0 (git tree `030419b8`) and the plugins `glm_dsa_indexer_split`
1.1.0, `glm53full_speedups` 1.1.0 and `glm_dcp_decode_comm` 2.0.1 are the
parent's, unchanged, as are the vLLM build
`sparkring-kraken-bc9ea774-tp2-dcp2-prefill` and the CSF loaders, so the
GLM-5.3-Flash two- and four-Spark profiles and `qwen38-flash-next-tp2` keep
installing their CSF checkpoints by default.

## Profiles and transports

The lock admits 20 profiles. The status column is the profile catalog's
([profiles](../../../profiles/README.md)).

| Profiles | Status | Notes |
|---|---|---|
| `deepseek-v41-flash-tp4`, `glm53-flash-nvfp4-spark-tp2`, `glm53-flash-nvfp4-spark-tp4`, `mimo-v26-flash-mopd-tp2`, `mimo-v26-flash-mopd-tp4`, `qwen38-flash-next-qad-tp4`, `qwen38-flash-next-tp2`, `swift15-qwen38-flash-next-tp2` | implemented | Release 2026.10.4's, with their 2026.10.4 settings; none was started on this image |
| `swift15-qwen38-flash-next-tp4`, `deepseek-v41-flash-tp8`, `glm53-flash-csf-tp8`, `glm53-nvfp4-tp8`, `glm53-nvfp4-tp8-dcp1`, `glm53-flash-nvfp4-spark-tp2-dcp2`, `glm53-nvfp4-csf-tp6` | research-only | Release 2026.10.4's; none was started on this image |
| `deepseek-v41-flash-sparkcache-tp4` | research-only | DeepSeek-V4.1-Flash TP4/DCP1 with SparkCache ([guide](../../../profiles/deepseek-v41-flash-sparkcache-tp4/README.md)); same checkpoint and API port as `deepseek-v41-flash-tp4` |
| `qwen38-flash-next-step5500-sparkcache-tp4`, `qwen38-flash-next-csf-sparkcache-tp2` | research-only | Qwen3.8-Flash-Next QAD step 5500 on four Sparks and CSF on two, with SparkCache ([TP4 guide](../../../profiles/qwen38-flash-next-step5500-sparkcache-tp4/README.md), [TP2 guide](../../../profiles/qwen38-flash-next-csf-sparkcache-tp2/README.md)) |
| `glm53-flash-nvfp4-sparkcache-crc32-tp2`, `glm53-flash-nvfp4-sparkcache-crc32-tp4` | research-only | GLM-5.3-Flash CSF with SparkCache on two and four Sparks ([TP2 guide](../../../profiles/glm53-flash-nvfp4-sparkcache-crc32-tp2/README.md), [TP4 guide](../../../profiles/glm53-flash-nvfp4-sparkcache-crc32-tp4/README.md)) |

The SparkCache profiles take three `sparkring install` options, kept through
`up`, restart and recovery: `--sparkcache-size SIZE` (the per-Spark cache
budget, default 32G, with the connector's low watermark at 7/8 of it),
`--sparkcache-integrity fast|sha256|none` (the payload check; `fast` by
default; each mode is its own cache schema, so a change of mode starts from a
miss) and `--sparkcache-debug` (detailed SparkCache logs with vLLM at INFO).
The connector logs one `restore hit` or `restore failed` line per request from
the scheduler; per-rank restore lines are at DEBUG.

Transport selection is release 2026.10.4's: SIRCL ring sessions with NCCL
off where `sudo sparkring setup` recorded the fabric, the prepared transport
elsewhere, and `--transport prepared`, `nccl` or `libsircl` (research-only)
on request ([transport and receipts](../../../docs/operations/install-reference.md#transport-and-receipts)).
The lock's `tuning_defaults_sha256`, `92bd95e6…`, is unchanged from the
parent's.

## Evidence

Lane **public-functional**; hardware: a ring of eight DGX Sparks (GB10),
ConnectX-7 RoCE, SIRCL ring sessions with NCCL off, GPU clocks not locked;
date 2026-10-16. The package was built from this release's tree. None of it
is serving qualification, and none measured performance. The full record is
[the SparkCache installation and recovery record](../../../performance/records/images/dev-20261016-sparkcache-4a17f1f5198e-status038-sparkcache-recovery-ring8-20261016.md).

On this image, with the installer's default settings:

| Profile | Sparks | Result |
|---|---|---|
| `deepseek-v41-flash-sparkcache-tp4` | 0-3 | Installed (rc=0, 522 s) at the same time as the row below; `sparkring check` passed; served requests. After a restart, replaying a saved session logged one scheduler line per request, `restore hit tokens=14592 ms=100.3` |
| `qwen38-flash-next-step5500-sparkcache-tp4` | 4-7 | Installed (rc=0, 487 s); `sparkring check` passed; served requests |

On the composition's recheck build, image `sha256:c6fb5c0f4abb…`, built with
SparkCache `6ef875ecd94b`, whose only difference from `4a17f1f5198e` is the
operator log lines and counter names, installed with `--image-lock`:

| Check | Result |
|---|---|
| Three disjoint installations started at once: `deepseek-v41-flash-sparkcache-tp4` on 0-3, `qwen38-flash-next-csf-sparkcache-tp2` on 4-5, `glm53-flash-nvfp4-sparkcache-crc32-tp2` on 6-7 | All rc=0 (405, 531 and 512 s), no refusal for Sparks held by another installation |
| `sparkring setup` without `--name`, twice and as a plan | The cluster name stays |
| DeepSeek TP4, integrity `fast`: rank 1's cache corrupted, restart, the same prompt twice | First read recomputes; rank 1 invalidates 4 damaged entries and re-saves; second read restores 14,592 of 14,875 tokens on all four ranks (0.54 s) |
| Qwen step 5500 TP4, integrity `sha256`: the same sequence | First read recomputes; second read restores 17,280 of 18,533 tokens (0.79 s) |
| A `sha256` installation replays a session saved in `fast` mode | First lookup misses with `integrity_schema_mismatch`; then it saves and restores in its own schema |

The dashboard on Node A showed a tokens-per-second figure per running request
during live use of the two installations, as observed by the operator; the
harness polls never caught a request mid-decode. Its theme switch was checked
at desktop and 375 px width.

The other 15 profiles of the lock were admitted by the lock's `record` step:
the installer's admission (`installer_image.admit`) with the image's isolated
`verify`, which checks the installed files and receipts, passed for each.
They were not started on this image; `glm53-flash-nvfp4-sparkcache-crc32-tp4`
was not started on the recheck build either.

## Qualification

| Gate | Result | Record |
|---|---|---|
| `image-lock`: image `f625053225d4` built by the composition's `build.sh`, the KV-block lease contract verified in the parent and the built image, and its lock recorded for 20 profiles | passed | [installer-image.json](installer-image.json) |
| `ring8-installs`: the two installations on this image | passed (status research-only) | [Evidence](#evidence) |
| `sparkcache-recovery`: cluster-name preservation, parallel disjoint installations and corruption recovery at both integrity modes, on the recheck build | passed (single runs) | [Evidence](#evidence) |
| `relay-marker`: source `c64c74535c06`, unchanged since release 2026.10.2; package builds take the binary that [relay-marker-artifact.json](../../../spark_transport/fabric/relay-marker-artifact.json) records | passed | [relay-marker-artifact.json](../../../spark_transport/fabric/relay-marker-artifact.json) |
| `publication`: registry push, anonymous manifest and configuration check of `sha256:ac313f040fe3…`, and the publication records | passed | [publication.json](publication.json) |
| `glm53-tp6-hardware`: an installation of `glm53-nvfp4-csf-tp6` on six Sparks of this image | not run | [guide](../../../profiles/glm53-nvfp4-csf-tp6/README.md) |
| `compose-v3`: Install Builder verification with this release's lock as the default (`python scripts/generate_compose_builder.py --output DIR --tag 2026.10.5 --verify`): 10 profiles, 2 images, 459 valid and 910 invalid generated sites agree with `compose.build` | passed | [compose-builder.md](../../../docs/operations/compose-builder.md) |

## Known limitations

- Release 2026.10.4's limitations apply
  ([known limitations](../dev-20261013-dcp2prefill-status038-sircl-tp6/README.md#known-limitations)).
- The SparkCache profiles are research-only: single runs, no soak, no
  correctness screen and no performance measurement on this image. The
  restores are attributed to SparkCache by its log lines, not by timing alone.
- `glm53-flash-nvfp4-sparkcache-crc32-tp4` has not started on any build of
  this composition.
- None of release 2026.10.4's 15 profiles was started on this image. The
  layer changes no file they load, except the connector metrics logger, whose
  transform only changes which connector statistics the stat logger prints.
- The dashboard's per-request output rate rests on the operator's
  observation; no harness captured it.
- The Install Builder lists the five SparkCache profiles as install-command
  cards with the cache size and payload-check fields; its Compose
  verification covers the 10 Compose profiles, not the install-only cards
  ([limits](../../../docs/operations/compose-builder.md#limits)).

## Publication

The image is published as
`ghcr.io/fujitsupolycom/sparkring@sha256:ac313f040fe3e04238ad8ff152cf7fb427a050f40e0848c2e326c5b6b8717dfd`,
tagged with the release name. Its manifest and configuration were checked
without credentials. These records name it:

- [installer-image.json](installer-image.json): the recorded lock with the
  registry reference as `image_reference` and the registry's compressed size
  as `download_bytes`;
- [publication.json](publication.json)
  (`sparkring-shared-image-publication/v1`), whose
  `derivation.parent_release` is `dev-20261013-dcp2prefill-status038-sircl-tp6`,
  so the image keeps its parent's capabilities, `--save-cpu`'s shared-memory
  reader window among them;
- [release.json](release.json) as a `published-immutable-reference` with
  both files' SHA-256;
- this release in the `releases` of its layer's builder,
  `installer-sparkcache-4a17f1f5198e` (`prepare_sparkcache.py`), in
  [builders.json](../../images/builders.json);
- `"2026.10.5": "dev-20261016-sparkcache-4a17f1f5198e-status038"` in
  [installer-releases.json](../installer-releases.json).

Listing the release in installer-releases.json makes it the image of every
`sparkring install` without `--image` from a package built from this tree,
and of every Compose export and `sparkring compose render` without `--image`.
The rollback is `--image 2026.10.4`, the parent image on the same transports;
its deployments keep their site configuration.
