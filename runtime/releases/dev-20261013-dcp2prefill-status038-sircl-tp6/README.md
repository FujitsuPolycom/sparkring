# Release 2026.10.4: the TP2/DCP2 prefill and GLM-5.3 TP6 installer image

SparkRing's profiles run three models created by Local Inference Lab, Inc. and
licensed under the Local Inference Lab License, Version 1.0:
[GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD)
(`glm53-flash-tp2`, `glm53-flash-tp4`, `glm53-flash-nvfp4-spark-tp2-dcp2`),
[GLM-5.3-NVFP4-CSF](https://huggingface.co/local-inference-lab/GLM-5.3-NVFP4-CSF)
(`glm53-nvfp4-csf-tp6`) and
[Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD)
(`qwen38-flash-next-tp2`).

GitHub release `2026.10.4` publishes installer image release
`dev-20261013-dcp2prefill-status038-sircl-tp6`. It is release 2026.10.2's
image with these additions:

- the vLLM source change that lets GLM-5.3-Flash run on two Sparks with
  decode-context parallelism 2 (TP2/DCP2);
- runtime-status 0.3.8, whose dashboard plugins also log vLLM's serving
  statistics and report the resident quantization of GLM-5.3 at tensor
  parallelism 6;
- a rebuilt SIRCL 0.3.2 layer whose ring sessions admit GLM-5.3 on six
  Sparks.

Its lock, `sparkring-installer-image/v3`, admits 15 profiles, two more than
release 2026.10.2's. This page is the release record.
[RELEASE_NOTES.md](RELEASE_NOTES.md) is the summary for users, and
[status.json](status.json) states the same status for programs.

Status: **implemented**; release maturity **candidate**. The image is in the
registry as
`ghcr.io/fujitsupolycom/sparkring@sha256:e21ba936d4fd6bfeb6e7a651e54a03ac461bf4e1e6f5721c1a34845c80acd76f`
([publication](#publication)). Three profiles were installed and served on
this image; the others were admitted by the installer's image checks and not
started on it ([evidence](#evidence)).

## Identity

| Item | Value |
|---|---|
| GitHub release tag | `2026.10.4`; `2026.10.3` precedes it and publishes no image |
| Image release name | `dev-20261013-dcp2prefill-status038-sircl-tp6`, the lock's `name` |
| Image configuration ID | `sha256:3cc9a81fd344fe52fe9cc261b46dc051e3c3f494e51b8142846a5f854835c502` |
| Registry reference | `ghcr.io/fujitsupolycom/sparkring@sha256:e21ba936d4fd6bfeb6e7a651e54a03ac461bf4e1e6f5721c1a34845c80acd76f`, tagged `ghcr.io/fujitsupolycom/sparkring:dev-20261013-dcp2prefill-status038-sircl-tp6`: a single-image `linux/arm64` manifest whose configuration is the image ID, checked without credentials |
| Size | 31,895,394,449 bytes (29.7 GiB) unpacked; 15,283,036,575 bytes (14.2 GiB) to download in 57 layers. The first 51 layers are release 2026.10.2's, digest for digest, so a Spark that holds 2026.10.2's image downloads the 6 layers this image adds, 2,720,889 bytes (2.6 MiB) |
| Recorded lock | Written by `sircl_layer.py record` when the image was built, SHA-256 `1f313ef7b7bb9fa6c3d7f85373da537f4ab0e4b233aa6025cf1d5cd318015716`. [installer-image.json](installer-image.json) is that lock with the registry reference as `image_reference` and the registry's compressed size as `download_bytes` |
| Build sources | Commit `32c41e7f03218762a59bd43e2b47f85bb85407ac` for the SIRCL layer and the runtime-status 0.3.8 descriptor; `17d3a1fa` for the TP2/DCP2 prefill layer; `85b98b01` for the runtime-status 0.3.7 layer. In this tree, `integrations/vllm/runtime_status` is the same git tree (`ae8b68dd`) and `spark_transport/sircl` differs only in `STATUS.md`, which the wheel does not carry |
| Parent and rollback image | `2026.10.2`: `dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036`, `ghcr.io/fujitsupolycom/sparkring@sha256:4fffc4dc3074d5539f4e9d3a013ff9ef4e0be570a95b74d4646ee341da1f6911`; `sudo sparkring install --profile PROFILE --image 2026.10.2` selects it |

## Layers

Each row is one build step; its parent is the row above.

| Layer | Builder | Adds |
|---|---|---|
| Parent | `2026.10.2`, image `sha256:d52737a109e0…` | CSF sources, SIRCL 0.3.2, libsircl 0.6.0, the GLM-5.3 vLLM plugins and runtime-status 0.3.6 ([release record](../dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036/README.md#layers)) |
| TP2/DCP2 prefill admission | [derive_dcp2_prefill.py](../../images/derive_dcp2_prefill.py) | vLLM's KDA prefill checkpoint admission and both mHC row-ownership gates admit tensor parallelism 2 with decode-context parallelism 2, behind `VLLM_B12X_KDA_PREFILL_COALESCING` and `VLLM_GLM53_MHC_PREFILL_SHARD`, both off unless a profile sets them. SIRCL's pins name the result as vLLM build `sparkring-kraken-bc9ea774-tp2-dcp2-prefill` ([composition](../../images/compositions/dcp2-prefill/README.md), [source change](../../../integrations/vllm/dcp2_prefill/README.md)) |
| runtime-status 0.3.7 | `installer-derived-layer`, [derived_layer.py](../../images/derived_layer.py) with [installer-dashboard-status037](../../images/compositions/installer-dashboard-status037/descriptor.json) | The dashboard's vLLM stat-logger plugin `sparkring_stats` |
| runtime-status 0.3.8 | `installer-derived-layer`, [derived_layer.py](../../images/derived_layer.py) with [installer-status038](../../images/compositions/installer-status038/descriptor.json) | The stat logger and the resident quantization facts that the `glm53-tp6-resident-quantization` start check of `glm53-nvfp4-csf-tp6` reads |
| SIRCL 0.3.2 | `installer-sircl-layer`, [sircl_layer.py](../../images/sircl_layer.py) | The wheel `sparkring_sircl-0.3.2-py3-none-any.whl` (SHA-256 `2ab69d93…`) built from commit `32c41e7f`: the universal routes that admit GLM-5.3 on six Sparks and the TP2/DCP2 prefill build identity. Its native libraries `roce_proxy-09db871538819191.so` and `p2p_proxy-c8ccfcb93ffa2d6a.so` were compiled again in the parent image and are byte-identical to the parent's. Native ABI 9 and the tuning key are unchanged |

libsircl 0.6.0 (git tree `030419b8`) and the plugins `glm_dsa_indexer_split`
1.1.0, `glm53full_speedups` 1.1.0 and `glm_dcp_decode_comm` 2.0.1 are the
parent's, unchanged.

## Profiles and transports

The lock admits 15 profiles. The status column is the profile catalog's
([profiles](../../../profiles/README.md)).

| Profiles | Status | Notes |
|---|---|---|
| `deepseek-v41-flash-tp4`, `glm53-flash-nvfp4-spark-tp2`, `glm53-flash-nvfp4-spark-tp4`, `mimo-v26-flash-mopd-tp2`, `mimo-v26-flash-mopd-tp4`, `qwen38-flash-next-qad-tp4`, `qwen38-flash-next-tp2`, `swift15-qwen38-flash-next-tp2` | implemented | Release 2026.10.2's. The GLM-5.3-Flash profiles and `qwen38-flash-next-tp2` install their CSF checkpoints by default on this image: the TP2/DCP2 prefill build changes no CSF loader file of the build it derives from ([image_lock.py](../../common/image_lock.py), `CHECKPOINT_BUILDS`). The Qwen CSF checkpoint has no serving record on this build |
| `swift15-qwen38-flash-next-tp4`, `deepseek-v41-flash-tp8`, `glm53-flash-csf-tp8`, `glm53-nvfp4-tp8`, `glm53-nvfp4-tp8-dcp1` | research-only | Release 2026.10.2's |
| `glm53-flash-nvfp4-spark-tp2-dcp2` | research-only | GLM-5.3-Flash CSF on two Sparks with decode-context parallelism 2 ([guide](../../../profiles/glm53-flash-nvfp4-spark-tp2-dcp2/README.md)). It runs only on an image whose vLLM is the TP2/DCP2 prefill build |
| `glm53-nvfp4-csf-tp6` | research-only | GLM-5.3-NVFP4-CSF on six Sparks ([guide](../../../profiles/glm53-nvfp4-csf-tp6/README.md)). No installation has run; the GLM-5.3-NVFP4-CSF checkpoint stays research-only (`CHECKPOINT_STATUS`) |

Transport selection is release 2026.10.2's: SIRCL ring sessions with NCCL
off where `sudo sparkring setup` recorded the fabric, the prepared transport
elsewhere, and `--transport prepared`, `nccl` or `libsircl` (research-only)
on request ([transport and receipts](../../../docs/operations/install-reference.md#transport-and-receipts)).
The lock's `tuning_defaults_sha256`, `92bd95e6…`, is the SHA-256 of
[sircl-tuning-defaults.json](../../common/sircl-tuning-defaults.json) at the
build, with the measured `pair`, `path-4`, `cycle-4` and `cycle-8` rows.

## Evidence

Lane **public-functional**; hardware: a ring of eight DGX Sparks (GB10),
ConnectX-7 RoCE, GPU clocks not locked; date 2026-10-13. Each installation
used `sudo sparkring install` with this image's recorded lock
(`--image-lock`), before the image was in the registry. None is serving
qualification, and none measured performance.

| Profile | Sparks | Result |
|---|---|---|
| `deepseek-v41-flash-tp4` | 4 | Installed and served requests |
| `glm53-flash-nvfp4-spark-tp2`, CSF checkpoint, decode-context parallelism 1 | 2 | Installed and served requests |
| `glm53-flash-nvfp4-spark-tp2-dcp2` | 2 | Installed; the API answered after 546 s; vLLM reported a KV cache of 4,126,165 tokens; both ranks installed SIRCL's `mhc_prefill_shard` shim with `VLLM_GLM53_MHC_PREFILL_SHARD=1` and `VLLM_B12X_KDA_PREFILL_COALESCING=0`; it served requests |

The other 12 profiles were admitted by the lock's `record` step: the
installer's admission (`installer_image.admit`) with the image's isolated
`verify`, which checks the installed files and receipts, passed for each.
They were not started on this image. The settings of the TP2/DCP2 profile
were measured on image `e91876e4997c`, the TP2/DCP2 prefill layer alone
([record](../../../performance/records/images/dcp2-prefill-e91876-glm53-flash-nvfp4-spark-tp2-dcp2-settings-20261013.md)).

## Qualification

| Gate | Result | Record |
|---|---|---|
| `image-lock`: image `3cc9a81fd344` built, `external-base.py verify` passed (6,326 files), and its lock recorded for 15 profiles | passed | [installer-image.json](installer-image.json) |
| `ring8-installs`: the three installations above | passed (status research-only) | [Evidence](#evidence) |
| `relay-marker`: source `c64c74535c06`, unchanged since release 2026.10.2; package builds take the binary that [relay-marker-artifact.json](../../../spark_transport/fabric/relay-marker-artifact.json) records | passed | [relay-marker-artifact.json](../../../spark_transport/fabric/relay-marker-artifact.json) |
| `publication`: registry push, anonymous manifest and configuration check of `sha256:e21ba936d4fd…`, and the publication records | passed | [publication.json](publication.json) |
| `glm53-tp6-hardware`: an installation of `glm53-nvfp4-csf-tp6` on six Sparks | not run | [guide](../../../profiles/glm53-nvfp4-csf-tp6/README.md) |
| `compose-v3`: Install Builder verification with this release's lock (`python scripts/generate_compose_builder.py --verify`) | not run | [compose-builder.md](../../../docs/operations/compose-builder.md) |

## Known limitations

- Release 2026.10.2's limitations apply
  ([known limitations](../dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036/README.md#known-limitations)).
- `glm53-nvfp4-csf-tp6` has never started on hardware.
- `glm53-flash-nvfp4-spark-tp2-dcp2` is research-only: one installation,
  no soak and no correctness screen on this image.
- runtime-status 0.3.8's stat logger runs in every serving container of this
  image; its serving overhead is not measured.
- Of the eight implemented profiles, only `deepseek-v41-flash-tp4` and
  `glm53-flash-nvfp4-spark-tp2` were started on this image.

## Publication

The image is published as
`ghcr.io/fujitsupolycom/sparkring@sha256:e21ba936d4fd6bfeb6e7a651e54a03ac461bf4e1e6f5721c1a34845c80acd76f`,
tagged with the release name. Its manifest and configuration were checked
without credentials. These records name it:

- [installer-image.json](installer-image.json): the recorded lock with the
  registry reference as `image_reference` and the registry's compressed size
  as `download_bytes`;
- [publication.json](publication.json)
  (`sparkring-shared-image-publication/v1`), whose
  `derivation.parent_release` is
  `dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036`, so the
  image keeps its parent's capabilities, `--save-cpu`'s shared-memory reader
  window among them;
- [release.json](release.json) as a `published-immutable-reference` with
  both files' SHA-256;
- this release in the `releases` of its final layer's builder,
  `installer-sircl-layer` (`sircl_layer.py`), in
  [builders.json](../../images/builders.json);
- `"2026.10.4": "dev-20261013-dcp2prefill-status038-sircl-tp6"` in
  [installer-releases.json](../installer-releases.json).

Listing the release in installer-releases.json makes it the image of every
`sparkring install` without `--image` from a package built from this tree.
Compose exports keep release 2026.10.1's v2 image. The rollback is
`--image 2026.10.2`, the parent image on the same transports; its
deployments keep their site configuration.
