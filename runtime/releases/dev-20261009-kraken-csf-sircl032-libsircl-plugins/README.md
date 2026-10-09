# Release 2026.10.2: the SIRCL, libsircl and GLM-5.3 plugin installer image

GitHub release `2026.10.2` publishes installer image release
`dev-20261009-kraken-csf-sircl032-libsircl-plugins`. It is a kraken-line image
(built on Local Inference Lab's `karmic-kraken-beta` vLLM and B12X branches)
whose lock, `sparkring-installer-image/v3`, carries three collective
transports: SIRCL ring sessions, libsircl (SIRCL's NCCL-compatible C library,
built from this repository's source) and the prepared B12X RoCE transport. It
also carries the GLM-5.3 vLLM plugins. This page is the release record.
[RELEASE_NOTES.md](RELEASE_NOTES.md) is the summary for users, and
[status.json](status.json) states the same status for programs.

Status: **implemented**; release maturity **candidate**. The image is built,
loaded on eight Sparks by its configuration ID, and its lock is recorded. It
is not in a registry. Every pending gate under
[Qualification](#qualification) must pass before publication; the
`relay-marker` and `publication` gates complete with it
([publication](#publication)).

## Identity

| Item | Value |
|---|---|
| GitHub release tag | `2026.10.2`; `2026.10.1` is the release before it |
| Image release name | `dev-20261009-kraken-csf-sircl032-libsircl-plugins`, the lock's `name` |
| Image configuration ID | `sha256:1a8c10354eb0bbaad898ebcfc1b31d5c5cfc40daab2f9b8eaaeb074524aea952` |
| Tag on the Sparks that hold it | `sparkring-dev/kraken:csf-sircl032-libsircl-plugins-20261009` |
| Source commit of the build | `c7c35fe0b24cb37d13a7a85c23d3281648ecf847`. The SIRCL, libsircl, plugin and tuning-table sources the image carries are byte-identical in the commit that adds this record |
| Lock | [installer-image-c7c35fe0.json](../../../performance/records/images/dev-20261009-kraken-csf-sircl032-libsircl-plugins-image-20261009/installer-image-c7c35fe0.json), SHA-256 `3bbcdfe378d7b0ad1577bb1e4e83517a428310a00fe5f388a35b79a4f2c95958`, 13 profiles ([image record](../../../performance/records/images/dev-20261009-kraken-csf-sircl032-libsircl-plugins-image-20261009.md)) |
| Registry reference | **Pending (`publication`).** `ghcr.io/fujitsupolycom/sparkring:dev-20261009-kraken-csf-sircl032-libsircl-plugins` and the manifest digest that publication records |
| Size | 31,883,141,118 bytes (29.7 GiB) unpacked. The lock's 15,308,236,724 download bytes are an upper bound; publication replaces them with the registry's compressed size |
| Parent and rollback image | `2026.10.1`: `dev-20261004-kraken-cuda1342-nccl2323-status034`, `ghcr.io/fujitsupolycom/sparkring@sha256:71d410571407fef3ce2959c6d392f5a2c3f44b757b856e853a71f6e3295620ad`, on the prepared transport |

## Layers

| Layer | Builder | Adds |
|---|---|---|
| Parent | `2026.10.1`, image `sha256:aba309e4610c…` | The `eugr/spark-vllm-b12x` nightly-20261001 base, vLLM `d51b4181` and B12X `a850d8eb`, CUDA 13.4.2 and NCCL 2.32.3, the prepared B12X RoCE transport bundle `tp2-rocenante-adaptive-prepared` and runtime-status 0.3.4 ([composition](../../images/compositions/external-kraken-20261004/README.md)) |
| CSF sources | `installer-kraken-csf-sources`, [derive_kraken_csf_sources.py](../../images/derive_kraken_csf_sources.py) | The 49 Python files of vLLM `bc9ea774` and B12X `cc36aa6f`, which read the GLM-5.3-Flash CSF checkpoint ([sources](../../images/compositions/kraken-csf-sources-20261007/README.md)) |
| SIRCL | `installer-sircl-layer`, [sircl_layer.py](../../images/sircl_layer.py) | SIRCL 0.3.2, native ABI 9: the wheel `sparkring_sircl-0.3.2-py3-none-any.whl` and the prebuilt `roce_proxy-09db871538819191.so` and `p2p_proxy-c8ccfcb93ffa2d6a.so`; SIRCL's pinned vLLM build `sparkring-kraken-beta-20261007-bc9ea774` |
| libsircl | `installer-libsircl-layer`, [libsircl_layer.py](../../images/libsircl_layer.py) | libsircl 0.6.0 compiled in the image from [spark_transport/libsircl](../../../spark_transport/libsircl/README.md) (git tree `dbf3607484dd47df4cf6c8238eb5b3272466effb`) with its four kernel packs, the fail-stop mode and NCCL API level 22705; its notices; the vLLM general plugin `libsircl` |
| GLM-5.3 plugins | [derive_glm53_plugins.py](../../images/derive_glm53_plugins.py) on [derived_layer.py](../../images/derived_layer.py) | The vLLM general plugins `glm_dsa_indexer_split` 1.1.0, `glm53full_speedups` 1.1.0 and `glm_dcp_decode_comm` 2.0.1 |

The runtime-status dashboard in the image is 0.3.4, the parent's. The
repository's runtime-status source is 0.3.6, which names the collective
transport and shows SIRCL's session facts on the dashboard; no image carries
0.3.5 or 0.3.6. The intermediate images of the four layers and their locks
are not in the repository.

## Profiles and transports

The lock admits 13 profiles. The status column is the profile catalog's
([profiles](../../../profiles/README.md)).

| Profiles | Status | Notes |
|---|---|---|
| `deepseek-v41-flash-tp4`, `glm53-flash-nvfp4-spark-tp2`, `glm53-flash-nvfp4-spark-tp4`, `mimo-v26-flash-mopd-tp2`, `mimo-v26-flash-mopd-tp4`, `qwen38-flash-next-qad-tp4`, `qwen38-flash-next-tp2`, `swift15-qwen38-flash-next-tp2` | implemented | The 2026.10.1 profiles. The GLM-5.3-Flash profiles install the CSF checkpoint (research-only) by default on this image, because the lock's `sircl.vllm_pins` names the build that reads it; `--checkpoint nvfp4-spark` installs NVFP4-Spark ([default checkpoint by profile](../../../profiles/glm53-checkpoints.md#default-checkpoint-by-profile)) |
| `swift15-qwen38-flash-next-tp4` | research-only | A 2026.10.1 profile |
| `deepseek-v41-flash-tp8`, `glm53-flash-csf-tp8`, `glm53-nvfp4-tp8`, `glm53-nvfp4-tp8-dcp1` | research-only | Eight-Spark profiles; they run only on SIRCL ring sessions. `glm53-nvfp4-tp8` and `glm53-nvfp4-tp8-dcp1` load the GLM-5.3 plugins |

`qwen38-flash-next-qad-tp8` is not admitted: its token-row ownership at eight
ranks is refused by installer admission and by the image's Qwen prefill
sources, which accept tensor-parallel groups of two and four.

`sparkring install` runs every collective on SIRCL ring sessions with NCCL
off on this image wherever `sudo sparkring setup` recorded the fabric with
its relay table, and on the prepared transport elsewhere.
`--transport prepared`, `--transport nccl` and `--transport libsircl`
(research-only) select the others
([transport and receipts](../../../docs/operations/install-reference.md#transport-and-receipts)).
SIRCL sessions take the default tuning table
[sircl-tuning-defaults.json](../../common/sircl-tuning-defaults.json), keyed to
SIRCL 0.3.2, the image's build. Its measured rows are `pair`, `path-4` and
`cycle-8`; groups of other shapes, a cycle of four among them, run on
SIRCL's own rules.

## Evidence

Every result below is in lane **public-functional**: it was measured with
images built from this repository and with its harnesses. Each states its
status, its maturity and its scope. None comes from a `sparkring install`
deployment, and none is serving qualification.

### GLM-5.3 on eight Sparks, two-token MTP

- Lane **public-functional**. Status **research-only**. Maturity
  **live-validated** within the serving A/B runner's arm checks.
- Hardware: eight DGX Sparks (GB10) cabled as one ring with ConnectX-7 RoCE,
  all eight serving; GPU SM clocks locked at 2,418 MHz.
- Configuration: profile `glm53-nvfp4-tp8` (TP8, decode-context parallelism
  4, MTP with two draft tokens, the GLM-5.3 plugins, `glm_dcp_decode_comm`'s
  items off), SIRCL ring sessions with NCCL off, image `27e9f75c0d09`. That
  image is the predecessor of the release image: SIRCL 0.3.1, libsircl from
  snapshot `a3477af2`, `glm_dcp_decode_comm` 2.0.0.
- Scope: one warm-up and two measured starts; temperature 0; 30 s cells of
  at most 1,024 output tokens; the serving A/B runner, not the installer.
- Result, output tokens/s at 1 / 2 / 4 / 8 streams, median of the two
  starts: 50.7 / 73.9 / 102.4 / 151.4 with no context, 44.2 / 64.0 / 89.8 /
  135.7 at 16K, 47.9 / 65.9 / 99.8 / 143.7 at 32K. Time to first token of a
  cold prompt: 10.0 s at 8K, 11.9 s at 16K, 24.4 s at 32K.
- Speculation: three MTP draft tokens, with or without the scheduler's
  acceptance-length adaptation, gave no gain over two; the DFlash2 drafter is
  refused at startup by the KV cache layout of the B12X DSA backend. The
  profile keeps two-token MTP.
- Record: [GLM-5.3 TP8 speculation arms](../../../performance/records/images/dev-20261009-kraken-csf-sircl-libsircl-plugins-dcp-glm53-tp8-speculation-20261009.md).
  The profile guide records a second run on the same image
  ([glm53-nvfp4-tp8](../../../profiles/glm53-nvfp4-tp8/README.md#evidence-and-open-items)).
- **Pending (`glm53-tp8-release-image`):** a GLM-5.3 TP8 run on the release
  image `1a8c10354eb0`. No record of one is in the repository; add it, or
  keep the scope above.

### Two and four Sparks on SIRCL ring sessions

- Lane **public-functional**. Status **research-only**. Maturity
  **live-validated** within the serving A/B runner's arm checks.
- Hardware: the same eight-Spark ring. Two-Spark profiles ran on a cabled
  pair of it; four-Spark profiles ran on four consecutive Sparks, a path of
  four whose end ranks reach each other through relays. GPU SM clocks locked
  at 2,418 MHz.
- Configuration: SIRCL ring sessions with NCCL off and the ring schedules
  that the default tuning table's `pair` and `path-4` rows hold; image
  `816c6d6a7e96` (SIRCL 0.3.0).
- Scope: one warm-up and two measured starts per profile; temperature 0; no
  prompt context unless stated; the serving A/B runner, not the installer.
- Result, output tokens/s at 1 / 2 / 4 / 8 streams, median of two starts
  (each campaign's `summary.json`):

  | Profile | Checkpoint | Sparks | No context | First token, 32K prompt |
  |---|---|---|---|---|
  | `glm53-flash-nvfp4-spark-tp2` | GLM-5.3-Flash CSF `dec48abd33ef` | pair | 42.0 / 61.4 / 88.9 / 133.5 | 15.3 s |
  | `glm53-flash-nvfp4-spark-tp4` | GLM-5.3-Flash CSF `dec48abd33ef` | path of four | 68.3 / 106.5 / 152.4 / 224.3 | 10.5 s |
  | `qwen38-flash-next-tp2` | Qwen3.8-Flash-Next QAD step 5500 with MXFP8 attention | pair | 63.0 / 100.5 / 153.0 / 217.0 | 8.4 s |
  | `qwen38-flash-next-qad-tp4` | Qwen3.8-Flash-Next QAD step 5500 with MXFP8 attention | path of four | 91.0 / 143.6 / 218.5 / 310.6 | 6.8 s |
  | `deepseek-v41-flash-tp4` | DeepSeek-V4.1-Flash `dba1be0a40aa`, `SIRCL_FUSED_NORM=1` | path of four | 61.9 / 94.3 / 138.8 / 181.9 | 7.6 s |

- The Qwen profiles' default checkpoint, QAD step 5500 (`qad-step5500-ple1000`),
  was not run; step 4000 measured within 1-6 % of the MXFP8-attention
  checkpoint on another pair.
- Record: [two- and four-Spark configurations on SIRCL](../../../performance/records/images/dev-20261008-kraken-csf-sircl-libsircl-tp2-tp4-matrix-20261009.md).

### libsircl on the cycle of eight Sparks

- Lane **public-functional**. Status **implemented** for the collectives
  listed; **unsupported** for broadcast at bandwidth on a cycle (below).
  Maturity **live-validated** with nccl-tests v2.21.1.
- Hardware: the eight-Spark ring as a cycle of eight; one rank per Spark.
- Configuration: libsircl snapshot `a3477af2`, built in image `aba309e4610c`;
  NVIDIA NCCL 2.32.3 from the same image as the baseline.
- Result: every nccl-tests row `#wrong 0` and every receipt healthy. Under
  the ring schedules, all-reduce of 256 MiB took 19,305 µs (24.33 GB/s bus
  bandwidth) against NCCL's 19,940 µs (23.56 GB/s); from 4 KiB to 1 MiB,
  libsircl's all-reduce, all-gather and reduce-scatter took 0.19 to 0.69 of
  NCCL's time. Broadcast reached 1.94 GB/s against NCCL's 24.22 GB/s.
- Record: [libsircl status](../../../spark_transport/libsircl/STATUS.md#hardware-the-path-of-four-at-positions-4-7-and-the-cycle-of-eight-snapshot-a3477af2).
- **Pending (`libsircl-ring8-release-image`):** the eight-Spark gate of the
  libsircl 0.6.0 that the release image carries, whose cycle plan runs ring
  all-reduces, all-gathers and reduce-scatters from 8 MiB by default.
  libsircl's status lists that plan as not run on Sparks. Commit the gate's
  outputs and verdicts, then state its default-schedule and small-message
  results here.

### Clean-room acceptance audit

- **Pending (`clean-room-audit`):** the acceptance audit of SIRCL 0.3.2, the
  three GLM-5.3 plugins and libsircl 0.6.0 against SparkRing's clean-room
  rule. Its report is not in the repository; commit it, with its inputs,
  result and the lines it asks to rewrite, before publication.

## Qualification

| Gate | Result | Record |
|---|---|---|
| Image built, loaded on eight Sparks, lock validated for its 13 profiles | passed | [image record](../../../performance/records/images/dev-20261009-kraken-csf-sircl032-libsircl-plugins-image-20261009.md) |
| SIRCL one-shot all-reduce and all-gather on the eight-Spark ring | passed (status qualified) | [SIRCL status](../../../spark_transport/sircl/STATUS.md#component-status) |
| Serving A/B measurements of two, four and eight Sparks recorded | passed (status research-only) | [Evidence](#evidence) |
| `ring8-installer`: installer qualification rounds on the eight-Spark ring with the release lock | **pending** | — |
| `ring4-setup`: `sudo sparkring setup --re-form` on each of two rings of four | **pending** | — |
| `ring4-install`: TP4 installations on the rings of four | **pending** | — |
| `ring4-benchmark`: the TP4 benchmark on a ring of four | **pending** | — |
| `ring4-sircl-tune`: `sudo sparkring fabric tune` for the `cycle-4` group shape and its promotion to the default table | **pending** | — |
| `ring4-libsircl-gate`: the libsircl gate on a cycle of four | **pending** | — |
| `libsircl-ring8-release-image`: the libsircl gate on the release image | **pending** | — |
| `clean-room-audit`: the clean-room acceptance audit | **pending** | — |
| `relay-marker`: the relay marker binary of source `c64c74535c06` published as a release asset and named in [relay-marker-artifact.json](../../../spark_transport/fabric/relay-marker-artifact.json), which names none; package builds without a compiler, `install.sh` among them, stop until it does | **pending** | — |
| `compose-v3`: Compose rendering and the Install Builder (`python scripts/generate_compose_builder.py --verify`) on this release's lock. [releases.md](../../../docs/development/releases.md) states that they use a v3 image through its v2 fields; `compose.build` passes the lock to `installer_image.for_profile`, which accepts v1 and v2 locks only | **pending** | — |
| `publication`: registry push, anonymous manifest and configuration check, and the publication records | **pending** | — |

A measured `cycle-4` row changes `sircl-tuning-defaults.json`, and therefore
the `tuning_defaults_sha256` that an image built afterwards records. This
image keeps its recorded value; installations apply the table of the
installing package ([release procedure](../../../docs/development/releases.md)).

## Known limitations

- `--transport libsircl` is research-only: no serving measurement uses it,
  the installer does not judge its receipts (verdict `unknown`), and it runs
  without decode-context parallelism.
- libsircl's broadcast on a cycle of eight reaches 1.9 GB/s against NVIDIA
  NCCL's 24.2 GB/s: it has no ring broadcast (unsupported). The default
  `sircl` transport does not use libsircl.
- The GLM-5.3-Flash CSF checkpoint is research-only and the default of the
  GLM-5.3-Flash profiles on this image.
- The eight-Spark profiles are research-only; no installation of them has a
  record.
- MiMo-V2.6-Flash-MOPD and Swift-1.5 profiles run on SIRCL sessions with no
  SIRCL serving record in the repository.
- A cycle of four, the four-Spark ring, has no measured tuning row until the
  `ring4-sircl-tune` gate passes; its sessions take SIRCL's own rules.
- Every measurement above locked the GPU clocks, which the installer does
  not do.
- The image's dashboard (runtime-status 0.3.4) shows neither SIRCL's session
  facts nor the collective transport; `sudo sparkring status` and
  `sudo sparkring check` report the transport and the receipt verdict.

## Publication

Publication pushes the image to `ghcr.io/fujitsupolycom/sparkring` under the
release name, checks the manifest and configuration without credentials, and
then records, in one commit:

- `installer-image.json` in this directory: the lock above with the registry
  reference as `image_reference` and the registry's compressed size as
  `download_bytes`;
- `publication.json` (`sparkring-shared-image-publication/v1`), whose
  `derivation.parent_release` is `dev-20261004-kraken-cuda1342-nccl2323-status034`,
  so the image keeps the parent's capabilities, `--save-cpu`'s shared-memory
  reader window among them;
- [release.json](release.json) as a `published-immutable-reference` with both
  files' SHA-256;
- a builder entry for the final layer's builder, `derive_glm53_plugins.py`,
  whose `releases` names this release, in
  [builders.json](../../images/builders.json), which lists no entry for that
  builder;
- the relay marker binary's SHA-256 and release-asset URL in
  [relay-marker-artifact.json](../../../spark_transport/fabric/relay-marker-artifact.json);
- `"2026.10.2": "dev-20261009-kraken-csf-sircl032-libsircl-plugins"` in
  [installer-releases.json](../installer-releases.json).

Listing the release in installer-releases.json makes it the image of every
`sparkring install` without `--image`, with SIRCL as its transport where the
fabric carries it. The rollback is `--image 2026.10.1`, the prepared
transport on the parent image; its deployments keep their site
configuration.
