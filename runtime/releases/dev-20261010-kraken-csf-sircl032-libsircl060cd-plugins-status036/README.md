# Release 2026.10.2: the SIRCL, libsircl and GLM-5.3 plugin installer image

GitHub release `2026.10.2` publishes installer image release
`dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036`. It is a
kraken-line image (built on Local Inference Lab's `karmic-kraken-beta` vLLM and
B12X branches) whose lock, `sparkring-installer-image/v3`, carries three
collective transports: SIRCL ring sessions, libsircl (SIRCL's NCCL-compatible C
library, built from this repository's source) and the prepared B12X RoCE
transport. It also carries the GLM-5.3 vLLM plugins and runtime-status 0.3.6.
This page is the release record. [RELEASE_NOTES.md](RELEASE_NOTES.md) is the
summary for users, and [status.json](status.json) states the same status for
programs.

Status: **implemented**; release maturity **candidate**. The image is built and
its lock is recorded. It is not in a registry. Every pending gate under
[Qualification](#qualification) must pass before publication; the
`relay-marker` and `publication` gates complete with it
([publication](#publication)).

Most of the evidence below ran on image `1a8c10354eb0`, built from commit
`c7c35fe0`. The release image has the same SIRCL layer and was built from the
same CSF and GLM-5.3 plugin sources; its libsircl adds the current-device fix
(tree `030419b8` against `dbf36074`) and its dashboard is runtime-status 0.3.6
against 0.3.4. Each evidence item names its image.

## Identity

| Item | Value |
|---|---|
| GitHub release tag | `2026.10.2`; `2026.10.1` is the release before it |
| Image release name | `dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036`, the lock's `name` |
| Image configuration ID | `sha256:d52737a109e083d3eef05c0fc0db4d09fc1bf34485a13467382130963ecfe66b` |
| Tag on the Sparks that hold it | `sparkring-dev/kraken:csf-sircl032-libsircl060cd-plugins-status036-20261010` |
| Source commit of the build | `d3d3315863a63e3e5848acbc47d788251a32278d`. The SIRCL, libsircl, plugin, runtime-status and image-layer sources the image carries are byte-identical in this branch: `spark_transport/libsircl` is tree `030419b8`, the lock's `source_tree`. The default tuning table has the measured `cycle-4` row, which installations take from the installing package |
| Lock | [installer-image-d3d33158.json](../../../performance/records/images/dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036-image-20261010/installer-image-d3d33158.json), SHA-256 `ed4332d92f03086f60679f7e4e78e053a5c08c899ecd56a4c809afccbee8598d`, 13 profiles ([image record](../../../performance/records/images/dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036-image-20261010.md)) |
| Registry reference | **Pending (`publication`).** `ghcr.io/fujitsupolycom/sparkring:dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036` and the manifest digest that publication records |
| Size | 31,884,969,464 bytes (29.7 GiB) unpacked. The lock's 15,310,065,070 download bytes are an upper bound; publication replaces them with the registry's compressed size. Against 2026.10.1's image the image adds 6 layers, a 38.9 MB delta archive |
| Parent and rollback image | `2026.10.1`: `dev-20261004-kraken-cuda1342-nccl2323-status034`, `ghcr.io/fujitsupolycom/sparkring@sha256:71d410571407fef3ce2959c6d392f5a2c3f44b757b856e853a71f6e3295620ad`, on the prepared transport |

## Layers

| Layer | Builder | Adds |
|---|---|---|
| Parent | `2026.10.1`, image `sha256:aba309e4610c…` | The `eugr/spark-vllm-b12x` nightly-20261001 base, vLLM `d51b4181` and B12X `a850d8eb`, CUDA 13.4.2 and NCCL 2.32.3, the prepared B12X RoCE transport bundle `tp2-rocenante-adaptive-prepared` and runtime-status 0.3.4 ([composition](../../images/compositions/external-kraken-20261004/README.md)) |
| CSF sources | `installer-kraken-csf-sources`, [derive_kraken_csf_sources.py](../../images/derive_kraken_csf_sources.py) | The 49 Python files of vLLM `bc9ea774` and B12X `cc36aa6f`, which read the GLM-5.3-Flash CSF checkpoint ([sources](../../images/compositions/kraken-csf-sources-20261007/README.md)) |
| SIRCL | `installer-sircl-layer`, [sircl_layer.py](../../images/sircl_layer.py) | SIRCL 0.3.2, native ABI 9: the wheel `sparkring_sircl-0.3.2-py3-none-any.whl` and the prebuilt `roce_proxy-09db871538819191.so` and `p2p_proxy-c8ccfcb93ffa2d6a.so`; SIRCL's pinned vLLM build `sparkring-kraken-beta-20261007-bc9ea774`. The layer is image `1a8c10354eb0`'s, reused |
| libsircl | `installer-libsircl-layer`, [libsircl_layer.py](../../images/libsircl_layer.py) | libsircl 0.6.0 compiled in the image from [spark_transport/libsircl](../../../spark_transport/libsircl/README.md) (git tree `030419b8a2b61acc1a74010308a1c3e758863e2f`) with its four kernel packs, the fail-stop mode and NCCL API level 22705; it creates a communicator on the current CUDA device when the creating thread has no current context. Its notices; the vLLM general plugin `libsircl` |
| GLM-5.3 plugins | [derive_glm53_plugins.py](../../images/derive_glm53_plugins.py) on [derived_layer.py](../../images/derived_layer.py) | The vLLM general plugins `glm_dsa_indexer_split` 1.1.0, `glm53full_speedups` 1.1.0 and `glm_dcp_decode_comm` 2.0.1 |
| runtime-status 0.3.6 | `installer-derived-layer`, [derived_layer.py](../../images/derived_layer.py) | runtime-status 0.3.6 in place of 0.3.4, built from `integrations/vllm/runtime_status` tree `de6358a8`: the dashboard names the collective transport and the SIRCL version of the tensor-parallel group |

The image record keeps the intermediate images' locks and the runtime-status
layer's descriptor.

## Profiles and transports

The lock admits 13 profiles. The status column is the profile catalog's
([profiles](../../../profiles/README.md)).

| Profiles | Status | Notes |
|---|---|---|
| `deepseek-v41-flash-tp4`, `glm53-flash-nvfp4-spark-tp2`, `glm53-flash-nvfp4-spark-tp4`, `mimo-v26-flash-mopd-tp2`, `mimo-v26-flash-mopd-tp4`, `qwen38-flash-next-qad-tp4`, `qwen38-flash-next-tp2`, `swift15-qwen38-flash-next-tp2` | implemented | The 2026.10.1 profiles. The GLM-5.3-Flash profiles install the CSF checkpoint (implemented; not soaked) by default on this image, because the lock's `sircl.vllm_pins` names the build that reads it; `--checkpoint nvfp4-spark` installs NVFP4-Spark ([default checkpoint by profile](../../../profiles/glm53-checkpoints.md#default-checkpoint-by-profile)) |
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
SIRCL 0.3.2, the image's build. Its measured rows are `pair`, `path-4`,
`cycle-4` and `cycle-8`; groups of other shapes run on SIRCL's own rules.

## Evidence

Every result below is in lane **public-functional**: it was measured with
images built from this repository and with its harnesses. Each states its
status, its maturity and its scope, including whether `sparkring install` or
the serving A/B runner started the model. None is serving qualification.

### GLM-5.3 on eight Sparks, two-token MTP

- Lane **public-functional**. Status **research-only**. Maturity
  **live-validated** within the serving A/B runner's arm checks.
- Hardware: eight DGX Sparks (GB10) cabled as one ring with ConnectX-7 RoCE,
  all eight serving; GPU SM clocks locked at 2,418 MHz.
- Configuration: profile `glm53-nvfp4-tp8` (TP8, decode-context parallelism
  4, MTP with two draft tokens, the GLM-5.3 plugins, `glm_dcp_decode_comm`'s
  items off), SIRCL ring sessions with NCCL off, image `27e9f75c0d09`. That
  image precedes image `1a8c10354eb0`: SIRCL 0.3.1, libsircl from snapshot
  `a3477af2`, `glm_dcp_decode_comm` 2.0.0.
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
- On image `1a8c10354eb0`, installed with `sparkring install` (GPU clocks not
  locked): 47.7 output tok/s at one stream and 16K context, 2.47 accepted
  tokens per step, 19.4 steps/s, against 44.2 tok/s at 2.27 and 19.5 steps/s
  in the same cell on `27e9f75c0d09`; the step rates agree and the output
  rates differ with the acceptance
  ([installer runs on the ring of eight](../../../performance/records/images/dev-20261009-kraken-csf-sircl032-libsircl-plugins-installer-ring8-20261009.md)).

### Installer runs on the ring of eight

- Lane **public-functional**. Status **research-only**. Maturity
  **live-validated**: each profile installed once and passed the installer's
  checks; single-run timing; not serving-qualified.
- Hardware: the eight-Spark ring, recorded by `sparkring setup --adopt`;
  package `498a7bc4` on every Spark; image `1a8c10354eb0` by its
  configuration ID with `--image-lock` and its recorded lock. GPU clocks not
  locked.
- Result: `glm53-nvfp4-tp8` on all eight Sparks (install 661 s, API ready
  after 392.5 s, 6 functional checks passed; the image check does not
  apply); `glm53-flash-nvfp4-spark-tp4` on four (919 s) and
  `glm53-flash-nvfp4-spark-tp2` on two (1,040 s), both on the CSF checkpoint;
  and
  `qwen38-flash-next-tp2` on two (775 s); the three smaller deployments
  served at once. Each passed `sparkring check`'s functional checks and
  answered three known-answer questions. The transport check passed for the
  three smaller deployments; for `glm53-nvfp4-tp8`, package `498a7bc4`
  reported "differs" because its sessions took SIRCL's built-in cycle-8 plan,
  and revision `00f95601` accepts that plan and judges the receipts as
  expected.
- Record: [installer runs on the ring of eight](../../../performance/records/images/dev-20261009-kraken-csf-sircl032-libsircl-plugins-installer-ring8-20261009.md).

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

- At 32K context, the release notes' speed table takes the two-Spark rows
  from these runs: GLM-5.3-Flash CSF 39.5 / 87.1 at 1 / 4 streams and
  Qwen3.8-Flash-Next 51.1 / 117.1 / 179.0 at 1 / 4 / 8 streams, output
  tokens/s, medians of the two starts. CSF's 8-stream cell, 123.2 tok/s, is
  the two repeats of that cell on a pair of a ring of four with image
  `1a8c10354eb0` whose readiness gate opened before an outside request
  arrived (123.2 and 123.3 tok/s; [ring-B checks](../../../performance/records/images/dev-20261009-kraken-csf-sircl032-libsircl-plugins-ring-b-checks-20261010.md)). In these runs
  that cell measured 130.4 and 66.2 tok/s, the second an underfilled
  measurement.
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

### libsircl of image 1a8c10354eb0 on the cycle of eight Sparks

- Lane **public-functional**. Status **implemented**. Maturity
  **live-validated** with nccl-tests v2.21.1 and libsircl's bit-exact check
  and timing sweep; no serving measurement.
- Hardware: the eight-Spark ring as a cycle of eight; one rank per Spark.
- Configuration: the image's `libsircl.so.0.6.0` (SHA-256 `8b180879…`, source
  tree `dbf36074`) on every rank; no NCCL arm, so NCCL's figures are those of
  the `a3477af2` gate above.
- Result: every verdict passed. The bit-exact check and every collective and
  point-to-point line are exact, and the two arms that must be refused were.
  Every eight-rank communicator without a schedule setting took the cycle
  plan, and from 8 MiB the default all-reduce ran within 1.15 times the ring
  schedules. All-reduce of 256 MiB: 19.3 ms in the timing sweep and in
  nccl-tests (24.3 GB/s bus bandwidth) against NCCL's 23.6 GB/s; 4 KiB:
  18.2 µs in the timing sweep, 29.1 µs in nccl-tests against NCCL's 120.3 µs.
  Broadcast stays at 1.9 GB/s.
- Record: [libsircl of image 1a8c10354eb0 on eight Sparks](../../../performance/records/transport/libsircl-ring8-image-1a8c10354eb0-20261009.md).
- The release image's libsircl, tree `030419b8`, is this library with one
  change: a communicator created on a thread that has no current CUDA context
  uses the current device's primary context. vLLM's PyNccl creates its
  communicator that way, so on image `1a8c10354eb0` `--transport libsircl`
  failed vLLM's initialization at TP2 with `NCCL error: invalid usage`. A
  research build of the change, in a container of that image, served the same
  deployment on a cabled pair: vLLM loaded, captured its CUDA graphs and
  answered two known-answer requests
  ([libsircl status](../../../spark_transport/libsircl/STATUS.md)). The
  release image's own requalification is the `release-image-requal` gate.

### Two rings of four Sparks: setup, SIRCL tune and libsircl

- Lane **public-functional**. Status **implemented**. Maturity
  **live-validated** for the setups and the libsircl gate; the installer
  scenarios and TP4 serving measurements below are research-only.
- Hardware: the eight Sparks recabled as two independent rings of four
  (ring A and ring B), one rank per Spark. Ring B's positions 0 and 1 ran GPU
  driver 580.178.04 with kernel 7.0.0-1019, its positions 2 and 3 driver
  580.173.02 with kernel 6.17.0-1029.
- Setup: `sudo sparkring setup --re-form` finished with exit status 0 on both
  rings, with `Fabric verified: 4 cables on 4 Sparks (cycle-4)`; cables
  measured 212.7 to 213.4 Gb/s. The package held the re-form fixes of commit
  `e2590e79`.
- SIRCL tune: the ring harness's quick tune of the `cycle-4` group shape
  passed on both rings, 1,102 and 1,098 cases, every output exact. The
  default table's measured `cycle-4` row merges the two rings' tunes
  ([SIRCL tune record](../../../performance/records/transport/sircl-cycle4-tune-two-rings-20261009.md)).
- libsircl of image `1a8c10354eb0` on both rings at once: byte checks, the
  bit-exact check and nccl-tests passed; `on-budget` passed, as the planner
  expects on a ring of four; the cycle plan held from 8 MiB on 24 of 24
  comparisons per ring. All-reduce of 256 MiB: 16.55 ms, 24.3 GB/s bus
  bandwidth, on both rings, which agree within 1 %. Broadcast reaches
  5.3 GB/s.
- TP4 serving on a ring of four, research-only: `sparkring install` of
  `glm53-flash-nvfp4-spark-tp4` (CSF), `deepseek-v41-flash-tp4` and
  `qwen38-flash-next-qad-tp4` (step 5500) on image `1a8c10354eb0`, GPU clocks
  locked at 2,418 MHz, the default table's `cycle-4` row, each deployment
  measured once through its API by the serving A/B runner. Against the same
  profiles on a path of four Sparks of the ring of eight (image
  `816c6d6a7e96`, SIRCL 0.3.0), on the cells both measured (0K and 32K):

  | Install | Output tok/s, median (range) | Accepted tokens per step, largest difference |
  |---|---|---|
  | GLM-5.3-Flash CSF | -0.8 % (-3.6 % to +0.7 %) | 0.08 |
  | DeepSeek-V4.1-Flash | +0.9 % (-4.9 % to +4.8 %) | 0.07 |
  | Qwen3.8-Flash-Next | -8.9 % (-16.1 % to -1.4 %) | 0.10 |

  CSF and DeepSeek-V4.1-Flash decode on the ring of four within the spread
  of the path of four. Qwen3.8-Flash-Next's 8.9 % gap is the checkpoint: on
  the same ring, `qad-step5500-mxfp8-attention` decodes +6.3 % and +7.8 %
  over the default `qad-step5500-ple1000`, and on the path of four's
  checkpoint the ring of four is -2.3 % and -2.6 %; a fresh installation
  repeats within -2.2 % and +0.7 % in the median
  ([second starts and Qwen control](../../../performance/records/images/dev-20261009-kraken-csf-sircl032-libsircl-plugins-tp4-starts-qwen-control-20261010.md)). Each comparison also
  changes the image, the SIRCL build and the transport settings. The release notes' speed table takes its
  four-Spark rows from these installs at 32K context, its eight-Spark row from
  [GLM-5.3 on eight Sparks](#glm-53-on-eight-sparks-two-token-mtp) and its
  two-Spark rows from
  [two and four Sparks](#two-and-four-sparks-on-sircl-ring-sessions).
- Installer scenarios, research-only, GPU clocks not locked, one run each:
  with image `1a8c10354eb0`'s lock, `sparkring install` installed GLM-5.3-Flash CSF at
  TP4, switched it to Qwen3.8-Flash-Next and back, ran GLM-5.3-Flash CSF and
  Qwen3.8-Flash-Next on the two pairs of one ring at once (ring A), and
  installed DeepSeek-V4.1-Flash and Qwen3.8-Flash-Next at TP4 (ring B). Each
  of these seven installations passed `sparkring check`'s seven checks and
  three known-answer questions, on SIRCL ring sessions with NCCL absent; the
  four-Spark sessions took the measured `cycle-4` row and the pairs the
  `pair` row. `--transport libsircl` on a pair of ring B failed vLLM's
  initialization with that image's libsircl (above). Six plan-only refusals
  and plans behaved as expected. Ring B's further checks passed or are
  stated in their record: the prepared and nccl transports, a repeated
  installation of a serving deployment, down and up, a killed worker
  container and its recovery, `fabric verify` and the `--image 2026.10.1`
  rollback on a pair ([ring-B checks](../../../performance/records/images/dev-20261009-kraken-csf-sircl032-libsircl-plugins-ring-b-checks-20261010.md)).
- Against 2026.10.1 on the same checkpoints, TP4 on ring A, one start per
  row: GLM-5.3-Flash NVFP4-Spark decodes +3.7 % and Qwen3.8-Flash-Next
  `qad-step5500-ple1000` +2.1 % and +3.3 % in the median, with the first
  token of a 32K prompt within 2 %; GLM-5.3-Flash on its default `csf`
  checkpoint decodes +7.8 % and +8.8 % against 2026.10.1's NVFP4-Spark
  ([image 1a8c1035 against 2026.10.1](../../../performance/records/images/dev-20261009-kraken-csf-sircl032-libsircl-plugins-vs-2026.10.1-tp4-20261010.md)).
- The measured `cycle-4` table against SIRCL's own rules, GLM-5.3-Flash CSF
  TP4 on ring A, one start per arm: the first token of an 8K-32K prompt comes
  7.6-9.3 % sooner; decode engine steps per second move -0.5 % in the median
  ([table against rules](../../../performance/records/images/dev-20261009-kraken-csf-sircl032-libsircl-plugins-csf-tp4-tuning-row-vs-rules-20261010.md)).
- CSF correctness screens, status implemented: the CSF installations of
  `glm53-flash-nvfp4-spark-tp4` on ring A and `glm53-flash-nvfp4-spark-tp2`
  on a pair of ring B, serving at the same time, each passed the acceptance
  harness's 7 functional checks and returned 256 responses of its
  correctness screen with none degenerate, wrong or failed. A 20-minute soak
  of the TP4 installation served 2,112 requests of the screen's mix with no
  error, degenerate or wrong reply, flat memory and engine step rates within
  3.4 % of their median: a short soak, not qualification ([soak](../../../performance/records/images/dev-20261009-kraken-csf-sircl032-libsircl-plugins-csf-tp4-soak-20261010.md)).
- Records: [setup and SIRCL tune](../../../performance/records/transport/ring4-setup-and-sircl-tune-20261010.md),
  [libsircl on two rings of four](../../../performance/records/transport/libsircl-ring4-image-1a8c10354eb0-20261010.md),
  [installer scenarios](../../../performance/records/images/dev-20261009-kraken-csf-sircl032-libsircl-plugins-installer-ring4-20261010.md),
  [CSF correctness screens](../../../performance/records/images/dev-20261009-kraken-csf-sircl032-libsircl-plugins-csf-screens-20261010.md),
  [TP4 on a ring of four](../../../performance/records/images/dev-20261009-kraken-csf-sircl032-libsircl-plugins-tp4-cycle4-20261010.md),
  [second starts and Qwen control](../../../performance/records/images/dev-20261009-kraken-csf-sircl032-libsircl-plugins-tp4-starts-qwen-control-20261010.md),
  [image 1a8c1035 against 2026.10.1](../../../performance/records/images/dev-20261009-kraken-csf-sircl032-libsircl-plugins-vs-2026.10.1-tp4-20261010.md),
  [table against rules](../../../performance/records/images/dev-20261009-kraken-csf-sircl032-libsircl-plugins-csf-tp4-tuning-row-vs-rules-20261010.md),
  [soak](../../../performance/records/images/dev-20261009-kraken-csf-sircl032-libsircl-plugins-csf-tp4-soak-20261010.md),
  [ring-B checks](../../../performance/records/images/dev-20261009-kraken-csf-sircl032-libsircl-plugins-ring-b-checks-20261010.md).

### Clean-room acceptance audit

- Verdicts: SIRCL 0.3.2 (`spark_transport/sircl` at `1273183e`, the image's
  tree) PASS with 0 lines to rewrite and 522 counted idioms; the three GLM-5.3
  plugins PASS at `cf478504` (`glm_dcp_decode_comm` 2.0.1 differs from the
  audited tree only in its version strings); libsircl 0.6.0 at `c7c35fe0`
  (tree `dbf36074`, image `1a8c10354eb0`'s) PASS with 0 lines to rewrite,
  and its vLLM plugin `integrations/vllm/libsircl` PASS. The report stays
  outside the repository because it quotes excluded-origin text. The release
  image's libsircl, tree `030419b8`, adds the current-device change
  (commit `c4f3a1af`), which the audit did not cover: gate
  `clean-room-audit-030419b8`.
- Record: [clean-room acceptance audit](../../../performance/records/repository/clean-room-acceptance-audit-20261009.md).

## Qualification

| Gate | Result | Record |
|---|---|---|
| Image `d52737a109e0` built from `d3d33158` and its lock validated for its 13 profiles | passed | [image record](../../../performance/records/images/dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036-image-20261010.md) |
| `release-image-requal`: image `d52737a109e0` requalified on Sparks: its libsircl gate and installations | **pending** | — |
| `clean-room-audit-030419b8`: the clean-room acceptance audit of libsircl tree `030419b8`'s current-device change | **pending** | — |
| SIRCL one-shot all-reduce and all-gather on the eight-Spark ring | passed (status qualified) | [SIRCL status](../../../spark_transport/sircl/STATUS.md#component-status) |
| Serving A/B measurements of two, four and eight Sparks recorded | passed (status research-only) | [Evidence](#evidence) |
| `ring8-installer`: installer runs on the eight-Spark ring with image `1a8c10354eb0`'s lock | passed (status research-only) | [record](../../../performance/records/images/dev-20261009-kraken-csf-sircl032-libsircl-plugins-installer-ring8-20261009.md) |
| `ring4-setup`: `sudo sparkring setup --re-form` on each of two rings of four | passed | [record](../../../performance/records/transport/ring4-setup-and-sircl-tune-20261010.md) |
| `ring4-install`: the installer scenarios on the rings of four, image `1a8c10354eb0` | passed (status research-only); the libsircl pair failed on that image's libsircl | [record](../../../performance/records/images/dev-20261009-kraken-csf-sircl032-libsircl-plugins-installer-ring4-20261010.md) |
| `ring4-benchmark`: in-place TP4 measurements on a ring of four of GLM-5.3-Flash CSF, DeepSeek-V4.1-Flash and Qwen3.8-Flash-Next; CSF and DeepSeek within the spread of a path of four, Qwen's 8.9 % gap its checkpoint | passed (status research-only) | [record](../../../performance/records/images/dev-20261009-kraken-csf-sircl032-libsircl-plugins-tp4-cycle4-20261010.md) |
| `ring4-sircl-tune`: SIRCL's ring-harness tune of the `cycle-4` group shape on both rings | passed | [record](../../../performance/records/transport/ring4-setup-and-sircl-tune-20261010.md) |
| `cycle4-row-promotion`: the measured `cycle-4` row in the default tuning table | passed | [record](../../../performance/records/transport/sircl-cycle4-tune-two-rings-20261009.md) |
| `ring4-libsircl-gate`: the libsircl gate on a cycle of four | passed | [record](../../../performance/records/transport/libsircl-ring4-image-1a8c10354eb0-20261010.md) |
| `glm53-tp8-release-image`: GLM-5.3 TP8 installed on image `1a8c10354eb0` | passed (status research-only) | [record](../../../performance/records/images/dev-20261009-kraken-csf-sircl032-libsircl-plugins-installer-ring8-20261009.md) |
| `libsircl-ring8-release-image`: the libsircl gate on image `1a8c10354eb0` | passed | [record](../../../performance/records/transport/libsircl-ring8-image-1a8c10354eb0-20261009.md) |
| `clean-room-audit`: the clean-room acceptance audit | passed | [record](../../../performance/records/repository/clean-room-acceptance-audit-20261009.md) |
| `relay-marker`: the relay marker binary of source `c64c74535c06`, compiled on an arm64 Spark with `gcc` and `libibverbs-dev` by `python3 scripts/build_deb.py --relay-marker require` ([release procedure](../../../docs/development/releases.md)), published as the release asset `sparkring-relay-marker-c64c74535c06-arm64` and named with its SHA-256 and URL in [relay-marker-artifact.json](../../../spark_transport/fabric/relay-marker-artifact.json), which names none; package builds without a compiler, `install.sh` among them, stop until it does. No arm64 compiler is available offline, and the release build compiles again and refuses a binary that differs from the recorded one | **pending** | — |
| `compose-v3`: Compose rendering and the Install Builder on this release's lock: `compose.runtime_lock` renders a v3 lock on its v2 fields, and the page's Default card is the checkpoint `sparkring install` installs on the image (`install_default`). With this release's publication records and a stand-in digest in a scratch checkout, `python scripts/generate_compose_builder.py --verify --cases 20` passed every case of 15 images | passed | [compose.py](../../common/compose.py), [compose-builder.md](../../../docs/operations/compose-builder.md#model-and-checkpoint) |
| `publication`: registry push, anonymous manifest and configuration check, and the publication records | **pending** | — |

The image's lock records the default tuning table of its build
(`tuning_defaults_sha256` `c6f82805…`), which has no `cycle-4` row.
Installations apply the table of the installing package, which carries the
row ([release procedure](../../../docs/development/releases.md)).

## Known limitations

- `--transport libsircl` is research-only: no serving measurement uses it,
  and it runs without decode-context parallelism. The installer does not
  judge its receipts (verdict `unknown`), so `sparkring check` reports
  `ok: false` for an installed libsircl deployment.
- libsircl's broadcast on a cycle of eight reaches 1.9 GB/s against NVIDIA
  NCCL's 24.2 GB/s: it has no ring broadcast (unsupported). The default
  `sircl` transport does not use libsircl.
- The GLM-5.3-Flash CSF checkpoint, the GLM-5.3-Flash profiles' default on
  this image, is implemented, not qualified: its installations passed the
  installer's checks and a 256-request correctness screen at TP4 and TP2,
  and one 20-minute soak at TP4; serving qualification needs more.
- The eight-Spark profiles are research-only; `glm53-nvfp4-tp8` has one
  installation that passed the installer's checks.
- MiMo-V2.6-Flash-MOPD and Swift-1.5 profiles run on SIRCL sessions with no
  SIRCL serving record in the repository.
- The `cycle-4` row comes from one quick tune per ring with GPU clocks not
  locked; one serving comparison against SIRCL's own rules, one start per
  arm, measured shorter prefill and unchanged decode.
- The serving A/B measurements locked the GPU clocks, which the installer
  does not do; the installer runs and the `cycle-4` tunes did not lock them.

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
- this release in the `releases` of its final layer's builder,
  `installer-derived-layer` (`derived_layer.py`), in
  [builders.json](../../images/builders.json);
- the relay marker binary's SHA-256 and release-asset URL in
  [relay-marker-artifact.json](../../../spark_transport/fabric/relay-marker-artifact.json);
- `"2026.10.2": "dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036"` in
  [installer-releases.json](../installer-releases.json).

Listing the release in installer-releases.json makes it the image of every
`sparkring install` without `--image` from a package built from this tree,
with SIRCL as its transport where the fabric carries it. GitHub release
`2026.10.2` is a pre-release on tag `2026.10.2`; `main`, packages built from
it and the Install Builder keep 2026.10.1 until this tree merges into
`main`. The rollback is `--image 2026.10.1`, the prepared transport on the
parent image; its deployments keep their site configuration.
