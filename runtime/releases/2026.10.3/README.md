# Release 2026.10.3: installer and profile changes on release 2026.10.2's image

SparkRing's profiles run three models created by Local Inference Lab, Inc. and
licensed under the Local Inference Lab License, Version 1.0:
[GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD)
(`glm53-flash-tp2`, `glm53-flash-tp4`),
[GLM-5.3-NVFP4-CSF](https://huggingface.co/local-inference-lab/GLM-5.3-NVFP4-CSF)
(`glm53-nvfp4-csf-tp8`) and
[Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD)
(`qwen38-flash-next-tp2`).

GitHub release `2026.10.3` carries SparkRing package changes that need no new
container image. It installs installer image release
`dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036`, the image
that release 2026.10.2 published; that image's
[release record](../dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036/README.md)
describes its layers, evidence and publication. This page is the record of
release 2026.10.3. [RELEASE_NOTES.md](RELEASE_NOTES.md) is the summary for
users, and [status.json](status.json) states the same status for programs.

Status: **implemented**; release maturity **accepted**: GitHub release
`2026.10.3` is the latest release, and `main` installs it. Its image is release
2026.10.2's, already published. The serving-defaults qualification gate stays
open ([open gates](#qualification)).

## Identity

| Item | Value |
|---|---|
| GitHub release tag | `2026.10.3`, the latest release; `2026.10.2` (a pre-release) and `2026.10.1` (the latest release before it) precede it |
| Image | Release 2026.10.2's, unchanged: image configuration ID `sha256:d52737a109e083d3eef05c0fc0db4d09fc1bf34485a13467382130963ecfe66b`, registry reference `ghcr.io/fujitsupolycom/sparkring@sha256:4fffc4dc3074d5539f4e9d3a013ff9ef4e0be570a95b74d4646ee341da1f6911`. A Spark that holds it downloads nothing |
| Lock | Release 2026.10.2's, unchanged: [installer-image.json](../dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036/installer-image.json), SHA-256 `d58cbfefedd47bd861d634c9337aa388b284c8759664d5e10580015a28ac8f79`, 13 profiles, which that directory's `release.json` pins. That directory's lock, `release.json`, `publication.json`, `status.json` and `RELEASE_NOTES.md` are byte-identical to release 2026.10.2's; its README adds the Local Inference Lab attribution notice after its heading |
| Image sources | The sources the image carries are byte-identical to release 2026.10.2's: `spark_transport/libsircl` is git tree `030419b8a2b61acc1a74010308a1c3e758863e2f`, the lock's `libsircl.source_tree`; `spark_transport/sircl` is tree `8a54c1abf591`, `integrations/vllm/runtime_status` `de6358a86f8c`, and the GLM-5.3 plugin directories are unchanged |
| Relay marker | Source `c64c74535c06` is unchanged, so package builds take the binary that [relay-marker-artifact.json](../../../spark_transport/fabric/relay-marker-artifact.json) records |
| Release-tag index | [installer-releases.json](../installer-releases.json) names no `2026.10.3`: this release publishes no image. Without `--image`, or with `--image 2026.10.2`, the package installs this image |
| Rollback | Release 2026.10.2's package on the same image; release 2026.10.1's image on the prepared transport with `--image 2026.10.1` |

## Changes

| Change | Status | Where |
|---|---|---|
| `sparkring dashboard`, a read-only ring status page on Node A (port 8090, its LAN address and 127.0.0.1) that sends no inference request and opens no SSH session; `sparkring dashboard tui`; the dashboard's quiet mode during benchmark load; placement and pacing settings of the host agent, the recovery timer and the tunnel check | implemented | [dashboard](../../../docs/operations/dashboard.md), `runtime/host/dashboard.py` |
| An install with `--on` places the serving image only on the arc's Sparks; a Spark outside the arc writes nothing and forwards only when no way to an arc Spark avoids it | implemented | [image distribution](../../../docs/operations/install-reference.md#spreading-along-the-cables), `runtime/host/spread.py`, `runtime/host/install_assets.py` |
| Automatic recovery judges a four-rank group on a fabric other than a four-Spark ring by its containers and API, without the mesh reports that only a four-Spark ring's Sparks give | implemented | [automatic recovery](../../../docs/operations/install-reference.md#automatic-recovery), `runtime/host/recovery.py` |
| The installer's memory settle refuses a start whose vLLM share plus a SparkCache connector's pinned host memory (capture slots and restore arenas) exceeds a Spark's total less 6 GiB, naming the numbers | implemented | [Memory](../../../docs/operations/install-reference.md#what-a-run-does), `runtime/host/memory_settle.py` |
| `qwen38-flash-next-tp4` and `qwen38-flash-next-tp8` select `qwen38-flash-next-qad-tp4` and `qwen38-flash-next-qad-tp8` | implemented | `runtime/host/models.py` |
| `qwen38-flash-next-tp2` installs local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD at `dc5b8ab2fa14` without `--checkpoint` on an image whose vLLM reads the nvfp4_csf format (this one), and step 5500 on every other image; `--checkpoint qad-step5500-ple1000` installs step 5500 on any image | research-only | [install reference](../../../docs/operations/install-reference.md), `profiles/qwen38-flash-next-tp2/config.json`, `runtime/common/image_lock.py` |
| Every installer profile states `--async-scheduling`; the nine that did not, `deepseek-v41-flash-tp4` and `-tp8`, `mimo-v26-flash-mopd-tp2` and `-tp4`, `qwen38-flash-next-tp2`, `qwen38-flash-next-qad-tp4` and `-qad-tp8`, and `swift15-qwen38-flash-next-tp2` and `-tp4`, gain it. vLLM already schedules every one of them asynchronously, so the flag changes no resolved setting and no speed is claimed for it | implemented | [serving defaults](#serving-defaults), [catalog](../../../performance/enhancements.md#async-scheduling) |
| `glm53-flash-tp2` and `glm53-flash-tp4` set `B12X_W4A16_A4_PREFILL_MIN_TOKENS=1536`: on the `csf` checkpoint's W4A16 experts, every prefill row of a MoE call outside a CUDA graph runs 4-bit activations, whatever its length. In this image's vLLM the variable is only an on/off switch, so the value sets no 1,536-token cutoff and short prefills also run 4-bit activations; decode rows, graph-captured calls and the MTP draft layer stay W4A16. The output quality of short prefills is not separately qualified | implemented; measurements research-only | [serving defaults](#serving-defaults), [catalog](../../../performance/enhancements.md#b12x-w4a16-a4-prefill) |
| `qwen38-flash-next-tp2`, `qwen38-flash-next-tp4` and `qwen38-flash-next-qad-tp8` keep the MTP draft's KV cache in the model's dtype (`"kv_cache_dtype": "auto"` in `--speculative-config`), BF16 beside the target's FP8 KV cache in 32-token blocks, and quantize the target LM head to MXFP8 at load (`VLLM_MXFP8_LM_HEAD=1`). The four-Spark profile's default checkpoint stays step 5500 | implemented; measurements research-only | [serving defaults](#serving-defaults), [catalog](../../../performance/enhancements.md#qwen38-mtp-draft-kv-auto) |
| `deepseek-v41-flash-tp4` shards its two Engram projections over the four ranks (`"projection_tp": true` in `--engram-config`); `deepseek-v41-flash-tp8` keeps them replicated | implemented; measurements research-only | [serving defaults](#serving-defaults), [catalog](../../../performance/enhancements.md#ds41-engram-projection-tp) |
| `glm53-nvfp4-csf-tp8`: GLM-5.3-NVFP4-CSF at `a4d406932f2d` on eight Sparks with every setting of `glm53-nvfp4-tp8`, served from a serving directory derived without a donor. No lock of this package admits it: it installs with `--image-lock FILE`, and the owner has parked it | research-only | [profile guide](../../../profiles/glm53-nvfp4-csf-tp8/README.md) |
| The column-split start check reads only `glm53full_speedups`' own log lines | implemented | `runtime/host/start_checks.py` |
| SparkCache variants `glm53-flash-nvfp4-sparkcache-tp2` and `-tp4` of the GLM-5.3-Flash profiles, with 64 MiB CUDA placement arenas inside a 256 MiB restore budget and capture slots of 512 MiB (TP2) and 8 GiB (TP4), and the recipe of their derived image layer, `installer-sparkcache-gab86b70`. They are listed in the research catalog. Only the lock that layer's record step writes admits them; a Spark has built the layer's image, `e1607979b7f0`, whose lock is not in this repository | research-only | [layer recipe](../../images/compositions/installer-sparkcache-gab86b70/README.md), [recipe record](../dev-20261010-sparkcache-gab86b70-plugins-status036/README.md) |
| The enhancement catalog states the serving defaults above with their measurements, and lists Local Inference Lab's launch settings that the profiles do not adopt (five GLM-5.3-Flash draft tokens; Qwen `custom_ops: ["all"]`, the value this image's vLLM already gives the profiles' own setting), the Qwen CSF checkpoint, the two-Spark GLM-5.3-Flash split KV pages, the SparkCache layer image and libsircl 0.7.0's copy-relay broadcast | implemented | [catalog](../../../performance/enhancements.md) |
| Every README and release page that names a Local Inference Lab CSF checkpoint opens with its attribution notice: the repository README, this record, its release notes, the profile catalog, the guides of `qwen38-flash-next-tp2` and its Compose recipe, `glm53-flash-csf-tp8`, `glm53-nvfp4-csf-tp8` and the GLM-5.3-Flash SparkCache variants, the CSF source layer's README, the records of image releases `dev-20261007-kraken-csf-sircl-cuda1342-nccl2323-status034` and `dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036`, and the recipe record `dev-20261010-sparkcache-gab86b70-plugins-status036`; the libsircl architecture page states the eight routing variables the installer transport carries | implemented | [README](../../../README.md), [libsircl](../../../docs/architecture/libsircl.md) |

The changes came from these commits, merged or applied on release
2026.10.2's branch head `dc0300ac`: `claude/qwen-csf-tp2` at `8e398fd4`,
`claude/dashboard-feature` at `86baf73b`, `glm/sparkcache-flash` at
`2f12d483` with `a6d1e309` from `claude/ring-test-20261010`,
`claude/glm53-csf-tp8` at `809fc4b1`, and `3e991b93` from
`claude/csf-start-check-fix`. `claude/enhancements-catalog-20261011` at
`407139c2` is merged on top of the serving defaults; the rest are commits of
`release/2026.10.3`.

## Serving defaults

Each row names a serving setting this release changes, the profiles that set
it and its measurement. Every measurement is research-only and shares these
conditions:

- eight DGX Sparks cabled as one ring of eight, with this release's image
  (`d52737a109e0`), GPU clocks locked at 2,418 MHz and outside clients paused,
  on 2026-10-11;
- the settings applied through the research-only install options of SparkRing
  package `afe1561f`, which is `5faff595` of this release's branch with those
  options added;
- four-Spark groups ran as a path of four inside the ring of eight, never as a
  ring of four, and two-Spark groups on one cabled pair of it;
- decode is aggregate output tok/s at 32K context for 1 / 4 / 8 streams unless
  a row says otherwise, and prefill is the prompt's tokens divided by the cold
  time to the first token;
- *runner* is the serving A/B runner (`performance/harnesses/serving_ab`):
  llm-inference-bench at temperature 0, at most 1,024 output tokens, 30 s
  cells.

| Setting | Profiles | Measurement | Conditions |
|---|---|---|---|
| `--async-scheduling`, stated explicitly | every installer profile | Nothing to claim. In the image's vLLM (`vllm/config/vllm.py`, lines 1725-1774), an unset `async_scheduling` turns on for MTP, DFlash and DSpark speculation on the `mp` executor, which every profile uses, so the flag resolves to vLLM's default. A `deepseek-v41-flash-tp4` start and a `qwen38-flash-next-tp4` start with the flag resolved to the same settings as starts without it, and their speed differences are start-to-start spread | inferred from vLLM's source; vLLM does not log the resolved value |
| `B12X_W4A16_A4_PREFILL_MIN_TOKENS=1536` | `glm53-flash-tp4` | 32K prefill 3,334 against 3,004 tok/s for the matched start without it (+11 %); campaign arms that also change the draft head or the top-k weight precision do not isolate it; first token 10-11 % sooner at 8K, 16K and 32K; decode 61.7 / 125.0 / 213.6 against 62.7 / 122.6 / 206.5, within the start-to-start spread. Prompt log-probabilities move: rank-1 agreement 0.983 with the start without it, against 0.992 between two starts without it | one start each, runner (llm-inference-bench 0.4.29), `csf`, Sparks 0-3 |
| `B12X_W4A16_A4_PREFILL_MIN_TOKENS=1536` | `glm53-flash-tp2` | 32K prefill 2,319 tok/s; decode 38.7 / 84.1 / 123.9. No start without it ran on this image: the README row, 2,138 tok/s and 39.5 / 87.1 / 123.2, ran on other images | one start, runner (0.4.29), `csf`, Sparks 4-5 |
| MTP draft KV cache `auto`: BF16 beside the target's FP8 KV cache (unset, the draft inherits FP8) | `qwen38-flash-next-tp2`, `qwen38-flash-next-tp4`, `qwen38-flash-next-qad-tp8` | Four Sparks: 81.6 / 185.9 / 234.0 against 60.9-62.8 / 148.6-149.9 / 237.2 for two starts without it, from 2.52-2.57 instead of 1.88-2.08 accepted tokens per step; prefill unchanged | one start, runner (0.4.29), `--checkpoint qad-step5500-mxfp8-attention`, Sparks 4-7 |
| Target LM head quantized to MXFP8 at load (`VLLM_MXFP8_LM_HEAD=1`) | the same three | Four Sparks: 64.7 / 153.4 / 231.9 against the same two starts; prefill equal | one start, runner (0.4.29), the same checkpoint and Sparks |
| Both Qwen settings with the `csf` checkpoint, the profile's full configuration | `qwen38-flash-next-tp2` | Runner: 64.0 / 130.4 / 213.7, 2.56 accepted tokens per step at one stream, 32K prefill 3,696 tok/s. llm-inference-bench 0.7.6 at temperature 1 (at most 2,048 output tokens, 20 s cells, 5 s warm-up): 71.4 / 110.2 / 164.8 / 229.3 at 1 / 2 / 4 / 8 streams, 32K prefill 3,920 tok/s, coding-peak median 93.8 tok/s | one start, Sparks 6-7; the temperature-1 run's last cells, at 128K, overlapped another client and are not quoted |
| Engram projections sharded over the ranks (`"projection_tp": true` in `--engram-config`) | `deepseek-v41-flash-tp4` | Engine steps/s +5.5 % at one stream and +12 % at eight, prefill 1-2 % lower. With llm-inference-bench 0.7.6's defaults, 8K 64.6 / 213.0 / 268.8 tok/s at 1 / 8 / 16 streams against 60.5 / 198.8 / 273.3 without it, 128K 63.9 / 207.7 / 266.8 against 61.7 / 194.8 / 261.9, 32K prefill 4,332 against 4,376 tok/s; runner, 8K 64.5 / 206.4 / 273.9 against 63.6 / 199.5 / 269.7. Greedy outputs of eight fixed prompts were correct and coherent; no matched log-probability comparison with replicated projections ran. The owner approved the default on this evidence | two starts with it, each against a start with otherwise the same resolved settings; the tool's defaults are the server's default temperature and at most 8,192 output tokens; runner on llm-inference-bench 0.7.6; Sparks 0-3 |

The Qwen target KV cache stays FP8 in 32-token blocks: with the draft's KV
cache at `auto`, BF16 in 16-token blocks, the setting of Local Inference Lab's
launch catalog, decoded 20-32 % slower from 8K to 64K context at temperature 1
on two Sparks ([catalog](../../../performance/enhancements.md#qwen38-bfloat16-kv-block16)).
In a separate comparison on that pair, with BF16 target KV in 16-token
blocks, the MXFP8 LM head and the runner at temperature 0, the `csf`
checkpoint decoded 7 % faster at 32K and one stream than
`qad-step5500-mxfp8-attention` and 5 % in the median cell
([catalog](../../../performance/enhancements.md#qwen38-flash-next-csf-checkpoint)).
No comparison of the two checkpoints has run with the profile's FP8 target KV
cache or at temperature 1.

Not measured:

- No hardware start has run `--async-scheduling` on `deepseek-v41-flash-tp8`,
  `qwen38-flash-next-qad-tp8`, `swift15-qwen38-flash-next-tp2` and `-tp4`, or
  `mimo-v26-flash-mopd-tp2` and `-tp4`. vLLM's rule makes it a no-op there as
  well.
- The Qwen MXFP8 LM head and BF16 draft KV cache have run together only on two
  Sparks with `csf`: not on four Sparks, not with the four-Spark default
  checkpoint, step 5500, and not on eight Sparks, where
  `qwen38-flash-next-qad-tp8` has run neither.
- `deepseek-v41-flash-tp8` has not run with sharded Engram projections.
- Apart from DeepSeek's sharding, no changed setting has a second start, and no
  changed profile has had a correctness screen or a soak with its settings.

## Compatibility

- **Every profile:** a deployment lock records the package's source revision,
  so the first `sparkring install` of any profile with this package is a
  separate deployment from the one release 2026.10.2's package installed; it is
  prepared and then replaces the running model. Every changed profile's serving
  command differs as well, and every Compose export's `io.sparkring.deployment`
  label changes.
- **`qwen38-flash-next-tp2`:** without `--checkpoint`, on an image whose vLLM
  reads it, this release's among them, it installs
  Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD instead of step 5500: a separate
  deployment with its own 95.2 GiB checkpoint directory, downloaded when the
  Sparks lack it, and cache namespace, served as
  `Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD-TP2`. Clients that request
  `Qwen3.8-Flash-Next-NVFP4-QAD-TP2` must use that name, or the profile must
  be installed with `--checkpoint qad-step5500-ple1000`, which keeps step 5500
  and its name. On an image that reads the preferred checkpoint, the commands
  the installer suggests for repeating or switching back to a default request
  name that default (`--checkpoint qad-step5500-ple1000`, or
  `--checkpoint nvfp4-spark` for the GLM-5.3-Flash profiles). Images that cannot
  read CSF, 2026.10.1's among them, keep step 5500.
- **KV cache:** the Qwen MTP draft's KV cache is BF16, twice FP8's bytes per
  token, within the unchanged `--kv-cache-memory-bytes`; the target's KV cache
  stays FP8 in 32-token blocks.
- **Rollback:** release 2026.10.2's package on the same image restores the
  previous serving settings.

## Evidence

Eight DGX Sparks cabled as one ring of eight, with this image and SparkRing
package `46eeb18b`, 2026-10-10 and 2026-10-11 (research-only, one start each):

- `qwen38-flash-next-tp2 --checkpoint csf` on positions 6 and 7 loaded,
  served and answered the fingerprint prompts correctly; 32K decode 48.3 /
  119.6 / 171.5 tok/s at 1 / 4 / 8 streams
  ([record](../../../performance/records/images/dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036-qwen38-flash-next-tp2-csf-pair-20261011.md)).
- `glm53-nvfp4-csf-tp8` on all eight Sparks: column-split loading took effect
  through the nvfp4_csf loader, which package `46eeb18b`'s start check
  misjudged and commit `3e991b93`'s rule judges correctly; one measured start
  at `--reasoning-effort high`, so its comparison with `glm53-nvfp4-tp8` at
  the default `max` is confounded ([record](../../../performance/records/images/dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036-glm53-nvfp4-csf-tp8-ring8-20261010.md)).
- `glm53-flash-nvfp4-sparkcache-tp4` on positions 0 to 3, on the derived
  image: Stage 1 of the test plan passed steps 1-7 and 9 on every rank; step 8
  and the two-start A/B did not run; with 256 MiB placement arenas every rank
  refused its connector configuration ([record](../../../performance/records/images/dev-20261010-sparkcache-gab86b70-plugins-status036-glm53-flash-nvfp4-sparkcache-tp4-path4-20261011.md)).
- `glm53-flash-nvfp4-sparkcache-tp2` on positions 0 and 1, with two 8 GiB
  capture slots: both Sparks stopped answering during kernel compilation,
  most likely through memory exhaustion
  ([record](../../../performance/records/images/dev-20261010-sparkcache-gab86b70-plugins-status036-glm53-flash-nvfp4-sparkcache-tp4-path4-20261011.md#two-sparks-with-8-gib-capture-slots)).
  This release gives it two 512 MiB slots. With them, a private record
  (2026-10-11, package `5faff595`) reports one start on a pair of the ring
  that passed steps 1-7 and 9 on both ranks, including prefix restoration
  after a planned restart and after SIGKILL, with at least 9.52 GiB of
  `MemAvailable` ([catalog](../../../performance/enhancements.md#sparkcache-persistent-prefix)).
  No public record holds that run; step 8 and the two-start A/B did not run.
- The dashboard's page, terminal view, loopback listener and reload worked
  (reported by the ring test; no record).

The same runs showed two conditions this release corrects, tested offline
only: an install on positions 0 to 3 copied the image, 14.2 GiB, to each of
positions 4 to 7 while they served, and automatic recovery reported a TP4
group on positions 0 to 3 as `unknown` ("its mesh units are not reported"),
so it would not have restarted it ([SparkCache record](../../../performance/records/images/dev-20261010-sparkcache-gab86b70-plugins-status036-glm53-flash-nvfp4-sparkcache-tp4-path4-20261011.md)).

The serving defaults' measurements are in [Serving defaults](#serving-defaults).
On the same ring on 2026-10-11, with GPU clocks locked, a package built from
revision `4c9528fb`, which carries this release's Qwen and GLM-5.3-Flash
serving defaults but not DeepSeek's Engram sharding, installed two profiles
without research options (private record):

- `qwen38-flash-next-tp2 --on 6,7` chose `csf` without `--checkpoint`; its
  engine resolved FP8 target KV in 32-token blocks, the draft's KV cache at
  `auto`, asynchronous scheduling and compilation mode none. It completed and
  served.
- `glm53-flash-tp2 --on 4,5` chose `csf` with
  `B12X_W4A16_A4_PREFILL_MIN_TOKENS=1536` set. It completed and served.

## Qualification

| Gate | Result | Record |
|---|---|---|
| `image-lock`: release 2026.10.2's lock, unchanged, validates for its 13 profiles | passed (CPU) | `runtime/common/test_installer_image.py`, `runtime/common/test_image_lock.py` |
| `image-sources`: the image's sources are byte-identical to release 2026.10.2's | passed | [Identity](#identity) |
| `cpu-checks`: the CPU jobs of `.github/workflows/ci.yml` that run on Windows 11 with Python 3.12.10: ruff 0.15.17, the generators' `--check`, the layout, link and release-safety checks, the image composition check, the Compose and Install Builder jobs, and pytest over the tests job's paths, `runtime/images` and `spark_transport/sircl` | passed apart from tests that also fail on release 2026.10.2's head `dc0300ac` on the same host. Pytest at commit `187bf795`: 11,161 passed, 77 failed, 37 errors, 682 skipped; 111 of the 114 failing tests also fail on `dc0300ac`, and the other 3 (a timing-bound spread test, a container-cleanup test and a sweep-harness timeout) pass and fail from run to run on both | [Validation](#validation) |
| `qwen-csf-tp2`: an installation of `qwen38-flash-next-tp2 --checkpoint csf` that loads, serves and answers correctly | passed, one start (research-only) | [record](../../../performance/records/images/dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036-qwen38-flash-next-tp2-csf-pair-20261011.md) |
| `dashboard-hardware`: the dashboard on the ring | reported, not recorded | [Evidence](#evidence) |
| `arc-image`: an install with `--on` on a ring of eight leaves the Sparks outside the arc without image writes | passed: `--on 6,7` prepared its two Sparks only while GLM on 4-5 and DeepSeek on 0-3 kept serving, with no restart and no new image | [record](../../../performance/records/images/dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036-release-2026.10.3-plain-ring8-20261012.md) |
| `arc-recovery`: automatic recovery restarts a stopped four-rank group on part of a ring of eight | passed: one stopped rank of `deepseek-v41-flash-tp4` on positions 0-3; serving again 6 min 47 s after the stop, the pairs beside it untouched | [record](../../../performance/records/images/dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036-release-2026.10.3-plain-ring8-20261012.md) |
| `glm53-nvfp4-csf-tp8-install`: an installation that starts and serves, with column-split loading in effect | passed, one start (research-only); the profile is parked | [record](../../../performance/records/images/dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036-glm53-nvfp4-csf-tp8-ring8-20261010.md) |
| `sparkcache-variants`: the hardware test plan's stages | Stage 1 (four Sparks): steps 1-7 and 9 passed, step 8 and the two-start A/B not run; Stage 2 (two Sparks): failed with 8 GiB capture slots; with 512 MiB slots, steps 1-7 and 9 passed in one start that only a private record reports, step 8 and the two-start A/B not run | [record](../../../performance/records/images/dev-20261010-sparkcache-gab86b70-plugins-status036-glm53-flash-nvfp4-sparkcache-tp4-path4-20261011.md) |
| `serving-defaults-measured`: each changed serving setting measured against a start without it | partial: research-only, one start per setting and two for DeepSeek's Engram sharding; the profiles listed under [Not measured](#serving-defaults) have no start | [Serving defaults](#serving-defaults), [catalog](../../../performance/enhancements.md) |
| `serving-defaults-install`: installations of the changed profiles without research options | partial: `deepseek-v41-flash-tp4` (a path of four in a ring of eight), `glm53-flash-tp2` and `qwen38-flash-next-tp2` (`--checkpoint csf`, pairs) installed from a package of this candidate, resolved the measured settings (run receipts) and served; measured once each at temperature 1. the same three installed on two new rings of four (DeepSeek on a whole ring, the pairs on the other) with the same model settings and SIRCL's ring-of-four tuning ([record](../../../performance/records/images/dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036-release-2026.10.3-plain-ring4-20261012.md)); the other changed profiles not run | [record](../../../performance/records/images/dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036-release-2026.10.3-plain-ring8-20261012.md) |
| `serving-defaults-qualification`: a second start, a correctness screen and a soak of each changed profile with its settings | pending | |
| `publication`: tag, package built with the relay marker, GitHub release | passed: tag `2026.10.3` on this commit; GitHub release `2026.10.3`, marked latest, carries the arm64 package and the relay marker with their SHA-256 files; `main` fast-forwarded to the tag | [GitHub release](https://github.com/FujitsuPolycom/sparkring/releases/tag/2026.10.3) |

## Validation

The CPU checks ran on Windows 11 with Python 3.12.10, in full at commit
`187bf795`; the commits after it ran the generators' `--check`, the layout,
link and release-safety checks and the test files they affect. The libsircl job
(`make check`, which needs `gcc` and `nvcc`) and the pinned LIL bridge's Go
build did not run, and the SIRCL tests ran without that job's private mount
namespace. Most pytest failures on this
host come from Windows paths, line endings and process timing, and they fail
on release 2026.10.2's head too.

The commits after `5faff595` that carry the serving defaults, the catalog
merge and this record ran the checks of
[Local checks and CI](../../../docs/development/testing.md) on Ubuntu 24.04
under WSL 2, in a clone outside the worktree, with Python 3.12.3,
`requirements-dev.txt` (ruff 0.15.17, pytest 8.4.2), Node.js 18 and no torch.
The layout check, the generators' `--check`,
`check_enhancements.py --validate`, the link and release-safety checks and
ruff pass at each of them. At this record's commit, pytest of
`runtime/common` (1,678 passed), `runtime/host` (1,609 passed), `scripts`
(1,679 passed) and `performance/harnesses` (1,427 passed) fails only where
`5faff595` fails in the same environment: 2 Install Builder tests that need a
newer Node.js, and one test module that imports torch and is not collected.

CPU checks do not validate CUDA, RDMA, serving or performance.

## Known limitations

- Release 2026.10.2's limitations hold
  ([known limitations](../dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036/README.md#known-limitations)).
- The installed-profile evidence is one start per profile: on a ring of eight
  (`deepseek-v41-flash-tp4` as a path of four inside it, the two-Spark profiles as
  pairs) and on two new rings of four set up with `sparkring setup --re-form`
  ([record](../../../performance/records/images/dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036-release-2026.10.3-plain-ring4-20261012.md)).
- Engram projection sharding stays a four-Spark setting: one start of
  `deepseek-v41-flash-tp8` with it showed no consistent gain on a ring of eight
  ([record](../../../performance/records/images/dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036-release-2026.10.3-plain-ring8-20261012.md#not-adopted-engram-projection-sharding-at-tp8)).
- The serving defaults rest on research-only measurements with the limits
  that [Serving defaults](#serving-defaults) lists. The GLM-5.3-Flash
  SparkCache variants do not set their cache-off profiles'
  `B12X_W4A16_A4_PREFILL_MIN_TOKENS`, so their prefill runs W4A16.
- `B12X_W4A16_A4_PREFILL_MIN_TOKENS=1536` puts every uncaptured prefill row of
  the GLM-5.3-Flash `csf` entries on 4-bit activations, including short
  prompts and short prefill chunks: this image's vLLM uses the value only as
  an on/off switch. The output quality of short prefills with it is not
  separately qualified.
- The repository README's speed table shows measurements made on earlier
  images before these serving defaults; its two-Spark Qwen row is step 5500.
- No image this package carries runs the SparkCache variants: only the lock
  their derived layer records admits them, and that lock is not in this
  repository. They are listed in the research catalog.
- No image lock this package carries admits `glm53-nvfp4-csf-tp8`; it
  installs with `--image-lock FILE`, and the owner has parked it.
- libsircl's README and NOTICE do not state the attribution rule for a file
  that adapts NCCL source. That change (`claude/libsircl-provenance-policy`,
  commit `014dd1f6`) alters `spark_transport/libsircl`'s git tree, which this
  image's lock records, so it belongs to a release that rebuilds libsircl.
