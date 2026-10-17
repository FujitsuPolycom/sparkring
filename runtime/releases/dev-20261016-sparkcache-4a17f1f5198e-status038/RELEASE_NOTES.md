# SparkRing 2026.10.5

SparkRing's profiles run three models created by Local Inference Lab, Inc. and
licensed under the Local Inference Lab License, Version 1.0:
[GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD)
(`glm53-flash-tp2`, `glm53-flash-tp4`, `glm53-flash-nvfp4-spark-tp2-dcp2`,
`glm53-flash-nvfp4-sparkcache-crc32-tp2`, `glm53-flash-nvfp4-sparkcache-crc32-tp4`),
[GLM-5.3-NVFP4-CSF](https://huggingface.co/local-inference-lab/GLM-5.3-NVFP4-CSF)
(`glm53-nvfp4-csf-tp6`) and
[Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD)
(`qwen38-flash-next-tp2`, `qwen38-flash-next-csf-sparkcache-tp2`).

Packages built from tag `2026.10.5` install this release's image by default.
The
[release record](https://github.com/FujitsuPolycom/sparkring/blob/2026.10.5/runtime/releases/dev-20261016-sparkcache-4a17f1f5198e-status038/README.md)
has the evidence and what remains open.

## What's in it

Installer image `dev-20261016-sparkcache-4a17f1f5198e-status038`
(`ghcr.io/fujitsupolycom/sparkring@sha256:ac313f040fe3e04238ad8ff152cf7fb427a050f40e0848c2e326c5b6b8717dfd`):
2026.10.4's image plus SparkCache, a persistent KV cache on each Spark's
disk, for three models. A conversation whose prefix was served before is
restored from the cache instead of recomputed.

- `deepseek-v41-flash-sparkcache-tp4` (research-only): DeepSeek-V4.1-Flash on
  four Sparks with SparkCache.
- `qwen38-flash-next-step5500-sparkcache-tp4` and
  `qwen38-flash-next-csf-sparkcache-tp2` (research-only): Qwen3.8-Flash-Next on
  four and two Sparks with SparkCache.
- `glm53-flash-nvfp4-sparkcache-crc32-tp2` and
  `glm53-flash-nvfp4-sparkcache-crc32-tp4` (research-only): GLM-5.3-Flash CSF
  on two and four Sparks with SparkCache.
- Options for them: `--sparkcache-size` (per Spark, default 32G),
  `--sparkcache-integrity fast|sha256|none` and `--sparkcache-debug`, kept
  through restart and recovery. The cache survives a corrupted rank: damaged
  entries are dropped and saved again.

Every 2026.10.4 profile is admitted on this image with its 2026.10.4 settings.
`deepseek-v41-flash-tp4` offers DeepSeek's 2026-10-17 weights as
`--checkpoint 1017` (`deepseek-ai/DeepSeek-V4.1-Flash-1017`, served as
`DeepSeek-V4.1-Flash-1017-TP4`); without the option it installs the original
checkpoint as before. Contributed by Sabatino Masala with a hardware record
on a four-Spark ring.

Installer and dashboard, in the package:

- `sparkring setup` without `--name` keeps the cluster's name.
- Installations on disjoint Sparks run at the same time.
- `sparkring status` names the serving profiles and their Sparks.
- The dashboard shows the output rate per running request, has a light and
  dark theme switch, and each model's Logs panel shows a follow command.
- A deployment starts again after `sparkring setup` rewrote the fabric
  document: each Spark accepts a copy that records the cabling the
  deployment was made on.
- The dashboard keeps its page when a stopped record names a checkpoint the
  installed package does not list.
- The Install Builder offers the five SparkCache profiles as install-command
  cards with the cache size and payload-check fields.

## Tested

On a ring of eight DGX Sparks, with `sparkring install`:

- `deepseek-v41-flash-sparkcache-tp4` on four Sparks and
  `qwen38-flash-next-step5500-sparkcache-tp4` on the other four: installed at
  the same time and served; a saved conversation was restored from the cache
  after a restart.
- On the same composition's recheck build: three installations on disjoint
  Sparks at once, and cache recovery after one Spark's cache files were
  corrupted, for DeepSeek (fast integrity) and Qwen (sha256).
- Each of the five SparkCache profiles with the default payload check:
  installed, a session saved and restored after a restart, and recovered
  after one Spark's cache files were corrupted; `deepseek-v41-flash-tp4` also
  installed and answered.
- The restart-after-setup fix and the dashboard fix, on a deployment that
  served DeepSeek-V4.1-Flash-1017 with the SparkCache TP4 profile.

The other profiles passed the installer's image checks but were not started
on this image. No speed was measured on it.

## Try it

On Node A:

```bash
curl -fsSL https://raw.githubusercontent.com/FujitsuPolycom/sparkring/2026.10.5/install.sh | bash -s -- --ref 2026.10.5 --package-only
sudo sparkring install --profile PROFILE --plan
sudo sparkring install --profile PROFILE
sudo sparkring check --report ~/sparkring-report
```

A Spark that holds the 2026.10.4 image downloads only the layer this image
adds, 692 KiB.

## Go back to 2026.10.4

```bash
sudo sparkring install --profile PROFILE --image 2026.10.4
```

That is the parent image on the same transports
([another image](https://github.com/FujitsuPolycom/sparkring/blob/2026.10.5/docs/operations/install-reference.md#another-image)).
