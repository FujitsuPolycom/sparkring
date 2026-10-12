# SparkRing 2026.10.3

SparkRing's `glm53-flash-tp2` and `glm53-flash-tp4` profiles run
GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD, created by Local Inference Lab, Inc., a
non-profit organization, available at
https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD.
SparkRing's `glm53-nvfp4-csf-tp8` profile runs GLM-5.3-NVFP4-CSF, created by
Local Inference Lab, Inc., a non-profit organization, available at
https://huggingface.co/local-inference-lab/GLM-5.3-NVFP4-CSF.
SparkRing's `qwen38-flash-next-tp2` profile runs
Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD, created by Local Inference Lab, Inc.,
a non-profit organization, available at
https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD.
GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD, GLM-5.3-NVFP4-CSF and
Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD are licensed under the Local Inference
Lab License, Version 1.0.

Pre-release for testing. It uses 2026.10.2's image unchanged
(`ghcr.io/fujitsupolycom/sparkring@sha256:4fffc4dc3074d5539f4e9d3a013ff9ef4e0be570a95b74d4646ee341da1f6911`),
so a Spark that holds that image downloads nothing. The
[release record](https://github.com/FujitsuPolycom/sparkring/blob/2026.10.3/runtime/releases/2026.10.3/README.md)
has the evidence and the open gates.

## What's in it

- `sparkring dashboard`: a read-only status page on Node A, port 8090, and
  `sparkring dashboard tui` for a terminal.
- `--on` places the serving image only on the Sparks it names. Models serving
  on the other Sparks get no image write.
- Automatic recovery restarts a four-Spark model on part of a ring of eight.
- The installer refuses to start a model whose GPU memory share plus
  SparkCache's pinned host memory leaves a Spark less than 6 GiB for the
  system, and says by how much.
- `qwen38-flash-next-tp2` installs Local Inference Lab's Qwen3.8-Flash-Next
  CSF checkpoint on this release's image (research-only);
  `--checkpoint qad-step5500-ple1000` installs step 5500.
- Research-only profiles: `glm53-nvfp4-csf-tp8`, GLM-5.3-NVFP4-CSF on eight
  Sparks, which installs only with `--image-lock FILE` and is parked, and
  SparkCache variants of the GLM-5.3-Flash profiles, which need an image this
  release does not publish.

## Profiles

`--profile` values: `qwen38-flash-next-tp2`, `qwen38-flash-next-tp4`,
`glm53-flash-tp2`, `glm53-flash-tp4`, `glm53-nvfp4-tp8`,
`mimo-v26-flash-mopd-tp2`, `mimo-v26-flash-mopd-tp4`, `deepseek-v41-flash-tp4`,
`swift15-qwen38-flash-next-tp2` and `swift15-qwen38-flash-next-tp4`.
`qwen38-flash-next-tp4` is the short name of `qwen38-flash-next-qad-tp4`, and
`glm53-flash-tp2` and `glm53-flash-tp4` of `glm53-flash-nvfp4-spark-tp2` and
`glm53-flash-nvfp4-spark-tp4`.

## Tested

- On the ring of eight with this image, one start each:
  `qwen38-flash-next-tp2 --checkpoint csf` loaded, served and answered
  correctly; `glm53-nvfp4-csf-tp8` started and served; the SparkCache variant
  on four Sparks passed every step it ran of its test plan's first stage (one
  step and the two-start comparison did not run). The
  dashboard's page, terminal view, loopback address and reload worked.
- The `--on` image placement and the recovery change are tested offline only.
- CPU checks pass, apart from tests that also fail on 2026.10.2 on Windows.

## Known issues

- 2026.10.2's: installed libsircl deployments report `ok: false` in
  `sparkring check`, and libsircl's broadcast is slow.
- The SparkCache variant on two Sparks made both Sparks unreachable in its
  first start, with larger capture slots than this release gives it. Its
  smaller slots have not run yet.

## Try it

On Node A:

```bash
curl -fsSL https://raw.githubusercontent.com/FujitsuPolycom/sparkring/2026.10.3/install.sh | bash -s -- --ref 2026.10.3 --package-only
sudo sparkring install --profile PROFILE --plan
sudo sparkring install --profile PROFILE
sudo sparkring check --report ~/sparkring-report
```

## Go back to 2026.10.2

Reinstall 2026.10.2's package; the image is the same:

```bash
curl -fsSL https://raw.githubusercontent.com/FujitsuPolycom/sparkring/2026.10.2/install.sh | bash -s -- --ref 2026.10.2 --package-only
```
