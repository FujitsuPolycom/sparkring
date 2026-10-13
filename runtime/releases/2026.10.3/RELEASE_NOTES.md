# SparkRing 2026.10.3

SparkRing's profiles run three models created by Local Inference Lab, Inc. and
licensed under the Local Inference Lab License, Version 1.0:
[GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD)
(`glm53-flash-tp2`, `glm53-flash-tp4`),
[GLM-5.3-NVFP4-CSF](https://huggingface.co/local-inference-lab/GLM-5.3-NVFP4-CSF)
(`glm53-nvfp4-csf-tp8`) and
[Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD)
(`qwen38-flash-next-tp2`).

The latest SparkRing release: `main`, `install.sh` from `main` and the
[Install Builder](https://fujitsupolycom.github.io/sparkring/) install it. It uses
2026.10.2's image unchanged
(`ghcr.io/fujitsupolycom/sparkring@sha256:4fffc4dc3074d5539f4e9d3a013ff9ef4e0be570a95b74d4646ee341da1f6911`),
so a Spark that holds that image downloads nothing. The
[release record](https://github.com/FujitsuPolycom/sparkring/blob/2026.10.3/runtime/releases/2026.10.3/README.md)
has the evidence and the open gates.

## What's in it

- Faster serving defaults, measured research-only on a ring of eight
  ([serving defaults](https://github.com/FujitsuPolycom/sparkring/blob/2026.10.3/runtime/releases/2026.10.3/README.md#serving-defaults)):
  - Qwen keeps its MTP draft's KV cache in BF16 beside the FP8 target cache
    and quantizes its LM head to MXFP8. On two Sparks with the CSF checkpoint
    it decoded 71.4 tok/s at 32K and one stream, temperature 1.
  - GLM-5.3-Flash with the CSF checkpoint runs its prefill with 4-bit
    activations: 11 % faster 32K prefill on four Sparks. The setting,
    `B12X_W4A16_A4_PREFILL_MIN_TOKENS=1536`, is an on/off switch in this
    image, not a 1,536-token cutoff: short prompts also prefill with 4-bit
    activations, and their output quality is not separately qualified.
    Decode stays W4A16.
  - DeepSeek-V4.1-Flash on four Sparks shards its Engram projections: 5.5 %
    more decode steps per second at one stream and 12 % at eight. The owner
    approved it on two starts and correct outputs for eight prompts; no
    log-probability comparison has run.
  - Every profile states `--async-scheduling`. vLLM already turns it on for
    these models, so it changes nothing.
- `sparkring dashboard`: a read-only status page on Node A, port 8090, and
  `sparkring dashboard tui` for a terminal. The install's Model ready card
  ends with the dashboard's address while its service is enabled.
- Plans, status, checks and the dashboard name the transport that runs:
  SIRCL, NCCL or RoCEnante (b12x), which `--transport prepared` selects.
- `--on` places the serving image only on the Sparks it names. Models serving
  on the other Sparks get no image write.
- Automatic recovery restarts a four-Spark model on part of a ring of eight.
- The installer refuses to start a model whose GPU memory share plus
  SparkCache's pinned host memory leaves a Spark less than 6 GiB for the
  system, and says by how much.
- `qwen38-flash-next-tp2` installs Local Inference Lab's Qwen3.8-Flash-Next
  CSF checkpoint on this release's image (research-only);
  `--checkpoint qad-step5500-ple1000` installs step 5500.
- The enhancement catalog lists Local Inference Lab's launch settings, the
  Qwen CSF checkpoint, the SparkCache layer image and libsircl's copy-relay
  broadcast.
- Research-only profiles: `glm53-nvfp4-csf-tp8`, GLM-5.3-NVFP4-CSF on eight
  Sparks, which installs only with `--image-lock FILE` and is parked, and
  SparkCache variants of the GLM-5.3-Flash profiles, which need an image this
  release does not publish.

## Before you upgrade

- `qwen38-flash-next-tp2` without `--checkpoint` installs the 95.2 GiB CSF
  checkpoint, downloading it when the Sparks lack it, and serves it as
  `Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD-TP2`.
  Clients that request `Qwen3.8-Flash-Next-NVFP4-QAD-TP2` must change the name,
  or install with `--checkpoint qad-step5500-ple1000` to keep step 5500.
- Installing any profile with this package creates a separate deployment and
  restarts the model.

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
- The serving defaults: one start per setting on the ring of eight with GPU
  clocks locked, two for DeepSeek's Engram sharding. `deepseek-v41-flash-tp4`
  (four Sparks of the ring), `glm53-flash-tp2` and `qwen38-flash-next-tp2`
  (`--checkpoint csf`, two Sparks each) installed from a package of this
  release with no research options; their run receipts show the measured
  settings, and each served one measurement at temperature 1. DeepSeek on
  eight Sparks started and served once; Engram sharding showed no consistent
  gain there, so that profile keeps it off. The same three also installed on
  two new rings of four, set up as a new four-Spark user would: DeepSeek on
  one ring measured the same as on the ring of eight. `qwen38-flash-next-tp4`
  installed with both Qwen settings on both rings of four and on half of the
  ring of eight, beside `glm53-flash-tp4`, and each served one measurement at
  temperature 1. No start with these settings has run for Qwen on eight
  Sparks, Swift-1.5 or MiMo.
- On the ring of eight: an `--on` install on two Sparks left the deployments
  beside it serving, and automatic recovery restarted DeepSeek's four-Spark
  group 6 min 47 s after one of its Sparks stopped.
- CPU checks pass, apart from tests that also fail before these changes.

## Known issues

- 2026.10.2's: installed libsircl deployments report `ok: false` in
  `sparkring check`, and libsircl's broadcast is slow.
- The SparkCache variant on two Sparks made both Sparks unreachable in its
  first start, with larger capture slots than this release gives it. With
  this release's smaller slots, a private record reports one start that
  passed every step it ran of its test plan, including recovery after a
  restart and after SIGKILL; no public record holds it, one step and the
  two-start comparison did not run, and the variant stays research-only.

## Try it

On Node A:

```bash
curl -fsSL https://raw.githubusercontent.com/FujitsuPolycom/sparkring/2026.10.3/install.sh | bash -s -- --ref 2026.10.3 --package-only
sudo sparkring install --profile PROFILE --plan
sudo sparkring install --profile PROFILE
sudo sparkring check --report ~/sparkring-report
```

## Go back

To 2026.10.2, a pre-release on the same image:

```bash
curl -fsSL https://raw.githubusercontent.com/FujitsuPolycom/sparkring/2026.10.2/install.sh | bash -s -- --ref 2026.10.2 --package-only
```

To 2026.10.1, the latest release before this one, which installs image
`dev-20261004-kraken-cuda1342-nccl2323-status034`:

```bash
curl -fsSL https://raw.githubusercontent.com/FujitsuPolycom/sparkring/2026.10.1/install.sh | bash -s -- --ref 2026.10.1 --package-only
```
