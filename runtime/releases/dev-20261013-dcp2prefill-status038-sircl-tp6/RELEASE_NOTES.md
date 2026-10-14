# SparkRing 2026.10.4

SparkRing's profiles run three models created by Local Inference Lab, Inc. and
licensed under the Local Inference Lab License, Version 1.0:
[GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD)
(`glm53-flash-tp2`, `glm53-flash-tp4`, `glm53-flash-nvfp4-spark-tp2-dcp2`),
[GLM-5.3-NVFP4-CSF](https://huggingface.co/local-inference-lab/GLM-5.3-NVFP4-CSF)
(`glm53-nvfp4-csf-tp6`) and
[Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD)
(`qwen38-flash-next-tp2`).

Packages built from tag `2026.10.4` install this release's image by default.
The
[release record](https://github.com/FujitsuPolycom/sparkring/blob/2026.10.4/runtime/releases/dev-20261013-dcp2prefill-status038-sircl-tp6/README.md)
has the evidence and the open gates.

## What's in it

Installer image `dev-20261013-dcp2prefill-status038-sircl-tp6`
(`ghcr.io/fujitsupolycom/sparkring@sha256:e21ba936d4fd6bfeb6e7a651e54a03ac461bf4e1e6f5721c1a34845c80acd76f`):
2026.10.2's image plus:

- `glm53-flash-nvfp4-spark-tp2-dcp2` (research-only): GLM-5.3-Flash CSF on
  two Sparks with decode-context parallelism 2. The pair splits the KV cache,
  so it holds about twice the context and serves 16 requests at once.
- `glm53-nvfp4-csf-tp6` (research-only): GLM-5.3-NVFP4-CSF on six Sparks. It
  has not run on hardware yet.
- Dashboard 0.3.8: vLLM's serving statistics on the status page.

Every 2026.10.2 profile is admitted on this image with its 2026.10.2 settings.

## Tested

On a ring of eight DGX Sparks, with `sparkring install`:

- `deepseek-v41-flash-tp4` on four Sparks: installed and served.
- `glm53-flash-tp2` with the CSF checkpoint: installed and served.
- `glm53-flash-nvfp4-spark-tp2-dcp2`: installed and served.

The other profiles passed the installer's image checks but were not started
on this image. No speed was measured on it.

## Try it

On Node A:

```bash
curl -fsSL https://raw.githubusercontent.com/FujitsuPolycom/sparkring/2026.10.4/install.sh | bash -s -- --ref 2026.10.4 --package-only
sudo sparkring install --profile PROFILE --plan
sudo sparkring install --profile PROFILE
sudo sparkring check --report ~/sparkring-report
```

A Spark that holds the 2026.10.2 image downloads only the 6 layers this
image adds, 2.6 MiB.

## Go back to 2026.10.2

```bash
sudo sparkring install --profile PROFILE --image 2026.10.2
```

That is the parent image on the same transports
([another image](https://github.com/FujitsuPolycom/sparkring/blob/2026.10.4/docs/operations/install-reference.md#another-image)).
