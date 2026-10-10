# SparkRing 2026.10.2

SparkRing's `glm53-flash-tp2` and `glm53-flash-tp4` profiles run
GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD, created by Local Inference Lab, Inc., a
non-profit organization, available at
https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD.
GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD is licensed under the Local Inference Lab
License, Version 1.0.

Pre-release for testing. Packages built from tag `2026.10.2` install this
release's image by default. `main`, `install.sh` from `main` and the
[Install Builder](https://fujitsupolycom.github.io/sparkring/) stay on
2026.10.1 until this release merges. The
[release record](https://github.com/FujitsuPolycom/sparkring/blob/2026.10.2/runtime/releases/dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036/README.md)
has the full evidence and gates.

## What SIRCL is

SIRCL is SparkRing's own collective-communication layer for DGX Sparks
cabled directly to each other over ConnectX-7, with no switch. It replaces
NCCL for vLLM's tensor-parallel and decode-context-parallel collectives.

- Each Spark talks RDMA straight to its cabled neighbours. Sparks that aren't
  cabled to each other are reached through the ConnectX-7's hardware relay on
  the Sparks in between, so a ring of four or eight behaves like one fabric.
- It picks a schedule by message size: one-shot for small messages, two-shot
  for medium, a ring for large. For GLM-5.3 it can also fuse the all-reduce
  with the following RMSNorm.
- Its settings come from a tuning table measured on real rings. You can
  measure your own with `sudo sparkring fabric tune`.
- On eight Sparks, measured through libsircl, a 4 KiB all-reduce takes about
  18 µs, and a 256 MiB all-reduce runs at 24.3 GB/s bus bandwidth, faster
  than NCCL 2.32.3 on the same cables
  ([record](https://github.com/FujitsuPolycom/sparkring/blob/2026.10.2/performance/records/transport/libsircl-ring8-image-1a8c10354eb0-20261009.md)).

libsircl puts the same protocol behind NCCL's C API (it loads as
`libnccl.so.2`), so unmodified programs such as nccl-tests and PyTorch can use
it. It's SparkRing's own implementation under Apache-2.0, not NVIDIA NCCL.

## What's in it

Installer image `dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036`
(`ghcr.io/fujitsupolycom/sparkring@sha256:4fffc4dc3074d5539f4e9d3a013ff9ef4e0be570a95b74d4646ee341da1f6911`): 2026.10.1's image plus SIRCL 0.3.2, libsircl 0.6.0 and the GLM-5.3 plugins.

- Collectives run on SIRCL with NCCL off on any fabric that `sparkring setup`
  recorded.
- Setup handles pairs, lines and rings of up to eight Sparks. It can adopt an
  existing ring (`--adopt`) or set up a recabled one (`--re-form`).
- GLM-5.3 runs on eight Sparks (`glm53-nvfp4-tp8`), and GLM-5.3-Flash installs
  the CSF checkpoint on this image (`--checkpoint nvfp4-spark` installs
  NVFP4-Spark).
- Running vLLM on libsircl (`--transport libsircl`) is research-only.
- `sparkring check` also checks GPU clocks, and `--report` writes a test
  report to attach to an issue.

## Speed

Decode at 32K context, output tok/s:

| Model | Checkpoint | Sparks | `--profile` | 1 / 4 / 8 streams | Prefill, 32K prompt (tok/s) |
|---|---|---|---|---|---|
| GLM-5.3 | NVFP4 | 8 | `glm53-nvfp4-tp8` | 47.9 / 99.8 / 143.7 | 1,343 |
| GLM-5.3-Flash | CSF | 4 | `glm53-flash-tp4` | 64.2 / 126.6 / 218.7 | 3,127 |
| GLM-5.3-Flash | CSF | 2 | `glm53-flash-tp2` | 39.5 / 87.1 / 123.2 | 2,138 |
| Qwen3.8-Flash-Next | QAD step 5500 | 4 | `qwen38-flash-next-qad-tp4` | 62.1 / 149.4 / 234.0 | 4,666 |
| Qwen3.8-Flash-Next | QAD step 5500 | 2 | `qwen38-flash-next-tp2` | 42.9 / 109.9 / 162.9 | 3,649 |
| DeepSeek-V4.1-Flash | FP8/MXFP4 | 4 | `deepseek-v41-flash-tp4` | 63.8 / 127.5 / 182.8 | 4,328 |

On four Sparks, `--checkpoint qad-step5500-mxfp8-attention` decodes 6–8 % faster than the default checkpoint in the median of two starts on the same ring; it was not compared on two Sparks.

`glm53-flash-tp2` and `glm53-flash-tp4` are short names of the profile IDs `glm53-flash-nvfp4-spark-tp2` and `glm53-flash-nvfp4-spark-tp4`; both names install the CSF checkpoint on this image.
Each row's record is under `performance/records/`.

## Tested

On eight DGX Sparks, cabled as one ring of eight and then as two rings of
four: every profile above installed with `sparkring install` and passed
`sparkring check` and its known-answer requests. So did a model switch and
two pairs on one ring. libsircl passed its collective tests on both layouts.
CPU checks and the release-safety scan pass.

## Known issues

- Installed libsircl deployments report `ok: false` in `sparkring check`,
  because the installer doesn't judge libsircl receipts yet.
- libsircl's broadcast is slow.

## Try it

On Node A:

```bash
curl -fsSL https://raw.githubusercontent.com/FujitsuPolycom/sparkring/2026.10.2/install.sh | bash -s -- --ref 2026.10.2 --package-only
sudo sparkring setup --plan     # only if `sudo sparkring fabric show` finds no fabric; then sudo sparkring setup
sudo sparkring install --profile PROFILE --image 2026.10.2 --plan
sudo sparkring install --profile PROFILE --image 2026.10.2
sudo sparkring check --report ~/sparkring-report
```

A Spark that holds the 2026.10.1 image downloads only the 6 layers this image adds, 8.3 MiB.

To tune SIRCL to your cables, run `sudo sparkring fabric tune --execute`
while no model serves, then install again. It takes up to half an hour per
group shape.

Post the report on the 2026.10.2 pull request or in an issue titled
`[test] 2026.10.2 PROFILE`.

## Go back to 2026.10.1

- Keep this package and run a profile on 2026.10.1's image and transport:
  `sudo sparkring install --profile PROFILE --image 2026.10.1`
  ([another image](https://github.com/FujitsuPolycom/sparkring/blob/2026.10.2/docs/operations/install-reference.md#another-image)).
- Keep this image without SIRCL: add `--transport prepared`.
- Reinstall `main`'s package, whose default image is 2026.10.1's:
  `curl -fsSL https://raw.githubusercontent.com/FujitsuPolycom/sparkring/main/install.sh | bash -s -- --package-only`.
  This hasn't been tested on a cluster that this package's setup re-formed.
