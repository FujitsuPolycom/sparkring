# Installer runs of four profiles on the SIRCL 0.3.2 image, on a ring of eight

Status: **research-only record of installer runs; each profile installed once and passed the installer's
functional and transport checks; single-run timing; not serving-qualified**.

`sparkring install` installed four profiles on eight DGX Sparks with the image lock of the
[SIRCL 0.3.2, libsircl-from-source and GLM-5.3 plugin image](dev-20261009-kraken-csf-sircl032-libsircl-plugins-image-20261009.md):
GLM-5.3 on all eight, then GLM-5.3-Flash on four, Qwen3.8-Flash-Next on two and GLM-5.3-Flash on two,
serving at the same time. The two GLM-5.3-Flash installations are the first installations of the CSF
checkpoint (`local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD` at
`dec48abd33efa73c3bb7c95b74eee10cad34f9be`) that passed the installer's checks.

## Conditions

- **Hardware:** eight DGX Sparks (GB10) cabled as a ring of eight with ConnectX-7 RoCE. SparkRing's setup
  had recorded the fabric by adoption (`sparkring setup --node ... --adopt`), which keeps the existing
  links, addresses and routes and adds only the relay table.
- **Installer:** SparkRing package revision `498a7bc4` on every Spark, run on Node A.
- **Image:** configuration ID `sha256:1a8c10354eb0bbaad898ebcfc1b31d5c5cfc40daab2f9b8eaaeb074524aea952`,
  selected with `--image-lock` and the recorded lock
  [installer-image-c7c35fe0.json](dev-20261009-kraken-csf-sircl032-libsircl-plugins-image-20261009/installer-image-c7c35fe0.json)
  (SHA-256 `3bbcdfe378d7b0ad1577bb1e4e83517a428310a00fe5f388a35b79a4f2c95958`). Every Spark held the image
  under that ID; no image was pulled.
- **Checkpoints:** every plan listed no file to download (`hub_files` empty, `hub_bytes` 0). The installer
  hard-linked the weight files from copies on each Spark and copied the small configuration files.
- **Installs:** `--plan`, then `--yes`, with no other option. The three smaller deployments installed one
  after another; then all three served together. Each start was the first of its profile on this image, so
  compile and tuning caches were empty. GPU clocks were not locked.
- **Checks:** after each installation, `sparkring check` (its functional requests: counting, arithmetic,
  code, a tool call, a forced tool call, an image and thinking; then the transport check of each SIRCL
  deployment and the GPU clock reading), and three temperature-0 questions with one exact answer each
  (17×23, the capital of France, "ring" spelled backwards).

## Result

| Profile | Sparks | Checkpoint | Install time | API readiness | Functional checks | Transport |
|---|---|---|---|---|---|---|
| `glm53-nvfp4-tp8` | 0-7 (cycle of eight) | `local-inference-lab/GLM-5.3-NVFP4` at `b472e4ee` | 661 s | 392.5 s | 6 passed; image not applicable | SIRCL, NCCL absent (see below) |
| `glm53-flash-nvfp4-spark-tp4` | 0-3 (path of four) | CSF at `dec48abd` | 919 s | 777.1 s | 7 passed | SIRCL, NCCL absent, as expected |
| `qwen38-flash-next-tp2` | 4, 5 (pair) | `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` at `60215d26` | 775 s | 647.2 s | 7 passed | SIRCL, NCCL absent, as expected |
| `glm53-flash-nvfp4-spark-tp2` | 6, 7 (pair) | CSF at `dec48abd` | 1,040 s | 919.1 s | 7 passed | SIRCL, NCCL absent, as expected |

Install time runs from `--yes` to the installer's exit, which includes linking and verifying the checkpoint
files (84-88 s per Spark for GLM-5.3, 19-32 s for the others) and waiting for the API. Every deployment
answered the three questions correctly. Each GLM-5.3-Flash plan and installation result named the CSF
checkpoint, which the installer chose without `--checkpoint` because the image's vLLM reads it. Every
`sparkring check` read GPU clocks of 2,405-2,411 MHz with no clock event reason other than idle.

The `glm53-nvfp4-tp8` transport check of package `498a7bc4` reported "differs": each tensor-parallel session
decided from table `ff8af0fdacad4ad4`, where the check expected no table. That table is SIRCL 0.3.2's built-in
plan for the cycle of eight (source `builtin:cycle:8`), which a session whose group is its whole fabric takes
when no measured table matches; every other transport finding held (NCCL mode `never` in every receipt, no
NCCL decision row, no NCCL communicator in any log, and the cycle-8 tuning row's 16 link slots of
524,288 bytes). The installer's transport check accepts that plan in revision `00f95601`, and judges these
receipts as expected.

One-stream decode of `glm53-nvfp4-tp8` at 16K tokens of context, measured with the
[serving A/B runner](../../harnesses/serving_ab/README.md)'s decode cell (llm-inference-bench, temperature 0,
at most 1,024 output tokens, one 30 s cell after 10 s of warm-up): 47.7 output tokens/s at 2.47 accepted
tokens per step, 19.4 engine steps/s. The
[speculation record](dev-20261009-kraken-csf-sircl-libsircl-plugins-dcp-glm53-tp8-speculation-20261009.md)'s
`mx-t8-mtp2` run on image `27e9f75c0d09`, with GPU clocks locked, measured 44.2 tokens/s at 2.27 accepted
tokens per step (19.5 steps/s) in the same cell; the engine step rates agree, and the output rates differ
with the acceptance.

## Limits

Each profile ran once, on one image, with one measurement series. The runs include no correctness screen,
concurrency sweep, full-context request, restart cycle or soak, and no measurement of the smaller
deployments' speed. These results do not establish serving qualification.
