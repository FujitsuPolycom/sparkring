# GLM-5.3 Flash asynchronous SparkCache image validation

Status: **qualified** for the immutable Linux/ARM64 image and DCP4 conditions
recorded in [`async-capture-image-receipt.json`](async-capture-image-receipt.json).

The image combines the Local Inference Lab GLM-5.3 runtime, BF16 DFlash2 at
depth seven, B12X GB10 kernels, switchless NCCL, fastsafetensors, and
SparkCache's bounded CUDA publication and restore paths.

## Artifact

| Item | Identity |
|---|---|
| Registry image | `ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:368973d2e67241479ff49f7898f5026a2a44a37dad78b36f26afa1c6d9684e0e` |
| Image ID | `sha256:4664bcba054d2cf383d3d7940189e26aa32774e755583652a6e93c0058500029` |
| SparkRing source | commit `897b566fa69671b804451b47404ef4f298d655e9` |
| SparkCache source | commit `506cc4a16581b5f62ae343cbd90cdd6bea13a6cd` |
| vLLM source | commit `22ffe1401ca9bd3e4503e62de7b414deca7661a1` |

The archive and image IDs were identical on all four ranks. Registry pull
read-back returned the same digest and Linux/ARM64 manifest.

## Conditions

The deployment used four GB10 systems at TP4/DCP4, 24 GiB of FP8 KV per rank,
a 1,048,576-token request limit, 16 sequences, an 8,192-token scheduler
budget, prefill interval eight, and DFlash2 depth seven. SparkCache used
`read-write`, complete `snapshot-v1` objects, and two 3 GiB mapped capture
slots per rank.

## Result

The artifact published a fresh 125,999-token prompt. Every rank committed the
124,928-token reusable boundary in 528.1–545.1 ms, and the exact needle was
returned.

After startup discovered three retained entries, no prime request was sent.
The first user request restored 899,072 of 899,998 prompt tokens in
2.05–2.35 seconds per rank and returned the exact 900K needle. The following
request restored 999,424 of 1,000,000 prompt tokens in 2.24–2.46 seconds per
rank and returned the exact 1M needle.

The result qualifies bounded asynchronous complete-snapshot publication,
automatic all-rank startup inventory, and persistent DCP4 restore for this
artifact and configuration.

## Limits

DCP1 and DCP2 asynchronous capture is implemented but not qualified by this
record. Asynchronous page-tail publication is not included. The 900K and 1M
entries were written by a namespace-compatible source composition; this image
performed their process-replacement restores and separately published the
fresh 126K entry. Long-duration serving and concurrent deep-context
publication were not measured.
