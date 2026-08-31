# GLM-5.3 R8 DCP1, DCP2, and DCP4 image validation

**Status: implemented with bounded live evidence.** The local Linux/ARM64
image passed publication and restart-restore checks at DCP1, DCP2, and DCP4.
This record is not a general qualification.

The machine-readable receipt is
[`local-image-receipt.json`](local-image-receipt.json).

## Image

The tested local image ID is
`sha256:77da063d1d51fa181eb39e519dda7c5ae4eb59a47e169cb4c33bd2cd42120225`.
Its compressed transfer archive is 8,467,812,978 bytes with SHA-256
`51b1aece26dad833ac2b2727a88429642d38b8c1b48b00f6d4b28214f7d840fc`.

Every rank loaded the same image ID. Construction verification checked the
complete vLLM and SparkCache Python manifests, 15 retained vLLM native
extensions, switchless NCCL, and the SparkCache CUDA placement library.

## Configuration

The DCP2 and DCP4 runs used four GB10 systems at TP4, an 8,192-token scheduler
budget, and 30 GiB of FP8 target KV per rank. The DCP1 deep-context run used
26 GiB per rank. Every run used DFlash2 depth seven, fastsafetensors, and
SparkCache CUDA placement.

The model limit was 1,048,576 tokens. Full-CKV gathering remained capped at
524,288 tokens. Complete `snapshot-v1` publication used the
`manager-pages-v2` identity.

## Results

| DCP | Reported KV capacity | Stored span | Snapshot | Commit | Restart restore | Exact output |
|---:|---:|---:|---:|---:|---:|---|
| 4 | 5,402,023 tokens | 8,192 | 19.1–33.9 ms | 131.3–161.1 ms | 116.5–151.7 ms | `R8_DCP4_RED` |
| 2 | 2,899,004 tokens | 9,216 | 50.0–55.4 ms | 172.3–204.9 ms | 155.9–176.2 ms | `R8_DCP2_BLUE` |

The DCP4 allocator reported capacity for approximately 5.15 requests at the
1M request ceiling. DCP2 reported approximately 2.76 requests. These are
allocator capacity ratios, not concurrency qualification.

Each run published one deterministic prompt, replaced all four model
processes, completed one inventory request, and replayed the identical prompt.
Every physical rank reported a verified SparkCache CUDA restore.

A separate DCP4 rerun published 14,336 tokens under digest prefix
`ea67c90d8d6f`, replaced all four processes, and restored the same digest on
all ranks in 135.0–167.1 ms. No restore rejection or recomputation appeared.
The rerun did not use generated text as its semantic oracle because the
probabilistic DFlash draft path produced different visible text across process
replacement despite a fixed request seed. The exact-output DCP4 result in the
table remains the semantic check.

## DCP1 KV capacity

The DCP1 capacity sweep used a 1,048,576-token request limit, 8,192 batched
tokens, 16 sequences, DFlash2 depth seven, and SparkCache. It measured pool
capacity separately from deep-request memory requirements.

| FP8 KV per rank | Reported capacity | Result |
|---:|---:|---|
| 39 GiB | 1,955,798 tokens | Served with healthy memory margin |
| 41 GiB | 2,056,272 tokens | Served; SparkCache restored after process replacement |
| 42 GiB | — | Host memory fell to approximately 1 GiB; rejected |
| 43 GiB | — | Host memory fell to approximately 1 GiB; rejected |
| 44 GiB | 2,206,984 tokens calculated | Rank 0 was OOM-killed during KV materialization |

At 41 GiB, SparkCache published 8,192 tokens under digest prefix
`15083199c308`. After replacing all four processes, every rank verified the
restore in 190.9–215.2 ms. Available host memory after restore was
11,537,212–13,138,388 KiB. A later 450,000-depth request exhausted host memory,
so the 41 GiB profile is a capacity result rather than the deep-context
default.

## DCP1 deep context

The 26 GiB profile exposed 1,303,701 KV tokens and completed the public
needle-retrieval probe's 1,000,000-depth case. The API reported a 942,767-token
prompt, a 478.1-second completion time, `finish_reason=stop`, and the exact
eight-digit needle. Available host memory remained at least 23.7 GiB during
prefill. No container or kernel OOM record appeared.

SparkCache published the completed request as a 942,592-token `snapshot-v1`
object. Snapshot capture took 16.6–17.9 seconds per rank and background commits
took 12.0–14.5 seconds. The snapshot was not replayed after process
replacement. See the
[`deep-context record`](../../performance/records/glm53-flash/dcp1-deep-context-boundary-20260831.md)
for the boundary walk and exact limitations.

## Limits

The image completed one 942,767-token request, but the resulting 942,592-token
snapshot was not replayed. This record does not prove large-context restore,
concurrent deep requests, fault recovery, soak behavior, or throughput for
this image.

The image has no registry digest. Publishing it requires a separate explicit
operator decision and an immutable registry receipt.
