# GLM-5.3 Flash asynchronous SparkCache image validation

Status: **qualified** for persistent restore and for asynchronous publication
of the recorded 124,928-token, 231.8 MiB-per-rank snapshot under the immutable
Linux/ARM64 image and DCP4 conditions recorded in
[`async-capture-image-receipt.json`](async-capture-image-receipt.json).

The image combines the Local Inference Lab GLM-5.3 runtime, BF16 DFlash2 at
depth seven, B12X GB10 kernels, switchless NCCL, fastsafetensors, and
SparkCache's bounded CUDA publication and restore paths.

## Artifact

| Item | Identity |
|---|---|
| Registry image | `ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:bc7d079f16ff4a418669c58c5250f2da52e989a0c5805569ba9429d41b765f65` |
| Published tag | `ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache:20260901-r10-async-telemetry` |
| Image ID | `sha256:35f397668c01075d0bdd28bbdb3398afd3744df6086646c6f68bcf7ebe7f918f` |
| SparkRing image source | commit `d2f8911427d64bbb89c275814777fc3f8112fd21` |
| SparkCache source | commit `c5dda75ec46bf235f6ece6e0d0174c1e41bd805a` |
| vLLM source | commit `22ffe1401ca9bd3e4503e62de7b414deca7661a1` |

The archive SHA-256 is
`5f3e02b85ace893e5f9ced34ffe7a6dacfe896bd026137e9486517b8e490db8d`.
The archive and image IDs were identical on all four ranks. Registry read-back
returned the recorded digest, image ID, Linux/ARM64 platform, and source
labels.

## Conditions

The deployment used four GB10 systems at TP4/DCP4, 24 GiB of FP8 KV per rank,
a 1,048,576-token request limit, 16 sequences, an 8,192-token scheduler
budget, prefill interval eight, and DFlash2 depth seven. SparkCache used
`read-write`, complete `snapshot-v1` objects, and two 3 GiB mapped capture
slots per rank.

Startup inventory checked and offered 29 manifests with zero rejected entries
before API readiness.

The live deployment overrode the operator template's semantic storage-root
name `glm53-flash-dcp4-snapshot-v1` with the durable evidence identifier
`jj-r10-async-ab-v1`. This value selects rank-local storage and JIT
directories; it does not alter `CacheIdentity` or the stored snapshot format.
The semantic operator default itself was not measured.

## Fresh publication

The artifact published a fresh 125,999-token prompt and returned its exact
needle. Every rank stored the 124,928-token reusable boundary. Capture
completion was observed 403.7–408.5 ms after submission. Durable commit took
520.6–567.6 ms per rank.

| Rank | Capture completion observed | Capture rate | Payload | Durable commit |
|---:|---:|---:|---:|---:|
| 0 | 408.5 ms | 306K tok/s | 231.8 MiB | 520.6 ms |
| 1 | 404.3 ms | 309K tok/s | 231.8 MiB | 528.6 ms |
| 2 | 403.7 ms | 309K tok/s | 231.8 MiB | 526.3 ms |
| 3 | 404.3 ms | 309K tok/s | 231.8 MiB | 567.6 ms |

## Persistent restore

The service restored two retained entries without sending either prompt after
startup to prime them. The 900K request restored 899,072 of 899,998
prompt tokens and returned its exact needle. The 1M request restored 999,424
of 1,000,000 prompt tokens and returned its exact needle.

| Prompt | Rank restore range | Rank rate range | End-to-end | Needle |
|---|---:|---:|---:|---|
| 900K | 2.123–2.180 s | 412–423K tok/s | 6.47 s | passed |
| 1M | 2.276–2.395 s | 417–439K tok/s | 6.73 s | passed |

### Deep-entry writer provenance

The retained 900K and 1M entries were published by a private writer artifact
before this public image restored them:

| Item | Identity |
|---|---|
| Writer image ID | `sha256:8e586e6ad9b4f30a8ccef1bfd8b76194524e156089c958907872d0f8735a09b2` |
| Writer archive SHA-256 | `47c800fd73130c1fe26b707caa2c64f81ed43c951fe2019d8836cd0b883dbe48` |
| Writer SparkCache | commit `6d83c7d8cb6ace96e657b3d0150116d0fe4e011c`, tree `0bb871bd1e8d3893a11686f0ba404bd4b6240e4d`, source SHA-256 `67edb651835b978cbaf2519f92e68251145c1368a22cc0339f706d5c2144f862` |
| Writer vLLM | commit `22ffe1401ca9bd3e4503e62de7b414deca7661a1`, tree `1bb7f10a5838d348ca2fcb0134b05ad768d3340b` |
| CUDA snapshot library | `4398f18b8913e743e7bf1ed8fe29560d4580e61b6a1e2ab8b16684b19b6573b5` |
| Cache contract | TP4/DCP4, `snapshot-v1`, storage root `jj-r10-async-ab-v1` |

The writer and public restore artifact used the same target and draft
checkpoint identities. The deep restores establish persistent compatibility;
they do not qualify asynchronous publication at 900K or 1M.

All four containers remained running with `OOMKilled=false`, exit code zero,
and the recorded image ID. The API remained healthy. No SparkCache, CUDA,
out-of-memory, capture-abort, semantic, or restore error appeared during the
checks.

The result qualifies bounded asynchronous complete-snapshot publication at
124,928 stored tokens and 231.8 MiB per rank, automatic all-rank startup
inventory, and persistent DCP4 restore for this artifact and configuration.

## Limits

DCP1 and DCP2 asynchronous capture is implemented but not qualified by this
record. Asynchronous publication above 124,928 stored tokens per entry is also
unqualified. Asynchronous page-tail publication is not included. Long-duration
serving and concurrent deep-context publication were not measured.
