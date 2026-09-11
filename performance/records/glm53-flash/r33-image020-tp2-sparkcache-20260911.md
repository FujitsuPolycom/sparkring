# Generic R33 image TP2 SparkCache qualification

Status: **TP2 bounded-qualified; TP4 and generic-image release pending**.

## Conditions

Two NVIDIA GB10 systems ran image
`sha256:3c7779ad71dd0d5d6fae4c98e04b94c377429306158c2259fc44635892b8b8e4`
with SparkRing launcher source
`c8646b000e0815f6ca6327539452c3597b9fc8ff`. The model was
`local-inference-lab/GLM-5.3-Flash-NVFP4-Spark` revision
`df116c4fb16b1d37ae43d2cfd624de26ffbc832e`.

The serving profile used TP2/DCP1, managed B12X target and MTP loading, MTP3,
7.5 GiB (`8053063680` bytes) of KV memory per rank, eight sequences, an
8,192-token scheduler budget, a 1,048,576-token configured request limit,
SparkCache, and active 2 GiB host-memory guards. The source closure contained:

| Component | Identity |
|---|---|
| vLLM integrated tree | `547f7091841728f21ab419012a766fd1df70a569` |
| B12X tree | `284e7df8caff930477a314fea20d826256844de4` |
| SparkCache tree | `86ef46de45dd0f4ed776b86109a30df6f83db557` |
| NCCL | 2.31.2; `libnccl.so.2.31.2` SHA-256 `84a4b8d83fb5fa1f0d640d311ad38b45140672dae9889775fe1e4a3990479e47` |

Both workers mapped the pinned NCCL library and selected the two physical
functions `rocep1s0f0` and `roceP2p1s0f0` of the single DAC. RoCEnante
initialized as the first tensor-parallel all-reduce backend.

## Results

- Three consecutive cold starts completed CUDA graph capture on both ranks.
  Neither log contains `capture_end` or `cudaErrorNotPermitted`.
- A short request after the third start returned the exact requested text with
  HTTP 200 and normal completion.
- Both workers restored the same 8,192-token cache entry after restart. The
  deterministic 8,193-token reader returned the exact answer and reported
  8,192 cached tokens with zero created tokens.
- Sixteen of sixteen structured-output requests passed at concurrency eight
  with a 1,024-token output budget.
- Both ranks repeatedly executed mHC with 8,192 input rows and 4,096 owner rows.
- Six decode cells completed with zero request errors.

| Prompt tokens | Cold prefill tok/s, median of 3 | C1 aggregate tok/s | C4 aggregate tok/s |
|---:|---:|---:|---:|
| 8,192 | 2,340.0 | 33.1 | 66.1 |
| 16,384 | 2,354.2 | 31.9 | 67.4 |
| 32,768 | 2,350.4 | 30.9 | 71.8 |

The first 32,768-token prefill sample took 17.83 seconds; the following two
took 13.93 and 13.94 seconds. The median includes the retained outlier.

| Context | Concurrency | MTP-normalized steps/s | Effective acceptance length |
|---:|---:|---:|---:|
| 8,192 | 1 | 13.1 | 2.52 |
| 16,384 | 1 | 13.0 | 2.47 |
| 32,768 | 1 | 13.1 | 2.36 |
| 8,192 | 4 | 26.1 | 2.53 |
| 16,384 | 4 | 25.2 | 2.67 |
| 32,768 | 4 | 26.3 | 2.73 |

Normalized steps per second represent target-model forward passes and separate
engine speed from data-dependent speculative acceptance. Raw and normalized
values remain in the decode receipt.

## Evidence

The raw evidence is retained outside the repository. These hashes identify the
exact inputs used for this record without exposing host addresses, cache paths,
prompts or generated secrets.

| Evidence role | Artifact identifier | SHA-256 |
|---|---|---|
| Image package/source receipt | `image-020/image-receipt.json` | `97dd159418da08b6e5d4c05ca36a7b7b7f02e0ec8b1cbad16cc6a3db564edbd6` |
| Qualification summary | `tp2-020-qualification.json` | `ec9b279c1e6e028c4fe2903984d74ff2a45d37b9d2f12cffb474bc3678138c8c` |
| Cache writer | `tp2-020-semantic-writer.json` | `6c16a5198efdbbb248406bc161c5279c7c710ed3175dc34ddb5790583f37eabc` |
| Restart reader | `tp2-020-semantic-reader.json` | `be636d4122de796f7711cdf2b5d63935982c7b888c7091ec38f537342aec23f0` |
| Writer/reader comparison | `tp2-020-semantic-comparison.json` | `ab00eb15b0e6404e5c4370cb2936bfc9478c787a45cdc3752ce70422d0c127d9` |
| Third-start semantic check | `tp2-020-third-start-semantic.json` | `8e0a7a6243c2020631101dfdb7889306f18f730104618a192febf222c2635ad8` |
| Concurrency-eight correctness | `tp2-020-kv750-c8-budget1024.json` | `fbe8f6549828741fc41d2112b61ecb82692bba80e69eb911e35fb308abe9d5e8` |
| Cold prefill | `tp2-020-kv750-prefill.json` | `80213a66b65e32b8fc9195e957b2c58ea0b425986fec36e1963aff0d46ba6ab8` |
| Decode | `tp2-020-kv750-decode.json` | `d59f99aa6650392093ca5613f4418e70e543bb7719b25cf7a96e53642fdb7a7b` |
| Rank-zero cumulative startup log | `tp2-020-three-starts-r0.log` | `04f25651c113c3649f50d1cc64c32ce7bc1bec5c5e90c510c0d66a1624fa76c9` |
| Rank-one cumulative startup log | `tp2-020-three-starts-r1.log` | `77c94bbf9f47ffddc18743482e1a8f4de63a0c8a1a83dec474706196cbcf52f6` |
| Rank-zero NCCL process maps | `tp2-020-nccl-maps-r0.txt` | `da8b913f38b6f317eb107ef0bf1ee85179da941d77abdbe7a07cbc9dfc215d8d` |
| Rank-one NCCL process maps | `tp2-020-nccl-maps-r1.txt` | `f26a5802f2f46aa4d2742893e733f144127f3e3a8f2323bfeddd5eed85ec55bd` |

## Qualification limits

The configured one-million-token limit was admitted by the reported KV pool;
no completed one-million-token request was run. Multimodal correctness and
sustained memory stability were not tested. The cumulative rank-zero log
contains an engine-dead exception during a coordinated shutdown after a
successful writer run; both containers exited with code zero and without an
out-of-memory condition. The event is retained and is not classified as a CUDA
graph startup failure.

This record qualifies only the TP2 profile on the exact image and source
identities above. TP4 qualification is independent. The generic image must not
be labeled release-qualified until the TP4 gate passes.
