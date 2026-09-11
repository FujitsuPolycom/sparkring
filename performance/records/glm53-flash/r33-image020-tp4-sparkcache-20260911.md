# Generic R33 image TP4 ring qualification

Status: **bounded-qualified; registry publication pending**.

## Conditions

Four NVIDIA GB10 systems ran image
`sha256:3c7779ad71dd0d5d6fae4c98e04b94c377429306158c2259fc44635892b8b8e4`.
The normal deployments used SparkRing renderer
`c8646b000e0815f6ca6327539452c3597b9fc8ff`. One diagnostic startup used
renderer `406546e0727c127790a99f3cc24ffa56977291bd`, whose only effective
container-environment change was `NCCL_DEBUG=WARN` to `NCCL_DEBUG=INFO`.
The diagnostic renderer did not change the image, model arguments, cache
configuration or transport settings.

The model was `local-inference-lab/GLM-5.3-Flash-NVFP4-Spark` revision
`df116c4fb16b1d37ae43d2cfd624de26ffbc832e`. The serving profile used
TP4/DCP1, InstantTensor target loading, static MTP3, 24 GiB
(`25769803776` bytes) of FP8 KV memory per rank, 16 sequences, an 8,192-token
scheduler budget, a 1,048,576-token request limit, hardware-forwarded ring
links, dual-domain NCCL, token-sharded mHC, continuation-prefill coalescing,
and SparkCache.

The image construction receipt binds these source and native identities:

| Component | Identity |
|---|---|
| vLLM integrated tree | `547f7091841728f21ab419012a766fd1df70a569` |
| B12X tree | `284e7df8caff930477a314fea20d826256844de4` |
| SparkCache tree | `86ef46de45dd0f4ed776b86109a30df6f83db557` |
| NCCL | 2.31.2; `libnccl.so.2.31.2` SHA-256 `84a4b8d83fb5fa1f0d640d311ad38b45140672dae9889775fe1e4a3990479e47` |
| SIRCL | `libspark_transport_capi.so` SHA-256 `bea00f2ba6051c2c0bcd2853aae894672aa7f1fe5a1d905edaa9120aabf74246` |

## Results

- Three coordinated startups of the same image completed CUDA graph capture
  on all four ranks. Two used normal WARN logging and one used the INFO-only
  diagnostic renderer.
- The INFO startup produced 24 `NET/IB RouteFinal` records on every rank.
  Every rank used `rocep1s0f0`, `rocep1s0f1`, `roceP2p1s0f0`, and
  `roceP2p1s0f1`, spanning PCI domains `0000` and `0002`. These records prove
  NCCL-connected paths through both host domains during startup. They do not
  quantify per-request traffic or throughput by HCA.
- A semantic request after the INFO startup returned the exact expected answer,
  created an 8,192-token cache entry, and completed normally.
- The managed lifecycle then returned all four ranks to the canonical WARN
  configuration. Every rank ran the same image, completed graph capture, and
  reported active mesh and model services. A post-return semantic request
  returned the exact expected answer and restored the same 8,192-token cache
  entry on every rank in 97.8–99.1 ms.
- One 114,688-token cold prompt returned the exact expected answer, reported
  zero cached tokens, completed normally, and left the API healthy.
- Six prefix-hit geometry cases completed after a 32,256-token cache hit,
  including scheduled suffixes of 515, 513, 531, and 1,027 tokens. This proves
  execution liveness across the one-checkpoint and packed-checkpoint metadata
  shapes. The harness did not assert raw output equivalence for those six
  cases.
- A deterministic 8,193-token writer/restart/reader sequence returned the exact
  expected answer before and after restart. The reader reported 8,192 cached
  tokens and zero newly created cache tokens. Worker logs on all four ranks
  verified restoration of the selected 8,192-token snapshot; observed restore
  times ranged from 91.5 to 195.7 ms.
- SIRCL snapshots matched the active workers on all four ranks. Each snapshot
  reported 205 published and 205 completed sequences with `fatal=false`.
- mHC diagnostics recorded 8,192 input rows and 2,048 owner rows. The scheduler
  selected positive 8,192-token continuation-coalescing plans, and B12X
  initialized checkpoint export with 16 local heads and capacity for four
  checkpoints. The worker logs do not emit a separate export-completion event,
  so the record establishes live plan selection and compatible worker
  configuration rather than independently observing checkpoint writes.

| Prompt tokens | Cold prefill tok/s, median of 3 | C1 aggregate decode tok/s | C4 aggregate decode tok/s |
|---:|---:|---:|---:|
| 8,192 | 3,274.5 | 51.4 | 132.5 |
| 16,384 | 3,467.9 | — | — |
| 32,768 | 3,580.6 | 47.4 | 127.4 |

All nine prefill samples reported zero cached tokens. Decode cells used
10-second measurement windows and completed with zero request errors.

| Context | Concurrency | MTP-normalized steps/s | Effective acceptance length |
|---:|---:|---:|---:|
| 8,192 | 1 | 21.1 | 2.44 |
| 32,768 | 1 | 21.4 | 2.21 |
| 8,192 | 4 | 48.4 | 2.74 |
| 32,768 | 4 | 46.1 | 2.76 |

Normalized steps per second represent target-model forward passes and separate
engine speed from data-dependent speculative acceptance. Raw and normalized
values remain in the decode receipt. The measurements are one configuration,
not a matched performance comparison.

## Evidence

Raw evidence is retained outside the repository. These hashes identify the
exact inputs used for this record without publishing host addresses, cache
paths, prompts, generated secrets, or container identifiers.

| Evidence role | Artifact identifier | SHA-256 |
|---|---|---|
| Image package/source receipt | `image-020/image-receipt.json` | `97dd159418da08b6e5d4c05ca36a7b7b7f02e0ec8b1cbad16cc6a3db564edbd6` |
| Cold prefill | `tp4-020-prefill.json` | `7a55be6b511883a2aeec6b77023ee81f48b6506ac6636f5382ec09e8bee4f24f` |
| Decode | `tp4-020-decode.json` | `dc195270d0fb3772aec40a0951bf85f762e0e8a2be800f514a9e118aa293c42b` |
| Prefix-hit geometry | `tp4-020-prefix-raw-regression.json` | `9b56208e8bced72e94b621b1babd73639d1a55557b46fe98a6b2b1246d433bc2` |
| 114,688-token exact-answer case | `tp4-020-long-oracle-003.json` | `88c337273839c52d587d804ab1204cf835062aa9f1dc898639130ba84a85d53a` |
| Cache writer | `tp4-020-cache-writer.json` | `88f1526c065a12bfa60a5555548a5b2d369672ea433fbb9fb2e258e9989de139` |
| Restart reader | `tp4-020-cache-reader.json` | `c091bff1899af7105786286de44392caacda6fa0ee8061da144d5ac197f275c4` |
| Writer/reader comparison | `tp4-020-cache-comparison.json` | `49d316f053b5c63ea1f3a13573bf4d536980850eb9fe3e7fe170c789d178aeb4` |
| Loaded-library and activation audit | `tp4-020-activation/evidence-summary.json` | `52016469566e6c37e753908f394a23138118d69964f63a237c0d0a67e04cd3a9` |
| SIRCL post-request summary | `tp4-020-activation/sircl-post-request-summary.json` | `86121b5b21a9e66bcb951f1dd1f4de94fc53efe8c44aa4c44680294369329781` |
| Dual-domain NCCL route attribution | `tp4-image020-nccl-info-plan/deployment/ready-attribution.json` | `5cccadfde05fb420442c271c2c034cc8d7eb10cc881a3e8f124f9360a5687838` |
| INFO-start semantic check | `tp4-020-info-semantic.json` | `c4b92e4845773e7433f506db79ad774576b53c31f694c9aeb434898ecc1010e3` |
| Canonical WARN return | `tp4-image020-nccl-info-plan/deployment/canonical-ready.json` | `cfbb837566ac01ef049566fd24815687102f558a479910c7353b1eaadbd653df` |

## Qualification limits

The configured one-million-token limit was admitted by the reported KV pool;
no completed one-million-token request was run. Multimodal correctness,
sustained memory stability, switched-fabric operation, and arbitrary
concurrency were not tested. The six prefix-hit geometry cases establish
liveness but not raw output equivalence. Route attribution proves NCCL path
creation through both PCI domains during startup, not traffic balance or bytes
transferred by a particular request.

The managed Noether lifecycle returned all ranks to the canonical WARN
renderer and passed health, graph-capture, semantic, and cache-restore checks.
The image has no public registry digest yet, and no published tag or quickstart
should identify it as the public default until immutable registry verification
completes.

This record qualifies only the TP4/DCP1 ring profile on the exact image and
source identities above. The TP2 result has a separate
[qualification record](r33-image020-tp2-sparkcache-20260911.md). Switched and
other model profiles do not inherit qualification from the generic image name.
