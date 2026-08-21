# GLM-5.2 3.5bpw transport, DCP, MTP, and KV qualification

## Scope

This document records a **public-functional-lane, live-validated candidate
campaign** on four directly cabled NVIDIA DGX Sparks / GB10 GPUs. Each change
was isolated behind a distinct generated profile and rollback artifact. A
candidate advanced only after bounded semantic output, finite-logprob,
transport, performance, and post-run health checks appropriate to that change.

The final retained service is the
[fixed-MTP4, 9.25 GB/rank KV candidate](../../../docs/GLM52_35BPW_FIXED_MTP4_PROFILE.md). The
campaign does not change the repository's advertised EXL3 3.25-bpw plus
LMCache default or its accepted NF3 alternative.

## Immutable model and runtime

| Item | Value |
|---|---|
| Model | `brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78` |
| Revision | `9ab9579774cc432df91567a36f6e9e863e0d4c9f` |
| Model config SHA-256 | `fabb73eb513ec64f3a365da396b38de8d55b3930edfb11baeecbf34ecafa6126` |
| Model index SHA-256 | `9fd852f69ed64442e31dce1cbc5fe7acd0a76bfb848e945d272fe98d00d0c9cd` |
| Image ID | `sha256:02881d5229d4f4d1cbba0cf40537492a2a505b9d4e43bbfe9a0b2a7bd0584513` |
| Topology | Four-rank direct-cable ring, TP4 |
| Weight path | checkpoint EXL3 routed experts plus target-only online EXL3 K6 |
| KV representation | `fp8_ds_mla` |

All throughput figures below are end-to-end serving observations for the named
workload. They are not reference-lane results or portable hardware claims.

## Qualification decisions

### Hybrid SparkRing transport plus direct-RDMA NCCL fallback

The hybrid profile used SparkRing native TP all-reduce and vocabulary
collectives for supported decode shapes while retaining checksum-pinned NCCL
2.30.7 NET/IB for unsupported communication. Its 1,024-prompt plus 128-output
endpoint probe returned the expected semantic output and 96 finite logprob
values. It measured 689.56 prompt tokens/s, 13.80 inter-token decode tokens/s,
and 11.98 end-to-end output tokens/s.

The endpoint artifact SHA-256 is
`2f98d3aab2a871f230a3634d9bd9099942f28191e21687cd6ca8fae51e29aa8b`.
The hybrid transport became the common TP transport for the later candidates.

### Custom sparse-indexer graph transport

The custom sparse-indexer arm returned valid semantic output and finite
logprobs. Its initial C1/C2/C8 matrix measured 13.64/21.36/62.40 aggregate
tokens/s. A corrected DCP4 derivative measured 11.20/20.40/29.76/55.36 at
C1/C2/C4/C8, compared with 12.28/20.72/34.88/55.68 for the matched stock-DCP4
control.

The corrected endpoint and matrix artifact SHA-256 values are
`0de2bc864109dc09eb5dff6ac5240b00c8406640d7b77dad4500f32fc0edc66f`
and
`b9a8347d349c20419e71fca857e1674fdc18421debfe7c6e95ec665b0ac148c2`.
The retained artifacts do not contain a dedicated native indexer-session
census, so live custom-session activation is not claimed. The corrected arm
also failed to improve the matched matrix. The promotion decision is
**unsupported for this profile**; all later candidates keep the stock indexer
path.

### DCP4 stock control and custom DCP graph transport

The stock DCP4 control passed semantic and finite-output gates and measured
12.28/20.72/34.88/55.68 aggregate tokens/s at C1/C2/C4/C8. Its endpoint and
matrix artifact SHA-256 values are
`77ddfd461d02df7658a5e66b55c90ee329f88311de19661eae0f9f6d0a8860a6`
and
`9094f47bbb27598f6d2a9cbc7b528997de175246ee68d958bb24feb04f45d2c1`.

The padded-stride custom DCP candidate was live-active: every rank captured
624 native query and 624 native combine graph nodes, replay sequences caught
up at 205,920, and fatal/overflow counters stayed zero. Its matrix measured
11.84/20.64/32.96/54.72 aggregate tokens/s, below the stock control in every
cell. A separate five-token eager-prefill probe also found one stock Q5 combine
call per layer because the head-major pitch exceeded the deliberately narrow
custom-layout contract.

The custom DCP matrix and counter-audit SHA-256 values are
`822758a88e1cd1e84f30fb605c973c6e0bb93b7f1a86529ed6ee98781000a30e`
and
`3f415dab8b72ac4ba82e77e98d1af2181e7d520db158bd8ce69d99f51a2f5dd5`.
The implementation was functionally active but was **not promoted** because it
lost performance and did not cover the short eager-prefill layout. The final
candidate uses stock DCP4 `ag_rs`.

### Fixed MTP2

Fixed MTP2 matched the MTP-disabled control on three 128-token and three
256-token greedy completions, returned 96 finite logprob values, and activated
both speculative positions. It accepted 814 of 818 draft tokens. The exact
endpoint measured 25.40 inter-token decode tokens/s, and the matrix measured
25.32/40.56/60.96/91.20 aggregate tokens/s at C1/C2/C4/C8.

Qualification, endpoint, matrix, and hardened transport-audit SHA-256 values:

```text
41a97bbf6cc59f65a015b283ac5f1ca85da2421bc342f2c442d785126bd1a269
fdf24c9aa414cb721b5c3c992871e088420067d4a3852f305c8807e47bd74111
1b77f691766e79cd075962e39d5e97af1c78f8098cae6e1ca18c8278a497b9f8
fa662966e5fdbb8051201ae9cebb3813d8644b973361c6378fc9070f94c4b6b8
```

Fixed MTP2 passed, then served as the rollback for fixed MTP3.

### Fixed MTP3

Fixed MTP3 preserved the MTP-disabled output hashes and finite logprobs,
activated all three speculative positions, and accepted 917 of 921 draft
tokens. Its 9.0 GB/rank matrix measured 30.80/44.32/71.28/100.40 aggregate
tokens/s at C1/C2/C4/C8, improving every cell over MTP2.

Qualification, endpoint, matrix, and transport-audit SHA-256 values:

```text
1d4992e6b76f01200643f0084b3cdb971366f70bfd7d4e37fa2073a711663801
79359021a8ec6343d842e9dc77c9605e737d2e2e0af6488e97dd6a0b3fef8b88
4763397fac0ec9b92414b1d9d61baa772327599b8f5b12cc368a032630d95e89
65742b2214aaf3ff23b23b8152463714d454f56484e1711e685930dbc5aec38c
```

Fixed MTP3 passed and became the control for KV and fixed-MTP4 testing.

### 9.25 GB/rank KV allocation

Increasing the KV allocation from 9,000,000,000 to 9,250,000,000 bytes per
rank raised reported capacity to 675,840 tokens. MTP3 output equivalence and
speculative counters were unchanged. The matched matrix measured
30.80/47.16/69.72/96.96 aggregate tokens/s; it introduced no systematic
throughput gain, so the allocation is retained for capacity rather than speed.

The first near-64K probe was invalidated by a whole-host/fabric outage, not a
memory failure. After the hosts were restored, the fixed-MTP4 capacity gate
proved the allocation with eight simultaneous 64K-class requests. The final
capacity evidence is part of the fixed-MTP4 record rather than attributed to
the invalid MTP3 attempt.

The MTP3 qualification, endpoint, and matrix SHA-256 values for the 9.25 GB
allocation are:

```text
262cc3e80ca2a2bc4efe0cf79cb51faf71400494e2b30bccb3cd6455281f1b27
68731741d297bb084a0179f9aa0428c2861cb52f6fe18f6adab9e80d5974d748
e1054db12b5917b21ae404bdff561b22536ca147d461b2300af5136bc20364cf
```

### Fixed MTP4 and retained result

Fixed MTP4 passed output equivalence, four-position speculation, Q1-Q40
transport, endpoint, two decode suites, coding peak, and long-context capacity
gates. It improved the matched C1-C4 cells over MTP3 but regressed C8 by
11.63%. The retained serving choice favors its single-user and moderate-
concurrency gains; MTP3 is the exact rollback and the better matched C8
control.

See [GLM52_35BPW_FIXED_MTP4_PROFILE.md](../../../docs/GLM52_35BPW_FIXED_MTP4_PROFILE.md) and
[mtp4-kv925-20260811.json](mtp4-kv925-20260811.json)
for the complete bounded result.

### Dynamic-NVFP4 CKV-gather serving update

The operator-running fixed-MTP4 service uses dynamic per-token NVFP4
latent KV, FP8 RoPE, a 262,144-token request limit, a 4,096-token prefill
ceiling, and 9.25 GB KV/rank. It reports 1,156,864 KV tokens. The final
prefill optimization changed only
`VLLM_B12X_MLA_CKV_GATHER=1` and
`VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS=262144` from the otherwise identical
dynamic-NVFP4 control.

The matched single-sample cold-prefill rows improved 14.85%, 39.16%, 29.71%,
and 29.83% at 8K, 16K, 64K, and 128K. The matched C8 16K decode cell measured
47.85 tok/s versus the control's displayed 45.4 tok/s. CKV gather is not a
decode path, so the latter is retained only as a no-regression result.
Deterministic MTP0 parity, finite logprobs, four-position MTP4 counters, Q40
transport parity, and post-run idle health passed on the exact profile.

See
[mtp4-nvfp4-ckv-gather-20260811.json](mtp4-nvfp4-ckv-gather-20260811.json)
for the configuration, hashes, measurement values, and limitations. The
larger compressed-KV pool has not yet repeated the FP8 predecessor's
near-capacity residency gate.

## Rollback and service availability

Every service-changing candidate used a distinct generated launch/site pair
and an exact predecessor rollback. The fixed-MTP4 rollback is byte-identical
to fixed-MTP3 with 9.25 GB/rank KV. Failed startup or load gates were stopped
before inference, and failed runtime arms were drained to zero running, zero
waiting, and healthy API state before another candidate advanced. The final
fixed-MTP4 capacity receipt ends at health HTTP 200 with zero scheduler and KV
use.

## Unsupported extension

Fixed MTP5 is not compatible with the qualified image as a profile-only
change. Eight sequences at depth five require Q48. The Python query contract
and native vocabulary/DCP/indexer decode-family caps are built for Q40. A
bounded MTP5 experiment therefore requires test-first Q48 contract changes, a
rebuilt native library and image, new graph-memory qualification, and an exact
fixed-MTP4 rollback.
