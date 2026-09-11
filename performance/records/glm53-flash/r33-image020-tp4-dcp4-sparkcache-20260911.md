# Generic R33 image TP4/DCP4 SparkCache qualification

Status: **qualified** for the bounded functional checks below; served from
the unchanged published R33 image. Throughput values are **research-only**
observations and were not recorded in this session's harness.

## Conditions

Four NVIDIA GB10 systems ran the published generic R33 image
`ghcr.io/fujitsupolycom/sparkring@sha256:1328a4f6f483014021a66a757012793629bd054d28d0fe4d5e581fa4aed776ef`
(config ID `sha256:3c7779ad71dd0d5d6fae4c98e04b94c377429306158c2259fc44635892b8b8e4`).
The deployment used the managed mesh renderer with the DCP4 profile
templates added to the R33 profile contract in the same change, rendered
against the existing `tp4-dcp1-sparkcache` site
(`runtime/glm53-spark-mtp3-mesh/profile.py`), with the updated
profile-contract directory and entrypoint supplied through
`R33_PROFILE_CONTRACT_HOST_ROOT` bind mounts. The published image bytes,
its artifact lock, and its NCCL library identity are unchanged.

The model was `local-inference-lab/GLM-5.3-Flash-NVFP4-Spark` revision
`df116c4fb16b1d37ae43d2cfd624de26ffbc832e` (config SHA-256
`e1c0246a44ebefb5fd6383fb57aebbf7ac69ff6e7b23e989c0571b279a0eca23`). The
serving profile used TP4/DCP4, PP1, `dcp_comm_backend=ag_rs`, InstantTensor
target loading, static MTP3, 24 GiB FP8 KV per rank, 16 sequences, an
8,192-token scheduler budget, a 1,048,576-token request limit,
hardware-forwarded ring links with the fabric plan installed on every rank
(routes, ingress qdiscs with tc flower rules, and two RDMA-TX rewrite
markers per rank), dual-domain NCCL, token-sharded mHC,
continuation-prefill coalescing, and SparkCache with async page capture.

The RoCEnante overlay pins `decode_context_parallel_size: 4`; DCP4
activations therefore require the managed fabric installation before
launch. DCP1 activations disarm that overlay and do not require it.

## Evidence

- Engine initialization on rank 0 reported TP4/DCP4, `dcp_comm_backend=ag_rs`,
  MTP3, InstantTensor, and the profile's exact
  `cudagraph_capture_sizes=[4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64]`.
- Graph capture completed on every rank (`Graph capturing finished in
  61–78 secs`, 2.95–3.11 GiB).
- The mapped NCCL library hashed to
  `84a4b8d83fb5fa1f0d640d311ad38b45140672dae9889775fe1e4a3990479e47`
  (`libnccl.so.2.31.2`, NCCL 2.31.2) on all four ranks, and the HCAs span
  PCI domains `0000` (`rocep1s0f0/f1`) and `0002` (`roceP2p1s0f0/f1`).
- Four exact-answer semantic requests returned exact responses.
- Two completed cold prompts of 37,032 tokens each with zero cached tokens,
  no `sample_tokens` timeout, and no fatal engine error.
- A repeated prompt with an identical 37,019-token prefix hit the cache on
  its second run: 36,352 cached tokens, 15.8 s → 5.0 s wall time.
- mHC token sharding executed on every rank:
  `GLM_MHC_PREFILL rank=<r> rows=8192 owner_rows=2048 rs=90 ag=90 aux=0`,
  two executions per rank.
- SparkCache reported `sparkcache_ranks=4 … healthy=1` throughout; page
  snapshots committed (`kind=page_snapshot outcome=committed payload up to
  100,200,614 B`).
- A deterministic writer/restart/reader sequence: the writer created a
  7,831-token entry; the whole ring restarted (planned container recreation,
  cold engine start); the reader restored 6,144 tokens from SparkCache on
  every rank in 50.6–54.8 ms (112–121 K tok/s, 46.3 MiB per rank).
- A fault-injection recovery: all four containers were killed (`docker
  kill`, SIGKILL), relaunched cold, returned to healthy serving, and the
  reader restored 6,144 tokens on every rank in 51.9–57.7 ms (106–118 K
  tok/s).
- The engine admitted the one-million-token request limit and reported
  `GPU KV cache size: 8,364,901 tokens` (the DCP4 sharded pool; the DCP1
  profile reports ~2.28M on the same 24 GiB/rank).

The activation receipt is
[`evidence/tp4-dcp4-sparkcache-activation-20260911.json`](../../../runtime/sparkring/jovian-r33/profiles/evidence/tp4-dcp4-sparkcache-activation-20260911.json),
bound to the profile contract SHA-256 recorded inside it and passing
`verify_profile.py validate-activation`.

## Limitations

This record does not include three-coordinated-cold-start latency evidence,
an INFO-level NCCL diagnostic startup (`NCCL_DEBUG=WARN` throughout, so no
`NET/IB RouteFinal` records were emitted), or decode/prefill throughput
windows. Those additions follow the TP2/TP4 DCP1 record format in a later
run. No completed one-million-token request was run; multimodal correctness,
switched hardware, other models, and sustained-memory behavior do not
inherit this qualification. The configured limit is not a tested workload.
The mounted profile-contract overlay is verified for internal consistency
and by the profile contract SHA; its contract bytes are newer than the
baked image contract and the published image remains byte-unchanged.
