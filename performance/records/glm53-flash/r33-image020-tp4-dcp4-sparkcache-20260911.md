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

**Post-qualification restart event (2026-09-11 15:22:14 Central):** all
four containers stopped simultaneously (containerd shim teardown on
every host, coinciding with an SSH session logout and a transient
withdrawal of the fabric interface addresses at 15:22:07). Every
relaunch attempt since — including after full host reboots, managed
network-plan reinstallation (routes, qdiscs, flower rules verified with
`ib_read_bw` at ~109 Gb/s on every direct pair), memory-gate passes, and
correct overlay mounting — dies silently during engine initialization's
first cross-rank collective (determine-available-memory phase); no OOM,
XID, cgroup limit, or kernel trace. The 13:09–14:20 qualifications ran
before this event and are unaffected; later observations in this record
are stamped and the ring has not served since 14:20 Central.

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
- DCP prefill execution used the DCP4-specific path, verified in the
  published image source: the `_use_b12x_full_ckv_gather` gate
  (`dcp_world_size > 1`, prefill-only) selects the local
  `cp_gather_cache` + DCP-group all-gather path, and top-k selection
  resolves per-token ownership arithmetically over the gathered cache in
  `_map_global_topk_to_gathered_ckv_kernel`
  (`owner = (tok // DCP_INTERLEAVE) % DCP_SIZE`). Worker logs on every
  rank report the `Using full-CKV gather for GLM5Next B12X DCP prefill`
  gate line. Note: this image has no separate topk-to-owner message
  exchange step — ownership is resolved after the all-gather, not
  exchanged — so the evidence is the gate line plus the
  all-gather/owner-mapping code path, not a distinct exchange primitive.

  **Scope change vs the issuing requirement:** #264 asked for "DCP top-k
  owner-exchange activation evidence from worker logs". This image
  implements DCP4 prefill top-k **without** a distinct owner-exchange
  primitive (ownership is arithmetic over the all-gathered cache), so the
  original requirement is satisfied only under an amended reading: the
  evidence establishes the DCP4 gate, the DCP-group all-gather, and the
  owner-mapping kernel — not a separate exchange. The requirement was not
  silently reinterpreted: this note records the implementation change and
  that a different implementation does not automatically satisfy the
  original acceptance item; any reviewer of this record should treat the
  owner-exchange item as amended, not passed.
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

## Measurement

(2026-09-11 evening pass, cache-enabled ring. An additional diagnostic
RouteFinal startup and a cache-disabled observation follow this section.)

Prefill values are medians of three cold requests per prompt size; all nine
requests reported zero cached tokens (unique prompt text per sample; prompt
lengths within ~1% of the target sizes are recorded in the sample table).
Decode used 10-second windows with zero request errors; each window was
seeded by a warmup request over the full context so decode ran against a
resident prefix. MTP-normalized steps/s counts target-model forward passes
only (`spec_decode_num_drafts_total` delta per window); the effective
acceptance length divides `spec_decode_num_accepted_tokens_total` deltas.
Throughput values remain **research-only** observations: the harness does
not pin clocks, warm-up policy, or a timing revision, so the tables are not
a reproducible benchmark or a speedup claim. **Fabric-overlap caveat:**
the DeepSeek V4.1 distribution jobs (`deepseek-distribution-fabric.service`,
~37–52 GB copies) ran on this ring's hosts from 13:09:30–13:42:50 Central
on 2026-09-11 (ranks 0, 1, and 3; rank 2 none). The first measurement
pass — the two 37K prefill probes, the prefill medians, and the C1/C4
decode windows — ran between approximately 13:28 and 13:39 Central,
**overlapping the tail of those jobs**, and the windows are labeled
**potentially affected**: distribution copies saturate the same QSFP
fabric links the DCP collectives use. Container census at each
stop/launch cycle showed no second serving tenant, and #260's cycle
ring is separate, but neither fact rules out host-level copy traffic.
The RouteFinal diagnostic (13:39+), nocache pass (13:56+), and final
decode windows (14:08+) started after the last job deactivated at
13:42:50 and are not overlapping. A post-job rerun was attempted;
the ring could not be restarted after the 15:22:14 Central multi-rank
stop event (see Conditions note), so the affected windows have not yet
been re-measured.

| Prompt tokens (actual) | Cold prefill tok/s, median of 3 |
|---:|---:|
| 8,192 (8,185–8,186) | 2,398.4 |
| 16,384 (16,341–16,342) | 2,766.2 |
| 32,768 (32,652–32,653) | 2,998.4 |

| Context | Concurrency | C1/C4 aggregate decode tok/s | MTP-normalized steps/s | Effective acceptance length |
|---:|---:|---:|---:|---:|
| 8,192 | 1 | 50.2 | 18.2 | 2.76 |
| 32,768 | 1 | 48.8 | 18.3 | 2.67 |
| 8,192 | 4 | 100.7 | 35.7 | 2.82 |
| 32,768 | 4 | 104.4 | 37.2 | 2.80 |

## Result

C1 decode matches the TP4/DCP1 record within 2–3%; C4 aggregate decode
reaches 0.76–0.82× of DCP1. **No attribution is claimed for the C4 gap:**
the full-CKV gather is a prefill-only path (`num_decode_tokens == 0` in
its gate), so it cannot explain a decode-window difference, and no
profiling evidence exists for any cause. Candidate explanations — the
DCP4 cross-rank exchange in decode-path collectives, scheduling/overlap
differences under concurrency, and (for the affected windows) the fabric
overlap documented above — are unverified. Effective acceptance length
is at or slightly above the DCP1 record (2.21–2.76). The DCP4 exchange
buys ~3.7× KV capacity (8.36M vs 2.28M tokens on the same 24 GiB per
rank).

## RouteFinal dual-domain diagnostic startup

A diagnostic start rendered the same site with `nccl_debug: INFO`
(`NCCL_DEBUG=INFO`) through the unchanged published image. The ring reached
HEALTH-OK and served an exact semantic answer. Every rank emitted 32
`NET/IB RouteFinal` lines covering all four HCAs across both PCI domains:
`rocep1s0f0` and `rocep1s0f1` (domain 0000) and `roceP2p1s0f0` and
`roceP2p1s0f1` (domain 0002), 8 RouteFinal records per HCA, with
`crossNic 1`, 32 channels, and `Connected all rings, use ring PXN 0 GDR 0`.
This is the dual-domain route attribution the DCP1 record uses, now
confirmed active for DCP4.

## Six-case prefix-hit regression (1,027-token suffix included)

After a 32,256-token cache hit established the base entry, six suffix
geometries completed with exact expected answers: 515, 513, 531, **1,027**,
512, and 2,048 tokens, with 28,160–28,672 cached tokens per case. As in the
DCP1 record the harness checked completion liveness and exact answers, not
raw output equivalence across shapes. No #220-style zero-hit behavior was
observed: every case hit the cache.

## Cache-disabled tp4-dcp4 observation

A separate ring start rendered the cache-disabled `tp4-dcp4` profile
(`SPARKCACHE_ENABLED=0`). Prefill medians of three cold samples per size
(all zero-cached): 8,192 → 2,451.4 tok/s; 16,384 → 2,885.7; 32,768 →
3,141.7 (one 8,192 sample absorbed a fresh-start JIT cost, 524 tok/s, and
the median remains the middle of three). Decode windows: 8,192 c1 → 50.2
(norm 18.4, acc 2.73); 32,768 c1 → 52.8 (norm 18.5, acc 2.85); 8,192 c4 →
61.1 (norm 21.9, acc 2.80); 32,768 c4 → 104.6 (norm 35.2, acc 2.97). No
snapshot writes or restores occurred in any worker log; the SIRCL
capability vote remained (transport handshake) and vLLM's in-engine prefix
caching stayed enabled (28,160-token hits in the suffix cases), so
`SPARKCACHE_ENABLED=0` disables the SparkCache snapshot/capture layer only.

## Conclusion

The TP4/DCP4 SparkCache profile is bounded-qualified on the unchanged
published R33 image through the profile-contract overlay: managed four-rank
startup with graph capture, exact-answer serving, DCP4 prefill
all-gather/owner-mapping evidence, dual-domain RouteFinal attribution,
the six-case prefix-hit regression including the 1,027-token suffix,
planned-restart and SIGKILL fault-injection restore, and the 8.36M-token
KV pool with 1M request admission. #220's zero-hit behavior was not
observed. The cache-disabled `tp4-dcp4` profile starts and serves as an
observation row: functional
checks (four-rank start, semantic answer, 1M admission) passed on that
start as well.

## Reproduction (overlay and quickstart)

The published image contains only the DCP1 profile contract. A DCP4 start
requires the contract overlay from this repository:

1. Clone this repo on every host and export
   `R33_PROFILE_CONTRACT_HOST_ROOT=<repo>/runtime/sparkring/jovian-r33/profiles`
   (the launcher bind-mounts it read-only at the contract host root and the
   entrypoint prefers the overlaid contract; DCP1 starts do not need it).
2. Render the site once on the head host:
   `python3 runtime/glm53-spark-mtp3-mesh/profile.py render --site <site>.json
   --bundle <bundle_root> --output <rendered> --image-receipt
   <image-receipt.json>` with the site's `runtime_profile` set to
   `tp4-dcp4-sparkcache` (or `tp4-dcp4`). The renderer emits one
   `rank<N>.env` per rank; copy them to each host (the rank-0 env stays on
   the head).
3. The DCP4 overlays arm `decode_context_parallel_size: 4`; the managed
   fabric plan (routes, tc flower qdiscs, RDMA-TX markers) must be
   installed before launch — DCP1 silently disarms this and DCP4 does not.
   The `R33_PROFILE_CONTRACT_HOST_ROOT` bind mount and the fabric plan are
   the only two requirements that differ from a DCP1 start.
4. Launch rank by rank:
   `R33_PROFILE_CONTRACT_HOST_ROOT=... bash <launch-site>/launch-rank.sh <rank>
   <rendered>/rank<rank>.env` (or the site's own rank env paths). Rank 0
   reaches readiness ~4–5 minutes after container start; `/health` on the
   rank-0 API returns 200 and a semantic request returns an exact answer.
5. Cache-disabled starts use the same steps with `runtime_profile`:
   `tp4-dcp4`; the renderer sets `SPARKCACHE_ENABLED=0` and the rank-local
   cache root directory must still exist (the launcher requires it even
   when the cache is disabled).

The rendered env files are rank-specific (HOST_IP differs per rank); do not
reuse a rank-0 env on a peer host.

The engine admitted the one-million-token request limit and reported
`GPU KV cache size: 8,364,901 tokens` (the DCP4 sharded pool; the DCP1
profile reports ~2.28M on the same 24 GiB/rank).

The activation receipt is
[`evidence/tp4-dcp4-sparkcache-activation-20260911.json`](../../../runtime/sparkring/jovian-r33/profiles/evidence/tp4-dcp4-sparkcache-activation-20260911.json),
bound to the profile contract SHA-256 recorded inside it and passing
`verify_profile.py validate-activation`.

## Limitations

This record does not include three-coordinated-cold-start latency evidence.
No completed one-million-token request was run; multimodal correctness,
switched hardware, other models, and sustained-memory behavior do not
inherit this qualification. The configured limit is not a tested workload.
The mounted profile-contract overlay is verified for internal consistency
and by the profile contract SHA; its contract bytes are newer than the
baked image contract and the published image remains byte-unchanged.
Prefill and decode throughput windows are research-only observations (see
Measurement); the INFO RouteFinal startup was a separate diagnostic start,
not the qualification serving configuration. The cache-disabled `tp4-dcp4`
row is an observation, not a standalone qualification, and the in-engine
vLLM prefix cache remained enabled on that start (SparkCache snapshot layer
disabled only).
