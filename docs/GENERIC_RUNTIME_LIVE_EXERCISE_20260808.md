# Generic runtime composition supervised exercise, 2026-08-08

> **Lane:** public-functional | **Tooling maturity:** offline-validated |
> **Canonical EXL3+LMCache deployment evidence:** live-validated, bounded |
> **Generic bundle execution:** not performed; the canonical bridge is
> intentionally plan-only | **Hardware:** four directly cabled NVIDIA DGX
> Sparks | **Exercise status:** the canonical stack was restored cleanly after
> an operator-coordinated reboot, but an isolated sustained C8 attempt
> reproduced rank-1 host/fabric loss; that cell is invalid and the corrected
> sustained matrix, cache-boundary gate, rollback, and restore remain open

This record separates two things that are easy to conflate:

1. the generic multi-service bundle layer was validated offline, including its
   deterministic plan, ownership guards, dependency order, failure behavior,
   and invocation-local rollback ledger; and
2. the current pinned EXL3 3.25-bpw plus LMCache CS512 stack was exercised live
   through its canonical launcher.

The bundle's `canonical-exl3-lmcache-cs512` source remains plan-only. A live
canonical launch does not turn that bridge into a generic native execution
path or establish acceptance for arbitrary bundles.

## Exact target

The live target was the repository's default public-functional recipe:

- model `willfalco/GLM-5.2-EXL3-TR3-3.25bpw` at revision
  `d7d79c2d14599dfce7a5d12b85f7ad73f40e623d`;
- image
  `sha256:20c4099f2e7e3dd3c8ab64f7d7930bde4f372df1895aa3ffa593252ca04ae96f`
  on all four ranks;
- TP4/DCP4, fixed MTP2, 4,096 batched tokens, eight sequences, Q32 graph
  ceiling, 524,288 maximum model length, and 4.5 GB KV allocation per rank;
- NVFP4 latent plus FP8 RoPE KV, with 562,688 tokens reported;
- native prefix caching enabled;
- one LMCache MP server per rank, 512-token chunks, lazy 1-GiB L1, LRU;
- SparkCache disabled.

Addresses, SSH identities, hostnames, local paths, and registry identities are
intentionally excluded from this document.

## Offline composition result

The rank-scoped bundle implementation and its final safety hardening are in
commits `2d42e95` and `83a8e22`. The canonical bridge plan was generated twice
from the same resolved private inputs. Both byte streams were 59,021 bytes and
had SHA-256:

```text
8a6acc811d3b5b850b88e12031c42235af91122eca1c26b2ecb528879144dec0
```

The plan targeted only the four exact-label engines and four exact-label
LMCache servers. It did not target historical containers or the independent
observer. The bridge reported `execution_supported: false`, as required.

Native generic execution cannot faithfully express this EXL3 deployment
without weakening the public schema. The canonical launcher owns required
runtime environment, site-derived LMCache server URLs, model attestation,
privilege/device policy, entrypoint behavior, phased readiness, and exact-label
rollback. The supported execution path therefore remains
`scripts/sparkring_exl3_lmcache_launcher.py`.

## Live preflight and lifecycle evidence

After management connectivity recovery, the read-only public preflight passed
116/116 checks. It verified SSH, the four management identities, all four
direct 200-Gb/s MTU-9000 links, active Ethernet-mode RDMA, RoCE v2 GID 3,
8,972-byte jumbo pings, peer control, free serving/rendezvous ports, cache
paths and free space, and the identical exact image ID on every rank.

The canonical `restart-stack` lifecycle then completed successfully. A later
operator-coordinated reboot recovered all four hosts after the first rank-1
loss. The complete read-only preflight was rerun from a clean post-reboot
state and again passed all 116/116 checks. The same exact canonical stack then
completed another successful `restart-stack`, with all four engines and all
four LMCache servers healthy. Across the successful canonical lifecycle:

- four LMCache servers passed health before engine start;
- all 81 EXL3 shards passed the outer and in-container verifier on every rank;
- all four engines loaded 84.43 GiB/rank with the expected 192 K3 plus 64 K4
  mixed-expert tiers;
- LMCache pointer-transfer integration registered one GPU context per server;
- 16/16 piecewise and 12/12 full CUDA-graph captures completed;
- every engine and server reported running, zero restarts, and no OOM;
- `/health` returned HTTP 200 and `/v1/models` reported the expected served
  name and 524,288-token maximum length; and
- a repeated fixed-seed request returned byte-identical reasoning and exact
  content `SPARKRING LIVE OK`.

Before the reboot, the repository's bounded `exl3_live_gate.py` passed with
deterministic 128-token output and no threshold failures:

| Cell | Aggregate completion tok/s | Published floor |
|---:|---:|---:|
| C1 | 22.14 | 15 |
| C2 | 30.91 | 24 |
| C8 | 74.86 | 35 |

These are short deployment-gate samples, not the sustained 16K performance
matrix. They must not be substituted for the sustained 16K table in
`docs/RESULTS.md`.

The first cold deterministic gate after the reboot had a last-token
divergence. Three immediately following fixed-input repetitions matched one
another, and the warmed bounded gate then passed every configured floor:

| Cell | Aggregate completion tok/s | Published floor |
|---:|---:|---:|
| C1 | 21.9374 | 15 |
| C2 | 30.2472 | 24 |
| C8 | 69.3897 | 35 |

The matching warmed repetitions are useful deployment evidence, but they do
not erase the cold last-token divergence or satisfy the broader deterministic
correctness gate. These remain short finite-request samples, not sustained
16K decode rates.

## LMCache observation

After the workload, all four LMCache servers independently reported the same
healthy state: 197 L1 objects, 806,912,000 bytes used of 1 GiB, no locked or
temporary objects, no pending or in-flight work, and one live registered GPU
context. Each reported 101 layers, 2,198 blocks, and the same two MLA kernel
groups. This proves population and consistent server geometry during this
exercise. It does not by itself prove reuse across an engine restart or
persistence across a server restart.

## Invalid sustained-matrix attempts and failure boundary

The first `llm_decode_bench.py` v0.4.31 attempt used one 16K context row,
C1/C2/C4/C8, 25-second duration windows, 2,048 maximum tokens, temperature
zero, fully unique contexts, DCP4, the exact 562,688-token KV budget, and the
harness's automatic readiness policy.

Only C1 was valid, at 15.37 tok/s, and it remains a diagnostic rather than a
replacement matrix. C2/C4/C8 missed readiness and were correctly suppressed.
Their estimated peak demand was only 147,456 tokens at C8, so the result did
not demonstrate KV exhaustion. The automatic policy gave C2 and C4 roughly 60
seconds; each interval included a cold stream-zero scout plus additional
unique prefills that took roughly 32-35 seconds apiece. C8 then spent 601
seconds in a polluted readiness phase because abandoned work from earlier
cells remained running or queued. The corrective design is to start from an
idle engine, run cells independently from C8 to C1, disable the hidden decode
warmup, and allow 600 seconds for readiness.

Before that corrected run, rank 1 disappeared from its management interface
and both independent direct-ring addresses. Rank 0 continued returning HTTP
200 from `/health`, but distributed generation stopped: running/waiting
requests and the generation-token counter remained unchanged. This is useful
negative evidence that the API health endpoint alone is insufficient for a
multi-rank readiness claim. The exercise stopped rather than publishing
partial rates or treating the shallow health response as cluster health.

After the operator-coordinated reboot, clean preflight, canonical restart, and
warmed bounded gate, the corrected campaign began with an isolated sustained
C8 cell. Under that load, rank 1 again became unreachable on its management
interface and both neighbor-facing production links lost carrier. Ranks 0, 2,
and 3 remained alive. The C8 benchmark cell is invalid: it did not complete a
healthy eight-stream measurement and no throughput is claimed from it. This
independent reproduction narrows the failure boundary to rank 1 or its
host/fabric path under load; it does not by itself establish a root cause.

The lower-memory experimental profile
`kv4gb-480k-l1-0.5gb` has been prepared for a controlled follow-up. It reduces
the per-rank KV allocation to 4.0 GB, caps model length at 491,520 tokens, and
uses a 0.5-GiB LMCache L1. It has not been launched on the cluster, so it has
no live stability, capacity, or performance result and is not the canonical
default.

## Remaining gates

This exercise does not close any of the following:

1. diagnose or otherwise resolve the reproducible rank-1 host/fabric loss;
2. recover rank 1, repeat the full read-only preflight, and restore an idle
   exact-identity stack;
3. run independent fully unique 16K C8/C4/C2/C1 cells with explicit readiness
   allowance and reject any underfilled cell;
4. live-test the named lower-memory profile against the canonical control;
5. demonstrate LMCache reuse across an engine-only restart with cache servers
   preserved;
6. execute canonical rollback, verify rollback, and restore the best stack;
7. finish the broader correctness and release-promotion gates in
   [EXL3_ACCEPTANCE_RUNBOOK.md](EXL3_ACCEPTANCE_RUNBOOK.md).

Until those items are complete, the repository's existing Aug. 3 clean-image
matrix remains the current sustained performance evidence. This interrupted
exercise adds lifecycle, cache-population, warm bounded-gate, and reproducible
failure-detection evidence only.
