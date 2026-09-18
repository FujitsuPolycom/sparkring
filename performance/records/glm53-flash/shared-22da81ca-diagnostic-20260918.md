# GLM shared-image memory and replay investigation

Status: **research-only; unresolved failure causes**. This analysis examines
retained evidence from the [bounded GLM cache trials](shared-22da81ca-tp2-tp4-cache-20260918.md),
not additional hardware qualification.

The tested ARM64 image is
`sha256:22da81cae0572ae2985a5c34a125db4f3cc58e871fa7e6256dffe59828f1ae5d`;
the GLM checkpoint revision is `df116c4fb16b1d37ae43d2cfd624de26ffbc832e`.
Both topologies used DCP1, MTP3 and the B12X loader.

## TP2 memory evidence

| KV pin per node | Context / sequences / batch | Rank-0 reported model-load memory | Reported KV tokens | Result |
|---|---|---:|---:|---|
| 7.5 GiB | 1M / 8 / 8192 | 100.59 GiB | 1,081,922 | Did not reach readiness; memory-pressure trial was terminated |
| 4 GiB | 262K / 8 / 8192 | 100.59 GiB | 447,949 | Reached readiness and passed smoke; later cache-seeding request disconnected |
| 2 GiB | 64K / 4 / 4096 | 94.49 GiB | 73,631 | Bounded generation and restart/restore passed with separately stated answer checks |

With 7.5 GiB pinned and two compiler workers, available host RAM was 393/368 MiB
and swap usage was 8,090/6,791 MiB across the pair. Both Docker records report
`OOMKilled=false`; recorded exit 137 followed operator termination. These
observations establish unsafe pressure, not a proven kernel OOM kill. Reducing
compiler workers from eight to two did not establish a viable configuration.

The 4 GiB trial is specifically **not a startup failure**. Its seed request
failed with `RemoteProtocolError: Server disconnected without sending a response`.
Retained startup evidence does not identify the termination mechanism.

The 2 GiB configuration also reduced context, concurrency, batch budget and
maximum graph size, changed target/recurrent pages from 2048/256 to 512/512,
and disabled KDA coalescing. The 6.10 GiB difference in reported model-load memory
therefore cannot be attributed to KV pinning or any one setting.

Manual KV pinning bypasses automatic memory profiling. Initial free memory and
negative allocator deltas during graph capture do not establish post-load
headroom. SparkCache's 4 GiB persistent-store limit is a disk budget, not a
4 GiB RAM reservation; capture slots and restore arenas are separate allocations.

## Replay evidence

The initial TP4 replay passed the 7,870-token fixture with 7,680 tokens restored
on every rank, then failed exact-answer validation for the 8,194-token fixture.
**The failed raw response was not retained.** A formatting failure cannot be
distinguished retrospectively from an incorrect key or a restoration defect.

An unchanged full-worker restart subsequently returned both exact keys with
7,680/8,192 restored tokens on all four ranks. This is a passing repeat, not a
fix or proof of the original failure's cause. A different request salt still
received cache credit in that image, so that request was not a valid cold
control. An attempted reset returned HTTP 404 and establishes no reset result.

A TP2 cold request independently returned the correct key wrapped in Markdown
bold, with zero cached tokens. That demonstrates that exact-format failures can
occur without restoration; it does not explain the unrecorded TP4 response.

## Conditions needed to resolve the causes

For replay, compare identical fixtures and sampling on fresh workers with an
empty persistent namespace against fresh workers with a verified seeded
namespace. Keep SparkCache enabled in both arms. Retain raw responses before
assertions, and report finish status, exact formatting, literal-key identity and
all-rank external restore separately. Alternate fixture order; inspect boundary
lengths only after reproducing the original symptom. Salt changes are not a
coldness guarantee for the tested image.

For memory, preserve the successful 64K configuration and vary one allocation
at a time. Sample available RAM, swap growth, process RSS/PSS and cgroup events
through loading, graph capture, prefill and idle. Retain operator termination
separately from runtime failure. The existing records do not establish a safe
higher KV ceiling or identify the allocation responsible for the 6.10 GiB delta.

Neither investigation supports GLM profile promotion. These limitations remain
independent of the bounded Qwen performance and cache-restoration results.
