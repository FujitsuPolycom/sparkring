# GLM-5.2 EXL3 3.5-bpw normalized base benchmark

Lane: **public-functional candidate**. Maturity: **live-validated candidate**;
not qualified. Hardware: four directly cabled NVIDIA DGX Sparks in a cycle,
TP4/DCP4. Evidence scope: fixed MTP4, dynamic NVFP4 MLA plus FP8 RoPE,
1,048,576-token request limit, 16 sequences, 4,096 batched tokens, 9.25 GB KV
per rank, block size 64, native prefix caching, target-only exact Q40, and no
SparkCache.

## Prefill

| Context | Prompt tokens | TTFT | Prompt tok/s |
|---:|---:|---:|---:|
| 2K | 2,050 | 2.95 s | 694 |
| 8K | 8,194 | 12.14 s | 675 |
| 16K | 16,386 | 24.41 s | 671 |
| 32K | 32,770 | 49.56 s | 661 |
| 64K | 65,538 | 100.92 s | 649 |
| 128K | 131,073 | 206.55 s | 635 |

## Sustained decode

Temperature datasets changed only temperature and left `top_p` unset.
Aggregate values are generated tokens per second.

| Context | T=0 C1 | C2 | C4 | C8 | T=1 C1 | C2 | C4 | C8 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2K | 23.13 | 35.34 | 50.95 | 71.82 | 22.001 | 28.283 | 46.981 | 67.622 |
| 8K | 22.87 | 34.17 | 51.26 | 74.50 | 19.153 | 30.207 | 47.697 | 65.534 |
| 16K | 21.60 | 33.45 | 49.14 | pending | 20.151 | 32.384 | pending | pending |
| 32K | 22.40 | 34.62 | 48.98 | pending | 21.611 | 30.524 | pending | pending |

The temperature-1 rows had 99.7–100% client coverage, exact client/server
agreement, requested concurrency, zero queue/errors, and clean all-rank logs.
The temperature-0 rows passed token checks but their hardware summaries predate
the exact measurement-boundary fix.

## Temperature probe and coding workload

| Temperature-0.9 C1 probe | 2K | 8K |
|---|---:|---:|
| Aggregate tok/s | 19.999 | 22.056 |

GLM Coding Peak at temperature 1.0 completed five normal requests: mean 25.391,
median 25.565, range 22.699–26.919 tok/s. All five stopped normally.

## Lifecycle and evidence limits

The normalized stack reached ready state from a fresh rank-specific JIT and
create-once receipt namespace. Restarting the preserved containers in the same
namespace fails before model startup because the exact-Q40 producer refuses to
overwrite an existing receipt. This is an artifact-lifecycle limitation, not a
model crash. Preserve existing receipts; use a fresh namespace until matching
receipt revalidation/reuse is implemented and tested.

- Temperature-1 C4/C8 at 16K/32K, longer-context temperature-1 cells, and
  temperature-0 C8 at 16K/32K are pending.
- Rows named `DISCARD`, `ABORTED`, or JIT-affected are excluded.
- Saved last-scrape acceptance is not used as a run average.
- Pre-boundary-fix hardware summaries are invalid for exact thermal attribution.
- Available memory was approximately 1.3–3.5 GB per rank during this campaign;
  the campaign stopped at 128K and makes no 256K-or-longer performance claim.
- [The machine-readable summary](normalized-base-20260822.json) records table
  values, pending coordinates, and exclusions.
