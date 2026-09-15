# GLM-5.3-Flash TP4 decode and cold-prefill performance

These measurements describe the R35 SparkRing image on four NVIDIA GB10 nodes.
They are **performance evidence, not stability qualification**. A native TP4
collective stall observed during serving remains unresolved. Long-duration
stability testing is deferred; these records do not establish a stable release.

The [measurement record](r35-tp4-direct-performance.json) contains exact image,
model revision, source identities, raw-result hashes, settings and ranges.
Raw files are retained locally because they contain private deployment metadata;
the record includes the curated measurements needed to read these results.

The model is GLM-5.3-Flash with NVFP4-Spark weights at revision
`df116c4fb16b1d37ae43d2cfd624de26ffbc832e`. The image identity is
`sha256:7b698d4299aaaebb359e287d75c7f18275311b6a6d56322d9767b0b4f35cd60b`.
Serving uses TP4/DCP1, MTP3, direct doorbell submission, dual-domain NCCL,
SparkCache read-write, mHC and prefill coalescing. Each rank has 24 GiB of FP8 KV
memory; the request limit is 1M tokens, batch limit 8,192 tokens and sequence
limit 16. OpenMP uses one thread; graph submission and progress use CPUs 10 and
11. These are the measured settings, not a claim that each enhancement improves
throughput independently.

## Sustained decode

The unmodified `llm_decode_bench.py` 0.6.2 harness ran three repetitions of eight
cells, each measured for 60 seconds after readiness, with a 30-second global
warmup, temperature 1, an 8,192-token output limit and `ignore_eos=true`.
All 24 cells reported zero errors, no underfill, no capacity limit, no warmup
timeout and no detected exact-text loop. That heuristic does not establish
semantic correctness. Prefill timing was skipped; the 16K requests still
contained tokenized input. Zero added context also includes the harness prompt.

| Added context | Concurrent requests | Median aggregate output tokens/s | Run range |
| --- | ---: | ---: | ---: |
| 0 | 1 | 54.37 | 54.33–54.93 |
| 0 | 4 | 122.14 | 120.26–124.25 |
| 0 | 8 | 172.37 | 168.94–177.01 |
| 0 | 16 | 248.71 | 239.28–257.33 |
| 16K | 1 | 54.06 | 53.48–60.25 |
| 16K | 4 | 120.73 | 119.46–121.29 |
| 16K | 8 | 176.28 | 172.90–178.42 |
| 16K | 16 | 253.03 | 249.23–253.80 |

Aggregate throughput grows sublinearly with concurrency. Server verifier-step
counters in the JSON are aggregate counters, not whole GPU batch forward passes.
There is no matched R33 baseline or complete command-ring comparison here.

## Cold prefill

Each context size has three independent run-level measurements at concurrency
one. The 30-second sampling target produces seven samples per 8K run, three per
32K run, and one per larger-context run. The table reports medians across runs,
not pooled sample medians. Client throughput includes first-token latency.

| Requested context | Actual prompt tokens | Total samples | Client tokens/s | First token, seconds | Server computed tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8K | 8,194 | 21 | 3,497 | 2.343 | 3,537 |
| 32K | 32,770 | 9 | 3,537 | 9.265 | 3,561 |
| 128K | 131,074 | 3 | 3,399 | 38.563 | 3,419 |
| 512K | 524,290 | 3 | 2,864 | 183.078 | 2,879 |
| 900K | 900,002 | 3 | 2,499 | 360.078 | 2,513 |

Nonce-based prompts produced zero cached tokens in every retained server
validation, with `kv_computed` accounting and no invalid result reason. Model
kernels were warm; KV prefixes were cold. SparkCache remained enabled, but these
measurements do not test cache reuse or restart restoration.

A separate [managed restart record](r35-managed-restart.json) verifies one
coordinated four-rank restart. Every worker restored the 8K prefix; the request
returned the expected answer with 8,192 cached tokens and zero recreated cache
tokens in 0.72 seconds. This bounded check does not establish long-run stability.

Contexts through 128K use the unmodified 0.6.2 harness. The 512K and 900K runs
use a task-local variant that honors requested standalone-prefill sizes up to
the reported model limit minus 64 tokens, retaining a 128K fallback when the
limit is unavailable. Timing, nonce construction, token calibration, metric
validation and model/KV safety checks are unchanged. Both harness hashes are
recorded in the JSON; the larger runs must not be attributed to unmodified
0.6.2 behavior.
