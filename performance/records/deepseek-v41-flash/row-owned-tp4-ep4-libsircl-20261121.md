# DeepSeek V4.1 expert parallelism beside row-owned prefill on a four-Spark libsircl ring

Status: **research-only**. DeepSeek-V4.1-Flash-1017 ran with its experts
split across four Sparks (TP4+EP4) and with the whole expert set on every
Spark (TP4), on the same libsircl ring, image and checkpoint. Expert
parallelism raised the KV pool per Spark by 18.6%, lowered decode at every
concurrency that campaign measured, and raised 64K cold prefill by 5.8%.

A libsircl link geometry of a 1 MiB slot and 1 MiB pieces, with a 64 KiB
one-shot ceiling, then raised the TP4+EP4 deployment's cold-prefill medians by
5.6%, 9.7% and 11.5% at 16,384, 65,536 and 131,072 tokens, left its decode at
parity, and is what `sparkring install --expert-parallel --row-owned-prefill`
renders. A 16,384-token prefill batch in place of the profile's 8,192 raised
prefill further and cost 10 to 17% of decode at one, four and sixteen
requests, so it is not rendered.

The image's B12X weight loader reads an expert-parallel rank's shard of this
checkpoint correctly, cuts weight load from 116.36 s to 13.88 s and total
API-ready startup from 320 s to 239 s, and matches prefill throughput (6,418.0
and 6,284.5 tok/s warm at 16K and 64K), at the cost of 2.15 GiB of KV cache
(15.62 against 17.77 GiB) and non-bit-identical greedy output; an
expert-parallel install therefore keeps vLLM's loader by default
(`EXPERT_LOADER_EP = False`).

A separate data-parallel deployment (DP4+EP4 over SIRCL-EP's own all-to-all)
served about a third of the tensor-parallel prompt rate, and with DSpark5 and
the rewritten zero-allocation dynamic all-to-all reached 436.13 tok/s at 32
concurrent requests (up from 410.35 tok/s). A fixed-stride static all-to-all
(`SIRCL_EP_STATIC_A2A=1`, cap 96) captures cleanly into CUDA graphs (3,816
captured libsircl calls across 42 FULL graphs) and serves cleanly across all
concurrencies after fixing cross-rank capacity agreement, but its 96-row
padding lowers `cc=1` decode from 33.74 to 19.65 tok/s.

## Conditions

- Four DGX Sparks that SparkRing records as ring 4-7, running in rank order
  8, 7, 5 and 3, in one libsircl ring. Chain 0-3 was not touched.
- Installer image `dev-20261013-dcp2prefill-status038-sircl-tp6`,
  `sha256:3cc9a81fd344fe52fe9cc261b46dc051e3c3f494e51b8142846a5f854835c502`;
  checkpoint `deepseek-v41-flash-tp4` variant `1017`,
  `deepseek-ai/DeepSeek-V4.1-Flash-1017@1bae0e1088993b9b82de08f59de5be5a34e42170`;
  libsircl `0.6.0`, build `8005169b…`.
- The pinned profile's settings, which R0 and R1 ran: TP4, DP1, PP1, DCP1, a
  262,144-token context limit, 16 sequences (`--max-num-seqs`), 8,192-token
  prefill chunks, W4A8 MXFP4 experts (`--moe-backend b12x`,
  `--attention-backend B12X`), disk Engram. The D2 deployments' settings
  beyond their parallel layout and SIRCL-EP's all-to-all are not recorded
  here.
- Each rank reaches every peer over `rocep1s0f0` or `rocep1s0f1`, two 200 Gb/s
  ports, 50 GB/s each.
- vLLM's PyNccl carries the collectives on libsircl. At the tensor width,
  expert parallelism runs no all-to-all kernels: every rank keeps the batch's
  tokens, gathers the rows its experts take and reduce-scatters the result, so
  the grouped collectives are libsircl's all-gather and reduce-scatter
  (`--all2all-backend allgather_reducescatter`). The live log names vLLM's
  `MoEPrepareAndFinalizeNoDPEPModular`, which issues one all-gather and one
  reduce-scatter per mixture-of-experts layer through `ncclAllGather` and
  `ncclReduceScatter`.
- The decode and prefill rates follow the definitions of the earlier records
  of this profile ([row-owned prefill on a SIRCL path](row-owned-sircl-path4-20261119.md)):
  decode counts output tokens after the first token, and cold prefill counts
  prompt tokens of a request that reports no cached prefix.
- Two readings exist of the TP4+EP4 deployment, and the tables keep them
  apart. The campaign tables below hold one rate per measurement, from the
  harness that measured R0 and the DP4 deployments too. The transport sweep
  holds a median of seven consecutive cold requests per size and one run per
  configuration, with 128 output tokens per decode request and no sample
  count for decode.

## Configurations

| ID | Parallel layout | Rank settings | Speculation |
|---|---|---|---|
| R0 | TP4 | `ds41_row_shard` row-owned prefill | adaptive DSpark5 |
| R1 | TP4+EP4 | `ds41_row_shard` row-owned prefill, `SIRCL_B12X_EP_MOE=1`, `--load-format auto` | adaptive DSpark5 |
| R1-link | TP4+EP4 | R1 with `SIRCL_ONESHOT_MAX_BYTES=65536` and a 1 MiB link slot and 1 MiB gather, reduce and scatter pieces | adaptive DSpark5 |
| R1-link-batch | TP4+EP4 | R1-link with `--max-num-batched-tokens 16384` | adaptive DSpark5 |
| D2 | DP4+EP4 | SIRCL-EP all-to-all (`SIRCL_EP_ALL2ALL=1`) | none |
| D2 + DSpark5 | DP4+EP4 | SIRCL-EP all-to-all | adaptive DSpark5 |

`sparkring install` renders the rank settings of R0 and R1, and now renders
R1-link's link geometry with R1
(`serving.EXPERT_PARALLEL_LINK` in `runtime/common/serving.py`); the campaign's
R1 rows were measured before that geometry, so they are R1 without it. The
DP4 deployments have no installer option, so no `sparkring install` request
renders them.

## Decode

Output tokens per second at a given number of concurrent requests (`cc`):

| Configuration | cc=1 | cc=4 | cc=8 | cc=16 | cc=32 |
|---|---:|---:|---:|---:|---:|
| R0 | 69.80 | 163.79 | 258.49 | 337.70 | not measured |
| R1 | 56.47 | 158.04 | 221.70 | 331.11 | not measured |
| D2 | 12.32 | 54.58 | 98.21 | 156.00 | 249.91 |
| D2 + DSpark5 | 34.55 | 64.51 | 110.72 | 258.11 | 410.35 |

R1 is lower than R0 at every concurrency this campaign measured, by 19.1% at
one request and by 14.2% at eight. The gap narrows to 1.9% at sixteen, which
suggests the expert parallelism's per-step collectives are amortized by the
batch; no counter in these runs confirms it. In the DP4 deployments, DSpark5
raises decode by 2.8x at one request and by 1.6x at 32, and D2 + DSpark5's
410.35 tok/s at 32 requests is the highest decode rate of the four
configurations.

The sweep measured the same deployment at 86.55, 185.11, 255.41 and 418.76
tok/s, up to 53% above this campaign's R1 rows (see Transport tuning), so the
two harnesses' rates do not compare; R1's ordering against R0 holds only
inside this table.

## Cold prefill

Prompt tokens per second of a cold request:

| Configuration | 16,384 tokens | 65,536 tokens | 4 x 16,384 tokens |
|---|---:|---:|---:|
| R0 | 5,471.9 | 5,794.3 | not measured |
| R1 | 4,634.6 | 6,132.9 | not measured |
| D2 | 1,813.2 | 1,891.5 | 3,695.3 |
| D2 + DSpark5 | 1,953.5 | 2,086.5 | 4,005.2 |

R1's 65,536-token rate is 5.8% above R0's. Its 16,384-token figure is a
first-hit cold request on a freshly started deployment, which R0's samples are
not, so the two 16K figures do not measure the same steady state; the 64K
figures do. The DP4 deployments serve less than half of R0's and R1's prompt
rate at both lengths, and their four concurrent 16K requests aggregate
3,695.3 and 4,005.2 tok/s. Both R1 figures lie inside the spread of the
sweep's seven samples at their length, whose medians supersede them
(Transport tuning).

## KV cache

| Configuration | KV cache per Spark (GiB) | Tokens |
|---|---:|---:|
| R0 | 15.76 | not recorded |
| R1 | 18.69 (+18.6%) | not recorded |
| R1, 262,144-token limit | 17.72 | 7,286,986 |
| D2 | 14.39 | not recorded |
| D2 + DSpark5 | 12.19 | not recorded |

R1's 18.69 GiB is the pool at 131,072 tokens of context, the reading its 18.6%
compares; a start at the full 262,144-token limit reported 17.72 GiB, or
7,286,986 tokens. At the same 131,072-token reading, expert parallelism costs
2.93 GiB per Spark more than R0; the recorded runs do not separate the expert
slices from the row-owned prefill's buffers. D2 + DSpark5's pool is 2.20 GiB
smaller than D2's. The sweep's three rungs reported 17.72, 17.77 and 17.52 GiB
at the same 262,144-token limit (Transport tuning).

## Transport tuning

Three rungs of the same deployment, each one run of seven consecutive cold
requests per size:

| Rung | 16,384-token prefill | 65,536-token prefill | 131,072-token prefill |
|---|---:|---:|---:|
| R1 | 5,980.8 (3,462.1–6,204.3) | 5,579.4 (4,582.9–6,271.5) | 5,184.0 (4,383.1–5,499.4) |
| R1-link | 6,317.7 (5,163–6,422) | 6,118.7 (5,892–6,143) | 5,782.8 (5,658–6,009) |
| R1-link-batch | 6,369.3 (6,341.1–6,379.0) | 6,219.1 (6,138.2–6,235.3) | 6,078.8 (6,035.4–6,097.7) |

| Rung | decode cc=1 | cc=4 | cc=8 | cc=16 | KV cache (GiB) | API ready |
|---|---:|---:|---:|---:|---:|---:|
| R1 | 86.55 | 185.11 | 255.41 | 418.76 | 17.72 | 5 min 42 s |
| R1-link | 87.95 | 192.16 | 279.43 | 404.22 | 17.77 | 5 min 20 s |
| R1-link-batch | 77.07 | 167.09 | 293.16 | 347.77 | 17.52 | 5 min 35 s |

The link geometry carries most of the prefill gain: 5.6%, 9.7% and 11.5% at
the three lengths. The 16,384-token batch adds 0.8, 1.6 and 5.1 points on top
of it, reaches 6.4 ktok/s at 16,384 tokens, and misses it at 65,536 (6,219.1)
and 131,072 (6,078.8). It is also what costs single-request decode: R1-link
holds 87.95 tok/s at one request where R1-link-batch falls to 77.07.

R1's samples are bimodal at 64K, alternating between about 6.2 and about
5.2–5.6 ktok/s, and its worst three (3,462.1, 4,582.9, 4,383.1) are stalls
rather than throughput: one 65,536-token prefill ran 165 s and recovered on its
own inside libsircl's 600 s startup completion limit. R1-link's samples are
tight, so most of the median gain is the stalls disappearing, not a higher
ceiling. Where the ceiling is measurable it did move — at 131,072 tokens R1's
best sample (5,499.4) is below every R1-link-batch sample (6,035.4–6,097.7) —
while at 65,536 tokens the two rungs' peaks agree (6,271.5 against 6,235.3).

The link is not what limits these rates. The libsircl receipt covering roughly
twelve minutes of this deployment — which also served 21 measured cold prefills
and the decode runs — reports 127,109 native operations and 2,988.5 GB, of
which 31,368 are `ring_gather` and the rest ring reduce and ring scatter, with
`healthy: true`, `error: null` and `transport: verbs`. Delivering 2,988.5 GB at
the ports' line rate costs 60 s, so the wire carried about 4 GB/s per rank,
under a tenth of what the two ports offer; the remaining wall time is the FP4
expert GEMMs, attention over 8,192 rows and the tensor-parallel row-shard
collectives. Transport settings can therefore move this deployment's prefill
by a few percent, not past 6,400 tok/s at every length.

Each configuration is one run, so the percentages carry no run-to-run error
estimate; the within-run spread is the only one available, and it is why the
0.8-point difference between R1-link and R1-link-batch at 16,384 tokens is not
separable from noise. The three rungs' KV pools span 0.25 GiB (17.77, 17.72
and 17.52 GiB), which is the size of the link slots the transport reserves, not
a difference in what the engine holds.

## The B12X weight loader on an expert-parallel rank

`b12x_loader` reads an expert-parallel rank's shard of this checkpoint
correctly. Its support check (`_has_b12x_loader_support()`) is true in the
container, both of its read paths apply vLLM's
`should_skip_weight(name, local_expert_ids)`, and the checkpoint names its
expert tensors `layers.N.ffn.experts.E.w{1,2,3}.weight`, which that filter
shards. Filtering the checkpoint's 96,085 tensors by each rank's local expert
set gives 61,237 tensors for rank 0 and 60,661, 60,373 and 60,373 for ranks 1
to 3; the union covers all 96,085, 48,853 dense, shared, embedding and `.scale`
tensors are kept by all four ranks, and no expert weight is kept by two ranks.
That is an expert-parallel shard.

With `--load-format b12x` and the container's `/cache` mount pointed at the
host's persistent compile cache, `R1-link` reached a serving API end to end:

| Measurement | default loader (`auto`) | `b12x` loader | Delta |
|---|---:|---:|---:|
| target-model weight load | 116.36 s | 13.88 s | 8.4x faster |
| drafter weight load | 6.12 s | 1.08 s | 5.7x faster |
| `Model loading took` | 76.96 GiB / 153.88 s | 76.96 GiB / 32.07 s | -121.8 s |
| API ready (launch to `/v1/models` 200) | 320 s | 239 s | -81 s |
| `Available KV cache memory` | 17.77 GiB | 15.62 GiB | -2.15 GiB |

Warm exact-token prefill on the `b12x`-loaded deployment measured 6,418.0
tok/s median at 16,384 tokens (warm 6,432.7; samples 6,313.6 / 6,422.1 /
6,418.0) and 6,284.5 tok/s median at 65,536 tokens (warm 6,304.1; samples
6,284.5 / 6,289.1 / 6,177.5), matching `R1-link`'s default-loader peaks.
Greedy output is not bit-identical to vLLM's loader (different packed GEMM
reduction order; two of four probe continuations matched token for token, and
first-token log-probabilities differed by `1e-6` to `1e-3`). Because `b12x`
trades 2.15 GiB of KV cache (12% of the pool at 262,144 context) for 81 s of
startup time, `serving.EXPERT_LOADER_EP` stays `False` by default and remains
a single-line opt-in switch.

## DP4+EP4 static all-to-all and CUDA-graph capture

The rewritten `SirclEpAll2AllManager` (`integrations/vllm/libsircl/sircl_ep_all2all.py`)
reuses a grow-only per-geometry arena (retaining displaced buffers so captured
CUDA graphs stay valid) and adds an opt-in zero-host-sync static path
(`SIRCL_EP_STATIC_A2A=1`, constant cross-rank capacity
`SIRCL_EP_STATIC_A2A_MAX_TOKENS=96`, decided from shared
`ForwardContext.dp_metadata` CPU counts or capture sizes). Measured on Ring
4-7 with `D2 + DSpark5` (`DP4+EP4`, `max-model-len 131072`, `max-num-seqs 32`,
`B12X_AUTOTUNE=0`):

| Measurement | Dynamic arena (`STATIC_A2A=0`) | Static fixed-stride (`STATIC_A2A=1`, cap 96) |
|---|---:|---:|
| startup to `/v1/models` | 599 s | 575 s |
| KV cache per Spark | 11.27 GiB | 11.36 GiB |
| decode `cc=1` | 33.74 tok/s | 19.65 tok/s (-42%) |
| decode `cc=4` | 77.13 tok/s | 68.72 tok/s (-11%) |
| decode `cc=8` | 128.67 tok/s | 160.60 tok/s (+25%) |
| decode `cc=16` | 215.93 tok/s | 242.64 tok/s (+12%) |
| decode `cc=32` | 436.13 tok/s | 376.76 tok/s (-14%) |
| prefill 16,384 (median) | 2,124.0 tok/s | 2,093.3 tok/s |
| prefill 65,536 (median) | 2,103.5 tok/s | 2,062.3 tok/s |
| 4 x 16,384 concurrent | 3,910.0 tok/s | 3,301.6 tok/s |
| rank-0 EP dispatches / receipt | 11,249 / healthy | 11,374 / healthy |

Three findings from the DP4 runs:

- **DP+EP MoE captures into FULL CUDA graphs with zero refusals:** the
  `libsircl` receipt recorded 3,816 captured calls across 42 FULL CUDA graphs
  on rank 0 with `SIRCL_EP_STATIC_A2A=1`.
- **Cap 96 penalizes `cc=1` decode:** padding a 6-token step to `world * 96`
  (384) rows cuts single-request decode from 33.74 to 19.65 tok/s, while the
  dynamic zero-allocation arena (`STATIC_A2A=0`) reached 436.13 tok/s at
  `cc=32`. Lowering `SIRCL_EP_STATIC_A2A_MAX_TOKENS` to match the deployment's
  actual decode ceiling (`max_num_seqs * (1 + num_speculative_tokens)`) is the
  remaining lever for static A2A.
- **Cold `B12X_AUTOTUNE=1` does not converge inside a DP4 start:** tuning
  progressed to 827 of 3,272 preparation units in 6.5 minutes before vLLM's
  60-second shared-memory broadcast watchdog closed the DP store
  (`DistNetworkError`). Autotune caches must be populated outside a DP4 start
  (or with raised store/shm timeouts) before enabling `B12X_AUTOTUNE=1` on D2.

## What the installer renders

`sparkring install --expert-parallel` with the pinned image and
`--transport libsircl` names the grouped all-to-all, loads the checkpoint with
vLLM's own loader, leaves the image's B12X weight loader plugin out, and sets
`SIRCL_B12X_EP_MOE=1` on every rank. With `--row-owned-prefill` (R1) it also
stages `integrations/vllm/ds41_row_shard` on `PYTHONPATH` and in
`VLLM_PLUGINS`, sets the `DS41_ROW_SHARD_*` switches, keeps the row-owned
command adjustments (compilation mode 0 with `FULL_DECODE_ONLY`, Engram
`projection_tp` off, adaptive verification on, a 262,144-token limit), and sets
the R1-link link geometry. The offline checks are
`runtime/host/test_install_sircl.py` (the plan, the lock and every rank's
container), `runtime/common/test_libsircl.py` (the switch and the link
geometry survive the transport's replacement of `SIRCL_*` variables, and
libsircl's own checks pass on the command the deployment serves), and
`runtime/common/test_serving.py` (the geometry follows the pairing of the two
switches, which is what it was measured on).

## Open items

- The link geometry's effect on expert parallelism without `--row-owned-prefill`
  is unmeasured, so `sparkring install` applies `EXPERT_PARALLEL_LINK` only
  when both switches are set.
- A lower `SIRCL_EP_STATIC_A2A_MAX_TOKENS` cap and pre-warmed `B12X_AUTOTUNE=1`
  cache remain to be benchmarked on D2.

## Limitations

- The per-sample records of these runs are not in this repository; the tables
  hold the rates the runs reported.
- The campaign tables hold one rate per measurement and the transport sweep
  one run of seven samples per configuration, so no percentage has a
  run-to-run error estimate. On this ring a single-sample difference below
  roughly 20% is not distinguishable from noise.
- The 16,384-token R1 figure is a first-hit cold request, so it is not
  comparable with R0's steady-state samples (see Cold prefill).
- The DP4 deployments' expert parallelism runs SIRCL-EP's own all-to-all, a
  different mechanism from R1's grouped all-gather and reduce-scatter; their
  rates do not transfer to R1 or the reverse.
- The KV cache sizes are the starts' reported pools, which vLLM sizes from the
  memory free at start under `--gpu-memory-utilization 0.83`; a different
  allocation changes them without changing the configuration.
- The engine ran libsircl's startup wait regime while serving, with libsircl's
  600 s completion limit, and one 65,536-token prefill stalled 165 s before
  recovering. `LIBSIRCL_WAIT_REGIME=serving` would bound a stall at 60 s; no
  run here did.
- No NCCL receipt check covers these deployments: the libsircl transport does
  not judge libsircl's per-communicator receipts.
