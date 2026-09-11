# DeepSeek-V4.1-Flash on a four-Spark cycle — TP4, DSpark k=5, CUDA graphs (2026-09-10)

Live benchmark of the `deepseek-v41-flash-cycle` recipe on four directly cabled GB10 DGX Sparks
(`0-1-2-3-0`, two RoCE devices per rank, MTU 9000) with SparkRing's patched NCCL 2.30.7, a locally built
image (`sha256:af86a3d2bb0d267faa7f31777cdbe855addc1348f0b9f8323016ebf17d3dae3c`, see
[`runtime/deepseek-v41-gb10/image-receipt.json`](../../../runtime/deepseek-v41-gb10/image-receipt.json)) and the stock
checkpoint `deepseek-ai/DeepSeek-V4.1-Flash @ dba1be0a` on every rank's local NVMe. The benchmark tables below are
from the first serving boot: 300,000-token context, 8 sequences, 8,192 batched tokens, `gpu-memory-utilization 0.80`,
`block-size 128`, Engram tables on NVMe (32 reader threads), DSpark k=5 (probabilistic draft, block rejection, adaptive
verification off), `FULL_AND_PIECEWISE` CUDA graphs with exact capture sizes, tools and vision on, thinking off.
The [lever campaign](#lever-campaign-one-variable-per-rebooted-boot) that follows moved the recorded recipe to a
430,080-token limit, `gpu-memory-utilization 0.83`, greedy draft, 64 Engram threads and
`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`; that profile carries the 400K needle and the six-hour soak.
All four ranks were rebooted before every boot. Raw data: [`cycle-tp4-dspark5-graphs-20260910.json`](cycle-tp4-dspark5-graphs-20260910.json).

## Method

Fixed prompt set of eight categories (code, JSON, math, reasoning, tables, summary, prose, narrative) plus a
counting ceiling, 30–120-token prompts, 150–256-token budgets, temperature 0, thinking off, streaming. At
concurrency C, C streams are released together; one batch per cell. Token counts come from the server's
`usage` block. **Decode** = tokens after the first / time after the first token, per stream. **Aggregate** =
all streams' tokens / batch wall time (TTFT included). Counting is excluded from the eight-category means.
Prompt set and harness: `prompts-v1.json` / `v41bench.py` from tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark (MIT).

## Headline

| C | per-stream decode, 8-category mean (tok/s) | aggregate (tok/s) | mean TTFT (s) |
|---|---:|---:|---:|
| 1 | 56.2 | 49.8 | 0.32 |
| 2 | 44.1 | 74.7 | 0.57 |
| 3 | 37.5 | 96.7 | 0.37 |
| 4 | 33.8 | 115.8 | 0.39 |
| 5 | 31.5 | 136.2 | 0.45 |
| 6 | 31.4 | 159.9 | 0.48 |

## Decode per stream (tok/s, after the first token)

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---:|---:|---:|---:|---:|---:|
| Code | 77.3 | 58.0 | 54.4 | 48.3 | 45.7 | 42.9 |
| JSON | 58.9 | 44.7 | 36.5 | 31.3 | 27.9 | 28.6 |
| Math | 76.9 | 58.5 | 47.7 | 44.9 | 41.6 | 52.2 |
| Reasoning | 60.0 | 42.0 | 35.5 | 34.3 | 30.5 | 30.1 |
| Tables | 76.4 | 70.2 | 63.6 | 54.4 | 52.6 | 49.1 |
| Summary | 36.8 | 29.8 | 22.6 | 20.4 | 19.0 | 16.8 |
| Prose | 33.9 | 25.0 | 21.3 | 19.1 | 19.4 | 16.4 |
| Narrative | 29.9 | 24.2 | 18.7 | 17.4 | 15.4 | 14.7 |
| Counting (ceiling) | 90.9 | 79.3 | 66.7 | 63.3 | 65.4 | 57.2 |

## Aggregate throughput (tok/s, wall time incl. TTFT)

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---:|---:|---:|---:|---:|---:|
| Code | 70.0 | 107.6 | 146.7 | 174.7 | 208.3 | 231.9 |
| JSON | 49.4 | 72.8 | 85.7 | 92.9 | 121.4 | 131.2 |
| Math | 69.3 | 103.0 | 126.5 | 157.5 | 184.5 | 279.5 |
| Reasoning | 56.1 | 76.5 | 94.8 | 125.8 | 134.9 | 161.7 |
| Tables | 61.6 | 111.5 | 154.2 | 175.5 | 216.4 | 226.5 |
| Summary | 31.8 | 34.1 | 56.2 | 65.8 | 73.1 | 78.7 |
| Prose | 31.8 | 47.8 | 58.5 | 71.5 | 82.8 | 90.2 |
| Narrative | 28.4 | 44.6 | 50.8 | 63.1 | 67.9 | 79.7 |
| Counting (ceiling) | 83.3 | 147.2 | 184.0 | 233.8 | 303.7 | 315.1 |

## Mean TTFT (s)

| category | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---:|---:|---:|---:|---:|---:|
| Code | 0.28 | 0.29 | 0.33 | 0.34 | 0.35 | 0.41 |
| JSON | 0.29 | 0.36 | 0.34 | 0.38 | 0.40 | 0.41 |
| Math | 0.30 | 0.34 | 0.34 | 0.35 | 0.38 | 0.39 |
| Reasoning | 0.25 | 0.31 | 0.33 | 0.35 | 0.37 | 0.39 |
| Tables | 0.33 | 0.34 | 0.40 | 0.39 | 0.41 | 0.44 |
| Summary | 0.53 | 2.40 | 0.69 | 0.77 | 1.04 | 1.18 |
| Prose | 0.27 | 0.27 | 0.28 | 0.29 | 0.30 | 0.31 |
| Narrative | 0.27 | 0.28 | 0.29 | 0.29 | 0.34 | 0.32 |
| Counting (ceiling) | 0.25 | 0.24 | 0.29 | 0.27 | 0.29 | 0.29 |

## Cold prefill (unique prompt, one-token reply)

| prompt tokens | TTFT (s) | prefill tok/s |
|---:|---:|---:|
| 2,950 | 2.1 | 1376 |
| 11,592 | 8.1 | 1432 |
| 46,810 | 29.9 | 1568 |
| 93,335 | 59.3 | 1574 |

## Other measurements on the same boot

| check | result |
|---|---|
| Needle-in-haystack, depth 0.5 (`v41needle.py`) | 131K: pass, 130,258 prompt tokens, TTFT 83.2 s, 1,565 tok/s prefill · 262K: pass, 260,119 tokens, TTFT 180.0 s, 1,445 tok/s |
| Vision + tool calling end-to-end (`vision_tools_demo.py`) | 7/7: three-stripe image, two images in one message, 2×2 grid; tool call with arguments, full round trip, parallel calls, forced `tool_choice` |
| DSpark acceptance | mean 3.77 tokens per step over the benchmark (2.00–6.00 across 10 s windows); 6.00 on counting |
| 20-minute soak, 8 concurrent streams (temperature 1.0, top-p 0.95, 256-token budgets, eight categories rotating) | 81 waves, 648 requests, 0 failures, 0 streams silent for 90 s, aggregate median 95.0 tok/s (65.5 on the warm-up wave, max 102.6), first-five-wave mean 89.5 → last-five 93.9, TTFT ~0.7 s after warm-up; MemAvailable 14–16 GiB per rank before and after |
| Six-hour soak, 8 concurrent streams, on the recorded profile (430,080 tokens, gmu 0.83, greedy draft; temperature 1.0, top-p 0.95, 256-token budgets, eight categories rotating; 2026-09-11 05:28–11:28Z) | 1,417 waves, 11,336 requests, 1,985,316 completion tokens, **0 failures, 0 streams silent for 90 s**; aggregate median 92.1 tok/s (min 78.0, max 104.1); no drift: first / middle / last 100 waves median 90.4 / 92.9 / 91.1 tok/s, TTFT p50 0.71 s throughout; MemAvailable sampled every 60 s (344 samples) 14 / 15–16 / 13–14 / 13–14 GiB per rank, first and last sample identical; `/health` 200 after the soak |
| Smoke, cold first request (TTFT included) | counting 68.1 tok/s, code 56.0 tok/s at C1 |
| Text-only eager boot, no speculation (131K, 8 seqs) | 14.5–14.8 tok/s at C1; `Model loading took 78.79 GiB`; KV 13.74 GiB = 1,687,422 tokens (12.87× 131K) |
| Serving-shape memory | consumed 85.71 GiB per rank at startup (weights + non-torch); graphs 0.54 GiB; KV 8.39 GiB = 1,171,588 tokens (3.91× 300K); 15–16 GiB MemAvailable per rank while serving |
| Time to serving | 8 min from launch (weights ~4 min from local NVMe, DSpark draft 57 s, graphs + autotune ~2 min) |

## Lever campaign (one variable per rebooted boot)

Compact probe set per boot, one run each: `dsv4-ab-ladder.py` decode rungs (short prompts, 256-token budgets, C1/C4/C8),
cold prefill at 16K and 64K, an 8×4K TTFT burst, three decode shapes (draftable list, code, free prose) and the prompt set
at C1 and C6. `baseline-final` is the first boot's profile re-run the same way, so the band between it and the `boot 2`
column (≈ ±5 %) is the run-to-run noise; a lever has to clear it. KV pool sizes vary 1.09–1.33M tokens between boots of
identical profiles (memory-profiler variance) and are not lever effects.

| probe | boot 2 | baseline-final | L1 NCCL 8 ch | L2 Engram 64 thr | L3 seqs 16 | L4 batched 16K | L5 greedy draft | L6 430K · 0.85 · profiler off | L7 = L2+L5+L6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| decode C1 / C4 / C8 aggregate (tok/s) | 33.4 / 63.5 / 89.1 | 32.0 / 59.5 / 89.7 | 34.2 / 55.4 / 93.2 | 35.5 / 64.6 / 92.9 | 34.6 / 55.8 / 89.7 | did not boot | 36.1 / 61.1 / 91.6 | 32.0 / 59.9 / 95.3 | 36.0 / 59.6 / 92.8 |
| prefill 16K / 64K (tok/s) | 1,719 / 1,728 | 1,528 / 1,658 | 1,484 / 1,677 | 1,624 / 1,755 | 1,368 / 1,654 | — | 1,528 / 1,658 | 1,543 / 1,650 | 1,590 / 1,745 |
| burst 8×4K TTFT p50 / max (s) | 11.1 / 16.8 | 11.6 / 17.5 | 11.5 / 17.2 | 11.1 / 16.6 | 11.6 / 17.4 | — | 11.9 / 17.7 | 11.7 / 17.4 | 11.1 / 16.5 |
| shapes code / prose (tok/s) | 81–83 / 31–33 | 81.4 / 31.5 | 81.9 / 33.2 | 84.4 / 33.4 | 81.8 / 31.9 | — | 81.8 / 31.7 | 79.3 / 31.9 | 80.4 / 30.3 |
| prompt set C1 aggregate (per stream) | 49.8 (56.2) | 47.8 (53.5) | 46.6 (52.2) | 49.3 (55.5) | 47.7 (54.2) | — | 49.7 (56.6) | 48.7 (55.2) | 48.7 (54.8) |
| prompt set C6 aggregate | 159.9 | 150.0 | 150.7 | 150.2 | 156.1 | — | 151.5 | 152.8 | 152.9 |
| KV pool (tokens) | 1,171,588 | 1,086,792 | 1,309,195 | 1,326,231 | 1,131,965 | 1.84 GiB left < 1.9 GiB needed | 1,136,197 | 3,097,185 | 3,085,606 |
| MemAvailable per rank (GiB) | 15–16 | 15–18 | 14–16 | 14–16 | 13–15 | — | 15–17 | 7–9 | 7–9 |

No lever moves decode speed beyond the noise band: the step is set by the model and its ~88 all-reduces, and the fabric probes
above show the collectives already cost ~5 ms of a ~57 ms step. L2 is slightly positive on nearly every probe and free; L1
(the +50 % reported on a Thunderbolt-only ring) does not reproduce here; L3 is neutral at C≤8 and pays off only as admission
(a second `max-num-seqs 16` boot probed C16: ladder 130.2 tok/s aggregate, prompt set C8/C12/C16 164/229/285 tok/s aggregate at
26.0/22.6/21.5 tok/s per stream, 13–15 GiB MemAvailable); L4 cannot boot at 0.80 because the 16K chunk raises the profiler's
peak; L5 is neutral to +5 % on this hardware (spark-bench measured a larger gain on theirs); L6 buys capacity, not speed — 400K
needle pass at 397,753 prompt tokens, TTFT 302.6 s, 1,314 tok/s prefill, and the L7 combination repeats it at 288.6 s /
1,378 tok/s. The recipe records L7 with `gpu-memory-utilization 0.83` instead of 0.85 to keep 13–15 GiB MemAvailable per rank
(KV 2,182,642 tokens, 5.07× at 430K, `Available KV cache memory: 10.94 GiB`); the six-hour soak in the table above ran on that value.

Raw per-boot summaries and prompt-set JSON are in the operator's repository; the headline numbers above are the complete
compact-set output for each boot.

## Engram loader rebalance (2026-09-11): where the prefill time went

A rank-0 torch trace of one 15,693-token prefill (9.6 s) put 48.6 % of GPU time in `ncclDevKernel_AllReduce` (184 calls,
≈23.5 ms each on 84 MB tensors). An idle-fleet PyNccl sweep then showed the same all-reduce completing in 9.7–12.5 ms under
every protocol/channel/buffer variant (84–109 Gb/s bus bandwidth — the PCIe Gen5 ×4 ceiling of the ConnectX-7; raw
`ib_write_bw` between neighbours: 109 Gb/s one way, 213 bidirectional), so roughly half of the in-serving all-reduce time was
ranks waiting for each other. Per-rank traces of the same prefill located the skew:

| rank | GPU busy | NCCL kernel time | GPU idle gaps > 20 ms | `preadv` calls in the Engram staging pool |
|---|---:|---:|---:|---:|
| 0 | 94 % | 4,349 ms | 527 ms | 60,664 |
| 1 | 84 % | 3,317 ms | ~1,550 ms | — |
| 2 | 74 % | 2,417 ms | ~2,440 ms | — |
| 3 | 67 % | 1,718 ms | 3,162 ms (one gap per chunk) | 320,066 |

The checkpoint lays its 24 hash columns out order-major and the loader split them contiguously, so rank 3 owned six four-gram
columns (nearly every row unique after `torch.unique`) and rank 0 six bigram columns (heavily repeated), and every row cost two
`preadv` calls (weight and scale ~24 GB apart). `engram.py` gained two env-gated additions — strided columns
(`DSV41_ENGRAM_BALANCED=1`) with the all-gather permuted back, and packed single-read shards (`DSV41_ENGRAM_PACKED_DIR`,
`tools/pack_engram_rows.py`) — measured on the same profile after a fleet reboot:

| probe | stock loader (L7/L10/L11 boots) | balanced + packed (L12) |
|---|---:|---:|
| prefill 16K / 64K (tok/s) | 1,590–1,668 / 1,745–1,758 | **1,873 / 2,058** |
| burst 8×4K TTFT p50 / max (s) | 11.1–11.6 / 16.5–17.5 | **9.76 / 14.58** |
| needle 131K / 262K (prefill tok/s, pass) | 1,565 / 1,445 | **1,826 / 1,689**, pass |
| decode C1 / C4 / C8 aggregate (tok/s) | 32–36 / 57–64 / 88–98 | 32.0 / 59.5 / 94.0 |
| draft acceptance code / prose | 90–93 % / 23–25 % | 92.3 % / 23.5 % |
| prompt set C1 per stream / C6 aggregate | 52–55 / 150–157 | 54.26 / 157.47 |
| 20-minute c=8 soak (temperature 1.0, 256-token budgets) | 648 requests, 0 failures (boot 2) | **632 requests, 0 failures, 0 hangs, median 92.0 tok/s (79.0–98.8), MemAvailable flat 10 / 11–12 / 9 / 9 GiB** |

Also tried on this profile and found neutral (one rebooted boot each): FlashInfer `b12x` MXFP8 dense GEMM backend in place of
vLLM's pinned CUTLASS SM120 kernel (eager microbench 1.9–3.2× at decode M, no change inside the captured graph), `NCCL_PROTO=Simple`,
128 Engram reader threads, and `rejection_sample_method=standard` (prose acceptance identical to `block`). The decode trace on
this profile: MoE grouped GEMM 42 %, dense MXFP8 GEMM 21 %, bf16 `wo_a` emulation 15 %, NCCL 13 %, ≈66 ms per step at C1.

## Fabric probes before the boot (same image and NCCL environment, GPUs idle)

Four-rank all-reduce latency through vLLM's PyNccl wrapper (`nccl_lat.py`, tonyd2wild, MIT), ring `0 1 2 3`
over both RoCE devices:

| operation | p50 | p90 |
|---|---:|---:|
| all-reduce 8 KB | 47 µs | 428 µs (21 % of ops > 1.25× p10) |
| all-reduce 32 KB | 56 µs | 59 µs |
| all-reduce 60 KB (decode-shaped) | 64 µs | 68 µs |
| all-reduce 128 KB | 82 µs | 87 µs |
| all-reduce 256 KB | 97 µs | 102 µs |
| all-reduce 1 MB | 185 µs | 194 µs |
| decode-shaped step: 88 all-reduces + ~0.6 ms compute gaps, eager | 56.6 ms (collective share ~4.0 ms) | |
| same, CUDA-graphed | 57.5 ms (collective share ~5.0 ms) | |
| 88 all-reduces back-to-back, graphed | 4.9 ms | |

GPU fast/slow-state probe (`gpuflip.py`): all four GPUs in the fast state for the full 62 s window, 0 slow
seconds, 2,177–2,190 MHz, 23–25 W under the GEMV. These nodes pin `nvidia-smi -lgc 0,2200` via a systemd unit.

## Limitations

One batch per cell, no repeated-run interval; the lever table is one compact run per boot with a ≈ ±5 % band. Output quality
beyond needle recall and the end-to-end checks was not evaluated. 1M context was not run; 430,080 tokens with a 400K needle
pass is the recorded limit. The image is a private local build; the recipe
records how to reproduce it, not a public digest.
