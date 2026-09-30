# GLM-5.3-Flash installer profiles: KV capacity and Node A memory

Status: **implemented; measured in installer deployments on one Spark pair and one four-Spark ring; single runs; not serving-qualified**.

This record gives the measurements behind the KV cache, page size, per-request
media limits and memory settings of the two GLM-5.3-Flash installer profiles,
`glm53-flash-nvfp4-spark-tp2` and `glm53-flash-nvfp4-spark-tp4`. Both serve
`local-inference-lab/GLM-5.3-Flash-NVFP4-Spark` revision `a608241037e4` on
installer image `dev-20260928-plainstatus-cuda1342-nccl2323-status033`
(configuration `sha256:4b7049d1e00f263c65713b62247a4497eba72fb38977087941830cec38609a8c`).
[`installer-memory-20260929/programs`](installer-memory-20260929/programs)
holds the load and sampling programs and
[`installer-memory-20260929/results`](installer-memory-20260929/results) the
raw results.

## Conditions

- **Memory model:** a Spark's CPU and GPU share one 128 GB memory. Model
  weights, the KV cache, CUDA workspaces and every process's host memory come
  from it, so Linux `MemAvailable` is what is left for the whole node.
  Exhausting it has left a Spark unresponsive until it was power-cycled.
- **Node A:** rank 0's Spark. Besides its share of the model, it runs the API
  server, which decodes and preprocesses every image and video, and the engine
  core, which holds each request's preprocessed pixel data until the request
  finishes. Node A had the lowest `MemAvailable` in every measurement below.
- **Configurations:** each was installed with `install.sh` from a Git bundle
  of a branch that differs from main commit `59b253fd` in no file the
  installer reads except the profile's `config.json`. The baselines are the
  profiles at `59b253fd`:

  | Profile at `59b253fd` | KV cache per Spark | Page | Context | Images per request | NCCL channels |
  |---|---|---|---|---|---|
  | `glm53-flash-nvfp4-spark-tp2` | 8.75 GiB | 1,024 tokens | 262,144 | 3 | NCCL default |
  | `glm53-flash-nvfp4-spark-tp4` | 24 GiB | 512 tokens | 1,048,576 | 4 | 4 |

  "Page" is `--block-size`, `--mamba-block-size`,
  `VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE` and `VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE`
  together. "KV cache" is `--kv-cache-memory-bytes`.
- **Settings in this record's commit**, in addition to the KV cache, page,
  context and media limits in the Result tables:
  - `--mm-processor-kwargs` caps each image at 4,096 tokens and each video at
    8,192 tokens; `--media-io-kwargs` samples 16 frames per video.
  - `MALLOC_MMAP_THRESHOLD_=1048576` and `MALLOC_ARENA_MAX=4` make glibc
    return freed buffers of 1 MiB or more to the system.
  - `B12X_COMPILE_WORKERS=2` limits the B12X kernel compiler to two worker
    processes.
  - On the pair, `NCCL_MIN_NCHANNELS=4` and `NCCL_MAX_NCHANNELS=4`; the ring's
    profile already set them.
  - On the ring, `--mm-processor-cache-gb 0`; the pair's profile already set
    it.
- **Loads** ([programs](installer-memory-20260929/programs)):
  - [`image_load.py`](installer-memory-20260929/programs/image_load.py):
    C concurrent chat requests, each with N random-noise JPEG images cycling
    through 2048×2048, 3840×2160 and 1600×1200 pixels. The processor resizes
    each to at most 4,096 tokens, about 3,550 tokens per image on average.
    Every step starts at its own seed offset, so no image repeats one the
    server has cached, except where a result says otherwise.
  - [`video_load.py`](installer-memory-20260929/programs/video_load.py): one
    1920×1080 MP4 at 2 frames per second per request.
  - [`mixed_load.py`](installer-memory-20260929/programs/mixed_load.py):
    llm-inference-bench decode streams at a fixed context, image requests and
    video requests at the same time.
  - Text loads: llm-inference-bench `llm_decode_bench.py` 0.6.2 at
    temperature 1.0.
  - Requests carry `{"chat_template_kwargs": {"reasoning_effort": "low"}}`
    and at most 64 output tokens for image and video requests.
- **Guards:** every image, video and mixed load ran under a guard that stops
  the load client when a watched Spark's `MemAvailable` falls below a floor.
  On the pair, [`guarded.py`](installer-memory-20260929/programs/guarded.py)
  read Node A once a second with a 1.2 GiB floor. On the ring,
  [`fleet_guard_fast.py`](installer-memory-20260929/programs/fleet_guard_fast.py)
  read all four Sparks every 0.2 s with a 2.0 GiB floor in the configurations
  with 32 images per request;
  [`fleet_guard.sh`](installer-memory-20260929/programs/fleet_guard.sh) read
  them every 2 s with the same floor in the others.
- **Cluster:** one directly cabled Spark pair (`direct-pair-2`) and one
  four-Spark ring (`direct-cycle-4`); a separate client machine sent every
  request.

## Measurement

- **KV capacity:** the rank-0 engine's startup log line
  `GPU KV cache size: N tokens`.
- **Memory low:** the minimum `MemAvailable` reading on a Spark during a
  load step. Samplers read `/proc/meminfo` once a second
  ([`memsample.sh`](installer-memory-20260929/programs/memsample.sh)); in
  the ring's configurations with 32 images per request, every step's low
  comes from the fleet guard's 0.2 s readings instead.
- **Prefix-cache reuse:**
  [`cache_probe.py`](installer-memory-20260929/programs/cache_probe.py) sends
  each of four prompts twice with identical text and reads the second
  response's `usage.prompt_tokens_details.cached_tokens`.
- **NCCL staging buffers:**
  [`nccl_maps.py`](installer-memory-20260929/programs/nccl_maps.py) counts
  each vLLM process's anonymous mappings of 9,633,792 bytes, the host
  buffers NCCL keeps per connection when GPUDirect RDMA is unavailable.

## Result

### KV capacity

Two Sparks ([engine log lines](installer-memory-20260929/results/tp2-kv-capacity.txt)):

| KV cache per Spark | Page (tokens) | Context window | KV capacity (tokens) |
|---|---|---|---|
| 8.75 GiB | 1,024 | 262,144 | 747,630 |
| 10.5 GiB | 1,024 | 262,144 | 897,157 |
| 10.5 GiB | 2,048 | 262,144 | 1,437,393 |
| 11 GiB | 2,048 | 262,144 | 1,506,008 |
| 9 GiB | 2,048 | 262,144 | 1,231,548 |
| 8.5 GiB | 2,048 | 262,144 | 1,162,934 |
| 8.5 GiB | 2,048 | 1,048,576 | 1,300,391 |
| **10 GiB** | **2,048** | **1,048,576** | **1,530,566** |

The first row is the profile at `59b253fd` and the last is the profile in
this record's commit. On the pair, 2,048-token pages hold 1.60 times as many
tokens as 1,024-token pages in the same memory. At a 262,144-token context
window, capacity grows by about 136,900 tokens per GiB; with the
1,048,576-token window the engine reports 12% more tokens for the same 8.5 GiB.

Four Sparks, all with a 1,048,576-token context window
([summary](installer-memory-20260929/results/tp4/summary.txt)):

| KV cache per Spark | Page (tokens) | KV capacity (tokens) |
|---|---|---|
| 24 GiB | 512 | 2,173,412 |
| 24 GiB | 1,024 | 3,676,901 |
| 24 GiB | 2,048 | 3,656,321 |
| 32 GiB | 1,024 | 4,902,535 |
| 38 GiB | 1,024 | 5,821,269 |
| **40 GiB** | **1,024** | **6,128,169** |
| 44 GiB | 1,024 | 6,740,986 |
| 46 GiB | 1,024 | 7,046,902 |

The first row is the profile at `59b253fd`. On the ring, 1,024-token pages
hold 1.69 times as many tokens as 512-token pages, 2,048-token pages hold no
more than 1,024-token pages, and capacity grows by about 153,200 tokens per
GiB. The [page-size record](../images/dev-20260927-mimovision-glm53-flash-tp2-pages-20260928.md)
describes why smaller pages hold fewer tokens; this record does not establish
why 2,048-token pages add nothing on the ring.

### Prefix-cache reuse

Cached tokens on the second send of a prompt
([pair](installer-memory-20260929/results/tp2-cache-reuse.txt),
[ring](installer-memory-20260929/results/tp4/prefix-cache-reuse.txt)):

| Prompt (tokens) | Ring, 1,024-token pages | Ring, 2,048-token pages | Pair, 2,048-token pages |
|---|---|---|---|
| 2,458 | 1,024 (41.7%) | 0 | 0 |
| 9,457 | 8,192 (86.6%) | 6,144 (65.0%) | 6,144 (65.0%) |
| 23,731 | 22,528 (94.9%) | 20,480 (86.3%) | 20,480 (86.3%) |
| 38,703 | 36,864 (95.2%) | 34,816 (90.0%) | 34,816 (90.0%) |

The pair's prompts were one token longer. Every value equals
(⌊(P − 1) / B⌋ − 1) × B for a prompt of P tokens and pages of B tokens.

### NCCL staging buffers on the pair

With NCCL's default channel count, the profile at `59b253fd` held 256
connection buffers of 9,633,792 bytes in Node A's worker, 2,352 MiB. With
four channels each worker holds 16, 147 MiB
([counts](installer-memory-20260929/results/tp2-nccl-buffers.txt)).

### Node A memory on the pair

The profile in this record's commit, installed and then measured after the
[acceptance run](../images/dev-20260928-plainstatus-glm53-flash-tp2-20260929.md)
(functional checks, correctness screen and throughput matrix). Every image
step used images the deployment had not seen
([raw results](installer-memory-20260929/results/tp2-10gib)):

| Load | Node A before | Node A low | Node 1 low |
|---|---|---|---|
| None, 60 s | 4.32 GiB | 4.16 GiB | 5.87 GiB |
| 4, then 8 text streams at 128K tokens of context | 4.29 GiB | 4.09 GiB | 5.82 GiB |
| One request with 8 images | 4.11 GiB | 1.99 GiB | 5.38 GiB |
| Two requests with 8 images at once | 2.55 GiB | below 1.2 GiB after 3 s; load stopped | 5.35 GiB |
| Four requests with 8 images at once | 2.24 GiB | below 1.2 GiB after 6 s; load stopped | 5.09 GiB |
| 4 text streams at 64K, two 3-image requests and one 16-frame video at once | 2.89 GiB | 2.08 GiB | 4.68 GiB |

- Another installation of the same settings had 6.28 GiB free on Node A
  right after it became ready; after the acceptance run's text loads, this
  deployment's Node A had 4.3 GiB.
- After the first image request, Node A settled at 2.56 GiB, 1.55 GiB below
  its level before that request, and stayed there.
- The guard's one-second readings caught the two stopped steps at 1.16 and
  1.01 GiB; the samplers' lowest readings in the same steps were 1.73 and
  1.28 GiB, so Node A's free memory fell in dips shorter than a second.
- With the per-request limit raised to 16 on a freshly installed deployment,
  one 16-image request of 57,501 prompt tokens took Node A from 6.11 to
  1.40 GiB ([output](installer-memory-20260929/results/tp2-16-image-requests.txt)).

### Node A memory on the ring

The profile in this record's commit (40 GiB, up to 32 images per request)
and the same settings at 37 GiB. Every image step used images the deployment
had not seen, and the fleet guard read all four Sparks every 0.2 s; each
32-image request carried 32 images of 2048×2048 pixels, 131,160 prompt tokens
([summary](installer-memory-20260929/results/tp4/summary.txt), runs `P` and
`Q`):

| Load | Node A low, 40 GiB | Node A low, 37 GiB | Lowest worker, 40 / 37 GiB |
|---|---|---|---|
| None, after startup | 18.24 GiB | 21.17 GiB | 20.27 / 23.27 GiB |
| Text: prefill to 128K tokens, 4 to 16 streams at 64K and 128K | not run | 20.05 GiB | — / 21.90 GiB |
| One 32-image request, the first | 11.72 GiB | 13.02 GiB | 17.56 / 20.63 GiB |
| One 32-image request, repeated | 8.98 and 8.90 GiB | 12.42 GiB | 17.60 / 20.58 GiB |
| Two 32-image requests at once | 6.38 GiB | 8.96 GiB | 16.29 / 19.30 GiB |
| Three 32-image requests at once | not run | 6.84 GiB | — / 18.17 GiB |
| 16 requests with 8 images at once, 2 rounds | not run | 5.40 GiB | — / 19.35 GiB |
| Six more rounds of those 16 requests | not run | 4.78 GiB | — / 19.32 GiB |
| 8 text streams at 64K, 16 three-image requests and 4 videos at once | 9.10 GiB | 11.01 GiB | 16.75 / 19.81 GiB |

- Node A kept 4.07 GiB (40 GiB) and 2.84 GiB (37 GiB) of its first 32-image
  request after the request finished, so the repeats start lower.
- Three and four 32-image requests at once were not run at 40 GiB, and four
  were not run at 37 GiB: extrapolating from the one- and two-request steps
  as low(2) − k × (low(1, repeated) − low(2)) gives 3.8 GiB for three and
  1.2 to 1.3 GiB for four at 40 GiB, and 2.0 GiB for four at 37 GiB. A step
  ran only when its extrapolated low was at least 4 GiB, so that a spike
  shorter than the guard's 0.2 s reading interval could not exhaust Node A.
- The workers stayed above 16 GiB in every step; only Node A preprocesses
  images.

The memory settings, each compared at otherwise equal settings in the
[summary](installer-memory-20260929/results/tp4/summary.txt) (1-second
samplers):

- **Image caps and glibc settings:** at 40 GiB, 16 concurrent 3-image
  requests took Node A to 6.65 and 7.53 GiB in two installations without
  them, and to 11.00 GiB with them. At idle, the glibc settings left 1.5 GiB
  more free on Node A and 1.8 GiB more on each worker.
- **Two B12X compile workers:** the lowest point of a start that compiles
  kernels is the bind preparation right after the KV cache is allocated. At
  24 GiB and 1,024-token pages, eight compile workers took Node A 4.1 to
  4.4 GiB below its idle level there, and two workers 1.1 to 1.3 GiB. Starts
  that find every kernel in the compile cache run no workers.

### Speed on the ring

The text load ran on every ring configuration above except the 40 GiB one
with 32 images per request. At 1,024-token pages,
prefill was 3,732 / 3,626 / 3,481 tokens/s at 16K / 64K / 128K tokens
against 3,745 / 3,610 / 3,512 for three runs of the profile at `59b253fd`,
and decode with 4 to 16 streams at 64K and 128K tokens was within 3% of the
same runs' engine steps per second
([summary](installer-memory-20260929/results/tp4/summary.txt)).

## Conclusion

- **Pair:** 10 GiB of KV cache per Spark in 2,048-token pages holds
  1,530,566 tokens with a 1,048,576-token context window, 2.05 times the
  747,630 tokens of the profile at `59b253fd`. Text loads up to 8 streams at
  128K tokens left Node A 4.09 GiB. Images set the limit: after the
  deployment had served long text and one 8-image request, Node A idled at
  about 2.6 GiB, one 8-image request took it to 1.99 GiB, and two or four at
  once fell below 1.2 GiB within seconds. The profile allows 8 images per
  request, so two concurrent requests with that many images can exhaust
  Node A.
- **Ring:** 40 GiB per Spark in 1,024-token pages holds 6,128,169 tokens,
  2.82 times the 2,173,412 tokens of the profile at `59b253fd`, with no
  measurable change in prefill or decode speed. With up to 32 images per
  request, one such request left Node A 8.9 to 11.7 GiB and two at once
  6.38 GiB; by extrapolation three at once leave about 3.8 GiB and four about
  1.2 GiB. The same settings at 37 GiB (5,668,802 tokens) left 6.84 GiB with
  three at once.
- Page size, not KV bytes alone, sets capacity: 2,048-token pages on the pair
  and 1,024-token pages on the ring hold the most tokens per GiB for their
  topology, and a repeated prompt reuses all but its last one to two pages.

## Limitations

- One pair and one ring; each step ran once unless a table shows repeats.
- Nothing bounds the images in flight. A request waiting for one of the
  engine's slots already holds its preprocessed images on Node A, so more
  concurrent image requests than measured lower Node A further at any KV
  size, whatever the per-request limit.
- The pair's lows come from one-second samplers. On the ring, one-second
  samplers read 0.5 to 1.0 GiB higher than 0.2 s readings during image steps,
  and the pair's guard caught readings 0.3 to 0.6 GiB below its samplers' lows in
  the same steps, so the pair's image lows are upper bounds. The ring's
  comparisons of memory settings also come from one-second samplers.
- The test images are random noise, most of them resized to the 4,096-token
  cap; smaller images cost proportionally less. Videos were 1920×1080 noise
  sampled at 16 frames.
- No request used the full 1,048,576-token context window.
- The engine reports 12% more tokens at a 1,048,576-token window than at
  262,144 tokens for the same KV bytes on the pair; this record does not
  explain the difference.
- The pair's measurements of the 8-image mix of text, images and video ended
  after 15 s, when an installation replaced the deployment; that step is in
  the raw results but not in the tables.
- The ring's configurations at 24 GiB with eight compile workers and 1,024-
  or 2,048-token pages were installed from a commit that also converted the
  line endings of host-side files; their serving configuration equals the
  one described.
