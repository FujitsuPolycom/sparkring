GLM-5.3-Flash NVFP4 on four DGX Sparks (installer profile glm53-flash-nvfp4-spark-tp4): memory and KV-cache capacity measurements, 2026-09-29

Scope. Each run tag below is one installation of the profile, from a local branch that changes only
profiles/glm53-flash-nvfp4-spark-tp4/config.json, on one four-Spark ring (tensor parallel 4) running installer image
dev-20260928-plainstatus-cuda1342-nccl2323-status033. Every Spark is a GB10 with 128 GB of memory shared by the CPU and
the GPU, so the vLLM KV cache, the model weights and every host process draw from the same MemAvailable. A GB10 that
runs out of memory can stop responding until it is power-cycled, so the quantity measured throughout is the lowest
MemAvailable reached on each Spark. r0 is Node A (tensor-parallel rank 0, which also runs the OpenAI-compatible API
server and the vLLM engine core); r1-r3 are the three workers. All values are GiB.

Settings shared by every run. NCCL_MIN_NCHANNELS=4 and NCCL_MAX_NCHANNELS=4 (the value in the profile at main commit 59b253fd); with them every vLLM
worker holds 16 NCCL host buffer mappings of 9,633,792 bytes (0.14 GiB), counted by nccl_maps.py before and after each
text load. --max-num-seqs 16, --max-model-len 1048576, FP8 KV cache, MTP speculative decoding with 3 tokens.
The profile at main commit 59b253fd (tag A) uses 512-token split cache pages, the default of 8 B12X compiler processes, a 24 GiB KV
cache (--kv-cache-memory-bytes 25769803776), vLLM's default multimodal processor cache (4 GiB in the API server and
4 GiB in the engine core, both on Node A), the default vision token caps (8,000 per image, 240,000 per video), and up
to 4 images and 1 video per request.

Settings that runs change. "pages N": the four settings VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE,
VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE, --block-size and --mamba-block-size, all N tokens. "workers N": B12X_COMPILE_WORKERS=N
(compiler processes for every B12X preparation stage); "bind workers N": B12X_BIND_COMPILE_WORKERS=N (only the bind
stage, the one stage that runs after the KV cache is allocated). "no mm cache": --mm-processor-cache-gb 0. "vision caps":
--mm-processor-kwargs {"images_kwargs":{"max_image_tokens":4096},"videos_kwargs":{"max_image_tokens":8192}}. "glibc":
MALLOC_MMAP_THRESHOLD_=1048576 and MALLOC_ARENA_MAX=4 in the container environment. "images N": --limit-mm-per-prompt
{"image":N,"video":1} (vLLM accepts at most one video per request for this model family). "KV N": the KV cache budget in
GiB. Changing any B12X_* variable, or the KV size, makes the next start compile B12X kernels again (the "compilations"
field of each table row, per stage weights/state/bind, with the stage time).

Measurement. memsample.sh logs MemAvailable once a second on every Spark from install start (start-samplers.sh,
collect.sh). Steps of runs P and Q also ran under fleet_guard_fast.py, which reads every Spark every 0.2 s and kills the
load when any Spark falls below 2.0 GiB; there the lows come from the 0.2 s readings. The 1-second samplers miss short
spikes: on run P they read 0.5-0.75 GiB higher than the 0.2 s readings during 32-image requests, and on run Q 1.0 GiB
higher during 8-image bursts, so 1-second lows of image steps in runs E-O are that much optimistic. No guard fired in
any run, and no request failed.

Windows and columns (results/summary.txt, results/lows-*.txt, produced by table.py and fastlows.py).
startup: from the start of the new rank-0 container to "Application startup complete"; its low is the bind preparation
step right after the KV cache allocation whenever B12X compiles there. idle: from API ready to the text load.
load: the text load (run-load.sh: llm_decode_bench.py prefill at 16K, 64K and 128K tokens, decode at 4, 8 and 16
concurrent users at 64K and 128K context, temperature 1.0, 20 s per cell). img / img16: 16 concurrent requests with 3
unique random-noise images each (image_load.py, sizes cycling 2048x2048, 3840x2160, 1600x1200), 2 rounds, unless a run
says otherwise. burst6: 6 back-to-back rounds of those 16 requests. mixed: 8 text streams at 64K context, 16 image
requests (3 images) and 4 video requests (64 frames of 1920x1080, capped) at once, 2 rounds (mixed_load.py; in its
output, rounds of 4 requests are the video generator and rounds of 16 the image generator). mm: mm_stress.py, 4 images
of 2560x2560 per request and then one 16-frame 1920x1080 video per request, at 4 and 8 concurrent requests.
sNrunK / s2 / s3: N concurrent requests with 32 unique 2048x2048 images each (every image reaches the 4,096-token cap;
131,160 prompt tokens per request). "after X": the highest reading in the gap after step X, that is the settled level.
"Node A first reading": the level when a guarded step started. Request times are printed per round ("... s").

Fresh images. An image or video step is fresh when its seeds were not used by an earlier step on the same deployment
(each installation starts with an empty prefix cache). Repeated media hit the prefix cache, which skips the vision
encoder, so a repeated step understates memory. Steps after the LOAD_INDEX_START offset was introduced (runs K onward,
and the mixedfresh step of run J) are all fresh. Earlier repeats are flagged per run below.

Memory kept after images. results/summary.txt lists, per run, Node A's settled level in the 20 s before the first image
step and after it (retention.py). Node A keeps 1.6-4.2 GiB after the first image step; later bursts reuse that memory:
in runs J, K, M and N the settled level after six more bursts is at most 0.4 GiB below the level after the first.

Runs.
A: the profile at main commit 59b253fd (pages 512, workers 8, KV 24). 2,173,412 KV tokens. Text load only.
B: pages 1024, otherwise A. 3,676,901 KV tokens (+69% at the same 24 GiB). Text load only; startup compiled the state
and bind stages for 1,024-token pages with 8 compiler processes (4.1-4.4 GiB below idle during bind preparation).
C: pages 2048, otherwise A. 3,656,321 KV tokens (no gain over 1024; coarser prefix reuse, see
results/prefix-cache-reuse.txt). Text load only.
D: pages 1024, workers 2, KV 24. First start after the B12X_* change: every stage compiled (416, 224 and 4 kernels);
bind preparation 1.1-1.3 GiB below idle. Text load only.
E: pages 1024, bind workers 2, KV 32. 4,902,535 KV tokens. Text load, then mm at 4 and 8 concurrent requests. The mm 8
step repeated half of the mm 4 images and videos (seeds 0-15 of 0-31; videos 0-3 of 0-7).
F: E plus no mm cache. Text load; mm at 4 (fresh) and 8 (half repeated as in E); then image_load.py at 4 (fresh), 8
(round 1 repeated the 4-request images, round 2 fresh) and 16 (round 1 repeated, round 2 fresh) concurrent requests.
G: pages 1024, workers 2, no mm cache, KV 40. 6,128,169 KV tokens. Text load; image_load.py at 4, 8 and 16 with the
same repeats as F.
H: G at KV 44. 6,740,986 KV tokens. Same steps and repeats as G.
I: G's configuration started a second time (no compilations). Text load twice; 16 x 3 images, 2 rounds, fresh.
J: G plus vision caps and glibc, KV 40. 6,128,169 KV tokens. Text load; img16 fresh; burst6 (rounds 1-2 repeated the
img16 images, rounds 3-6 fresh); mixed (images repeated, videos fresh); mixedfresh (all media fresh).
K: J at KV 46. 7,046,902 KV tokens. Text load; img16, burst6 and mixed, all fresh.
L: J with images 16 at KV 32 (4,902,535 KV tokens), to measure requests with more images: iNcC = C concurrent requests
with N images each (sizes cycling as in img16), all fresh; per-step lows in results/lows-L.txt.
M: J with images 8 at KV 38. 5,821,269 KV tokens. Text load; img16 and burst6 with 8 images per request; mixed (3 images
per request); all fresh.
N: J with images 4 at KV 44. 6,740,986 KV tokens. As K (3 images per request), all fresh.
O: J with images 8 at KV 36. 5,515,352 KV tokens. Text load; the 8-image img16 step, fresh, was stopped during round 2.
P: J with images 32 at KV 40. 6,128,169 KV tokens. One 32-image request three times in a row (s1run1-3), two at once
(s2), then mixed; all fresh, all under the 0.2 s guard. Three and four concurrent 32-image requests were not run: the
extrapolation low(2) - k x (low(1 repeat) - low(2)) gave 3.8 GiB for three and 1.2-1.3 GiB for four, below the 4 GiB
threshold required to run them.
Q: J with images 32 at KV 37. 5,668,802 KV tokens. Text load; one 32-image request twice (s1run1-2), two at once (s2),
three at once (s3, run because the same extrapolation gave 5.5 GiB), then 16 x 8-image requests (img16, 2 rounds),
6 back-to-back rounds of them (burst6) and mixed (3 images per request); all fresh, all under the 0.2 s guard. Four
concurrent 32-image requests were not run (extrapolation from s1run2 and s2: 2.0 GiB).

Programs (programs/). Addresses, SSH targets and paths are environment variables or arguments (NODE_A, WORKERS or
WORKERS_CSV, API_HOST, OUT_ROOT, BENCH_DIR, BENCH_PYTHON, REMOTE_DIR, REPO); examples use 192.0.2.x. run-install.sh
installs a branch and starts the samplers; run-load.sh is the text load; run-final-checks.sh runs the text load, img16,
burst6 and mixed under guarded.py (runs J, K, M, N); run-images-fast.sh runs one image step under fleet_guard_fast.py
(runs P and Q; 32-image steps pass sizes 2048x2048). image_load.py, video_load.py, mixed_load.py and guarded.py are the
shared load generators and single-Spark guard; image_load.py and video_load.py read LOAD_INDEX_START, and mixed_load.py
takes --images. fleet_guard.sh is the 2-second all-Spark guard used alongside runs F-O.
