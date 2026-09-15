# SGLang TP4 streaming soak

Status: **Validated** for the bounded streaming stability checks below.
The tested 430080-context site profile completed six hours at concurrency eight.
This does not change the recipe's 262144-context defaults.

## Conditions

Four GB10 DGX Sparks on a directly cabled four-node cycle, TP4/EP4,
DeepSeek-V4.1-Flash revision `dba1be0a40aa45a94ad051997016db3960a90277`,
Mia adapter commit `e59e6eb67479aa68f6fa700c600dc90a0729b5ec`, and the
[pinned ARM64 SGLang base](../../../runtime/deepseek-v41-sglang/pins.json).
The tested image is the same artifact identified in the
[controlled replay comparison](sglang-decoder-replay-20260911.md).
SparkRing's patched NCCL 2.30.7 replaces the pip library in place.

Decoder-tail replay enabled, context 430080, chunk 4096, requested shared
token pool 1500000 (actual 1499904), maximum eight requests, memory fraction
0.90, DSpark block five, no SPS/STS tables, b12x MXFP8, MoE fused finalize
disabled, expandable segments disabled, packed Engram on local NVMe,
96 I/O threads, zero GiB row cache. The launch fingerprint was unchanged
before and after the soaks. No restart or cache flush separates them.

The deployed [harness text](sglang-soak-20260912/soak.py.txt) has SHA256
`876133b6141d0ef2464a43310eff5f56237536694f30f9cd75754ab18e20b4ee`;
the [prompt file](sglang-soak-20260912/prompts-v1.json) has SHA256
`f8106746dc729d287d3509376d683719862b13f3657847e86153856be6c80bb3`.
Each wave submits eight streaming requests using the same eight short category
prompts. The harness uses temperature 1.0, top-p 0.95, thinking disabled, and
a 256-token output cap; it overrides the prompt file's temperature and
per-category caps. The model was already warm. Prefix caching remains enabled;
repeated prompts are not a cold-prefill workload. The serving route remained
available to other clients, whose traffic was not measured separately.

## Measurement

The client sums server-reported completion tokens in each wave and divides by
that wave's wall-clock duration, including prefill, queueing and streaming.
The table reports the median and range across waves, not a six-hour weighted
token rate or a prefill benchmark. No warm-up observations are discarded.
This is one short run and one overnight run, with no repeated-run confidence
interval and no controlled comparison between them.

A successful request must receive nonempty content, positive completion-token
usage, a stop or length finish reason, and the streaming DONE terminator.
The harness checks for 90-second stream gaps/timeouts and exits nonzero on
any failed or hung request. Reaching the output cap is recorded separately.
Memory CSV columns are UTC time followed by rank 0–3 MemAvailable in GiB,
sampled sequentially about once per minute. Per-rank guards additionally sample
every five seconds and stop the model below 8 GiB twice or 4 GiB immediately.

Raw [short observations](sglang-soak-20260912/short.json),
[overnight observations](sglang-soak-20260912/overnight.json),
[short memory](sglang-soak-20260912/short-memory.csv), and
[overnight memory](sglang-soak-20260912/overnight-memory.csv) retain request
timings/counts and rank measurements without credentials, endpoints or outputs.
Request counts, failures, hangs, truncations and median throughput were
independently recomputed from these observations.

## Result

The short run started September 12 at 05:05:01 UTC and its coordinator accepted
PASS at 05:25:11 UTC. The overnight run started at 05:30 UTC and the coordinator
completed at 11:30:12 UTC.

| Measurement | 20-minute gate | Six-hour soak |
|---|---:|---:|
| Waves | 58 | 1653 |
| Requests | 464 | 13224 |
| Failures / hangs | 0 / 0 | 0 / 0 |
| Median aggregate tokens/s per wave | 79.65 | 106.7 |
| Aggregate tokens/s range | 29.0–100.5 | 86.4–122.5 |
| Responses reaching 256-token cap | 112 | 3261 |
| Memory samples | 20 | 348 |
| Lowest sampled MemAvailable, GiB | 18.8996 | 28.4652 |

Overnight per-rank memory floors were 29.5832, 30.7001, 28.4652 and 28.5318 GiB.
All four guards were healthy with zero strikes and no trip at the final check.
The backend health endpoint returned HTTP 200. An authenticated completion
through the existing public router returned the requested READY response in
0.48 seconds after the soak, with three generated tokens.

## Conclusion

The recorded profile sustained this repeated short-prompt C8 streaming workload
for six hours without request failures or detected hangs and remained usable
through its router afterward. No serving restart was needed.

## Limitations

Request files retain success/hang flags and finish reasons, not complete SSE
streams. Counts can be recomputed; DONE-marker and gap detection cannot be
replayed from those summaries. Fingerprint, guard-health and router checks
are reported run observations without separate raw receipts linked here.

This does not score response correctness, reasoning quality,
vision, tool use, long-output generation, or long-context memory pressure.
The context setting is recorded configuration, not a context-capacity result
from these short prompts. Output-cap finishes are accepted by this stability
gate and do not imply complete answers. Sampled memory floors can miss brief
allocation peaks. The higher overnight rate cannot establish a tuning gain
over the short run. No transport counters or controlled competing-client
measurements were captured. Six hours without errors does not establish
unattended recovery or high availability, and decoder-tail replay retains the
correctness limitations stated in the separate replay record.
