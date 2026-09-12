# SparkRing benchmark results

Selected benchmark records are summarized below. Throughput is in tokens per
second; decode throughput is aggregate across concurrent requests. `C1`, `C8`
and similar labels mean one, eight or that many simultaneous requests.
The records use different models, images, sampling settings and sample counts;
consult their conditions and limitations before comparing results.

## Recorded throughput at 16K context

| Profile | Prefill | C1 decode | C8 decode | Highest recorded concurrency at 16K |
|---|---:|---:|---:|---:|
| GLM-5.3 Flash B12X-KDA DCP4 public image, four Sparks | 2,649 | 37.97 | — | C4: 90.36 |
| GLM-5.2 EXL3 3.5-bpw, four Sparks | 671 | 20.15 | 64.13 | C8: 64.13 |
| DeepSeek-V4-Flash DSpark, two Sparks | 1,926 | 58.36 | 162.69 | C32: 307.13 |
| DeepSeek-V4-Flash-0731, four Sparks | 2,488 | 68.84 | 265.16 | C32: 508.11 |
| Qwen3.8-27B EXL3 K5/K6, two Sparks | 1,367 | 29.50 | 142.20 | C16: 184.39 |
| Qwen3.8-27B EXL3 K5/K6, four Sparks | 1,964 | 35.07 | 191.02 | C8: 191.02 |

Dashes indicate no value in the linked record. Full context/concurrency matrices,
coding workloads, sample counts and uncertainty estimates remain in those records.
The GLM-5.3 B12X-KDA row is one bounded observation per cell; it has no
repeated-sample performance qualification.

## Records and receipts

| Profile | Results | Matrix image | Receipts |
|---|---|---|---|
| GLM-5.3 Flash B12X-KDA DCP4 public image, four Sparks | [Bounded record](../performance/records/glm53-flash/b12x-kda-dcp4-20260903.md) | — | [Sanitized summary](../performance/receipts/glm53-flash/b12x-kda-dcp4-20260903/summary.json) |
| GLM-5.3 Flash DCP4 image `380283a5`, four Sparks | [Bounded record](../performance/records/glm53-flash/dcp4-24g-default-20260901.md) | — | [Sanitized summary](../performance/receipts/glm53-flash/dcp4-24g-default-20260901/summary.json) |
| GLM-5.2 EXL3 3.5-bpw, four Sparks | [Full record](../performance/records/glm-3.5bpw/normalized-base-20260822.md) | [Matrix image](../performance/records/glm-3.5bpw/normalized-base-20260822.png) | [Receipts](../performance/receipts/glm-3.5bpw/temp1/) |
| DeepSeek-V4-Flash DSpark, two Sparks | [Full record](../performance/records/deepseek-v4-flash/normalized-tp2-base-temp1-n5-20260823.md) | [Matrix image](../performance/records/deepseek-v4-flash/normalized-tp2-base-temp1-n5-20260823.png) | [Receipts](../performance/receipts/deepseek-v4-flash/temp1/) |
| DeepSeek-V4-Flash-0731, four Sparks | [Full record](../performance/records/deepseek-v4-flash/normalized-tp4-base-temp1-n5-20260823.md) | [Matrix image](../performance/records/deepseek-v4-flash/normalized-tp4-base-temp1-n5-20260823.png) | [Receipts](../performance/receipts/deepseek-v4-flash/temp1/20260823-tp4/) |
| Qwen3.8-27B EXL3 K5/K6, two Sparks | [Full record](../performance/records/qwen38-27b/normalized-tp2-1m-probmtp-temp1-20260823.md) | [Matrix image](../performance/records/qwen38-27b/normalized-tp2-1m-probmtp-temp1-20260823.png) | [Receipts](../performance/receipts/qwen38-27b/temp1/20260823-tp2/) |
| Qwen3.8-27B EXL3 K5/K6, four Sparks | [Full record](../performance/records/qwen38-27b/normalized-tp4-1m-probmtp-temp1-20260823.md) | [Matrix image](../performance/records/qwen38-27b/normalized-tp4-1m-probmtp-temp1-20260823.png) | [Receipts](../performance/receipts/qwen38-27b/temp1/20260823-tp4/) |

The normalized DeepSeek two-Spark profile also completed a
[three-hour llama-benchy prefix-cache benchmark](../performance/records/deepseek-v4-flash/llama-benchy-normalized-tp2-20260822.md).

The GLM-5.3 Linux/ARM64 image identified by the linked validation report completed a TP4/DCP1
[942,898-token needle retrieval](../runtime/glm53-flash-jj-r8-gb10/PUBLIC_IMAGE_VALIDATION.md)
with a 1M-token request limit and 26 GiB of FP8 KV per rank.
