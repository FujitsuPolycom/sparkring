# Scheduler liveness during a million-token prefill

Status: **qualified** for the single request described here. The scheduler
liveness endpoint remained healthy while a cold prefill exceeded the default
output timeout and KV allocation continued to advance. The native-MTP3 mesh
deployment remains research-only.

## Conditions

Four NVIDIA DGX Spark GB10 systems ran GLM-5.3-Flash-NVFP4-Spark on the SparkRing
mesh. One request used exactly 1,000,000 prompt tokens, temperature zero, and
a 64-token output limit. The API reported zero cached prompt tokens. Model
startup readiness had completed, and three consecutive idle checks preceded
the request. There was one repetition at concurrency one; this was not a
comparison between serving configurations.

| Setting or source | Value |
|---|---|
| Model repository | `local-inference-lab/GLM-5.3-Flash-NVFP4-Spark` |
| Model checkpoint identity | `357f6a86160ebd5caff25d9a10d9f29e8547b16c6c73e78751fa69fde11ac4e4` |
| Parallelism and speculation | TP4 / DCP1 / PP1 / native MTP3 |
| Context and batch limits | 1,048,576 context tokens; 8,192 batched tokens; 16 sequences |
| KV allocation | 24 GiB FP8 per rank; 512-token blocks |
| Prefill configuration | Chunked prefill; scheduler interval 2; coalescing and token-sharded mHC enabled; compact index cache disabled |
| Cache configuration | SparkCache read/write; 65,536-token retained-span ceiling |
| Output inactivity timeout | 300 seconds, using the wrapper default with no environment override on any rank |
| Metric scrape / endpoint poll | 10-second scrape; 2-second wait between endpoint polls |
| Local image config digest | `sha256:cd92adc4436c61290dbecc35362db447c7ae69e1d03bdb99704af0ed6c517b38` |
| vLLM serving source | `df62335d8248587f8d3fd1d9a234d1c162a9b84d` |
| SparkCache source | `2bc05bc9e94a4344758e48db36f69a46dafa6946` |
| B12X source | `0b6d61c37c87ae49d2f9d20d38b9da023146e243` |
| Startup source composition | `767522e399ad61d3b3cf02259414b1007becc030` |
| Isolated liveness change | `aea0e6819e0449a90b7807f41efb51ad4cfdc9df` on `e441057e17fb975358ec9f2abb7a23ac7fb2c23e` |
| Installed scheduler-liveness SHA-256 | `9078d796aeb8ad59e7a78e9739168596abecfb20c043925a0d4e728f3673a53c` |

The installed liveness module is byte-identical to the isolated source change.
The startup composition includes additional startup changes and preserves
the `glm53-thinking-low/v1` warmup transform. All four ranks verified the
installed startup file hashes. The image digest identifies a local image;
that image has not been released to a registry.

The [machine-readable record](prefill-aware-liveness-20260908.json) contains
an exact repeat recipe for the synthetic prompt, its UTF-8 hash, the request
parameters, checkpoint identity, source hashes, and per-rank verification
outcomes. The harness files are identified by SHA-256 because the measurement
harness was run from local source.
The [SHA-256 manifest](prefill-aware-liveness-20260908.sha256) covers this record,
its JSON projection, and the complete trace.

## Measurement

The harness calibrated the request through the serving tokenizer and checked
the completion's reported prompt-token count. The synthetic prompt places a
project code before repeated filler text and asks for that code at the end.
The response must match `RIVER-649271` exactly and finish with `stop`.

Client `time.monotonic()` measured the complete non-streaming request.
The outer observation includes request-receipt handling and generation;
it is not a separate prefill TTFT measurement. The inner request helper also
records its elapsed time so the two timing scopes remain explicit.

The [complete liveness trace](prefill-aware-liveness-20260908.liveness.jsonl)
preserves all 206 original JSONL rows with line endings normalized to LF.
No rows or values are omitted. Each row contains the
poll time, HTTP status, and liveness body. There were 198 active rows, defined
by `running_requests > 0`. The reported maxima are taken over those active
rows. Observed poll spacing ranged from 2.000 to 2.047 seconds, with a median
of 2.031 seconds. Endpoint polls can observe the same underlying metric scrape.

The raw output gap is `output_stalled_seconds`. The separate inactivity timer
is `progress_stalled_seconds`; its reset signal identifies output-counter
movement or a new KV/prompt maximum. Allocation growth does not reset the raw
output gap. The trace can be summarized from the repository root:

```python
import json
from pathlib import Path

path = Path("performance/records/glm53-flash/prefill-aware-liveness-20260908.liveness.jsonl")
rows = [json.loads(line) for line in path.read_text().splitlines()]
active = [row for row in rows if row["body"]["running_requests"] > 0]
print(len(rows), len(active))
print(max(row["body"]["output_stalled_seconds"] for row in active))
print(max(row["body"]["progress_stalled_seconds"] for row in active))
print(all(row["http_status"] == 200 and row["body"]["healthy"] for row in active))
```

The JSON record retains hashes of the original request, trace, startup audit,
source overlay, source verification, and CPU receipts. The large request
payload is represented by its exact repeat recipe. Private deployment
identifiers and unrelated logs are omitted. One repetition does not supply a
variability estimate.

## Result

| Measurement | Observed result |
|---|---:|
| Request window, UTC | 2026-09-08 21:29:43.088 to 21:36:34.878 |
| Prompt / generated tokens | 1,000,000 / 38 |
| Reported cached prompt tokens | 0 |
| Exact answer / finish reason | `RIVER-649271` / `stop` |
| Outer request observation | 412.282 seconds |
| Inner request helper | 411.246 seconds |
| Maximum raw output gap during active rows | 398.582 seconds |
| Maximum inactivity during active rows | 9.967231 seconds |
| Successful liveness polls | 206 / 206 HTTP 200 |
| Healthy active liveness polls | 198 / 198 |
| Active polls reporting KV allocation as last progress signal | 193 / 198 |
| Rank source/hash checks | 4 / 4 passed |
| Rank process generations / restarts during request | Unchanged on all four / zero |

The post-request idle check reported no running or waiting work. Separately,
the CPU suite passed 611 tests with five platform or optional-input skips.
Two focused long-prefill regressions fail on the exact base source and pass
with this change. CPU tests also cover flat stalls, allocation oscillation,
request churn, counter loss/reset, idle intervals, malformed metrics, and
packaged-module compatibility.

## Conclusion

The observed request produced the correct answer while its raw output gap
exceeded 300 seconds. Allocation maxima continued advancing, keeping
inactivity below 10 seconds and every observed active liveness response
healthy. This qualifies the prefill-aware policy for this bounded native-MTP3
case. Flat metrics still reach the configured failure threshold in CPU tests.

## Limitations

KV allocation is an activity proxy, not a GPU heartbeat. Fully preallocated
work or a long operation with no observable intermediate change still needs
a timeout above its measured gap, with margin. Prompt totals in the inspected
vLLM source are emitted with output statistics; this record does not assume
they advance after every prefill chunk. Aggregated metrics can hide one
stalled engine behind another progressing engine.

This single exact-answer text probe does not establish broad long-context
accuracy or qualify the issue reporter's DFlash/DCP4 configuration. No GPU
hang was deliberately injected. Endpoint polling and metric sampling do not
capture every instantaneous state. The reported duration includes generation
and client handling, and is not prefill throughput or a causal speedup.

The source fix does not change an existing immutable image. Installing this
behavior requires building and deploying an image containing the fixed
liveness module; the local image identified above is not a public release.
