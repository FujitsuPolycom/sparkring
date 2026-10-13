# Dashboard counters and graph verification

Status: **Experimental**, CPU and offline browser coverage only. This does
not qualify serving performance on a DGX Spark.

## Data contract

The Node A dashboard returns raw `/metrics` counters with the collector's
read time in `counters`. `plugin_counters` carries a `sparkring-stats/v1`
snapshot when one arrived through the existing host-status path. Node A
does not compute model rates or retain graph samples. The browser keeps
up to six hours, capped at 4,321 points per deployment; its five-minute
counter window is capped at 100 samples. Closing or reloading the page
discards this history. The terminal keeps the same bounded windows only
while its reader is open. `--once` has no earlier sample and shows dashes
for rates.

The browser implementation is
[dashboard_rates.js](../../runtime/host/dashboard_rates.js); the terminal
implementation is [dashboard_rates.py](../../runtime/host/dashboard_rates.py).
Their CPU test feeds both implementations the same sequence and compares
the resulting documents. The service embeds the JavaScript in its page
before computing the Content Security Policy hash; no asset request is added.

| Value | Computation in the viewer |
|---|---|
| Prefill | Increase in prompt tokens divided by elapsed time between the last two distinct samples |
| Output / decode | Increase in generated tokens over a 15-second window; if observations are farther apart, the last two observations |
| Five-minute output | Increase in generated tokens over the available five-minute window |
| Steps | Increase in observed, non-dummy iteration-detail records divided by elapsed time; unavailable without vLLM iteration details |
| Accepted drafts | Increase in accepted draft tokens divided by increase in verified draft tokens |
| Draft length | One bonus token plus accepted draft tokens divided by draft rounds over five minutes |
| First token | Increase in first-token latency sum divided by increase in count over five minutes |
| Prefix hits | Increase in `/metrics` cache hits divided by queries over five minutes, even when token counters come from the plugin |

Prefill is a dash on the first sample, when its counter did not increase,
when the observed running-request gauge is zero, or while quiet, stopped
or stale. A positive value describes the observed interval; sampling cannot
prove that prefill is still executing at the instant the screen is drawn.
The approximate `/metrics` method uses `vllm:prompt_tokens_total`, which can
include cached tokens. It is not GPU prefill compute throughput. Request
prefill histograms are deliberately unused: completion-time observations
and overlapping requests do not provide a matching wall-time denominator.

Plugin token rates are exact differences of its reported counters over its
monotonic snapshot interval. Computed prompt tokens exclude cache hits.
They remain interval averages, not instantaneous GPU timings, and tokens
appear when vLLM reports them to its frontend. Source labels distinguish
`plugin` from `/metrics`; read timestamps expose observation age. Prefix
hits always retain the `/metrics` source.

Duplicate timestamps add no history. A source or run identity change, a
backward timestamp, or a decreasing counter clears that source's window
and graph history. Missing counters remain unavailable. Plugin data older
than 180 seconds, more than five seconds in the future, or dated before
the `/metrics` process start falls back to `/metrics`. Cross-host wall
clocks therefore need synchronization for freshness checks; plugin rate
division uses monotonic time. A source transition needs two samples before
showing a rate. No rate is extrapolated through quiet mode.

## Plugin packaging and delivery

[SparkRingStatLogger](../../integrations/vllm/runtime_status/sparkring_runtime_status/stat_logger.py)
ships in `sparkring-runtime-status` 0.3.7 through the
`vllm.stat_logger_plugins` entry point `sparkring_stats`. The existing
[status wheel composition](../../runtime/images/installer-images.md#replacing-the-runtime-status-package)
accepts its wheel/source pins, records the added entry point and leaves
older pinned packages valid. The [pinned development descriptor](../../runtime/images/compositions/installer-dashboard-status037/descriptor.json)
replaces status 0.3.6 in the `sha256:d52737a109e0` parent. Its artifact
pins are local build inputs; no image was built or published. No published
image lock or default changes.
Build and record a development image with this package before selecting it
for installation; repository code alone does not add it to an existing image.

The installer recognizes `status_version` 0.3.7 or later in admitted 0.3.x
images. It adds the plugin to `VLLM_PLUGINS`, sets the deployment ID and
`SPARKRING_STATS_PATH=/run/sparkring/stats/snapshot.json`, and bind-mounts
`/run/sparkring/telemetry/DEPLOYMENT_ID` from each participating host.
The installer creates that directory during container creation and restores it before start after a reboot. Only the
API process emits statistics. Standalone Compose exports, which lack an
installer deployment identity, keep `/metrics` as their source.

The implementation follows the extracted serving image's
`vllm/v1/metrics/loggers.py` (`StatLoggerBase`), `stats.py`
(`SchedulerStats`, `IterationStats`) and `vllm/v1/spec_decode/metrics.py`.
The source snapshot is `image-d52737-sources`. `record()` only accumulates
already-computed CPU values, checks the clock and replaces a single pending
snapshot at most twice a second. One daemon writer serializes and atomically
replaces the file. It performs no retry or logging on write failure. A slow
disk can stall that writer, but cannot create an unbounded queue or wait in
`record()`. The existing vLLM `log()` callback also offers unchanged counters
while idle. No model, worker RPC, CUDA event or GPU synchronization is added.
A changed statistics interface disables the logger rather than failing
serving. Data-parallel configurations with more than one engine are disabled
to avoid presenting partial totals as deployment totals.

Step duration uses `SchedulerIterationDetails.elapsed_ms`. These details
exist only when vLLM's `enable_logging_iteration_details` is already enabled;
the plugin does not enable it. Absent details make steps/s and duration
unavailable. First-token durations are the existing frontend latency list.
The plugin provides no isolated prefill kernel duration.

The host agent includes bounded regular-file snapshots in its existing
status document. The recovery checker copies only the deployment head's
snapshot from its existing `node status` response into the recovery finding.
The dashboard reads that finding and its local agent status at the existing
30-second file cadence. The local path uses the lock's deployment ID, not
the controller directory name. Remote delivery depends on the existing
recovery checker being enabled and eligible to observe the deployment.
Its roughly one-minute interval, and any slower host-agent serving interval,
limit plugin freshness. There is no added SSH call, endpoint, polling loop
or websocket. The `/metrics` request schedule is unchanged. Dashboard quiet
mode suppresses its requests and rate updates; it does not disable the
independent vLLM logger or change the host-agent/recovery schedules.

## Coordinator browser checklist

1. Open the Node A page, allow two observations, and check Output, Prefill,
   steps, acceptance and their source labels. An idle deployment shows a
   prefill dash. An image without the plugin shows approximate `/metrics`.
2. Hover an output, prefill or KV graph: inspect the full stored value and
   ISO timestamp. Select **Expand graph** to inspect up to six hours already
   accumulated in that view. No history is fetched from the service.
3. Tab to an expand button, press Enter, then use the sample slider with
   arrows, Home and End. Escape closes the dialog and restores focus.
   Tap the control and drag the slider on a narrow touch screen.
4. Repeat in the OS light and dark themes. Check narrow cards, readable
   source labels and the expanded graph. The dialog freezes its selected
   history until closed, so incoming observations do not move the slider.
5. Watch the browser network panel: graph interaction adds no request.
   Activate dashboard quiet mode and verify counters/rates do not acquire
   synthetic samples. Release quiet mode and wait for real observations.
6. Restart a deployment through the normal authorized procedure. Confirm
   that the first sample is a dash and no cross-restart spike appears.
   Check `sparkring dashboard tui` for the same source and idle behavior.

## Hardware comparison plan (not executed by CPU tests)

Use an authorized maintenance window and the same development image with
and without `SPARKRING_STATS_PATH` set. Keep the DeepSeek checkpoint, TP
layout, speculative configuration, CPU affinity and all serving arguments
fixed. Keep iteration-detail logging at its original setting in both arms.
Hold dashboard quiet mode equally in both arms and keep agent/recovery
cadences fixed. Warm compilation and caches before measuring.

Run at least five alternating on/off pairs of the same single-stream
DeepSeek decode workload: fixed prompt, seed, maximum output length and
request count, with no other model traffic. Record output tokens/s,
per-token latency median/p95, TTFT, API-process CPU and writer CPU, snapshot
size/frequency, and any dropped/stale samples. Report every pair and the
paired relative differences, including run-to-run variance. Investigate
a repeatable decode regression exceeding 1%; do not claim zero overhead
from the CPU micro-benchmark. Repeat once with deliberately failed snapshot
writes to confirm serving continues, then restore the original environment.
Only the coordinator's hardware evidence can qualify this image change.
