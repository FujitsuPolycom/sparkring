# Dashboard counters and graph verification

Status: **Experimental**, CPU and offline browser coverage only. This does
not qualify serving performance on a DGX Spark.

## Data contract

The Node A dashboard returns raw `/metrics` counters with the collector's
read time in `counters`. `plugin_counters` carries a `sparkring-stats/v1`
snapshot when one arrived through the existing host-status path. Node A
does not compute model rates. It retains up to three hours of raw samples
in memory, with no history file or NVMe writes. Restarting the dashboard
clears this buffer. The browser fetches `/api/history` once before its first
status render, then receives deltas through `/api/status?since=CURSOR`.
No extra request is added per poll. The history endpoint shares the status
endpoint's Host and authentication checks and supports gzip encoding.

**Source-read:** Each deployment has at most 2,161 samples (three hours at
five-second spacing, including both endpoints). At most eight deployments,
in discovery order, receive server history; additional deployments retain
live status and browser session history. Departed deployments are removed
on the next collection cycle, freeing their history slots. Age eviction
also runs during quiet mode. Only fixed numeric counter/gauge fields, a
bounded run identity, source timestamps, a sequence number and a gap flag
are retained. Each retained sample explicitly labels its selected source as `plugin` or
`metrics`.
No derived rate is stored.

### Unmanaged model observations

The host agent adds `model_containers` and `gpu_compute` to its existing
`sparkring-node-status/v1` snapshot. It reuses the inventory's running
container IDs and GPU PIDs. Local Docker-socket GETs inspect at most 32
containers, with a 256 KiB response limit, 250 ms socket timeout and a
one-second budget for starting inspections. It starts no extra subprocess.
GPU launch lookup reads at most 256 known PIDs and eight same-container
ancestors per PID. Each argv is limited to 256 arguments and 16 KiB;
oversized argv is discarded. Only recognized launch fields are emitted;
raw argv, arbitrary labels and environment values are excluded.

`runtime/host/node_containers.py` parses vLLM TP/PP/DP/DCP, node count/rank,
headless state, model name, HTTP binding/port and master rendezvous. SGLang
aliases and `--dist-init-addr`/`--nccl-init-addr` produce the same fields.
SparkRing's `/opt/sparkring/toolchain/toolchain.py serve` wrapper and
`python3 -m vllm.entrypoints.cli.main serve` supply the same vLLM launch facts.
Literal defaults are used only for recognized server launches without a
configuration-file argument. Shell variables are not expanded. Unknown
GPU containers remain visible with engine `unknown`; absent launch facts
cannot establish a cross-Spark group. Bridged Docker port translation and
launches discoverable only through environment/configuration files are not
inferred.

The recovery timer writes an allowlisted projection to
`/var/lib/sparkring/dashboard/nodes.json` (`sparkring-dashboard-nodes/v1`).
It reads all setup-recorded SSH targets through its existing client and
reuses remote status answers for the initial recovery check only. Post-restart
checks read fresh status. It makes no Docker or
model request for this cache. This includes a controller without active
deployments. Quiet mode and dashboard disable suppress these added reads;
they do not change managed recovery policy. The dashboard itself still
starts no process and uses only its existing `ReadOnlyClient` and `READ_PATHS`.

Facts match Sparks by node identity. Their original agent timestamp expires
after `max(90, 2 * interval_seconds + 30)` seconds; cache publication never
renews it. Failed status reads retain the setup-recorded position and previous
facts at their original age, with `status_read` and `checked_at` describing the
read attempt. A first failed read has no model facts. Cache refreshes use a
process lock, reject mismatched node identities and preserve newer observations.

Groups join by engine, head Spark and master port. The worker's cable peer
owning the master address takes precedence over a globally unique address
owner, which also resolves relay paths. Repeated per-cable addresses therefore
remain separate. Rank zero identifies itself when ownership agrees; when no
address ownership is available, a unique observed rank zero supports legacy
reports. Ambiguous workers remain separate, with no DNS lookup or container-name
heuristic. Cable proximity alone never establishes membership. Localhost rendezvous is
scoped to one Spark. Group size follows `nnodes` and observed ranks, not a
parallel-size product. Only a unique rank-0 head supplies the model name and
API port; an explicit API bind must match that head's recorded management,
LAN or fabric address. No discovered rendezvous expands the address allowlist.
HTTP redirects are refused, so an endpoint cannot cause an additional read
at another address, port or path.

Unmanaged models use the same polling, bounded history, rate calculation,
graph controls and integer token-rate formatting as managed models.
SGLang's generation/prompt counters, first-token histogram sum/count,
running/waiting gauges and token-usage gauge map to the shared numeric
contract. Other absent counters remain unknown. Metric names follow the
[SGLang metrics reference](https://docs.sglang.io/docs/references/production_metrics)
and its [collector](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/observability/metrics_collector.py).
Launch defaults follow the [vLLM serve reference](https://docs.vllm.ai/en/stable/cli/serve/).

The agent unit preserves its shared `/run/sparkring` directory across
stop/start, including package upgrades. Operator quiet files and holder
records retain their contents and expiry; reboot still clears runtime state.

Offline coverage is in `test_dashboard_unmanaged.py`, `test_dashboard_ring_observation.py` and
`test_dashboard_charts.py`: both engines, one through eleven Sparks,
non-contiguous membership, mixed parallel dimensions, incomplete groups,
stale/stopped records, ownership, bounds, secret exclusion, restricted reads,
quiet mode and in-place browser updates. These tests do not qualify live
Docker, SSH latency, GPU serving or recovery on Spark hardware.

The initial history response contains raw `counters` and optional
`plugin_counters` envelopes for replay at their `observed_at` time.
A global `history_cursor` identifies the last retained sequence. Status
responses carry only subsequent `counter_samples`, capped at 120 per
deployment after a delayed poll; a truncated delta starts with a gap.
Without a cursor, status carries just the latest sample. The browser
merges by sequence, then by source/run/timestamp, so the initial fetch and
live polling cannot duplicate observations. A service restart resets rate
baselines while preserving the open browser's graph history.

The browser keeps up to six hours, capped at 4,321 graph points per
deployment, including gap markers; its five-minute counter window is
capped at 100 samples. Reloading replaces its session history with the
available server history. Expanded graphs include both stored history and
samples acquired during that browser session. The terminal consumes the
latest sample in its existing status request and keeps its own bounded
windows; it does not fetch the three-hour archive. `--once` has no earlier
sample and shows dashes for rates.

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
backward timestamp, or a decreasing counter clears that source's rate
window and inserts a graph boundary, preserving earlier graph history. Missing counters remain unavailable. Plugin data older
than 180 seconds, more than five seconds in the future, or dated before
the `/metrics` process start falls back to `/metrics`. Cross-host wall
clocks therefore need synchronization for freshness checks; plugin rate
division uses monotonic time. A source transition needs two samples before
showing a rate. Quiet mode and measurement holds stop requests and sample
appends; the first distinct observation after release carries a boundary.
Periods without observations appear as gaps, never fabricated zeros. A
sample interval longer than 185 seconds also breaks the rate window and
graph line. Token rates display as whole tokens per second in both viewers.

## Browser rendering and display precision

The browser keeps cards in arrival order, keyed by member ID. A member's
card, tiles, controls and SVG plots survive incoming status documents.
Detached markup for metadata is reconciled into existing text and attribute
nodes; graph subtrees are owned by their plot updater. Open details, keyboard
focus and selections in unchanged text remain intact. Plot paths change only
when history or the selected window changes, or a resume requires a time-axis
catch-up. A hovered plot resolves the sample at the pointer again after redraw.

Status updates are applied in one animation frame. A separate one-second
timer schedules only age-label updates. Hidden documents defer DOM work and
catch up in one frame on visibility return; status request cadence is unchanged.
Fixed tile/value/readout slots and responsive SVG view boxes prevent changing
numbers from moving neighbouring controls. Missing values retain their slots.

Each member shares one window and pause state across its three plots. Auto
uses available history up to three hours. Explicit 5 min, 15 min, 1 h and 3 h
windows also fill the width until enough samples exist, then scroll. The
window is stored per member in browser local storage with an exception-safe
fallback. Pause copies the graph history and freezes its axis; tiles continue
updating. Resume uses live history. The dialog copies a frozen snapshot on
opening and adds a 6 h window. Its slider selection survives status updates
and window changes that still contain the selected sample.

Output and prefill axes scale to the visible samples with headroom and whole
ticks. KV stays at 0–100%. Gap markers and intervals over 185 seconds break
paths. Short windows label elapsed seconds/minutes; longer windows use clock
time. SVG paths use non-scaling strokes at phone and desktop widths.

Both viewers round displayed measurements to whole numbers, including
steps/s, memory in GB, request counts, percentages and durations of 10 seconds
or more. Raw API values and rate calculations retain their precision.
The fractional display exceptions are:

| Measurement | Precision | Reason |
|---|---|---|
| Draft acceptance length | 2 places | A fractional token per draft step distinguishes draft effectiveness |
| KV concurrency below 10× | 2 places | Fractions show whether another full-context request fits |
| Memory below 10 TB (TiB in the terminal) | 1 place | Whole terabytes hide substantial capacity differences |
| First-token latency below 10 s | 1 place | Whole seconds hide interactive latency differences |

Timestamps, software versions, addresses, model names and copied commands are
identifiers rather than rounded measurements. SVG coordinates retain fractional
precision for drawing; they are not displayed numeric text.

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
or websocket for plugin delivery. Dashboard quiet
mode suppresses its requests and rate updates; it does not disable the
independent vLLM logger or change the host-agent/recovery schedules.

## Collection cadence and local cost

**Source-read:** A serving deployment without a fresh plugin snapshot gets
one `/metrics` GET every 10 seconds while unviewed, or every 5 seconds while
viewed. `/v1/models` remains every 60 seconds: respectively 7 or 13 GETs per
minute per head in steady state (0.117 or 0.217 GET/s), plus one runtime-status
GET per model run and explicit refreshes. A fresh plugin uses the existing
60-second unviewed metrics schedule for auxiliary gauges/prefix statistics:
2 GETs/minute per head unviewed, 13 viewed. Plugin snapshots themselves add
no requests; file reads remain every 30 seconds and remote resolution is
limited by recovery delivery. A fresh recovery copy is used when the local plugin snapshot is stale.
If neither is valid and fresh, collection falls back to metrics.
Stopped deployments retain 60-second reads; failed active reads retry no
faster than every 15 seconds. Quiet mode and measurement holds send zero
upstream requests. Browser requests only read collector memory.

**Source-read:** The service retains idle CPU scheduling, nice 19,
efficiency-core placement, `MemoryHigh=96M`, `MemoryMax=128M` and no swap.

**Measured, local Windows Python 3.12.10:** Three synthetic deployments,
2,161 samples each over three hours at the viewed cadence, occupied
50,724,864 resident bytes (48.4 MiB), versus 45,416,448 bytes before filling.
Holding the encoded history response raised RSS to 53,288,960 bytes
(50.8 MiB). Compact JSON was 1,873,890 bytes; gzip level 1 was 180,251 bytes
(176.0 KiB). Payload size depends on counters, timestamp precision and source.

The measurement uses the repeated-batch method of the stat-logger cost
measurement: nine batches, here of 1,000 full three-deployment collection
cycles each. Median process CPU was 140.625 microseconds/cycle, maximum
batch mean 156.250; median wall time was 141.903 microseconds/cycle. CPU
accounting granularity can make a batch's CPU estimate exceed wall time.
These cycles include real local file checks and parsing synthetic HTTP
responses through the read-only client, with network waits excluded.
The process also includes test-fixture imports. RSS is the Windows working
set, not a Linux cgroup measurement. Browser rendering and history encoding
are outside collection-cycle timing. The run does not measure Spark serving
impact; that remains **unknown** until hardware comparison.

**Measured:** Filling all eight history slots with 2,161 samples containing
both plugin and auxiliary metrics fields used 74,448,896 resident bytes
(71.0 MiB), or 92,094,464 bytes (87.8 MiB) while holding the compact and
gzip response bodies. This local synthetic bound check does not certify
Linux memory use or concurrent-client peaks.

## Coordinator browser checklist

1. Open the Node A page after the collector has observations, and check Output, Prefill,
   steps, acceptance and their source labels. An idle deployment shows a
   prefill dash. An image without the plugin shows approximate `/metrics`.
2. Hover an output, prefill or KV graph: inspect its formatted value and
   ISO timestamp. Select **Expand graph** to inspect up to six hours already
   accumulated in that view, including the stored three-hour prefill.
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
