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
No derived rate is stored. The live envelope updates every two seconds;
archive rows are retained at least five seconds apart (normally six seconds
at the two-second sample cadence). Source/run changes and counter resets
bypass thinning and insert boundaries. Every live sample is checked for a
reset before thinning, so a reset between archive rows cannot create a spike.
The browser merges live envelopes into its own two-second session history.

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

The browser keeps up to six hours, capped at 10,801 graph points per
deployment, including gap markers; its five-minute counter window is
capped at 181 samples. Reloading replaces its session history with the
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
| Prefill | Increase in prompt tokens over about six seconds, shown only when the latest distinct interval has prompt activity |
| Output / decode | Increase in generated tokens over a six-second window; if observations are farther apart, the last two observations |
| Five-minute output | Increase in generated tokens over the available five-minute window |
| Steps | Increase in observed, non-dummy iteration-detail records divided by elapsed time; unavailable without vLLM iteration details |
| Accepted drafts | Increase in accepted draft tokens divided by increase in verified draft tokens over six seconds |
| Accepted tokens per step | Increase in `step_accepted` divided by observed non-dummy steps over six seconds; excludes bonus tokens and is unavailable without matching iteration details |
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

Idle output and steps/s are zero when both running and waiting gauges are zero.
Stopped groups clear their live rate window, including when stale counters are still present.
Duplicate timestamps add no history. A source or run identity change, a
backward timestamp, or a decreasing counter clears that source's rate
window and inserts a graph boundary, preserving earlier graph history. Missing counters remain unavailable. Live plugin data older than 12 seconds falls back to the fresh metrics read;
archived observations are validated at their observation time with the compatibility
180-second limit. Data more than five seconds in the future, or dated before
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
catch up in one frame on visibility return; their two-second status polling
continues and renews the viewer lease until the page is closed.
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
| Draft acceptance | 1 place | Small acceptance changes remain visible |
| Accepted draft tokens per engine step | 1 place | A fractional token distinguishes draft effectiveness |
| KV concurrency below 10× | 2 places | Fractions show whether another full-context request fits |
| Memory below 10 TB (TiB in the terminal) | 1 place | Whole terabytes hide substantial capacity differences |
| First-token latency below 10 s | 1 place | Whole seconds hide interactive latency differences |

Timestamps, software versions, addresses, model names and copied commands are
identifiers rather than rounded measurements. SVG coordinates retain fractional
precision for drawing; they are not displayed numeric text.

## Plugin packaging and delivery

[SparkRingStatLogger](../../integrations/vllm/runtime_status/sparkring_runtime_status/stat_logger.py)
ships in `sparkring-runtime-status` 0.3.7 and later through the
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

To opt in, append `--enable-logging-iteration-details` to the serving profile's
`vllm_args`, or to a direct vLLM serve command. Keep it absent to avoid its CPU
bookkeeping and per-iteration log I/O. The dashboard does not change this
serving setting. [vLLM's observability reference](https://docs.vllm.ai/en/v0.15.0/api/vllm/config/observability/)
describes the fields and opt-in flag. The logger's CPU regression budget is
25 microseconds per `record()` call with blocked storage; this is a test-machine
budget, not measured Spark serving overhead. No DGX performance claim is made.

Step duration uses `SchedulerIterationDetails.elapsed_ms`. These details
exist only when vLLM's `enable_logging_iteration_details` is already enabled;
the plugin does not enable it. Absent details make steps/s and duration
unavailable. First-token durations are the existing frontend latency list.
The plugin provides no isolated prefill kernel duration.

The host agent retains the ordinary minute-cache path for inventory. For live
statistics, the unprivileged web service touches `/run/sparkring-dashboard/viewer`
at most once per two seconds after an authenticated, non-passive status request.
The root agent trusts only this file's modification time, never its contents.
It derives heads and deployment IDs from the root-owned controller cluster and
dashboard site records. A missing, future-dated or older-than-six-second lease,
quiet hold, or dashboard-disable marker closes forwarding on its next two-second
tick. A controller installed after the agent starts is detected at its normal
observation cycle. Worker agents start no forwarding monitor.

`telemetry_live.py` uses recovery's `discovery.ssh_argv` transport and its existing
SSH configuration and host-key checks. At most eight head channels are open.
Each remote `sparkring node telemetry` command reads the existing local stat
files when renewed on that persistent channel, at most once per two seconds.
There is no SSH process launch or authentication handshake per sample, Docker
inspection, worker RPC, inference request, or new port. EOF or six seconds with
no renewal ends the remote command. Node A's head is read in process. Failed
channels back off for 15 seconds while healthy heads and metrics keep working.
The existing reader caps enumeration at 128 files and each regular file at
4 KiB. Stream frames are capped at 512 KiB and projected to finite numeric fields
and bounded run IDs; only snapshots for the selected head's deployments survive.

The agent atomically replaces `/run/sparkring/dashboard-live.json` on tmpfs;
the collector reads it during its normal viewed cycle. It prefers the newest
eligible snapshot, and rejects a live-forwarding envelope older than six seconds.
While viewed, snapshots older than 12 seconds fall back to `/metrics`. This
permits vLLM's idle logger tick without allowing the minute cache to displace
fresh metrics. Quiet mode suspends forwarding and dashboard requests; the
independent stat logger and ordinary recovery/agent schedules are unchanged.

The collector requests `/metrics` every two seconds only while viewed, with
one request per group API, independent of TP or DCP rank count. Up to eight
workers prevent a slow head from serializing the others. Each HTTP request
retains the two-second socket timeout and 8 MiB response bound. A body-read
deadline checked between single socket reads also bounds slow-drip responses
(two seconds plus at most one pending socket timeout). Initial model
metadata reads and failed-endpoint backoff retain their existing schedules.
After the viewer lease lapses, API reads stop and the next viewer starts a
history boundary. Two new samples are needed to establish a live rate.

## Collection cadence and local cost

**Source-read:** A viewed serving group sends 30 metrics GETs and one model-list
GET per minute in steady state, plus the runtime-status read once per run and
explicit refreshes. The web service sends no API reads without viewers or during
quiet holds. Failed endpoints retry no faster than every 15 seconds; stopped
records are not probed. One persistent agent channel per remote head forwards
bounded snapshots every two seconds while leased. A newer idle metrics observation
wins over an older busy plugin snapshot.

The service retains idle CPU scheduling, nice 19, efficiency-core placement,
`MemoryHigh=96M`, `MemoryMax=128M` and no swap. With eight groups and 2,161 raw
metrics samples each, a local Windows Python 3.12 fixture process used 77,766,656
resident bytes (74.2 MiB), versus 55,308,288 bytes before filling. Holding the
JSON and gzip bodies raised RSS to 86,867,968 bytes (82.8 MiB). JSON occupied
7,520,440 bytes; gzip level 1 occupied 808,346 bytes. This synthetic fixture
measurement includes browser-test imports and does not certify Linux cgroup
usage, concurrent history requests or Spark serving overhead. The source-selected
history cap is 17,288 samples across eight groups; no persistent history is written.

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
