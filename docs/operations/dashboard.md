# Status dashboard

Every model that `sparkring install` starts serves a live status page from
the Spark that serves its API:

```text
http://API_HOST:PORT/v1/sparkring/status/view
```

- `API_HOST` is Node A for a pair, a four-Spark model or the model on Sparks
  0 and 1 of a ring. For the model on Sparks 2 and 3 (`--on 2,3`), it is
  Spark 2 ([where each half serves](install-reference.md#two-models-on-one-ring)).
  A deployment installed with `--api-bind` serves only at that address.
  `sudo sparkring status` prints each model's address, and the card after
  `Model ready:` prints it as `Model status:`.
- `PORT` is the profile's API port: 8000, 8015 or 8020
  ([profile table](../../README.md#profiles)), or the port that `--api-port`
  sets ([API endpoint](install-reference.md#api-endpoint)).

The page refreshes every 2 seconds. It only reads status: it runs no
benchmark and changes no setting.

![Dashboard summary and draft acceptance](assets/dashboard-overview.png)

## What it shows

| Section | What you see |
|---|---|
| Summary | Sparks reporting, data age and settings to check |
| Draft acceptance | MTP draft tokens accepted since the last refresh and since start, overall and per draft position |
| Memory and disk | Used and available memory on each Spark (CPU and GPU share it) and free disk space |
| Versions | NVIDIA driver, CUDA, NCCL, library and package versions on each Spark |
| Transport | The transport that carries the collectives (with runtime-status 0.3.6); whether the RoCE all-reduce adapter (B12X) and NCCL are available, the NICs in use, link rates and RDMA traffic; with runtime-status 0.3.5 or later on SIRCL ring sessions, each worker's SIRCL settings, session, NCCL policy, relays and receipt age. When SIRCL is enabled it replaces the RoCE adapter's all-reduce slot |
| Settings | Each serving setting as configured and as the model resolved it, and whether all Sparks agree |
| Workers | Draft tokens, KV transfer and kernel setup on each worker |
| Build information | Image, packages and source commits |

Links under the summary jump to each section. Add `#nodes` to the address to
open the page at memory and disk.

The `Chat and tools` settings table shows how the model API handles chat
requests:

| Row | What you see |
|---|---|
| Reasoning parser | The parser that separates the model's reasoning from its answer, or `None` |
| Tool-call parser | The parser that reads tool calls from the model's output |
| Default chat template arguments | Defaults passed to the chat template, such as `{"enable_thinking": false}`, or `Not set (the chat template decides)` |

The last two rows read `API server only`, because only the API server, on
`API_HOST`, holds them. Thinking is set per request: a request's `chat_template_kwargs` or
`reasoning_effort` take precedence over these defaults, as the note under the
table says.

The `Decode and speculation` table ends with `Shared-memory reader window`: how
long vLLM's waiting processes poll after each step before they sleep. It reads
`2 ms (sparkring install --save-cpu)` with
[`--save-cpu`](install-reference.md#serving-settings) and `1 s (vLLM default)`
without.

![Memory and disk space on each Spark](assets/dashboard-memory.png)

![Transport groups and NIC link rates](assets/dashboard-transport.png)

![Decode and speculation settings](assets/dashboard-settings.png)

## From a terminal

```bash
curl http://API_HOST:PORT/v1/sparkring/status.txt   # the report as text
curl http://API_HOST:PORT/v1/sparkring/status       # the report as JSON
vllm-top --url http://API_HOST:PORT                 # live rates
```

[vllm-top](https://github.com/mratsim/vllm-top) shows live prefill and decode
rates, queue, KV cache use and MTP acceptance.

`sudo sparkring status` checks the installation on every Spark and whether the
model is up; the dashboard shows how the running model is configured and
behaving.

## Access

Like the model API, the dashboard has no key and listens on every interface of
`API_HOST`. It shows hostnames, NIC MAC addresses and software versions, so keep
that Spark on a trusted network ([Security](install.md#security)).

## Every model on one page

The ring figure follows the cable topology, with one numbered rectangle per
Spark and an allocation strip below it. Managed groups use teal, green, violet
and blue in lowest-position order; each group's cables, card edge and graph
lines share that colour. Unmanaged groups and GPU jobs use muted outlines;
stopped deployments are dim. State pills have separate health colours.
Amber diamonds mark cable segments on topology-derived shortest routes with
an intermediate Spark, rather than measured traffic. The page uses system
fonts and follows the browser's dark or light theme without fetching assets.

Node A also serves one page for the whole ring (Experimental):

```text
http://NODE_A_LAN_ADDRESS:8090/
```

- `NODE_A_LAN_ADDRESS` is the address setup recorded for Node A's own LAN
  connection, where the model API of a model on Node A serves; without one,
  Node A's management address. Setup's plan prints the exact address, and
  while the page is on, the card after `Model ready:` prints it as
  `Ring dashboard:`.
- It listens on port 8090 on that address and on `127.0.0.1`, never on the
  fabric or the administration network. On Node A itself,
  `curl http://127.0.0.1:8090/api/status` reads it.
- It shows the ring and every model of the fabric, and links each model to the
  page above. Each model's transport is the one that carries its collectives:
  `SIRCL`, `NCCL` or `RoCEnante (b12x)` (the `prepared` transport).
- Installing the SparkRing package or running `sudo sparkring setup` turns it
  on. `sudo sparkring dashboard disable` turns it off for good, across reboots,
  setups and upgrades; `sudo sparkring dashboard enable` turns it back on.
- It shows a model that `sparkring install`, `up` or `down` or automatic
  recovery added, changed or stopped within a second, without a restart: those
  commands rewrite its site file and run `systemctl reload sparkring-dashboard`,
  and the service also notices a changed site file by itself.

It sends no inference request. The web service opens no SSH session; Node A's
host agent forwards plugin snapshots through the existing authenticated SSH
path while a viewer is present. The
[command reference](commands.md#dashboard) lists what it reads and how often.

### Spark details

Hover a numbered Spark in the ring, focus it with Tab, or tap it to open its
details card. Tap the same Spark again or tap outside to close; Escape also
closes it. Enter or Space pins or closes the focused card. The card fits the
viewport on phone widths and scrolls when the screen is short.

The card shows the hostname, management address, ring position, serving group
and model, reported rank and TP/DCP, and that Spark's startup progress or
serving state. Container state and health appear when observed; a failed
Spark includes its last error line. Cable peers come from the configured
fabric. NIC carrier and link state appear only for cached observations of
that interface. GPU memory used/total and temperature appear only when the
node cache supplies them; agents that do not publish these fields leave
them out. Missing values are omitted.

**Updated** dates the startup/API observation, or the status snapshot when no
such observation is available. **Node cache** separately dates the hardware,
link and cached container facts, which can be older. Opening the card sends
no request. It updates from the page's normal status refresh and does not
start hardware probes, log reads or additional polling.

### Startup progress

The ring and each group card show startup phases before the API is ready:
checkpoint check, download or copy, image load, derivation, starting, loading
weights, compiling, B12X tuning and CUDA graph capture. Counts appear when
the logs report them, for example `Loading weights 12/43`. Transfer progress
appears as a whole percentage. The card shows elapsed time since the operation
or container started, and Spark details when ranks differ. An exited container
or failed health check shows **Failed** with the last available error line.
An answering model API becomes **Serving** unless a fresh container observation
reports failure; a frontend can still answer while a worker has failed.

Node A's agent reads the installer's existing operation log and forwards
container observations over the existing authenticated telemetry channels,
at most one channel per managed Spark and eight channels per ring. The web
service opens no Docker socket and no extra port. Reads run only while the
six-second viewer lease is renewed; quiet mode and dashboard disable stop
them. A pending installation appears before its final site export.

Each local sample lists at most 32 managed containers and has a 1.5-second
admission budget. Each Docker GET has a 0.4-second deadline and a 64 KiB
response cap; an in-flight request can finish after the admission budget.
Container logs use `tail=200`, restricted to the container's current start.
Healthy containers need no log tail. Installer reads are capped at eight
slice records and 64 KiB per operation log; command detail logs are excluded.
Only known messages change phase. Unknown output leaves the previous phase
unchanged within the same container run. A restart resets it; missing or
expired observations fall back to API status without inventing progress.

Startup observations require the host-agent package on the participating
Sparks. They cover managed containers with deployment and rank labels.
Unavailable ranks remain unobserved; an observed rank's phase does not assert
that every Spark reached it. Log recognition is covered by archived vLLM,
B12X and installer excerpts and offline browser tests, not live hardware
qualification.

The same status shows in a terminal on Node A, for example over SSH:

```bash
sparkring dashboard tui            # q quits; d expands the selected model's card
sparkring dashboard tui --once     # one frame as plain text
```

It reads the page's service on Node A at `http://127.0.0.1:8090`, or at
`NODE_A_LAN_ADDRESS` when nothing answers on loopback, and refuses any other
host. It sends nothing to any other Spark itself. Like an open page, it makes the service read each model's metrics
every 2 s; reads stop six seconds after the last viewer poll. `--passive`
does not renew that viewer lease. Each model's card ends with its Sparks' rank, LAN address, memory
and free disk, which is red below 2 % of the disk.

Each card's **Thinking** line shows what applies when a request does not
choose: the mode, effective effort level and available levels, such as
`Thinking: always · high (levels: low, high or max)`. The terminal view shows
the same text in its detail rows. Expand **Details**, then **Thinking details**,
for one line of default provenance; the JSON retains the effective and model defaults. Installed
checkpoints without a recorded thinking behaviour have no line. The raw
**Chat defaults** row remains in the web details.

### Models installed outside SparkRing

The ring page and terminal also show running vLLM and SGLang containers
reported by the host agents. Their cards say **Not installed by SparkRing:
no automatic recovery, no install record**. They show the actual Sparks,
including non-contiguous positions, engine, image short ID, served name and
TP/DCP/PP/DP sizes. Node counts are independent of those parallel sizes.

Containers using the same head Spark and rendezvous port form one group.
The fabric's cable addresses distinguish pairs reusing the same IP address. Missing
ranks show **incomplete: N of M Sparks reporting**. A container without an
answering API keeps its Sparks busy; expand its card for launch facts.
A stopped install record overlapping a live unmanaged group appears in
that group's details.

Node A's existing recovery timer caches each Spark's agent status about
once a minute, including Sparks with no install record. The dashboard reads
that cache every 30 seconds. Quiet mode suppresses the cache's additional
status reads and the dashboard's API reads. Recovery's existing deployment
checks remain separate. Disabling the dashboard suppresses its cache reads.
Package upgrades preserve operator quiet mode and active quiet holds.
The dashboard starts the recovery timer; only managed records with recovery
enabled can trigger recovery actions.

Only rank 0's reported API port is read, on an address recorded for that
Spark. There is no port scan. SGLang counters need `--enable-metrics`;
without counters, an answering model API still appears with unavailable
graphs. Environment values and configuration-file contents are never read
for discovery. Launch facts hidden in those inputs remain unknown.
See the [observation contract](../development/dashboard-counters.md#unmanaged-model-observations)
for bounds and freshness rules.

For these models, **Thinking** uses explicit chat defaults from runtime status
or agent launch facts. Without them it says **unknown**, without guessing
a level. This display adds no model API requests.

### Rates and graphs

Serving cards show four tiles (output, prefill, running/max requests and KV
in use as a percentage of tokens), then three compact graphs and node chips.
Steps/s, draft acceptance (one decimal), and accepted draft tokens per engine
step share a secondary line. Steps require iteration details; missing details
show a dash. Empty graphs show a baseline
and **no data yet**. Stopped deployments show their state, last model and a
copyable start command. **Details** expands settings, recovery, totals, logs
and container facts; node chips for stopped deployments are there too.

The ring page and terminal show prefill tokens/s during intervals with
prompt activity. Idle shows a dash. `/metrics` prefill is labelled
**approximate**; compatible images can supply computed-token counters through
the SparkRing stat logger. Each card names its source and observation time.
While a viewer is present, each group is sampled every 2 seconds. Output and
steps/s use about six seconds of samples. Idle output and steps/s are zero;
stopped groups retain no live rate. A restart or counter reset starts a fresh
window.

The host agent forwards head snapshots every 2 seconds, independently of the
minute node cache and automatic recovery. Node A reads its own head locally;
remote heads use one persistent channel on the existing SSH path and credentials.
No listening port is added. The six-second viewer lease expires after the last
page or terminal poll; forwarding stops at the next two-second agent tick.
Quiet mode also stops forwarding. Plugin snapshots older than 12 seconds fall
back to `/metrics` while viewed, so the minute cache cannot hold rates back.

Iteration details are opt-in: add `--enable-logging-iteration-details` to a
vLLM serving profile's `vllm_args` (or the equivalent direct vLLM serve command).
It enables vLLM's CPU iteration bookkeeping and per-iteration log output, which
can add log I/O. Spark serving overhead has not been measured. The dashboard
does not enable it automatically. See the [cost and counter contract](../development/dashboard-counters.md#plugin-packaging-and-delivery).

Hover a graph for its value and timestamp. **Window** selects Auto, 5 min,
15 min, 1 h or 3 h for all cards' graphs from the page header. Auto fills
the width with available history, up to three hours; shorter histories fill
any selected window.
**Pause** freezes all graphs while the tiles keep updating. **Resume**
jumps to now. The browser remembers the page window for this viewer when
storage is available.

Click or tap a graph, or focus it and press Enter or Space, for a frozen
snapshot of stored and session history, up to six hours. Its window selector also offers 6 h. Use the slider with a
pointer, touch, arrow keys, Home or End; Escape closes the view and restores
focus. Graphs follow the OS light or dark theme. Interaction sends no request.

Rates and graph history are calculated in the viewer. The page starts with
up to three hours of stored observations, then adds session samples. Reloading
replaces session history with the available stored history. The terminal does
the same calculation only while open; `--once` has no rate history.
See [counter methods and validation](../development/dashboard-counters.md)
for precision, plugin installation and the hardware comparison plan.

### Why loopback as well

Programs on Node A, such as the terminal view, `curl` or a monitoring agent,
reach the service at `127.0.0.1` without knowing which LAN address setup
recorded, and that address keeps working when the LAN address changes. It
widens nothing: every program on Node A can already reach the LAN address, and
no other host can reach Node A's loopback. Limiting the terminal view to the
listen address instead would have kept `curl http://127.0.0.1:8090/` failing
and tied every local program to the fabric document.

`serve --bind ADDRESS` replaces the LAN address and keeps loopback beside it;
a loopback or wildcard address (`0.0.0.0`, `::`) is used alone. When another
program holds port 8090 on loopback, the service logs one line and serves on
the LAN address only.

## Ring display name

On Node A, run `sudo sparkring name "Draupnir"` to label the ring dashboard
and terminal view. The header, ring panel title (for example,
`Draupnir · ring of 8`) and browser tab use this name. `sparkring name`
prints it; `sudo sparkring name --clear` restores the cluster name.
[`--display-name`](commands.md#name) also sets it during setup or installation.

The controller keeps the override in
`/var/lib/sparkring/controller/display-name.json` and publishes it in the
existing dashboard site file. The collector re-reads that file at its normal
cadence and delivers the name in the existing status response. No extra
request, service reload or restart is needed. Exported site files carry the
label too; cluster and deployment identities stay unchanged.

## Source

SparkRing's runtime-status vLLM plugin serves the page. Its source, response
schema, collection cost and offline tests are in
[integrations/vllm/runtime_status](../../integrations/vllm/runtime_status/README.md).
## Managing slices

Management is off by default. On Node A, an administrator can opt in to
the existing privileged helper and give the unprivileged dashboard a token:

```bash
sudo systemctl enable --now sparkring-helper.socket
sudo sh -c 'umask 077; python3 -c "import secrets; print(secrets.token_urlsafe(32))" > /etc/sparkring/dashboard-manage.token'
sudo systemctl edit sparkring-dashboard.service
```

Use this service override:

```ini
[Service]
SupplementaryGroups=sparkring
LoadCredential=manage-token:/etc/sparkring/dashboard-manage.token
ExecStart=
ExecStart=/usr/bin/sparkring dashboard serve --manage --manage-token-file %d/manage-token
```

Restart the dashboard service. Give the token only to operators allowed to
change serving models. Use a trusted administration connection or an HTTPS
reverse proxy; a bearer token grants management access to anyone holding it.
The token stays in the page's password field and is never saved in browser
storage or a URL. Removing the override and restarting returns to read-only
mode. Rotate a token by replacing its file and restarting the service.

Each slice's buttons select install/swap, down, restart, recovery on/off or
logs. Choose a profile and optional checkpoint, context length, KV cache or
download limit, then select **Review plan and activity**. Read the plan and
warning before selecting **Confirm reviewed action**. Confirmations expire
after five minutes and can be used once. A changed deployment, fabric or package
requires another preview. A conflict never runs, even after confirmation.

Actions use the helper's allow-list and journal audit. They add no periodic
requests. Long installs keep the action request open; use the displayed
`sparkring logs --on 4-5 --follow` command in another terminal to follow
progress. **Follow logs** reads one snapshot and provides that command.
The dashboard itself does not continuously stream or poll logs.

See the [slice design](../architecture/independent-slices.md) for lock scopes,
shared-resource warnings, compatibility and qualification limits.
