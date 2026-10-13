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

The page refreshes every 5 seconds. It only reads status: it runs no
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

Node A also serves one page for the whole ring (Experimental):

```text
http://NODE_A_LAN_ADDRESS:8090/
```

- `NODE_A_LAN_ADDRESS` is the address setup recorded for Node A's own LAN
  connection, where the model API of a model on Node A serves; without one,
  Node A's management address. Setup's plan prints the exact address, and
  while the page is on, the card after `Model ready:` prints it as
  `SparkRing dashboard:`.
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

It sends no inference request and opens no SSH session. The
[command reference](commands.md#dashboard) lists what it reads and how often.

The same status shows in a terminal on Node A, for example over SSH:

```bash
sparkring dashboard tui            # q quits; d expands the selected model's card
sparkring dashboard tui --once     # one frame as plain text
```

It reads the page's service on Node A at `http://127.0.0.1:8090`, or at
`NODE_A_LAN_ADDRESS` when nothing answers on loopback, and refuses any other
host. It sends nothing to any other Spark itself. Like an open page, it makes the service read each model's metrics
every 5 s instead of every 10 s (every minute with fresh plugin snapshots); `--passive` leaves the service's reads
unchanged. Each model's card ends with its Sparks' rank, LAN address, memory
and free disk, which is red below 2 % of the disk.

Each card's **Thinking** line shows what applies when a request does not
choose: on, off or always, followed by the effort level and available levels
when known, such as `always · high (levels: low, high or max)`. The terminal view shows the same text in its detail rows. Installed
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
or agent launch facts. Without them it says **model default**, without guessing
a level. This display adds no model API requests.

### Rates and graphs

Serving cards show four tiles (output, prefill, running/max requests and KV
in use as a percentage of tokens), then three compact graphs and node chips.
Steps and draft acceptance share a secondary line. Empty graphs show a baseline
and **no data yet**. Stopped deployments show their state, last model and a
copyable start command. **Details** expands settings, recovery, totals, logs
and container facts; node chips for stopped deployments are there too.

The ring page and terminal show prefill tokens/s during intervals with
prompt activity. Idle shows a dash. `/metrics` prefill is labelled
**approximate**; compatible images can supply computed-token counters through
the SparkRing stat logger. Each card names its source and observation time.
Plugin updates follow the existing host-agent and recovery cadence, so they
can arrive less often than `/metrics`. Missing or stale plugin data falls
back to `/metrics`.

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
