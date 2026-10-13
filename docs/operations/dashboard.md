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
every 5 s instead of every minute; `--passive` leaves the service's reads
unchanged. Each model's card ends with its Sparks' rank, LAN address, memory
and free disk, which is red below 2 % of the disk.

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
