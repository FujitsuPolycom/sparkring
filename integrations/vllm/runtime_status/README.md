# SparkRing runtime status endpoint

This optional vLLM plugin adds `GET /v1/sparkring/status` and browser/text views. It reports the
configuration of the running API process and passive CPU metadata from its
workers. It does not benchmark, change a setting, inspect prompts, or execute a
GPU operation. Its implementation status is **Development**: offline API and
collector checks do not qualify a serving deployment.

## Installation and activation

Install this directory as a Python package in the same environment as vLLM.
The image composition must record the package files with its other source-bound
assets. Add `sparkring_status` to the existing `VLLM_PLUGINS` allowlist on the
API server and every worker. Preserve any other required plugin names. Activation
requires process startup; copying files into a running server does not activate
the endpoint.

The package declares two official vLLM entry points with the same allowlist
name: `vllm.endpoint_plugins` attaches the router and initializes application
state; `vllm.general_plugins` registers the unique worker RPC method
`sparkring_status_snapshot_v1` on `WorkerBase`. It does not wrap model execution
or replace vLLM's application builder. vLLM must provide the
`EndpointPlugin.attach_router/init_state` interface and
`EngineClient.collective_rpc(method, timeout, args, kwargs)` API. The plugin
applies to the `generate` task.

The `/v1` route inherits vLLM's authentication middleware. If API authentication
is enabled, supply the same bearer token used for inference. Tokens are never
included in the response. Without server authentication, this route has the
same network exposure as the other `/v1` routes.

## Query from a terminal

No installed status client is required. Open `/v1/sparkring/status/view` in a
browser for a self-contained dashboard, or use the text endpoint from Windows
Command Prompt:

```bat
curl.exe --fail --silent --show-error http://localhost:8000/v1/sparkring/status.txt
```

Replace `localhost:8000` with the inference server's address. The existing
`/v1/sparkring/status` route still returns its original JSON schema. The browser
and text routes use the same cached worker reports; they introduce no separate
GPU operation or benchmark. All views return HTTP 503 until plugin initialization.

The browser refreshes every five seconds while visible, with a pause checkbox
and a manual refresh button. Failed refreshes retain the previous display and
mark it outdated. Sections preserve their expanded/collapsed state. JavaScript
is optional for the initial server-rendered report; refresh the page manually
when JavaScript is disabled. No fonts, scripts or styles are downloaded from
another server. Runtime values are escaped for HTML and stripped of terminal
control characters for text.

All three routes inherit the inference API's authentication. For a protected
API, curl requires the same bearer header shown below. Browser navigation needs
an authenticated proxy/session that supplies that header; these views do not
bypass authentication or accept API keys in URLs. Responses are not cached.

The header uses the recorded SparkRing/eugr startup identity when supplied by
the deployment, otherwise it says `SparkRing runtime status`. Its model line
shows the served model name, the name clients put in the `model` field of a
request: the first `--served-model-name` alias, or the `--model` value when no
alias is given. The checkpoint architecture (`model_type` from the Hugging Face
config, for example `qwen4_exp`) appears on its own `Model architecture` line.
When the served name is unknown, the model line says so instead of showing the
architecture. The Model and topology settings list both with per-rank values.

The package version is 0.3.6. It provides passive transport, resource, library
and acceptance views, and the settings tables described below. Its Transport
table names the transport that carries each worker's tensor-parallel
collectives (SIRCL with its version, RoCEnante or NCCL) and shows each worker's
SIRCL ring-session facts: the `SIRCL_MODE`,
`SIRCL_NCCL`, `SIRCL_FABRIC` and `SIRCL_RANK_POSITIONS` settings and, from the
worker's tensor-parallel SIRCL receipt in `SIRCL_RECEIPT_DIR`, the session
identity and state, the NCCL policy, whether PyNccl was built, the most relays
on a lane and the receipt's age. Deploy its wheel
through a new source-recorded image composition and restart the server when a
deployment window is available. Published image receipts and running
installations are not changed by building this package.

### Settings tables

Each settings row has five columns:

| Column | Content |
| --- | --- |
| Setting | Plain name of the setting |
| Configured | Parsed launch argument, including its default, or allowlisted environment variable. `Not set in environment` marks an absent variable; `No separate setting` marks a value without its own launch option. |
| Runtime value | The resolved `VllmConfig` value in the API process, or the workers' common value when the API process has none. Environment-only settings show `Not checked at runtime`. |
| Source | `vLLM config`, `Running worker`, `Launch setting only` or `Not reported`; the tooltip names the fact source and reason in words. |
| Workers | How the per-rank reports compare, for example `Same on both workers`; the browser expands it to per-rank values. |

Colors mark rows to check. Red marks workers that report different values and a
value that differs between the API process and the workers. Orange marks an
explicit launch argument that vLLM replaced, a report missing from some workers,
and missing or repeated rank numbers. Every other row is neutral. The
`Settings to check` card counts red and orange rows.

Two differences that vLLM makes by design are neutral:

- A dtype that names a more specific form of the configured or API-process
  dtype, such as `fp8_ds_mla` or `fp8_e4m3` for `fp8`, or any dtype for `auto`.
  A worker value of this kind is shown as `fp8 (fp8_ds_mla on workers)`. The
  rule applies only to settings whose key ends in `_dtype`; other prefixes, such
  as CUDA graph modes `FULL` and `FULL_AND_PIECEWISE`, remain differences.
- A KV cache block size that vLLM changed from the configured value. The
  runtime value carries a `*` marker and a footnote under its table.

### Chat and serving rows

The `Chat and tools` table shows how the API server handles chat requests:

| Setting | Configured | Runtime value | Workers |
| --- | --- | --- | --- |
| Reasoning parser | `--reasoning-parser`; vLLM's empty name shows as `None` | `structured_outputs_config.reasoning_parser` | Compared |
| Tool-call parser | `--tool-call-parser` | `Not checked at runtime` | `API server only` |
| Default chat template arguments | `--default-chat-template-kwargs` as JSON text, such as `{"enable_thinking": false}`; `Not set (the chat template decides)` when absent or `{}` | `Not checked at runtime` | `API server only` |

vLLM's API server passes the tool-call parser and the default chat template
arguments from its launch arguments to its chat renderer unchanged. `VllmConfig`
holds no copy, so workers cannot report them and their rows have no per-worker
comparison. The default arguments are shown as given; they do not include the
`cohere_format` default that vLLM adds for Cohere templates. JSON text longer
than 512 characters, with more than 16 keys, or with values JSON cannot express
reads `Value not shown`. Thinking itself is chosen per request with
`chat_template_kwargs` or `reasoning_effort`, which take precedence over these
defaults; a note under the table says so. Whether a chat template accepts such
arguments is not shown: vLLM determines it while rendering a request and stores
no setting for it.

The `Decode and speculation` table ends with `Shared-memory reader window`,
the `SPARKRING_SHM_BUSY_LOOP_S` container variable: how long vLLM's
shared-memory readers poll after a read before they sleep until notified. Images
derived with
[derive_spin_wait.py](../../../runtime/images/derive_spin_wait.py) read it. It
shows `2 ms (sparkring install --save-cpu)` for `0.002`, the value of
`sparkring install --save-cpu`; `1 s (vLLM default)` when unset; and the value in
seconds, such as `0.05 s`, otherwise. The JSON document records the value as a
number of seconds.

Configured and runtime values, worker agreement and kernel preparation describe
configuration; they do not show which kernel or transport served a request. An
unset NCCL variable is not treated as proof of a native default. The text view
prints the same tables and lists per-rank values under each red row.

Offline view tests are included in the component suite:

```bash
python -m pytest integrations/vllm/runtime_status -q
```

The optional browser test uses Playwright Chromium and a loopback-only FastAPI
server with synthetic snapshots. It covers automatic/manual refresh, pause,
failure recovery, preserved section state and a narrow viewport. It skips when
Playwright or Chromium is unavailable. Set `SPARKRING_BROWSER_ARTIFACTS` to a
local directory to retain its desktop/mobile screenshots.
`SPARKRING_CHROMIUM_EXECUTABLE` can select an already installed Chromium binary.

Replace the example address with the existing inference API address. These
commands use `SPARKRING_API_KEY` when the server requires authentication.

Windows Command Prompt:

```bat
curl.exe --fail --silent --show-error -H "Authorization: Bearer %SPARKRING_API_KEY%" http://localhost:8000/v1/sparkring/status
```

PowerShell:

```powershell
$headers = @{ Authorization = "Bearer $env:SPARKRING_API_KEY" }
$status = Invoke-RestMethod -Headers $headers -Uri http://localhost:8000/v1/sparkring/status
$status | ConvertTo-Json -Depth 20
$status.workers.ranks | ForEach-Object { $_.effective.hc_projection_tp_size }
```

Bash:

```bash
curl --fail --silent --show-error \
  -H "Authorization: Bearer ${SPARKRING_API_KEY}" \
  http://localhost:8000/v1/sparkring/status
```

Omit the authorization header if the server does not require one. The endpoint
accepts no mutation, refresh, arbitrary RPC, file path, or environment-variable
parameter. `POST` returns 405. A request before plugin initialization returns
503 with `state: initializing`; other responses return 200 with explicit
collection status. HTTP caches are disabled.

## Response contract

The schema identifier is `sparkring-runtime-status/v1`; the JSON Schema is
[schema-v1.json](schema-v1.json). Facts contain a `state`, `source`, and a `value`
when known. An absent setting is `unknown`, not implicitly false. A known JSON
null means that the stored setting is null. `not_observed` means this collector
has no execution evidence, even when the corresponding optimization is enabled.

| Field | Meaning |
| --- | --- |
| `configured` | Parsed server arguments and a strict allowlist of explicitly set process environment controls, captured during plugin initialization. Parsed arguments may include defaults. `arguments.default_chat_template_kwargs` is the JSON text of `--default-chat-template-kwargs`, or null when it is absent or `{}`; `environment.SPARKRING_SHM_BUSY_LOOP_S` is a number of seconds. |
| `effective` | Stored resolved `VllmConfig` fields in the API process, captured at initialization. Includes parallelism, MTP depth, cache/checkpoint policy, batch capacity, capture capacity, loader and backend choices. `served_model_name` is the primary name clients request (`model_config.served_model_name`); `model_type` is the checkpoint architecture (`model_config.hf_config.model_type`). Both have source `resolved_vllm_config`. `reasoning_parser` is `structured_outputs_config.reasoning_parser`, an empty string when none is selected. |
| `observed` | Request execution evidence. Uninstrumented paths remain `not_observed`; configuration or preparation is not proof that a request used a kernel. |
| `workers.ranks` | Per-worker identity, rank-local configured environment, resolved config and passive resident state at the worker's `collected_at_unix_ns`. These can differ from the API process. |
| `workers.state` | `complete`, `partial`, `pending`, `error`, or `unavailable`. Complete means the expected count and unique rank identities were returned; it is not a health or correctness certification. |
| `provenance` | Bounded installed receipt summary read once at plugin startup, including receipt/composition/source hashes and the parent image config ID when recorded. `b12x_comm_bundle` names the b12x communication bundle the image carries (its published name, such as `tp2-rocenante-adaptive-prepared`, and manifest digest); it is not the transport that carries the collectives, which the workers report as `tp_collective_transport`. This read does not perform a fresh source-file audit. |

### Transport, versions and node resources

Each worker may add `transport`, `versions` and `resources`. Failure of one
optional collector leaves that section unknown and preserves the core report.
The tables keep rank-local values rather than treating different local NIC names
or library paths as configuration mismatches.

- `transport.groups` inspects existing TP, PP, DP, EP, DCP and PCP communicators.
  RoCEnante availability, AR and AG shard ceilings, selected HCAs and GID index
  come from resident objects. PyNCCL supplies its runtime version and library
  handle name. NCCL algorithm/protocol selection and channel counts remain
  unknown because these objects do not expose them; selection can vary by
  collective or communicator. An available backend need not have served a request.
- `tp_collective_transport` names the transport that carries the worker's
  tensor-parallel collectives: `sircl` when the group has a SIRCL receipt
  (also where SIRCL's shim takes the b12x RoCE slot), else `rocenante` when
  the RoCEnante communicator is enabled, else `nccl`. With `sircl`,
  `tp_sircl_version` is the version the image's SIRCL layer receipt
  (`/opt/sparkring/receipts/sircl-layer.json`) records.
- `transport.nics` reads bounded sysfs fields for up to eight selected HCAs:
  PCI address/domain, current and maximum PCIe link, associated interface,
  negotiated link rate, MTU, MAC and RDMA port counters. Functions sharing a PCI
  device key may share one uplink, so their capacities must not be summed.
  Counters are host cumulative, not process attributed. RDMA data counters use
  four-byte words in JSON and are converted to bytes in the views. Negotiated
  link rate is not measured payload bandwidth; no latency test is run. IP
  addresses and saved host diagnostics require the existing host-agent adapter.
- `versions` separates the host NVIDIA kernel driver, Torch build CUDA, image
  toolkit/NVCC, package metadata and libraries mapped by that worker. A newer
  runtime does not imply that framework native extensions were rebuilt. Library
  filename versions are labeled separately from the NCCL runtime version.
- `resources.memory` uses Linux total and available memory; used is their
  difference. On Spark this host pool is shared by CPU and GPU. Cgroup current
  usage and limits are separate views and must not be added to host usage.
  Filesystem reads cover only `/`, `/cache` and `/models/target`, deduplicate
  repeated devices, and skip remote or unrecognized mount types. Available space
  uses the unprivileged-process count, which can differ from total free blocks.
  The worker hostname is process metadata, not a verified persistent node ID.

Host/deployment joins must use identities supplied by the existing installer
and host agent, with their own collection timestamps. A matching hostname or
rank alone does not bind historical measurements to the current deployment.

### Speculative acceptance

`observed.speculative_acceptance` reads only already registered in-memory
Prometheus counters for draft rounds, drafted tokens, accepted tokens and draft
positions. It does not scrape an endpoint, collect unrelated metrics, inspect
tokens or change verification. The first snapshot has no recent interval;
later snapshots distinguish idle windows, counter resets and unavailable data.
Lifetime and recent acceptance are `accepted / drafted`. Estimated tokens per
verification is `1 + accepted / rounds`, including one bonus token. This is not
measured throughput and cannot predict acceptance on another prompt or sampling
configuration. The metric scope is the reporting API process's engine counters.

The rank snapshot includes the resident HC workspace's projection TP size and
the model's token-row ownership mode when those Python fields exist. These are
effective construction choices. Lookup follows at most four stored `model` or
`language_model` wrapper edges; it does not traverse vision modules or tensors.
The source-bound HC fusion hook's first-use
markers, when available, are reported with phase
`unspecified_includes_warmup`; they do not establish current request execution.

`effective.kernel_preparation` on each rank inspects at most 32 already resident
B12X session plans. It includes the prepared selection source, selected
allowlisted tile/backend/split-K fields and declared shapes. The sample reports
its inspected count and truncation; it is not an exhaustive kernel inventory.
Device compute capability and SM count come only from already stored preparation
metadata. No CUDA device query is made. Closed or unprepared plan payloads are
not resolved. Prepared selections have `phase: preparation` and
`execution: not_observed`, including cached/default selections. Their presence
does not establish that a specific request used them.

Coalescing, current prefill execution, actual transport execution, kernel
execution and cache publication remain `not_observed` without existing safe
per-request counters. In particular, this collector does not call
transport `stats()` methods that read device-backed tensors. It does not parse
logs, so historical tests and startup messages cannot
silently become current execution claims.

An installed image cannot reliably discover its own OCI image config ID.
`provenance.runtime_image_id` therefore remains `unknown` unless a separately
bound deployment identity is provided by a future composition. The parent image
config ID is labeled separately. Version labels, source pins and observed
execution are different identities. This endpoint is not release admission or
a replacement for startup receipt verification.

## Installer identity and freshness

The installer may bind a stopped container to its deployment with
`SPARKRING_RUNTIME_BINDING=/run/sparkring/runtime-binding.json`. It must stage
this read-only mount as a regular local file, finalize it after Docker assigns
the container ID and before startup, and never modify it under a running model.
The JSON object has exactly these fields:

| Field | Value |
|---|---|
| `schema` | `sparkring-runtime-binding/v1` |
| `deployment_id` | Full 64-hex immutable deployment-lock ID |
| `node_id` | Canonical persistent node UUID |
| `container_id` | Full 64-hex Docker container ID |
| `image_id` | Docker image config ID, `sha256:` plus 64 hex digits |
| `rank` | Expected nonnegative rank, below 256 |

Plugin registration reads at most 16 KiB once, before serving. Worker snapshots
then read cached CPU facts only. The actual boot UUID comes independently from
`/proc/sys/kernel/random/boot_id`; process ID and sample time come from the
worker. These installer-provided assertions are not cryptographic attestation.
Missing, malformed, duplicate-key or rank-mismatched bindings remain unknown;
binding validation failure does not abort model startup. The installer must
provide a regular local file, not a remote mount or a FIFO.

The pure `binding.join(expected, host, model, runtime, now=...)` helper compares
one expected rank against authenticated host-agent, Docker-inspection and
runtime-status records. `expected` supplies `deployment_id`, `node_id`,
`image_id` and `rank`. The model observation uses
`sparkring-model-observation/v1`; host observations use
`sparkring-node-status/v1`. The helper rejects incomplete/duplicate runtime
rank identities and never substitutes a hostname/rank-only match.

The result is `matched`, `unbound`, `mismatch` or `stale`. Host and model ages
default to 90 seconds each; worker age defaults to 10 seconds, with a five-second
future-clock tolerance. A producer's stale flag also prevents a match, and a
worker sample predating Docker's container-start timestamp is rejected. A
matched identity does not prove process liveness after the sample, model
readiness, RDMA correctness or performance. The helper performs no I/O or
lifecycle action. Dashboard/controller consumers must retain each source's
timestamp and qualification scope rather than using page reception time.

## Cost, freshness and failure behavior

Each API process has at most one worker RPC in flight. GET waits at most 250 ms
for it, then returns `pending` plus any cached result. Further GET requests reuse
that RPC. A completed result is cached for five seconds; failures are also
rate-limited. There is no background polling thread or new daemon.

The HTTP wait deliberately does **not** cancel the worker RPC or set a short
executor timeout. vLLM workers share ordered response queues with serving, and
late replies must be consumed. A slow or stuck worker leaves one pending RPC
until it completes or the process exits. It does not create a new RPC on every
GET. The RPC uses the existing engine control path; scheduling it can still add
a small CPU control-plane cost, so this endpoint is not a zero-overhead hot-loop
telemetry interface.

`cache_age_seconds` measures time since the response was received, while each
rank's `collected_at_unix_ns` records when that worker read its metadata.
`stale`, `rpc_pending`, and `pending_age_seconds` expose delayed refreshes.
`missing_rpc_slots` identifies absent positions in the executor's response list;
it does not invent global rank IDs. The scope is the **addressed engine executor**,
not a claim that every data-parallel engine was queried. Expected coverage uses
the stored executor world size, with TP × PP × PCP as a metadata-only fallback.
It does not multiply ordinary MP/Ray coverage by global DP. The external-launcher
executor owns one local worker, so its expected response count is one even when
its distributed world contains additional ranks. Worker collection errors
are returned per rank when available. An engine RPC failure may lose the entire
response batch; the API reports that honestly without exposing exception text.

Only named CPU fields and fixed local metadata sources are read. No full
config/environment dump, model checkpoint path, connector extra-config, API key,
request ID, token ID or prompt is emitted. NIC names/MACs, worker hostname,
fixed mount paths and selected library paths are included for diagnostics.
Receipt reads are capped at 4 MiB and never enumerate runtime files.

Filesystem sampling pins directories component by component with Linux
`O_PATH`, `O_DIRECTORY` and `O_NOFOLLOW`. Each descriptor's mount ID is checked
against a fresh bounded `/proc/self/mountinfo` read before proceeding. Only
confirmed local mounts reach `fstatvfs`; symlinks, remote or automount mounts,
missing mount proof and unavailable descriptor support produce unknown results
without sampling that filesystem. A later pathname replacement cannot redirect
the pinned descriptor. Local filesystem metadata still has an I/O cost.
The directory-handle behavior follows Linux
[`open(2)`](https://man7.org/linux/man-pages/man2/open.2.html), including its
`O_PATH` rules for symlinks and untriggered automounts.

Single-rank groups without an available cross-rank communicator are shown in a
collapsed section. Unknown group sizes and available communicators remain in
the main table. NVIDIA driver parsing accepts proprietary and
architecture-qualified open-kernel module version lines; unsupported or
ambiguous text remains unknown.

## Offline verification

```bash
python -m pytest integrations/vllm/runtime_status/test_runtime_status.py -q
```

Set `SPARKRING_TEST_VLLM_ROOT` to the pinned vLLM source checkout to also exercise
its actual authentication middleware. That optional test skips when the source
is unavailable. JSON Schema validation requires `jsonschema`; the package wheel
smoke test uses installed `pip`, `setuptools` and `wheel`, with index access and
build isolation disabled.

The suite covers configured/effective distinctions, missing values, strict
field selection, bounded prepared-plan inspection, warmup labeling, receipt
limits, authentication with the pinned vLLM middleware when present, rank errors,
cache freshness, concurrent callers, client cancellation and read-only routing.
It uses fake worker metadata and an in-memory ASGI app; it never starts an engine
or accesses a GPU.

## Building the image artifacts

An installer image replaces its runtime-status package from a pure wheel and a
source archive pinned by file name and SHA-256
([Replacing the runtime-status package](../../../runtime/images/installer-images.md#replacing-the-runtime-status-package)).
Both are built from a commit of this directory into a directory outside the
repository. Run the commands from the repository root with Git 2.43 or later,
Python 3.12, setuptools 78.1.0 and pip 24.0:

```bash
COMMIT=$(git rev-parse HEAD)
EPOCH=$(git log -1 --format=%ct "$COMMIT")
VERSION=0.3.6
OUT=~/status-$VERSION
mkdir -p "$OUT" ~/status-$VERSION-stage
git -c core.autocrlf=false archive --format=tar.gz -9 --prefix=runtime_status/ --mtime="@$EPOCH" \
  "$COMMIT:integrations/vllm/runtime_status" > "$OUT/sparkring-runtime-status-$VERSION-source.tar.gz"
tar -xzf "$OUT/sparkring-runtime-status-$VERSION-source.tar.gz" -C ~/status-$VERSION-stage
cd ~/status-$VERSION-stage/runtime_status
SOURCE_DATE_EPOCH=$EPOCH python -m pip wheel --no-deps --no-build-isolation --no-index --wheel-dir "$OUT" .
sha256sum "$OUT"/*
```

`--mtime` stamps every archive member with the commit's time, and the wheel
takes the same time from `SOURCE_DATE_EPOCH`, so both files follow from the Git
tree of this directory and that time. Without `--mtime`, `git archive` stamps
the time it runs. `core.autocrlf=false` keeps Git for Windows from converting
line ends, because the archived tree carries no `.gitattributes`. To rebuild a
pinned pair, archive its tree with its archive time in place of
`$COMMIT:integrations/vllm/runtime_status` and `$EPOCH`:

| Version | Git tree of this directory | Archive time |
| --- | --- | --- |
| 0.3.2 | `74407675db01502e57ad6131103a8bbdb3db3bd8` | `1790540628` |
| 0.3.3 | `b82d56e0a8a5a04470fc679be9c7a665a7ab7fef` | `1790578668` |

[installer-images.md](../../../runtime/images/installer-images.md#replacing-the-runtime-status-package)
records the tree and time of each later pinned pair. Rebuilding 0.3.2 and 0.3.3
this way, the archive with Git 2.52 for Windows and the wheel with Python
3.12.3, setuptools 78.1.0 and pip 24.0 on Linux, reproduced both pinned files
of each version byte for byte.
