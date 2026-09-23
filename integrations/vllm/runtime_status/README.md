# SparkRing runtime status endpoint

This optional vLLM plugin adds `GET /v1/sparkring/status`. It reports the
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
| `configured` | Parsed server arguments and a strict allowlist of explicitly set process environment controls, captured during plugin initialization. Parsed arguments may include defaults. |
| `effective` | Stored resolved `VllmConfig` fields in the API process, captured at initialization. Includes parallelism, MTP depth, cache/checkpoint policy, batch capacity, capture capacity, loader and backend choices. |
| `observed` | Request execution evidence. Uninstrumented paths remain `not_observed`; configuration or preparation is not proof that a request used a kernel. |
| `workers.ranks` | Per-worker identity, rank-local configured environment, resolved config and passive resident state at the worker's `collected_at_unix_ns`. These can differ from the API process. |
| `workers.state` | `complete`, `partial`, `pending`, `error`, or `unavailable`. Complete means the expected count and unique rank identities were returned; it is not a health or correctness certification. |
| `provenance` | Bounded installed receipt summary read once at plugin startup, including receipt/composition/source hashes and the parent image config ID when recorded. This read does not perform a fresh source-file audit. |

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
execution, MTP acceptance and cache publication remain `not_observed` without
existing safe per-request counters. In particular, this collector does not call
transport `stats()` methods that read device-backed tensors. It does not parse
logs or Prometheus series, so historical tests and startup messages cannot
silently become current execution claims.

An installed image cannot reliably discover its own OCI image config ID.
`provenance.runtime_image_id` therefore remains `unknown` unless a separately
bound deployment identity is provided by a future composition. The parent image
config ID is labeled separately. Version labels, source pins and observed
execution are different identities. This endpoint is not release admission or
a replacement for startup receipt verification.

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

Only named CPU fields are read. No full config/environment dump, model path,
connector extra-config, API key, host address, request ID, token ID or prompt
is emitted. Receipt reads are capped at 4 MiB and never enumerate runtime files.

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
