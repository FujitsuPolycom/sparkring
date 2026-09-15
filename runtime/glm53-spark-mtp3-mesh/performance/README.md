# MTP3 cache and checkpoint performance composition

Status: **implemented** source composition; performance is **research-only**.
The build combines GLM-5.3 MTP3 compute, verified persistent caching,
explicit recurrent checkpoints, and stream-ordered hardware mesh transport.
The recipe preserves the parent model weights and does not change host fabric.

This optional research builder uses its own pinned parent and source overlays.
The retained GLM source-image profiles are defined separately in
[`runtime/sparkring/source_image`](../../sparkring/source_image/README.md).
This builder does not replace those profiles or their image receipts.
Use the [profile catalog](../../../profiles/README.md) for maintained
published-image deployments.

## Build inputs

Use a clean SparkCache checkout at `b5aca7cd3d3f7e7a14636bf6e5fa1f50a9650168`.
It contains the merged restore/publication improvements, periodic-capture
option, and backlog gauges. Periodic full capture defaults off; enabling it
trades more writes for shorter history reconstruction.

The compute parent is the published MTP3 image with config ID
`2e41b1e934a85ff7c21b780532db2f0a0e978df081e52f4ae2bf11f8992fb24f`.
`Dockerfile` identifies its immutable registry reference.

Supply the verified native placement and transport libraries named by SHA-256
in `prepare.py`. The transport [source package](transport/README.md) contains
build instructions, licensing, and the full source inventory. SparkCache's
`sparkcache/native/README.md` describes placement builds. Compiler/toolchain
differences can change binary hashes; a different binary requires separate
qualification and must not bypass the expected-artifact checks.

For exact artifact replay, the published image in `public-image.json` contains
both libraries. Use `docker create` and `docker cp` to extract
`/opt/sparkcache-src/sparkcache/native/build-cuda/libspark_cache_placement.so`
and `/opt/spark-sircl/libspark_transport_capi.so` without running a model.
Verify the digests in `prepare.py` before using them as build inputs.

```bash
python runtime/glm53-spark-mtp3-mesh/performance/prepare.py \
  --sparkcache /path/to/sparkcache \
  --placement-library /path/to/libspark_cache_placement.so \
  --transport-library /path/to/libspark_transport_capi.so \
  --output /path/to/absent-build-context
docker build -t sparkring-mtp3-cache-checkpoints /path/to/absent-build-context
docker run --rm --network none --entrypoint python3 \
  sparkring-mtp3-cache-checkpoints -S -B /opt/sparkring/bin/verify-performance.py
```

The preparer performs no remote action. The installer verifies source inputs,
applies strict runtime preimages, installs the complete checkpoint ownership
contract, and generates a file inventory checked before serving. A mismatched
source or ownership dependency fails the build rather than weakening checks.

## Behavior

- The optional [token-sharded mHC prefill package](mhc-prefill/README.md) runs
  repeated mHC on each rank's token quarter in supported eager 8K prefills.
  It is installed with `SPARK_MHC_PREFILL_SHARD=0`. Its original serving evidence
  does not qualify a rebuild that also contains request attribution.

- Bounded restore memory, authenticated history reconstruction, and protected
  publication lifetimes come from SparkCache's pinned source.
- Recurrent checkpoints are materialized and retained through the full
  scheduler/runner/model-state path. Cache reuse retains verification backoff.
- The native transport preserves CTA publication ordering, constructs all CPU
  proxy state before thread startup, and links payload/doorbell submissions.
- Graph-only collective geometry uses the content-addressed bundle described
  in `transport/README.md`; eager geometry is unchanged.
- GLM-5.3 chat requests cannot disable thinking with an unsupported flag.
  The API rejects that request before generation; warmup uses supported low
  reasoning effort. This does not implement reasoning-free generation.

The image verifier establishes file identity, not a throughput result.
Source equivalence with a serving image and bounded test evidence must be
recorded separately from any claim that this image completed a serving soak.

## Startup admission

Status: implemented; CPU request-boundary tests cover admission. A rebuilt
serving image must be validated before hardware qualification is claimed.

The serving wrapper installs vLLM middleware that returns HTTP 503 with
`Retry-After: 5` until its readiness marker exists. Public requests cannot
consume scheduler slots during request-shape and sampling warmup. Read-only
`/health`, `/v1/models`, and `/metrics` probes remain available; `/health` is
liveness, not completion of warmup. Docker readiness uses the marker.

The wrapper generates a random startup token before launching vLLM and passes
it to its warmup requests in an internal header. The middleware removes the
header before forwarding a request. A loopback address alone does not bypass
the gate. The token changes at every startup, and stopping the wrapper removes
the marker. Failed warmup keeps public inference blocked.

The build context includes the wrapper, warmup client, admission module, and
scheduler-liveness module; the image receipt verifies all four files. Published image receipts describe
immutable artifacts and do not claim this behavior until an image containing
these sources has been built and recorded.

## Scheduler-liveness packaging and timeout policy

Status: **implemented** with CPU packaging and interface tests. Source builds
install `scheduler_liveness.py` beside the serving wrapper instead of inheriting
that module from the parent image. The installed-file inventory records its
SHA-256. This change does not modify a published image or qualify a deployment.

The module preserves the raw output-counter gap in `output_stalled_seconds`
and tracks inactivity separately in `progress_stalled_seconds`. While requests
run, output-counter movement, a higher KV allocation maximum, or a higher
optional prompt-token counter maximum resets inactivity. The maxima belong to
the interval without output, so falling allocation, allocation oscillation,
and request-count changes do not repeatedly renew the timer. The
`sparkring-scheduler-liveness/v1` payload returns HTTP 503 with
`reason=engine_output_stall` when inactivity reaches the configured timeout.
Allocation is an activity proxy, not a GPU heartbeat. Fully preallocated work
or an operation without observable intermediate progress can still exceed it.

**The source-build rule defaults to 300 seconds.** Legitimate long-context
work can exceed that interval. Installing this module enables a rule absent
from the published cache/checkpoint image identified by config ID
`sha256:6921a6c163ea40b603e19a0332330efe3dbccbf4dce9f6cbbf6b756c9231835a`.
Do not assume that image's observed liveness behavior transfers to a rebuild.

Before enabling router removal or automated recovery from `/liveness`, select
`SPARKRING_LIVENESS_OUTPUT_SECONDS` above the longest measured legitimate
interval without observable progress, including prefill, restore, concurrency,
and a stated margin.
For managed deployments, set the optional site field `liveness_output_seconds`
to the chosen integer seconds, then regenerate launch inputs and recreate the
containers through the managed lifecycle. For example, `900` is an explicit
operator selection, not a universal recommended timeout. The accepted range
is 1 through 2,147,483,647 seconds so the value remains representable by the
launcher's integer checks; booleans, floating-point values, and strings are
rejected. Omitting the field retains the existing 300-second default.

The mesh renderer emits `SPARKRING_LIVENESS_OUTPUT_SECONDS` on every rank.
The shared launcher passes it into the container, and the wrapper passes it
to the liveness service. Inspect the effective container environment and the
module's output-stall fields before activating a liveness consumer. An inherited
module that lacks the rule will not gain it merely from an environment override.

This packaging change keeps the existing timeout policy. A workload-aware
progress signal remains separate work. CPU
tests distinguish allocation progress from a stalled engine; they do not
establish a safe universal timeout.

## Request reuse accounting

The image build includes [scheduler attribution hooks](attribution/README.md)
for the connector's opt-in request ledger. These hooks distinguish admitted
local reuse, finalized persistent restoration, and accepted prompt work across
preemption attempts. They are inactive unless the connector enables request
cache events. Source availability does not change a published image receipt;
a rebuilt image must be validated before serving evidence is claimed.

## Continuation checkpoint sources

Source builds preserve the [four continuation checkpoint files](continuation/README.md)
from the source-attested continuation serving image. Both recurrent checkpoint
flags are enabled in the build recipe. The installer validates checkpoint
ownership, applies those four replacements, then applies the matching request
attribution transform and generates the full image inventory. This source build
composition requires separate serving validation; it does not change the
immutable published image contract.
