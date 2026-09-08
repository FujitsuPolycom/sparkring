# Native MTP3 cache and checkpoint performance composition

Status: **implemented** source composition; performance is **research-only**.
The build combines GLM-5.3 native MTP3 compute, verified persistent caching,
explicit recurrent checkpoints, and stream-ordered hardware mesh transport.
The recipe preserves the parent model weights and does not change host fabric.

## Build inputs

Use a clean SparkCache checkout at `48bbd2be4a7b972e56632a2d7b934bac5460f272`.
It contains the merged restore/publication improvements, periodic-capture
option, and backlog gauges. Periodic full capture defaults off; enabling it
trades more writes for shorter history reconstruction.

The compute parent is the published native-MTP3 image with config ID
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

The module reports `output_iterations` and `output_stalled_seconds` in the
`sparkring-scheduler-liveness/v1` payload and can return HTTP 503 with
`reason=engine_output_stall`. The output rule measures output-bearing batches;
it cannot distinguish an incomplete long prefill from a stalled engine.
Increasing KV occupancy alone does not prove completed model execution.

**The source-build rule defaults to 300 seconds.** Legitimate long-context
work can exceed that interval. Installing this module enables a rule absent
from the published cache/checkpoint image identified by config ID
`sha256:6921a6c163ea40b603e19a0332330efe3dbccbf4dce9f6cbbf6b756c9231835a`.
Do not assume that image's observed liveness behavior transfers to a rebuild.

Before enabling router removal or automated recovery from `/liveness`, select
`SPARKRING_LIVENESS_OUTPUT_SECONDS` above the longest measured legitimate
prefill, restore, or output gap, including concurrency and a stated margin.
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
tests demonstrate that frozen output with growing KV occupancy still triggers
the configured timeout; they do not establish a safe universal timeout.
