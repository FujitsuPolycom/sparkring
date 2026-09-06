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
