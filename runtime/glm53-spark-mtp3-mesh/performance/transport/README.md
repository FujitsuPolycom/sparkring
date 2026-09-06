# MTP3 mesh transport sources

Status: **implemented**. This package prepares source trees and immutable overlay
directories offline. It does not install services, contact hosts, start models,
or qualify a rebuilt binary.

The fused-prefill transport synchronizes each CTA before its leader enters the
cross-CTA barrier. Its CPU proxy starts only after all queue, mutex, and atomic
members have finished construction. Payload and doorbell work requests use one
linked provider submission on each QP. The doorbell alone is signaled; its
completion retires both requests. Partial posting errors remain fatal and must
not be retried.

The CUDA graph all-reduce adapter uses two blocks for payloads through 16 KiB,
four through 64 KiB, and eight for larger payloads. Eager calls use eight blocks.
Weighted arrival counters preserve eight arrivals per operation. All ranks must
use the same source identity and geometry contract.

## Source and artifact identities

`source-manifest.json` lists every packaged native and overlay source file with
its SHA256. `native-source.tar.gz` contains the CMake project, compilation
sources, headers, test sources, and Apache license. It excludes Python bytecode
and unrelated research directories. The native source bytes derive from the
archive identified by `source_archive_provenance_sha256`; per-file identities
are preserved, while archive layout and compression are normalized.

The native library used for four-rank checks has SHA256
`056243fad27d224b82e437925ffa2aed42037e6bd29f239f56076a832f6ca5cb`.
The overlay manifest used with that library has SHA256
`c0fd5567442b08b908cc193f36d0864e262573c7e5d232509479a823cface742`.
`bundle-source/` preserves its text/source inventory; the native binary is not
versioned. The B12X RoCE sources derive from commit
`eac260a8257cc6b14e7d4ad674f51e9a09b8790f` with graph-geometry modifications
recorded by the bundle manifest. `B12X-LICENSE` supplies their Apache license.

The bundle configuration specifies the four-rank Spark mesh routing contract.
It contains no host credentials or deployment paths. It is not a topology
discovery tool: operators must qualify the physical mesh against that contract.

## Build and prepare

Run these commands from this directory on an ARM64 Linux build host with CMake
3.24 or later, a CUDA toolkit supporting architecture 121, a C++17 compiler,
and libibverbs development headers/libraries:

```sh
python3 package.py verify
python3 package.py extract --output /tmp/mtp3-transport-source
cmake -S /tmp/mtp3-transport-source/spark_transport -B /tmp/mtp3-transport-build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=121 \
  -DBUILD_TESTING=ON -DSPARK_TP4_ENABLE_FUSED_STREAM_SWITCH_SMOKE=ON
cmake --build /tmp/mtp3-transport-build -j4
ctest --test-dir /tmp/mtp3-transport-build --output-on-failure
sha256sum /tmp/mtp3-transport-build/libspark_transport_capi.so
python3 package.py bundle \
  --native-library /tmp/mtp3-transport-build/libspark_transport_capi.so \
  --native-sha256 SHA256_FROM_PRECEDING_COMMAND \
  --output /tmp/mtp3-transport-bundle
```

Output directories must not exist. All source hashes are checked before writing
output. The library digest must match the explicit argument. With the reference
library, the builder reproduces the reference overlay manifest byte for byte.
With a different library, it records that library's digest and emits a distinct
manifest with `research-only` status. Compilers and linkers can change binary
bytes; source availability does not establish bit-for-bit binary reproducibility.
Record the build environment and run native and serving checks before publishing
a rebuilt artifact as qualified. Generating a manifest alone proves no runtime
or ABI property of the supplied library.

## Evidence and limits

The reference library passed 28 native CTests and four four-rank sessions totaling
256 operations. Sessions covered changing exact and noninteger inputs, query
sizes 128/512/2048/8192, alternating streams, both operation slots, numerical
results, input and tail guards, and health completion. Two sessions used tracing
and therefore do not support timing claims. Provider-call tracing counted 144
calls per operation with linked posting versus 240 separate calls; work-request
and completion counts were unchanged. Those observations establish bounded
correctness and submission-count evidence, not a model-throughput speedup.

Offline package checks run with `python3 -m pytest test_transport_package.py -q`. They
validate inventories, corruption refusal, path restrictions, destination
preservation, and rebuilt-library manifest handling without CUDA or networking.
