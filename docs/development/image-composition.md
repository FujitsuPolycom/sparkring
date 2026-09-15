# Image composition and upstream updates

SparkRing's vLLM image combines a digest-pinned ARM64 foundation, pinned
application sources, and SparkRing integration patches. Model weights, request
caches and host networking remain deployment inputs. SGLang has a separate
application environment; a shared model name does not make framework binaries
interchangeable.

The R35 recipe implements source overlays on the existing foundation. It does
not yet consume independently packaged LIL foundation or application wheels.
The published R35 image and profile defaults remain unchanged by these build
metadata checks.

## Contracts

- [Source lock](../../runtime/images/sparkring-r35/source-lock.json): authoritative
  parent image, upstream commits, integrated trees and patch hashes.
- [Compatibility manifest](../../runtime/images/sparkring-r35/compatibility.json):
  platform, framework, foundation, recorded native libraries and ABI evidence.
  Null values mean unmeasured, never a wildcard match. The library inventory
  covers five integration libraries, not the entire native dependency closure.
  [Runtime ABI evidence](../../runtime/images/sparkring-r35/contracts/runtime-abi.json)
  records Python/Torch measurements from the identified R35 application image,
  not from its distinct parent image. GPU target coverage remains unmeasured.
- [Patch ledger](../../runtime/images/sparkring-r35/patch-ledger.json): patch
  purpose, every changed path, preserved integrations and regression-test paths.
  Component tests live in the patched upstream checkouts; packaging tests live
  in this repository. Listing a test is not evidence that it passed.

Run the offline contract check from the repository root:

```bash
python3 runtime/images/composition.py runtime/images/sparkring-r35
```

The context builder runs this check before creating its output directory.
It rejects mismatched foundation identities, missing component ledger entries,
changed patch bytes and incomplete changed-path inventories. Context receipts
also hash the copied compatibility manifest, ledger and source lock. Existing
source-tree packaging and installed-image verification remain required.

An optional saved R35 admission receipt can check recorded baseline components,
native library hashes and Torch package metadata:

```bash
python3 runtime/images/composition.py runtime/images/sparkring-r35 \
  --baseline-receipt /private/r35/image.json
```

This reads saved evidence; it does not contact Docker, authenticate a live image
or establish serving correctness. Obtain admission receipts through the
[R35 launch procedure](../operations/r35-local-launch.md#record-the-image).

## Native artifact admission

The [ABI inspector](../../runtime/images/inspect_abi.py) runs inside the image
without initializing a GPU. Capture its JSON output using the image's Python
interpreter and record the exact Docker image ID alongside it. It measures
Python, SOABI, Torch build/CUDA, C++11 ABI and glibc; it does not infer compiled
GPU targets from the toolkit version.

The [artifact checker](../../runtime/images/artifact_compatibility.py) reads a
`sparkring-native-artifact/v1` manifest containing `sha256`, `platform`, `abi`,
`gpu_targets`, and `provenance`. Provenance requires `source_repository`, a full
`source_commit`, `build_inputs_sha256`, and `compiler_identity`. ABI fields are
exactly the seven emitted by the inspector. GPU target evidence must be supplied
explicitly in both records; null coverage rejects admission.

```bash
python3 runtime/images/artifact_compatibility.py \
  --manifest /private/component.json --runtime-abi /private/runtime-abi.json \
  --artifact /private/component.so --gpu-target sm_121
```

The checker hashes the actual file and requires exact platform and ABI matches,
including glibc and Python patch version. This conservative policy can reject
binaries that would work; it does not guess compatibility. Additional ABI fields
require a schema change rather than being silently ignored. A pass allows only
isolated testing: supplier declarations are not authenticated, the full native
dependency closure is not audited, and neither wheel installation nor serving
promotion is performed. Artifact manifests must come from a reviewed build;
do not fill missing values merely to obtain a pass.

## Updating LIL application components

The [candidate source packager](../../runtime/images/candidate_sources.py) compares
the fully patched tree with the integrated foundation sources. Its explicit
native/build path inventory must cover the component's dependency inputs;
changes require rebuilding instead of source-only reuse. Python-defined JIT
kernels still require runtime tests even when compiled extension inputs match.

The [candidate image installer](../../runtime/images/candidate_image.py) consumes
a `sparkring-candidate-image/v1` descriptor, the parent's installed receipt and
verified source archives. The descriptor declares the parent image ID, composition
ID, distribution version, component records and parent-authored file inventories.
Optional `integration_contracts` bind new JSON contracts by path and SHA256;
existing contracts cannot be overwritten. The installer verifies inherited bytes
before overlay and preserves unrelated native libraries and release records.

Build with [Dockerfile.candidate](../../runtime/images/Dockerfile.candidate), using
a local parent tag whose inspected ID matches the descriptor. A bare Docker
configuration ID is not a portable `FROM` reference. The build emits a distinct
candidate receipt; `verify` checks the installed payload. `serve` verifies then
delegates to vLLM's CLI, including headless rank dispatch. Model and topology
admission remain the deployment profile's responsibility, separate from image
verification. Do not pass a candidate through a different release's validator.

Before lifecycle changes, the [candidate host gate](../../runtime/common/candidate.py)
compares actual image verification and raw installed receipt bytes with the trusted
descriptor, reviewed entrypoint and mandatory inherited native hashes. The caller
must obtain image ID, platform and verification from the same local Docker image.
Passing this gate proves payload agreement, not model correctness or stability.

The [bounded R37 TP4 evaluation](../../performance/records/glm53-flash/r37-tp4-source-upgrade.md)
exercises this source-overlay path through image admission, inference, cache
restoration after restart and a matched short performance comparison. It does
not qualify arbitrary LIL artifacts or other model profiles.

1. Record the exact upstream repository and commit in an isolated checkout.
   Compare source locks and source trees, not release-number labels or PR prose.
   Inspect each integration in the ledger for upstream overlap before removing,
   replacing or reapplying a patch. Keep contributor attribution.
2. Prepare a proposed source lock outside the released recipe. Compare it with
   the baseline without fetching, building or installing anything:

   ```bash
   python3 runtime/images/composition.py runtime/images/sparkring-r35 \
     --candidate-lock /private/proposed-source-lock.json
   ```

   The report identifies changed components and foundation changes. It requests
   the necessary reviews; it never authorizes binary reuse from metadata alone.
3. Compare native source, headers, generated inputs, build flags and dependency
   constraints. Rebuild affected extensions if these change. Before consuming
   wheels, measure and match CPU architecture, Python ABI, CUDA, Torch ABI and
   GPU targets. Unknown values block that reuse decision. AMD64 artifacts are
   not ARM64 GPU artifacts, and a package version is not an ABI contract.
4. Update source lock, patch ledger, connector contracts and compatibility
   evidence together for the candidate composition. Preserve SparkCache lease
   semantics, mHC ownership, continuation-prefill checkpoints and selected NCCL
   and SIRCL libraries. Do not transfer a baseline image's measurements to a
   rebuilt image. Keep profile configuration separate from image identity.
5. Run packaging and component regression tests, build in isolation, then run
   installed-image verification. Final assembly must not resolve unpinned
   dependencies or mount application source at serving time. Use a separate
   cache namespace for incompatible cache contracts.
6. Collect model/topology-specific functional, restart/cache, performance and
   stability evidence before promotion. Preserve exact published images and
   contributor profiles for rollback. Composition checks require no cluster
   hardware; maintainers own hardware qualification.

The next packaging increment is one compatible application update using the
retained ARM64 foundation. Separately packaged wheels can replace source
overlays only after their provenance and ABI checks are implemented. Foundation
extraction should preserve the working runtime and demonstrate a build-time or
storage benefit before replacing the existing base.
