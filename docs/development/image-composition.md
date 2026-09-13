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

## Updating LIL application components

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
