# R35 vLLM image

This recipe overlays pinned R35 vLLM and B12X sources on the immutable
SparkRing ARM64 CUDA 13.3 image. It retains the parent's SparkCache native
modules, SIRCL, routed NCCL, continuation-prefill coalescing and mHC sharding.
The serving entrypoint accepts a model and rendered profile; it has no default
model. This is a vLLM image, not an SGLang runtime.

Status: **experimental**. Component tests, TP4 functional/decode checks and
[bounded TP2 cache/decode checks](../../../performance/records/glm53-flash/r35-tp2-sparkcache.md)
have passed. TP4's prolonged mixed workload encountered a native collective
stall; long-duration stability qualification is deferred. Existing published
image identities and profile defaults are unchanged. The TP2 test required an R35-specific launcher
argument correction and does not establish long-duration TP2 stability.

## Source and build identity

The measured ARM64 image is published as
`ghcr.io/fujitsupolycom/sparkring:r35-arm64-7b698d4299aa`.
The [publication record](publication.json) contains its immutable registry digest,
local image identity and validation scope. Follow the
[launch guide](../../../docs/operations/r35-local-launch.md) to pull it and create
a verified deployment receipt. Registry availability does not establish stability.

[source-lock.json](source-lock.json) pins the immutable parent, upstream commits,
patched Git trees and patches. The patches preserve SparkRing integrations
across the R35 source changes. [baseline-files](baseline-files) lists the R33
authored package files so finalization can remove files absent from R35 without
removing inherited compiled extensions.

The vLLM native build inputs are unchanged between the pinned R33 and R35
upstream compositions. The recipe reuses those compiled extensions and verifies
the installed payload. B12X GPU kernels use the R35 source and runtime compiler;
they require GPU tests in addition to image verification.

The parent supplies SparkCache source commit
`f220230a5a85b94af8a296187241b6aacc3ed724`. The
[connector contract](contracts/vllm-connector-jobs.json) binds its lease interface
to the exact installed vLLM bytes. Image verification checks this contract before
serving. Source archives use LF line endings even on Windows.

## Prepare sources

Use dedicated source checkouts. Fetch the commits recorded in the source lock
from `https://github.com/voipmonitor/vllm.git` and
`https://github.com/voipmonitor/b12x.git`. Check out each `base_commit` detached,
then apply its bundled patch with `git apply --index`. Do not commit the staged
patch before packaging: the packager verifies the upstream HEAD and staged tree.
An existing checkout with other changes is rejected.
The vLLM checkout must also contain the R33 upstream comparison commit
`ae89131442359dc332d9c46009be3c1f8cdee0b4`; fetch it before packaging.

From the repository root, with paths adjusted to those dedicated checkouts:

```sh
python runtime/images/sparkring-r35/prepare_context.py \
  --vllm-source /path/to/vllm-r35 \
  --b12x-source /path/to/b12x-r35 \
  --output /path/to/absent-build-context
```

The shared entry point also exposes this context generator as the
`sparkring-r35` builder. `python scripts/build_image.py sparkring-r35 -- --help`
prints its command without executing it. Add `--execute` before the builder name
to run context assembly; Docker construction remains a separate step.

The context generator verifies the patched trees and patch hashes, packages the
source, copies the shared profile contract, and records all context file hashes.
Archive timestamps can change archive hashes between builds; the pinned Git tree
and installed file hashes identify the payload.

## Build locally on ARM64

Use an ARM64 Docker host with the immutable parent available. These commands
build local tags only. Replace `CONTEXT` with the generated context directory.

```sh
docker build --network=none -f CONTEXT/Dockerfile.source-stage \
  --build-arg PARENT=ghcr.io/fujitsupolycom/sparkring@sha256:1328a4f6f483014021a66a757012793629bd054d28d0fe4d5e581fa4aed776ef \
  -t local/sparkring:r35-source CONTEXT
docker build --network=none -f CONTEXT/Dockerfile.candidate \
  --build-arg SOURCE_STAGE=local/sparkring:r35-source \
  -t local/sparkring:r35 CONTEXT
docker run --rm --network=none local/sparkring:r35 verify
```

Verification reports installed-file integrity, distribution versions and source
identity. It deliberately does not declare the serving configuration qualified.
Record the resulting image ID and use that immutable ID for tests and deployment.
Do not repoint a published tag as part of these steps.

## Serving and qualification

Use the [local R35 launch guide](../../../docs/operations/r35-local-launch.md)
to record the image receipt and render TP2 or TP4 arguments. The receipt selects
R35's entrypoint and cache contract while retaining model, topology, memory and
managed-mesh admission checks. Give the candidate a separate cache namespace;
keep model mounts read-only. A transport or CPU-placement change requires a
coordinated stop and start of all ranks.

Qualification must record the image, source trees, rendered arguments, hardware,
cache state and benchmark harness identity. Check exact-answer requests, cache
reuse across a full model-process restart, concurrent and changed-tail requests,
coalescing and mHC activation, and transport health. Measure output tokens/s
alongside verifier steps/s and MTP acceptance. Repeat matched measurements before
attributing a rate difference to a setting. A short decode run is not a stability
qualification.

Rollback uses the saved parent image and original rendered configuration after
quiescing every candidate rank. Remove only the candidate's containers and owned
network objects; retain its evidence. Never erase model files or shared cache
volumes to perform a rollback.
