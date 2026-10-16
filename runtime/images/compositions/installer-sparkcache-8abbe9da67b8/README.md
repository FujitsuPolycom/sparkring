# SparkCache CRC32 source layer over release 2026.10.4

Status: **research-only**. The composition and CPU preparation checks are
implemented. One local build is recorded in [`built-image.json`](built-image.json);
install it on a Spark that holds that image with `--image-lock` naming this
file. No CUDA restore or serving run is recorded.

On an owner-selected Spark with the release 2026.10.4 image already local,
run from the SparkRing repository root:

```bash
bash runtime/images/compositions/installer-sparkcache-8abbe9da67b8/build.sh
```

The [recipe](build.sh) first verifies the parent's eleven lease-contract files
and two native binaries using network-disabled containers. It then extracts
two receipts, prepares a context and invokes the shared
[derived-layer builder](../../derived_layer.py).
It neither starts a serving deployment nor pushes an image. It needs no GPU
access. The resulting development lock is written under
[`dev-20261015-sparkcache-8abbe9da67b8-status038`](../../../releases/dev-20261015-sparkcache-8abbe9da67b8-status038/README.md).
The recipe refuses an existing output lock. Its temporary directory retains
the input receipts, resolved descriptor, Dockerfile, plan and resulting receipts.

## Pinned inputs

[sources.json](sources.json) binds:

- Parent: `dev-20261013-dcp2prefill-status038-sircl-tp6`, release 2026.10.4;
  local image ID `sha256:3cc9a81fd344fe52fe9cc261b46dc051e3c3f494e51b8142846a5f854835c502`.
  The parent lock itself and its two receipt hashes are checked before preparation.
- SparkCache: `https://github.com/FujitsuPolycom/sparkcache.git`, commit
  `8abbe9da67b8143831d21b43d0395c4f347a619a`, Git tree
  `1ee0294eccbf233d2c8f08d7a91bb43afea724e0`, fetched from `origin/main` on
  2026-10-15. This includes fast CRC32 payload checks, in-arena verification
  and the `tail-cow-v2` fast-integrity default from PRs #76 and #77.
- The complete set of 62 runtime Python/JSON files from that commit, each
  with a SHA-256 and [vendored source path](../../../../integrations/sparkcache/8abbe9da67b8).
  Tests and native source code are excluded from the installed payload.
  The upstream [Apache-2.0 license](../../../../integrations/sparkcache/8abbe9da67b8/LICENSE)
  accompanies those sources. SparkCache implementation stays upstream.
- The eleven-file vLLM lease contract
  `vllm-connector-jobs-gab86b70.json`, SHA-256
  `ab53c097d7ca408e995bac7c2bc915b64119881ed64b9cec0b3cc3b30ed2f990`.

The parent does not record a SparkCache commit. Preparation therefore derives
each inherited file hash from the **exact receipt authenticated by the pinned
parent lock**, not from an assumed checkout. It writes those hashes and the
resulting hashes into `context/descriptor.json` and the image's provenance
receipt. Unrecognized inherited runtime Python modules are refused so that
removed upstream modules cannot silently survive. All supplied files are
SHA-256 checked before the context is created. No inherited hashes or image
results are invented in the repository.

The image also contains `sparkcache/runtime_patches/sparkring-source-8abbe9da67b8.json`,
a copy of the source inventory bound by the rewritten external-base receipt.
The layer changes no vLLM or B12X files. Its lock retains the parent's SIRCL,
libsircl, plugins, transport and default-tuning contracts. The existing
distribution metadata remains inherited; this is a source overlay, not a
claim that a wheel was rebuilt.

## Native libraries

CUDA restore and asynchronous capture require the following inherited binaries:

| Path under `/opt/sparkring/sparkcache/lib/` | Required SHA-256 |
|---|---|
| `libspark_cache_placement.so` | `d89c9fdae8dc99ae3f7a151cc3dd9e92fdc8fd0b994069fc263027fd4d056c93` |
| `libspark_cache_snapshot.so` | `b50db45123c4faf0fdfb021da0a0f1b7f329869ae5dc13dcc7432dd4ce765b68` |

These are the binary pins already used by the GLM installer variants; the
snapshot artifact supports 64 groups. SparkCache's native source tree
`9d70fdafe087951277770181fcb4d081f9cc3343` last changed at
`1388764ae9a8dedcd25c91a1313d56fa321c3388`. The CRC32 changes are Python-only.
This composition carries the inherited binaries unchanged and requires both
hashes in the parent's verified file inventory. The shared image-admission
step hashes the resulting installed files, and the profiles also pin the
libraries at connector construction.

No native rebuild is required **if those exact artifacts are present**.
Their presence in the release image has not been inspected in this work.
If either is absent or different, preparation fails before producing a
context. A separate native build with reviewed binary hashes and corresponding
profile pins is then required; do not substitute a guessed inherited hash.
The final recipe step verifies all eleven lease-contract files against the
built image without CUDA. GPU capture and restore still require qualification.

## Profile and namespace choice

[build-inputs.json](build-inputs.json) lists the parent's fifteen profiles plus
[`glm53-flash-nvfp4-sparkcache-crc32-tp2`](../../../../profiles/glm53-flash-nvfp4-sparkcache-crc32-tp2/README.md)
and [`glm53-flash-nvfp4-sparkcache-crc32-tp4`](../../../../profiles/glm53-flash-nvfp4-sparkcache-crc32-tp4/README.md).
The earlier installer variants retain their IDs, serving settings and image
references because their hardware evidence belongs to that composition.
The CRC32 variants have no inherited serving qualification. Their explicit
`fast` integrity setting and independent TP-specific roots prevent reuse of
the earlier cache namespace. No published lock is edited and no image becomes
the installer default. The build output is a development lock, not a release
selection. There are no Compose exports for these unlisted profiles.

## Checkpoint identities

`dec48abd33efa73c3bb7c95b74eee10cad34f9be` is the GLM CSF repository revision.
The recorded target and draft SparkCache digest is
`6af0343976eefbfa0fb59d87befcf6d10f02fef90ba0e7881dd1201fe0ecd76e`.
The recorded Qwen step-4000 digest is
`036c2f7994466d32514130f0417ccd117705ca01e2038c1ce5745f84813829ba`.
Both remain unchanged. SparkCache's `deploy/glm53_flash/profile.py` checks a
supplied digest; it does not derive these values from a manifest.

Neither digest reproduces from the available revision-hash helper, raw or
canonical pin-manifest hashes, or the DeepSeek complete-manifest preimage
applied to the pinned inventory. The original preimage, algorithm and salt
are absent. The shared `sparkcache.checkpoint_identity` function and CLI are
developed separately in SparkCache; its versioned inputs bind repository,
revision, manifest/file hashes, quantization, encoding, model config and draft
configuration. Its audit results must not replace recorded deployment values
merely to make an equality test pass. This layer deliberately pins the fetched
upstream runtime, rather than an unpublished local tooling commit.

## Before Qwen and DeepSeek variants

Qwen needs topology fixtures for TP2/TP4, non-default topology identity fields,
derived identities for the CSF and step-5500 checkpoints, capture-slot span
limits, variant registrations and same-image restore/output-equality tests.
The step-4000 identity must not be reused for those checkpoints.

DeepSeek-V4.1 needs request-private/replay group policies, compressor-ring
classification, the 128-token replay cap, compression alignment and failure
recompute tests, a V4.1 model profile and checkpoint identity. Begin with
synchronous capture and Python restore; CUDA restore and asynchronous capture
need a separate compatibility review and hardware validation. Engram tables
are static weights and do not become cached request state.

Both need an authorized hardware window, the parent/native/lease checks,
recorded topology, restart/SIGKILL/corruption/output-equality gates and a
cache-off performance comparison before qualification. Add their IDs to build
inputs only when their implementations and CPU tests exist.
