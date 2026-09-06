# Indexer and sampling-readiness image update

Status: implemented and GPU-tested. No full-model soak or serving replacement
is claimed. This is a small child of the published SIRCL operator image, with
the tested B12X publication barrier, its new compile revision, output-stall
liveness detection, and temperature-one/thinking-enabled readiness warmup.

Published tag:
`ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache:20260906-indexer-barrier-warmup`

Immutable reference:
`ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:28e6a9c0dba07cec4852bf21352e5e2f6fc7bd07592a0edc3916d13f849cdcfe`

Local/config image ID:
`sha256:d2ba5894bd5499883acbb89ecb496965c44f78aa3a4b4be83c4698b2010ec787`

For the operator launcher, set `IMAGE_REF` to the immutable reference and
`IMAGE_ID` to the config image ID on every rank. A distinct
`JIT_CACHE_NAMESPACE`, such as `glm53-20260906-indexer-barrier-warmup`, keeps
the update's warmup evidence separate. The sampling request runs when
`DFLASH_WARMUP=1`; disabled warmup and headless-rank behavior are unchanged.
Use the deployment's coordinated stop/start procedure to install the update.

This image replaces the SIRCL operator parent whose digest starts `0d4029b3`.
It does not contain the additional managed-MTP3 mesh bundle from the separate
image whose digest starts `23f00af8`. Preserve that bundle when rebuilding a
managed mesh child; do not substitute this base tag for a managed mesh image.
The DeepSeek-specific #217 serializer patch is not part of this GLM runtime.

## Checks on the built image

- Source manifests and all retained native libraries verified before and after
  the patch. The updated source receipt records the transform and wrapper
  hashes; the original receipt is retained as parent provenance.
- 22 tests exercised the installed readiness and liveness modules.
- 15 full-indexer GPU correctness tests passed on GB10.
- 2,000 GLM-shaped graph replays passed with exact selected-index parity,
  contexts up to 200K tokens, and concurrent 64 MiB GPU copies.
- Anonymous registry pull succeeded on another DGX4 node.
- The image adds two layers and 428,386 uncompressed bytes to its parent.

The warmup tests validate the installed request/marker contract, not cold-cache
full-model specialization coverage. The long-running TP4/SparkCache workload
was not repeated. Report recurrence with the image digest, runtime settings,
EngineCore stack, and all worker-thread stacks.

## Verify the published child image

The parent runtime's `verify_image.py --image` command checks the base builder's
labels. The child image has different parent and runtime-status labels, so use
the immutable reference with the installed content verifier:

```bash
docker run --rm --network none --entrypoint python3 \
  ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:28e6a9c0dba07cec4852bf21352e5e2f6fc7bd07592a0edc3916d13f849cdcfe \
  /opt/sparkring/bin/verify-jj-r8-sparkcache-image.py --inside-image
```

This command verifies the image's source manifests and retained native-library
hashes without GPU access or a model process. Its result does not establish
serving qualification.

## Rebuild

Assemble an empty context with this `Dockerfile` and `install_hotfix.py`, plus
`patch_indexer_barrier.py`, `serve_with_warmup.py`, and `scheduler_liveness.py`
from the parent runtime directory. Use LF line endings. Run the installer in
an isolated container of the pinned parent with that context mounted at
`/hotfix`, passing `--output /hotfix/build-input.json`.

Build with the resulting `source_receipt_sha256` as the
`SOURCE_RECEIPT_SHA256` build argument and the source checkout commit as
`SPARKRING_REVISION`. The installer rejects unexpected parent source or
wrapper hashes and verifies the final source receipt against the build argument.
The published build uses source commit
`ef3c381bd41eef07abcb5daebfc9c81d5928ff88` and source receipt
`b27290a28e6d322d37b9cea11b01cd76d18a83f2dd58e6f4e1094cdc83c9b4bc`.
