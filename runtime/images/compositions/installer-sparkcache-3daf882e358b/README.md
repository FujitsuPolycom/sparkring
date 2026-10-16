# SparkCache Qwen topology source layer

Status: **implemented**, CPU tested; hardware qualification pending. One local
build is recorded in [`built-image.json`](built-image.json) (image
`sha256:20c0fd2041d5607fc9e0278bb42dfe3877a0905c6bb7f43c3fdfca62d676222b`).
Run from the SparkRing root on an authorized Spark with release 2026.10.4's
parent image present:

```bash
bash runtime/images/compositions/installer-sparkcache-3daf882e358b/build.sh
```

The [source inventory](sources.json) pins SparkCache `3daf882e358b0c0846f14b44122204a8774f2f66`
over parent `sha256:3cc9a81fd344fe52fe9cc261b46dc051e3c3f494e51b8142846a5f854835c502`.
Unchanged package files reuse the byte-identical `8abbe9da67b8` vendor directory;
three changed/added modules are vendored under `3daf882e358b`. Native libraries and
the eleven-file gab86b70 lease contract retain their recorded hashes.
The preparer authenticates both parent receipts, refuses unknown inherited
runtime modules, and binds the source commit and tree in the installed receipt.

This composition admits the parent's fifteen profiles plus GLM CRC32 TP2/TP4
and Qwen SparkCache TP2/TP4, as specified by [build-inputs.json](build-inputs.json).
A separate composition preserves the recorded GLM layer build and its exact
source inventory. Published locks and defaults retain their recorded bytes.
The build writes `built-image.json` beside this recipe only after admission;
it does not place an unpublished selection under `runtime/releases/`.
Use that file with `sparkring install --image-lock` after qualification gates.

The recipe probes both inherited native hashes and all eleven lease files
without GPU access, builds through the common derived-layer builder, then
verifies the resulting contract. It never pushes an image. Missing or different
native binaries require a separately reviewed rebuild and refreshed pins.
The [Qwen hardware plan](../../../../docs/research/qwen38-sparkcache.md)
requires an authorized serving interruption and records actual topology before
correctness or performance qualification.
