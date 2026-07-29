# Faststart native validation

## 2026-07-29: one-Spark ARM64 build passed

The public faststart lane was built natively on one DGX Spark from clean
SparkRing commit:

```text
d1f62bcd4fd2682c43f143e12eec7b765fc008e4
```

Resulting local image:

```text
sparkring/glm52-faststart:d1f62bcd4fd2
sha256:44082de04068ae72c6abf18fda5d2562b1ba12aef8749c7ab34d64729e7f9bfb
```

The build completed all of the following:

- pulled and verified the immutable ARM64 community base;
- classified the recovered 73-operation vLLM overlay;
- verified 65 operations already inherited by the base;
- cleanly rebased four operations;
- applied four exact, checksum-pinned supplemental patches;
- compiled patched NCCL for SM121;
- compiled the SparkRing native transport;
- assembled the runtime image and generated its manifest.

The read-only runtime verifier then passed:

- manifest self-hash;
- 102 installed source files;
- 16 native libraries;
- expected runtime ID.

The image-digest check was intentionally skipped in the direct `docker run`
verification because the four-rank launcher had not injected a registry or
distributed image digest.

## What this does and does not prove

This proves that a clean public checkout can build the ARM64 faststart image
on a DGX Spark. It does not yet prove four-rank distribution, preflight,
model startup, or inference acceptance from this newly built image. Those
remain the next release gate.
