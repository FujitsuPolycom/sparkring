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

At the time of this one-Spark result, it proved that a clean public checkout
could build the ARM64 faststart image but did not yet prove four-rank
distribution, preflight, model startup, or inference acceptance. The next
section records the later four-rank progress.

## 2026-07-29: partial four-Spark bring-up passed

The native candidate was distributed to all four ranks with one identical
image ID:

```text
sparkring/glm52-faststart:public-gates-v1
sha256:b261c42a80c57435c0cfe5ae9f00a83b93bb2db29e5b35c70060922c14f069b2
```

The run passed:

- 12/12 directed SSH-management edges;
- 116/116 clean public preflight checks;
- all public entrypoint and GLM capability gates;
- distributed NCCL initialization;
- all 79 model shards plus both MTP shards on every rank;
- 100.3 GiB model memory per rank;
- 34/34 bounded B12X prewarm cases with zero failures;
- a 4.28 GiB KV allocation per rank, reporting 465,663 logical tokens.

The first attempt did **not** reach API acceptance. After KV allocation, the
generic full-model FlashInfer 4096-token autotuner entered a rank-asymmetric
collective and triggered the NCCL watchdog. SparkRing commit `b8f8a5b` disables
that unsafe distributed tuner through
`--kernel-config '{"enable_flashinfer_autotune":false}'`, while
retaining the bounded B12X prewarm, and fixes the launcher JIT-cache mount.
That historical candidate stopped at this point. The later clean-checkout NF3
NVFP4+FP8-RoPE bootstrap completed the full four-rank serving gate; see
[NF3_NVFP4_PUBLIC_VALIDATION.md](../NF3_NVFP4_PUBLIC_VALIDATION.md).
