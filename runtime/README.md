# SparkRing runtime builder

A **source-pinned, reproducible builder** for the SparkRing serving runtime:
patched switchless NCCL + vLLM built from a pinned public commit + the pinned
kernel stack (sparkinfer, FlashInfer, DeepGEMM, torch) + `spark_transport`,
assembled into one aarch64 container image targeting sm_121 (NVIDIA GB10 /
DGX Spark, CUDA 13.2, Python 3.12).

This replaces the historical "frozen image + post-install edits" runtime with
a build where **every input is pinned in `runtime-lock.json`** and every
divergence from upstream is a reviewable, hash-verified patch. See
`docs/RUNTIME_GAPS.md` for the audit this design answers.

## The two-lane model

- **Public lane (this directory):** everything needed to rebuild the *public
  ancestry* of the runtime — pinned upstream commits, the frozen pip set, the
  in-repo NCCL patch series, `spark_transport`. Anyone can run it.
- **Reference lane:** the maintainer's vLLM patch overlay (61 modified + 12
  new files, ~12.9k lines) that reproduces the exact production behavior
  (SM121 sparse-MLA backend, low-bit MLA KV record formats, etc.). It is
  captured privately and is **not yet shippable** pending provenance cleanup
  (~0.8k unattributed lines must be attributed or rewritten first —
  `docs/RUNTIME_GAPS.md`, "Action items").

Consequently `runtime/patches/vllm/` **ships empty** (gate note in its
README). A public-lane build today yields stock upstream vLLM at the pinned
commit — correct ancestry, but the reference behavior still requires the
maintainer patch series.

## Pins (authoritative table)

| Component | Pin |
|---|---|
| vLLM | `vllm-project/vllm` @ `fcc614141e5e9ab18cb304c476f7feed2a9552e3` (0.11.2.dev279 lineage) |
| B12X kernels | `local-inference-lab/sparkinfer` @ `284a2eae83754ee1abd31c37b9ca66b68e20b8a8` |
| FlashInfer | `flashinfer-ai/flashinfer` @ `25dd814e03791e370f96c3148242f0dc8de504ac`; wheels `flashinfer-python 0.6.13+cu132`, `flashinfer_jit_cache 0.6.13+cu132` |
| DeepGEMM | `2.5.0+2073ddb` (full SHA: resolve-pending) |
| NCCL | `NVIDIA/nccl` tag `v2.30.7-1` + `spark_transport/experiments/nccl_switchless_ring/` patch series (skip-tree-pat, advertise-all-listener-gids) |
| Torch / Python / CUDA / arch | `2.12.0+cu132` / `3.12.3` / `13.2` / `sm_121` |
| Model | `aidendle94/GLM-5.2-MXFP4-Experts-GPTQ`; `config.json` sha256 `ffd30e72ab8bb7e8ad560f2aaab03cc595f3106f0acf793ef96eedaf90f66d69`; immutable HF revision pinned at build time (currently `PENDING`) |
| Base images | `nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04` (builder) + matching runtime image; digests `pending` until pinned at first successful build |
| Full pip set | `runtime/pip-freeze.txt` (200 packages, from the audited frozen runtime) |

## How the build works

1. `build-runtime.sh` parses `runtime-lock.json` and passes every pin as a
   `--build-arg` — nothing version-shaped is hardcoded in the `Containerfile`.
2. **Stage 1** builds patched NCCL from the pinned tag; each patch is
   `git apply --check`ed first and any preimage mismatch aborts.
3. **Stage 2** builds vLLM from the pinned commit for sm_121 only
   (`TORCH_CUDA_ARCH_LIST` / `CMAKE_CUDA_ARCHITECTURES`) — honestly, this is
   a **multi-hour compile** (~1.5-3 h native aarch64; do not attempt emulated) —
   installs the pinned torch/flashinfer/sparkinfer/deep_gemm set, then runs
   the **fail-closed patch apply** (`apply-patches.py`): for every patch the
   sha256 of the exact preimage file is verified *before* applying, no fuzz,
   abort on any mismatch. Philosophy inherited from
   `spark_transport/experiments/fail_closed_mod_overlay/` (exact hash
   contracts, no arbitrary script execution, fail-closed validation, target
   tree never mutated on mismatch). Finally it builds `spark_transport`.
4. **Stage 3** assembles the runtime image and runs `generate-manifest.py`,
   which emits an immutable `runtime-manifest.json` (all pins, wheel hashes,
   `.so` hashes, applied-patch hashes, model identity pin) into the image.

```bash
./runtime/build-runtime.sh          # docker; CONTAINER_ENGINE=podman also works
```

The script prints the resulting image digest and the exact manual steps to
write it back into the lock. **It never auto-mutates the lock.**

## Verify flow

1. Rebuild from the lock on a clean machine (`build-runtime.sh`).
2. Read `/opt/sparkring/runtime-manifest.json` out of the image and diff it
   against the expected manifest for that `runtime_id`.
3. Cross-check `pip freeze` inside the image against `runtime/pip-freeze.txt`.
4. `sha256sum` the NCCL and `spark_transport` artifacts against the
   manifest's recorded hashes; confirm the model identity pin
   (`config.json` sha256) before serving (SETUP.md Stage 6).
5. Compare the image digest against `images.built.digest` in the lock.

## Status

- `patches/vllm/` — **empty, gated** on provenance review of the private
  overlay. Reference-lane behavior still requires the maintainer patch series.
- Base image digests, model HF revision, DeepGEMM full SHA, built-image
  digest — **placeholders (`pending`)** until pinned at first successful build.
- `runtime-lock.json`, `apply-patches.py`, `generate-manifest.py`,
  `pip-freeze.txt` are companion files owned by other workstreams; this
  directory's Containerfile/build script consume their documented interfaces.

## Acceptance gates (11 steps, from the project plan)

A runtime build is accepted only when all pass, in order:

1. `runtime-lock.json` parses and every required pin key is present and
   non-empty (fail-closed accessor in `build-runtime.sh`).
2. Base devel/runtime images resolve; digests pinned (or explicitly flagged
   `pending` on first build only).
3. NCCL patch series applies with `git apply --check` clean — zero fuzz,
   zero offsets — and the patched library builds for sm_121.
4. vLLM source checkout matches the pinned commit exactly
   (`git rev-parse HEAD` == lock) before compilation starts.
5. Installed python set matches `pip-freeze.txt` (no resolver drift).
6. Patch overlay preimage verification: every `runtime/patches/**` entry's
   target-file sha256 matches its declared preimage before apply; any
   mismatch aborts the build.
7. `spark_transport` builds and its ctest suite passes 100% inside the build
   stage.
8. `generate-manifest.py` completes and the emitted manifest is internally
   consistent (all hashes recomputed and matching).
9. Image builds end-to-end for `linux/arm64` and is tagged by `runtime_id`.
10. Post-build verify flow (above) passes on at least one clean rebuild —
    manifest diff empty against the expected manifest for the `runtime_id`.
11. Model identity gate: pinned HF revision resolves and its `config.json`
    sha256 equals the lock value before the image is admitted to serving
    (then the image digest is written back into the lock by hand).
