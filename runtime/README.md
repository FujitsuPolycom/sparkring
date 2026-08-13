# SparkRing runtime builder

A **fail-closed, content-addressed builder** for the public SparkRing runtime
ancestry:
patched switchless NCCL + vLLM built from a pinned public commit + the pinned
kernel stack (sparkinfer, FlashInfer, DeepGEMM, torch) + `spark_transport`,
assembled into one aarch64 container image targeting sm_121 (NVIDIA GB10 /
DGX Spark, CUDA 13.2, Python 3.12).

This replaces the historical "frozen image + post-install edits" runtime with
a build where every lock field is either consumed by the build or rejected,
base images and source repositories are content-addressed, and every
divergence from upstream is a reviewable, hash-verified patch. See
`docs/RUNTIME_GAPS.md` for the audit this design answers.

## Faststart versus full rebuild

Most users should begin with:

```bash
OUTPUT_IMAGE=sparkring/glm52-faststart:trial ./runtime/build-faststart.sh
```

`Containerfile.faststart` starts from the exact public ARM64 community image
recorded in `faststart-lock.json`. It applies the same recovered patch series
with the same fail-closed preimage checks, builds patched NCCL and SparkRing's
native transport, emits the same kind of installed-source/native-library
manifest, and stops. It does not rebuild vLLM, Torch, FlashInfer, SparkInfer,
DeepGEMM, or the GB10 kernel stack.

`Containerfile` plus `build-runtime.sh` remains the full source-reproducible
lane. Use it when auditing provenance or rebasing the stack, not as a
prerequisite for the NF3 alternative. Its operator procedure is
[`docs/NF3_QUICKSTART.md`](../docs/NF3_QUICKSTART.md). The default EXL3 path is
[`docs/QUICKSTART.md`](../docs/QUICKSTART.md).

The operator's separately scoped EXL3 R7 3.5-bpw component lineage has its own
reviewable ARM64 builder at [`runtime/exl3-r7/`](exl3-r7/README.md). That
builder consumes immutable public source pins and emits a locally tagged image;
it does not include model weights, a registry push, site configuration, or the
operator's accepted exact-Q40 serving overlay.

## The two-lane model

- **Public lane (this directory):** everything needed to rebuild the *public
  ancestry* of the runtime — pinned upstream commits, the frozen pip set, the
  in-repo NCCL patch series, `spark_transport`, and the two independently
  written SparkCache compatibility patches. Anyone can run it.
- **Recovered reference lane:** the measured vLLM overlay (59 safe modified + 12
  new files, ~12.9k lines) that reproduces the exact production behavior
  (SM121 sparse-MLA backend, low-bit MLA KV record formats, etc.). It is
  published under `patches/00-reference-vllm/` and hash-pinned. Its native ARM64
  faststart build and partial four-Spark bring-up passed; API/request acceptance
  remains.

`runtime/patches/vllm/` contains the two SparkCache patches applied after it.
They apply fail-closed after the recovered overlay against the pinned vLLM
commit. Together the ordered series performs 73 verified operations.

## Pins (authoritative table)

| Component | Pin |
|---|---|
| vLLM | `vllm-project/vllm` @ `fcc614141e5e9ab18cb304c476f7feed2a9552e3` (0.11.2.dev279 lineage) |
| B12X kernels | `local-inference-lab/sparkinfer` @ `284a2eae83754ee1abd31c37b9ca66b68e20b8a8` |
| FlashInfer | `flashinfer-ai/flashinfer` @ `25dd814e03791e370f96c3148242f0dc8de504ac`; `flashinfer-python 0.6.13+cu132` is built from that tree and the `flashinfer_jit_cache 0.6.13+cu132` wheel is SHA-256-pinned |
| DeepGEMM | `deepseek-ai/DeepGEMM` @ `2073ddb2814892014c33ef4cd1c7d4c148baf1fe`; installed version must report `2.5.0+2073ddb` |
| NCCL | `NVIDIA/nccl` @ `73cf112295c33aee2b895f329f592f2a9b4b0f97` (the `v2.30.7-1` release commit) + the two hash-pinned switchless patches |
| Torch / Python / CUDA / arch | `2.12.0+cu132` / `3.12.3` / `13.2` / `sm_121` |
| Model | `aidendle94/GLM-5.2-MXFP4-Experts-GPTQ` @ `46537e0e16fcd156627800139b41b9c497fc7ee2`; `config.json` sha256 `ffd30e72ab8bb7e8ad560f2aaab03cc595f3106f0acf793ef96eedaf90f66d69` |
| Base images | ARM64 manifests `sha256:5c3675...b35d` (devel) and `sha256:360506...52e4` (runtime), resolved from the CUDA 13.0.1 Ubuntu 24.04 tags |
| Python environment | `runtime/pip-freeze.txt`; exact versions are verified after source installs. Six private-machine `file://` origins are closed explicitly; disabled reference-only InstantTensor is intentionally omitted. |

## How the build works

1. `build-runtime.sh` validates a closed lock schema, rejects unknown
   decorative fields and unresolved identities, then passes every consumed pin
   into the `Containerfile`.
2. **Stage 1** builds patched NCCL from the pinned commit; each patch is
   `git apply --check`ed first and any preimage mismatch aborts.
3. **Stage 2** builds vLLM from the pinned commit for sm_121 only
   (`TORCH_CUDA_ARCH_LIST` / `CMAKE_CUDA_ARCHITECTURES`) — honestly, this is
   a **multi-hour compile** (~1.5-3 h native aarch64; do not attempt emulated) —
   closes the audited freeze over public inputs, initializes pinned source
   submodules recursively, installs the pinned
   torch/FlashInfer/SparkInfer/DeepGEMM set, verifies final versions, then runs
   the **fail-closed patch apply** (`apply-patches.py`): for every patch the
   sha256 of the exact preimage file is verified *before* applying, no fuzz,
   abort on any mismatch. Philosophy inherited from
   `spark_transport/experiments/fail_closed_mod_overlay/` (exact hash
   contracts, no arbitrary script execution, fail-closed validation, target
   tree never mutated on mismatch). Finally it builds `spark_transport` and
   requires the complete CTest suite to pass.
4. The builder emits an allowlisted 30-file public Python overlay bundle with
   a per-file SHA-256 manifest. The bundle specification, builder, capability
   gates, and entrypoint are themselves hash-pinned by `runtime-lock.json`.
5. **Stage 3** assembles the runtime image and runs `generate-manifest.py`,
   which emits an immutable `runtime-manifest.json` (all pins, wheel hashes,
   `.so` hashes, applied-patch hashes, model identity pin) into the image.

```bash
./runtime/build-runtime.sh          # docker; CONTAINER_ENGINE=podman also works
```

The script prints the resulting local image ID and, after a push, any registry
digest reported by the engine. The output-image digest cannot be embedded in
the image that creates it; retain it as launch/evidence metadata.

## Verify flow

1. Rebuild from the lock on a clean machine (`build-runtime.sh`).
2. Read `/opt/sparkring/runtime-manifest.json` out of the image and diff it
   against the expected manifest for that `runtime_id`.
3. Confirm the in-build frozen-package verifier passed; optionally rerun
   `verify-frozen-packages.py` inside the image.
4. `sha256sum` the NCCL and `spark_transport` artifacts against the
   manifest's recorded hashes; confirm the model identity pin
   (`config.json` sha256) before serving (SETUP.md Stage 6).
5. Record the pushed image digest in launch/evidence metadata and inject it as
   `SPARKRING_IMAGE_DIGEST`; the in-image manifest cannot self-contain its own
   final registry digest.

## Status

- `patches/00-reference-vllm/` publishes the recovered 71-operation GLM-5.2
  delta; `patches/vllm/` adds two independently written SparkCache patches.
- `public-entrypoint.sh` and the bundled public overlay are offline-validated.
  Startup checks the model config hash, complete sharded-safetensors layout,
  MTP draft layout, runtime manifest, external image-digest binding, pinned
  leader/follower headless ABI, and required GLM capability surface before
  `vllm serve`. The headless gate deliberately preserves upstream's follower
  `collective_rpc` assertion: rank 0 owns EngineCore and RPC; ranks 1-3 only
  host worker subscribers. The pinned
  recovered SM121 sparse-MLA and packed low-bit MLA KV capability surface is
  now present and passed its native capability gate. The next gate is a
  corrected four-rank startup through API readiness and a deterministic request.
- Base-image ARM64 manifests, model revision, DeepGEMM source and NCCL source
  are immutable pins. The output image gets a registry digest only after push;
  the launcher must inject it for image-identity verification.
- `runtime-lock.json`, `apply-patches.py`, `generate-manifest.py`,
  `pip-freeze.txt` are companion files owned by other workstreams; this
  directory's Containerfile/build script consume their documented interfaces.

## Acceptance gates (11 steps, from the project plan)

A runtime build is accepted only when all pass, in order:

1. `runtime-lock.json` matches the consumed schema; unknown, missing, empty or
   unresolved immutable values fail before the container engine runs.
2. Base devel/runtime images resolve only by pinned ARM64 manifest digest.
3. NCCL checkout matches its commit and patches apply with
   `git apply --check` clean — zero fuzz, zero offsets — before the patched
   library builds for sm_121.
4. vLLM source checkout matches the pinned commit exactly
   (`git rev-parse HEAD` == lock) before compilation starts.
5. The public closure has no machine-local URL and final exact package versions
   match `pip-freeze.txt` (no resolver drift).
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
    sha256 equals the lock value before the image is admitted to serving; the
    pushed output-image digest is retained in launch/evidence metadata.
