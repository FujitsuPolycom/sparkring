# Hotfix: ll_bf16 router GEMM warmup crashes non-GLM models

Scope: deployed image `sparkring/glm52-exl3-r7-3.5bpw:r34-sm121a-flat2-20260810`
(runtime `glm52-gb10-faststart-19523482c298`). Verified preimage sha256 of the
installed `vllm/model_executor/warmup/kernel_warmup.py`:
`6d75f77735eb09129008ffc23e53c01726da45e9429ca9eeaa85db0381f8e997`.

The `kernel_warmup` patch in `patches/00-reference-vllm` gates the CuTe-DSL
GEMM autotune section on `kernel_config.enable_cutedsl_warmup` but the
`_warmup_ll_bf16_router_gemm()` call site runs unconditionally on any
capability-90+ device. The image's `quack` package is version-skewed against
its `cutlass` (`AttributeError: module 'cutlass.cute.core' has no attribute
'ThrMma'`), so the first non-GLM model served from this image dies inside
`determine_available_memory`. GLM serving never exercises the broken path.

Discovered 2026-08-17 during the eager-width validation leg 3 (pythia-70m
TP4); reproduced deterministically; the patched file was live-validated on
all four ranks by bind-mount.

This directory is deliberately OUTSIDE `runtime/patches/` so the fail-closed
applier never consumes it (its preimage matches the deployed image, not the
current pinned base, whose kernel_warmup preimage has since moved).

Fold into the next `00-reference-vllm` kernel_warmup patch refresh: gate the
call site on `worker.vllm_config.kernel_config.enable_cutedsl_warmup` and
wrap it best-effort, as in the patch here; then delete this directory.
