# Qwen prepared-runtime prefill hooks

Status: implemented; CPU admission and call-contract checks only. GPU serving
qualification is required before selecting this bundle in a profile.

The bundle retains two Qwen prefill optimizations for the prepared B12X runtime:
Hyperconnection (HC) up-projection/gate fusion and BF16 MTP projection GEMMs at
128 or more rows.
It targets the Qwen4Exp NVIDIA implementation, including checkpoints registered
under the Qwen3.8-Flash-Next architecture aliases. HC state has four streams;
that count is distinct from the tensor-parallel rank count.

The hooks require TP4, BF16, hidden size 2560 and HC rank 320. Smaller MTP batches
keep the upstream prepared projection callables. HC uses its declared
`scaled_silu` preparation binding. These are source-specific hooks, not general
model or TP2 support.

The HC hook supports both replicated and TP4-sharded projection weights. With
sharded projections it retains KK's FP32 down-projection before BF16 conversion,
gathers the bottleneck, fuses the local 640-coordinate up-projection and gate,
then gathers the block input. Batches below 128 rows use the original KK method,
including its prepared decode workspaces and projection dispatch.

Projection sharding requires identical token rows on every rank. The separate HC
token-row ownership mode must therefore be `off` when projection sharding is
enabled; combining the two would mix projections from different tokens. The
adapter must select this configuration explicitly. The sharded fusion path has
CPU dispatch checks; GPU numerical and serving qualification remain pending.

`package_prefill.py` emits a manifest-bound bundle for
`/opt/sparkring/qwen4-prefill`. The selected image feature installs only
`qwen4_prefill.pth`; do not activate the R37 `qwen-prefill` bundle alongside it.
The distinct Python module names prevent accidental reuse of the R37 hook modules.

Admission requires `SPARKRING_QWEN4_PREFILL_MANIFEST_SHA256`, exact installed
source hashes, and compiler caches beneath `qwen4-prefill-<manifest-prefix>`.
Mismatches terminate startup. Existing R37 source locks and package contents
remain unchanged. No external KV-cache identity or public profile pin is changed
by packaging this bundle.

## Multimodal dispatch and startup audit

HC fusion and HC token-row sharding are separate optimizations. The multimodal
wrapper must call the language-model forward method for row-sharding admission
to execute. Calling its nested model directly bypasses that admission even when
`VLLM_QWEN3_8_HC_PREFILL_MODE=shard` is set.

`multimodal_hc_routing.py` repairs that call while preserving vision deep-stack,
n-gram and pipeline arguments. Its API-source transformation inserts the audit
after engine/app initialization and before HTTP serving. Both transformations
reject unexpected source structure; changed source requires a distinct image
and matching cache-source contract, not an edit to a live container.

`startup_audit.py` reports the Qwen HC route, coalescing, compact MTP, projection
overlap and dynamic-materialization selection. Warnings identify disabled or
incompatible paths; the audit never changes settings. A verified source route
is not proof of runtime execution or better performance. Runtime HC execution
is separately reported by `QWEN_HC_PREFILL` after an eligible real prefill.
No automatic benchmark or inference request is part of the audit.

These helpers are inputs to source reconciliation, not additional activation
hooks in the manifest-bound fusion bundle. The audit is packaged in the vLLM
API source of images that explicitly include it. Existing images are unchanged.
