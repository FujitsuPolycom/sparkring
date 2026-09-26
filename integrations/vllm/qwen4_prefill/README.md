# Qwen prepared-runtime prefill hooks

Status: implemented. The installer image runs these hooks for the Qwen
profiles on two and four Sparks; the tests here are CPU admission and
call-contract checks.

The bundle retains two Qwen prefill optimizations for the prepared B12X runtime:
Hyperconnection (HC) up-projection/gate fusion and BF16 MTP projection GEMMs at
128 or more rows.
It targets the Qwen4Exp NVIDIA implementation, including checkpoints registered
under the Qwen3.8-Flash-Next architecture aliases. HC state has four streams;
that count is distinct from the tensor-parallel rank count.

The hooks run in tensor-parallel groups of two or four ranks with BF16 weights,
hidden size 2560 and HC rank 320, or with the HC projection split into four
parts of HC rank 80. A split projection runs batches of 128 or more rows
sharded: each rank projects its share, and the bottleneck and the output are
all-gathered. The fused gate kernel takes hidden widths of 640, 1,280 and 2,560.
Batches below 128 rows and smaller MTP batches keep the upstream prepared
projection callables. HC uses its declared `scaled_silu` preparation binding.
These are source-specific hooks, not general model support.

`package_prefill.py` emits a manifest-bound bundle for
`/opt/sparkring/qwen4-prefill`. The manifest records the image files that the
hooks patch and their SHA-256: `--image sparkring`, the default, for images
SparkRing builds, whose Python packages live in `/opt/venv`; `--image
external-base` for the installer image, whose packages live in
`/usr/local/lib/python3.12/dist-packages`. The selected image feature installs only
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
