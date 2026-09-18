# Qwen prepared-runtime prefill hooks

Status: implemented; CPU admission and call-contract checks only. GPU serving
qualification is required before selecting this bundle in a profile.

The bundle retains two Qwen prefill optimizations for the prepared B12X runtime:
HC up-projection/gate fusion and BF16 MTP projection GEMMs at 128 or more rows.
It targets the Qwen4Exp NVIDIA implementation, including checkpoints registered
under the Qwen3.8-Flash-Next architecture aliases. HC state has four streams;
that count is distinct from the tensor-parallel rank count.

The hooks require TP4, BF16, hidden size 2560 and HC rank 320. Smaller MTP batches
keep the upstream prepared projection callables. HC uses its declared
`scaled_silu` preparation binding. These are source-specific hooks, not general
model or TP2 support.

`package_prefill.py` emits a manifest-bound bundle for
`/opt/sparkring/qwen4-prefill`. The selected image feature installs only
`qwen4_prefill.pth`; do not activate the R37 `qwen-prefill` bundle alongside it.
The distinct Python module names prevent accidental reuse of the R37 hook modules.

Admission requires `SPARKRING_QWEN4_PREFILL_MANIFEST_SHA256`, exact installed
source hashes, and compiler caches beneath `qwen4-prefill-<manifest-prefix>`.
Mismatches terminate startup. Existing R37 source locks and package contents
remain unchanged. No external KV-cache identity or public profile pin is changed
by packaging this bundle.
