# Token-sharded mHC prefill

Status: **research-only**. The source package runs repeated manifold-constrained
hyper-connection (mHC) operations on each GPU's quarter of the tokens in eligible
8,192-token eager prefills. The feature is installed but defaults off through
`SPARK_MHC_PREFILL_SHARD=0`; enabling it requires the supported configuration
below and explicit deployment validation.

## Computation and ownership

Attention and feed-forward projections produce full-token TP partial outputs.
Instead of all-reducing and repeating mHC on every rank, the model reduce-scatters
those partials, runs mHC on 2,048 owner rows, and all-gathers the normalized input
for the next attention or FFN operation. Each rank owns one contiguous quarter
in conventional NCCL order. This is distinct from the native ring's ownership
layout and uses the existing TP PyNccl communicator.

The first mHC pre operation remains full-sized. Its residual, post, and combine
state become owner views, with no initial boundary gather. Repeated mHC retains
local state within the forward. Final and auxiliary hidden outputs are gathered
back into full token order before their consumers. A 45-layer forward with no
auxiliary captures performs 90 reduce-scatters and 90 all-gathers.

Per-call `defer_tp_reduction` arguments reach the outer attention projection,
dense FFN projection, and the combined shared/routed MoE final reduction. The
normal call paths retain their reductions; shared module flags are not toggled.
An already-reduced or unsupported MoE result is rejected before partial output
can reach mHC. Attention kernels, recurrent cache layout, model weights, and
B12X compute kernels are unchanged.

## Admission and fallback

The implementation accepts BF16 `[8192, 4096]` base-model inputs on TP4/DCP4,
PP1/DP1/PCP1, with expert/sequence parallelism and expert load balancing disabled.
All base layers must use the concrete supported B12X mHC, attention, projection,
and MoE classes. The batch ceiling must be 8,192; captured graph sizes must stay
below that ceiling.

Host GDN metadata must describe pure prefill with no decode or speculative rows.
Mixed batches, other row counts, absent/ambiguous metadata, compilation, and CUDA
graph capture/replay keep the original path. MTP layers cannot receive an owner.
All ranks agree on the flag at construction and vote on eligible prefill
capabilities before changing ownership. The vote uses host counts and adds host
work; it does not copy GPU metadata to the CPU. Invalid enabled capabilities
fail explicitly. Rank-local eligibility differences fall back collectively.

This optimization does not target decode. It is independent of B12X's source-split
mHC decode schedules and does not include those schedules or MTP capture sizing.

## Source composition

[The manifest](manifest.json) binds eight existing preimages and nine installed
sources to the verified serving image
`sha256:65f2b9181acd77db660f9c105554c4fca5c4df89d87d1374276e17c6831d1359`.
The [review diff](source.patch) exposes every change. Licensed source and
preimage archives preserve the exact bytes, including upstream notices;
[LICENSE](LICENSE) and [NOTICE](NOTICE) describe redistribution and modifications.

The performance installer applies this package after continuation and request
attribution. It verifies all runtime preimages and overlapping checkpoint
ownership entries before its first write, preserves symbol requirements, and
updates only overlapping ownership hashes. The builder generates the combined
installed-file inventory. Existing published-image receipts remain immutable.

The exact nine sources completed the
[recorded serving checks](../../../../performance/records/glm53-flash/mhc-token-sharding-20260906/README.md)
on their original continuation parent. A rebuild containing request attribution
and this package is a different composition and has not completed GPU/serving
validation. Leave the feature off until that composition passes its own checks.
No managed service, host fabric, or live deployment is changed by preparation.

Six preimages match unmodified vLLM git objects at the pinned
`e02b174693e13859de61811b5e8cd13d5308e259` revision. The GLM model preimage matches
the maintained compute archive; Kimi GDN matches the checkpoint payload. Verify
these sources offline with `python verify_preimages.py --vllm-checkout /path/to/vllm`
from this directory. The checkout needs that commit locally. This verifies
source compatibility; the full composed Docker build has not been exercised.

## Offline validation

```bash
python -m pytest runtime/glm53-spark-mtp3-mesh/performance/mhc-prefill -q
```

Tests execute the packaged model and projection methods with a four-rank CPU
collective facade. They check local residual ownership, full final/auxiliary
outputs, exactly-once reductions, unsupported MoE rejection, MTP exclusion,
host metadata fallback, rank/device agreement, and package/preimage rejection.
These tests establish neither GPU ordering nor numerical equivalence. Recorded
model probabilities differ after changing the collective reduction order;
broader quality validation remains required.
