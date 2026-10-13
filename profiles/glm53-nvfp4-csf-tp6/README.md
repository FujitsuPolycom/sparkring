# GLM-5.3 NVFP4-CSF on six Sparks

This experimental profile serves Z.ai's GLM-5.3 using Local Inference Lab's
[`local-inference-lab/GLM-5.3-NVFP4-CSF`](https://huggingface.co/local-inference-lab/GLM-5.3-NVFP4-CSF)
checkpoint. It has CPU configuration tests only. No TP6 serving, correctness,
memory or performance result is claimed.

The six ranks occupy positions 0-5 of an eight-Spark cycle. SIRCL carries the
collectives with NCCL off. The subgroup planner uses only the cables between
positions 0-5, routing its closing edge through four internal relays. This differs
from the whole-cycle relay table, whose shortest closing route crosses positions
6 and 7. That route mismatch is an unresolved hardware blocker. This profile
does not create or change relay services on the other pair.

## Checkpoint and padding

The installer acquires the pinned CSF container and constructs the existing
[CSF serving directory](../glm53-nvfp4-csf-tp8/README.md). The base and derived
manifest, recipe hash and checksum list are shared with TP8. No separate TP6
weight copy is needed: `--hf-overrides` in [config.json](config.json) sets
66 attention heads, `original_num_attention_heads=64`, 2304 expert intermediate
columns and `original_moe_intermediate_size=2048`. The deployment identity
covers these overrides; the serving directory's pinned config stays unchanged.
The image applies the target overrides to MTP as well.

There are 11 local query heads, rounded by B12X to groups of eight for attention.
The expert count stays 256; columns within every expert are padded. The local
expert width is 384, which satisfies the A4 prefill requirement of a multiple
of 128. The smaller 352-column shard (2112 global columns) satisfies the CSF
loader and has a W4A16 K32 fallback in the image, but fails A4 eligibility.
Neither padding choice has been exercised on hardware here.

## Serving limits

The draft uses DCP2, interleave 1, two-token MTP with draft TP6, eight sequences,
4096 batched tokens and 32 GiB of KV cache per rank. Its maximum individual
sequence is 135168 tokens, allowing a 128K prompt plus 2048 output tokens. Eight simultaneous contexts at that length may exceed
the cache; record queueing and preemption instead of claiming resident capacity.
DCP3 is an unmeasured alternative. DCP6 requires disabling the indexer row-split
plugin, which requires at least two KV copies. The full model has no Flash mHC
row-sharding restriction.

The TP8-only latent and MTP projection speedups are disabled. Their startup
check is omitted, as are the MXFP8 and draft checks that infer quantization from
TP8-specific weight thresholds. Before any hardware benchmark, inspect the
actual loaded quantization, draft acceptance, weights and cache capacity.
The CSF A4, scale-prefetch and barrier-reset checks remain enabled.

## Image admission and installation

No shipped image lock admits this profile. Use a separate v3 image lock for the
CSF-capable `sparkring-kraken-beta-20261007-bc9ea774` vLLM build, with SIRCL and
`glm_dsa_indexer_split`, explicitly listing this profile. The image must also carry the vLLM adapter fix that honors `SIRCL_MAX_RELAYS=4`.
The extracted d52737 image lacks that fix. A rebuilt image needs its own digest
and receipts; do not edit a published release lock or claim the original image
contains changed code. Before installation, reconcile the subgroup routes with
the installed universal relay table without changing positions 6-7. Raising the
relay limit alone does not resolve that mismatch.

Only after those blockers are resolved, with that lock and this local package staged by the operator:

```bash
sudo sparkring install --profile glm53-nvfp4-csf-tp6 --on 0-5 --image-lock /path/to/tp6-image-lock.json --no-auto-recover
```

The command is a future hardware procedure. Stop if the plan would displace a
deployment on positions 6-7 or alter the fabric. Path-6 has no measured tuning
row; the existing path rules apply until a bounded latency sweep supplies one.

## Registration

The research catalog and `direct-path-6` relayed topology provide automatic
installer, toolchain and fabric-adapter registration. There are no Compose
exports. Existing checkpoint pins, thinking records, display names, publisher
credit and storage planning apply unchanged. No measured capacity record or
recommended public install row is added.
