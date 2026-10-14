# GLM-5.3 NVFP4-CSF on six Sparks

This experimental profile serves Z.ai's GLM-5.3 using Local Inference Lab's
[`local-inference-lab/GLM-5.3-NVFP4-CSF`](https://huggingface.co/local-inference-lab/GLM-5.3-NVFP4-CSF)
checkpoint. It has CPU configuration tests and RTX 5090 collective emulation.
One installation on release 2026.10.4's image served requests with a KV pool
of 1,255,552 tokens, but the installer did not record it as complete
([record](../../performance/records/images/dev-20261013-dcp2prefill-status038-sircl-tp6-glm53-nvfp4-csf-tp6-ring8-20261014.md)).
No correctness or performance result is claimed.

The six ranks occupy positions 0-5 of an eight-Spark cycle. SIRCL carries the
collectives with NCCL off. The installer selects `SIRCL_ROUTE_SCOPE=fabric`:
TP, DCP and P2P lanes use the whole fabric's existing universal relay table.
The closing route is 5-6-7-0, with two relays; the longest peer route has three.
The transport receipt calls this `cycle-6` (six members on the eight-position
cycle), and uses the unmeasured generic cycle rules, never the measured cycle-8
row. The profile's placement topology remains `direct-path-6`.

No relay service is installed or changed by this selection. Positions 6-7
remain outside the model deployment but carry forwarded traffic, which can
contend with their serving traffic. CPU tests compare every lane against the
universal table's devices, addresses and tags. RTX 5090 emulation passes the
six-rank collective checks; neither establishes real RDMA behavior.

Because only six of the eight fabric positions host ranks, the existing
session rules disable chain/ring link areas and use pieces for large messages.
The link-based column-gather optimization is disabled for this profile
(`SIRCL_COLUMN_GATHER=0`), avoiding its per-call meta-tensor allocation and route
probes when neither link route can run. CPU six-rank Q/O-width gathers match
the enabled fallback bit for bit; no speed gain is measured. Other
profiles retain group-owned routing and their resolved settings.

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

The TP8-only latent and MTP projection speedups remain disabled: sharding
changes floating-point reduction order, and latent width 2624 is not divisible
by six. The new `glm53-tp6-resident-quantization` startup check requires all six
workers to expose 156 target Q/O MXFP8 methods, two draft Q/O MXFP8 methods,
one draft NVFP4 A16 expert method and an MXFP8 target head. Missing or truncated
resident metadata gives `unknown`; weight size is not quantization evidence.
This check inspects method metadata, not GPU execution or every linear layer.
Record the full loaded quantization inventory before qualification.

The CSF A4, scale-prefetch and barrier-reset checks remain enabled. The A4 check
counts distinct rank-0 layers. The old TP8 weight heuristics return `unknown`
when a different TP size is observed. Before benchmarking, also record draft
acceptance, every rank's weights, graph pools and allocated cache capacity.

## Image admission and installation

The default image of release 2026.10.4,
`dev-20261013-dcp2prefill-status038-sircl-tp6`, admits this profile
([release record](../../runtime/releases/dev-20261013-dcp2prefill-status038-sircl-tp6/README.md)).
It carries what the profile needs:

- a SIRCL 0.3.2 layer built with the universal route-scope setting
  (`SIRCL_ROUTE_SCOPE`);
- runtime-status 0.3.8, whose collector reports the resident quantization
  facts that the `glm53-tp6-resident-quantization` start check reads;
- the vLLM plugin `glm_dsa_indexer_split`;
- vLLM build `sparkring-kraken-bc9ea774-tp2-dcp2-prefill`, which reads the
  GLM-5.3-NVFP4-CSF checkpoint.

Release 2026.10.2's image has neither the route-scope setting nor the
resident check, and its lock does not list the profile.

```bash
sudo sparkring install --profile glm53-nvfp4-csf-tp6 --on 0-5 --no-auto-recover
```

The command is a future hardware procedure. Stop if the plan would displace a
deployment on positions 6-7 or alter the fabric. Six ranks over the universal cycle have no measured tuning row; generic cycle
rules apply until a bounded latency sweep supplies one.

## Registration

The research catalog and `direct-path-6` relayed topology provide automatic
installer, toolchain and fabric-adapter registration. There are no Compose
exports. Existing checkpoint pins, thinking records, display names, publisher
credit and storage planning apply unchanged. Its capacity record in
[profile-capacity.json](../../performance/profile-capacity.json) is the KV pool
of that installation. No recommended public install row is added.

## Hardware entry gates

No DGX hardware was contacted for these fixes. The future 90-minute window
still requires pre-staged images and weights. Before starting its clock:

1. Verify the rebuilt image's source receipts and `SIRCL_ROUTE_SCOPE=fabric`
   support, and check every TP/DCP/P2P lane against the saved installed table.
   Do not install a subgroup relay table. Preserve the deployment on 6-7.
2. Run eager and graph all-reduce, gather, scatter and P2P on the actual six-rank
   route at 12/24/28/32/36/64/72/144/288 KiB and 1/4/16 MiB. Verify reference
   outputs before timing. Record p50/p95/max and relay queue pressure. Only
   pieces schedules are legal with unhosted positions; do not relax the
   ring-tune qualification guard or record a physical six-cycle measurement.
3. Require all four startup checks to take effect. Inspect the 11-head and
   384-column shapes, MTP TP6, vocabulary tail masking and complete graph batch
   list. Measure per-rank loaded weights and peak allocated/reserved memory.
4. Run deterministic correctness and long-context retrieval before temperature-1
   throughput. Keep the original bounded sequence: preflight 0-5 min,
   collectives 5-12, start 12-32, correctness 32-40, benchmark 40-85,
   restore or leave the six-rank slot stopped at 85-90. Record incomplete cells.

Compare local352/A4-off versus local384/A4-on only in a later matched A/B window.
DCP3 is a separate capacity experiment; DCP6 needs both indexer split flags off
and its plugin removed. A single model fits only if measured startup peak plus
32 GiB KV leaves stable headroom; the 69-75 GiB weight estimate is not admission.
