# Qwen saved-image performance controls

Status: **qualified bounded measurements; profile promotion held**. These are whole-stack comparisons, not an isolated measurement of B12X PR #394.

Each control uses the same checkpoint, topology, 24 GiB KV pin per rank, 262K context, C16 scheduler limit, 8192 batch budget, MTP3 and SparkCache-enabled mode as its candidate. Saved-image engine settings are retained except the explicitly matched capacity, graph and media options; source/runtime components differ. Writable compiler and persistent caches are isolated. Both execute media, short correctness, midpoint needles, three decode repetitions, nine cold-prefill requests, publication, worker restart, restore and three further decode repetitions.

| Topology | Saved image | Prefill change 8K / 64K / 128K | Initial decode change C1 / C8 | Post-restart decode change C1 / C8 |
|---|---|---:|---:|---:|
| TP2 | `5bd9d05327d4` | -1.2% / +7.9% / +6.8% | +7.2% / -0.1% | +7.6% / +3.9% |
| TP4 | `b03062b032bb` | -17.5% / -7.5% / -4.0% | +4.1% / +4.7% | +8.1% / +3.7% |

Changes are candidate/control minus one, using the median of three repetitions. Positive means faster. The [comparison record](shared-5ea26fe19e7c-tp2-tp4-comparison-20260918.json) retains every sample, engine-step rates and acceptance lengths.

Cold-prefill uses exact token counts and requires zero cached-token credit on every request. Leading request nonces differ; request cache-salt is not used as a cold-control guarantee. Decode allows warm prefix reuse. The synthetic archival-text prompt is not interchangeable with historical benchmark-prefill prompts.

Run ordering is identical, including long needles before measured prefill. Saved-image compiler caches start empty; candidate compiler caches include earlier qualification. First-start and post-restart measurements are retained separately, and no inconvenient sample is deleted. The observed variance and three-sample budget do not support a broad statistical or patch-specific speedup claim.

The candidate passes bounded correctness and restart/restore checks, but its TP4 prefill loss prevents quickstart promotion. Investigating that loss is separate from this qualification record. Full-context/C16 throughput, prolonged cache pressure, arbitrary multimedia accuracy and request cache-salt isolation remain unqualified.

## Related evidence

The [four-configuration qualification](shared-5ea26fe19e7c-tp2-tp4-qualification-20260918.md) records the candidate image identity, bounded correctness and GPU limitations. See the [exact-token cold-prefill methodology](../../methodology/exact-cold-prefill.md) and [RC2 source description](../../../runtime/releases/shared-2026.09.0-rc.2/README.md). Raw site artifacts remain local; the JSON's private-source hashes identify those inputs rather than asserting they are publicly available. The public candidate record is linked separately from the original private report hash.

Saved-image provenance is recorded in the [source-built image record for
`5bd9d05327d4`](r37-source-prefill-5bd9d05327d4.json) and the [combined-image TP4
record for `b03062b032bb`](combined-image-tp4-sparkcache.json). Those records have
their own workload conditions; their historical throughput is not substituted
for the matched controls measured here.
