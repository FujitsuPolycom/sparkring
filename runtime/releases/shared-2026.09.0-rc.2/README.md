# Sparse-attention serving candidate: 2026.09.0-rc.2

Availability: **local build; not a published image selection**. Public profile
image defaults are unchanged.

The [source manifest](sources/manifest.json) reconstructs the vLLM and B12X
inputs for the ARM64/GB10 serving candidate. It retains the Kraken-based vLLM
composition and incorporates the sparse-attention selection implementation from
[B12X pull request 394](https://github.com/local-inference-lab/b12x/pull/394),
revision `7530ea27d7d923dbc71a76a356922bd8b6b1611a`.

| Artifact | Docker configuration ID |
|---|---|
| Runtime build | `sha256:556f6c882e14afdc51158ab20837ab704ed2a6bfcc5c9e9ef478c4773916c133` |
| Labels-only serving image | `sha256:5ea26fe19e7cc17f68c9adac0501c1639e87882614fc9e102bdd35c959b94867` |

These configuration IDs are not public registry manifest digests and cannot be
substituted into a GHCR pull command. The source manifest is not a publication
receipt. The [published RC1 candidate](../shared-2026.09.0-rc.1/README.md) retains
its own immutable inputs and narrower qualification record.

Qwen TP2/TP4 cache-on and cache-off trials passed bounded text, retrieval and
media checks; cache-enabled trials also passed all-rank restart/restore checks.
Matched saved-image controls show topology-dependent throughput trade-offs,
including lower TP4 cold-prefill throughput. No blanket performance improvement
or profile-default promotion is claimed.

The upstream GPU selection recorded 21 passes and four program-identity failures.
Those four failures also occurred on RC1. A six-test compatibility supplement
checks the retained kernel identities and selection behavior; it does not turn
the upstream failures into passes. Request-level SparkCache salt isolation,
full-context/C16 throughput and arbitrary video accuracy are not qualified by
these trials. A separately developed salt-isolation fix is not in this image.

## Recorded measurements

- [Qwen TP2/TP4 with and without SparkCache: bounded qualification](../../../performance/records/qwen38-flash-next/shared-5ea26fe19e7c-tp2-tp4-qualification-20260918.md)
- [Matched saved-image comparison and promotion hold](../../../performance/records/qwen38-flash-next/shared-5ea26fe19e7c-tp2-tp4-comparison-20260918.md)
- [Portable exact-token cold-prefill reproduction](../../../performance/methodology/exact-cold-prefill.md)

The records retain every numeric sample and identify locally retained raw
artifacts by hash. They do not imply those private site artifacts are publicly
downloadable or that this image has been published to a registry.
