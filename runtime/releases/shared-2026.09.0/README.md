# SparkRing shared runtime 2026.09.0

The ARM64/GB10 serving image contains the Kraken-based vLLM runtime, B12X #394,
Qwen multimodal HC routing, a pre-API startup audit and SparkCache. The separate
SGLang runtime remains included; its presence does not qualify a model/profile.

```bash
docker pull ghcr.io/fujitsupolycom/sparkring@sha256:8cfcfdaffd91af252c0eef2f46d325765f364d3ac812ee7f76343fa2393dd357
```

The readable tag is `ghcr.io/fujitsupolycom/sparkring:shared-2026.09.0`.
Quickstarts pin the immutable digest. The [publication record](publication.json)
binds that digest to image configuration `a6a2a5b35e77`, source hashes and an
anonymous Docker pull. Runtime layers/configuration are unchanged from the
functionally checked image `462b65ac8629`; only release labels differ.

## Qwen deployments

| Topology | Without external cache | With SparkCache |
|---|---|---|
| Two Sparks | [TP2 quickstart](../../../profiles/qwen38-flash-next-tp2/README.md#cache-disabled-alternative) | [TP2 quickstart](../../../profiles/qwen38-flash-next-tp2-sparkcache/README.md) |
| Four Sparks | [TP4 quickstart](../../../profiles/qwen38-flash-next-qad-tp4/README.md) | [TP4 quickstart](../../../profiles/qwen38-flash-next-qad-tp4-sparkcache/README.md) |

All four select the QAD checkpoint revision
`629bc3218833a38b475b719f34aa571666f4a03e`, DCP1, 262,144 context, C16 scheduler
limit, batch8192, MTP3 and 24 GiB FP8 KV per rank. Media limits are three images
and one video with 16 configured frames. TP4 enables HC row sharding; TP2 does
not. Both enable Qwen checkpoint coalescing, compact MTP and projection overlap.

The [qualification record](qualification.json) states the bounded text and
synthetic-media checks, including observed TP4 HC execution on every rank.
Configured limits are not claims of tested full-context/concurrency capacity.
No benchmark results or performance claims are published here.

Other models' public image pins are unchanged. This release does not promote
GLM, DeepSeek, DCP alternatives or arbitrary multimodal workloads.

## Startup checks and limitations

`SPARKRING STARTUP AUDIT` appears before HTTP readiness. It distinguishes
requested flags and reachable source paths from observed runtime execution,
warns about disabled/incompatible optimizations and never changes settings.
An eligible real TP4 prefill separately logs `QWEN_HC_PREFILL mode=shard`.

SparkCache uses aligned checkpoints and a source-matched native lease contract.
Request `cache_salt` does not isolate persistent entries in this image; use
separate deployments/cache namespaces for tenant isolation. The separate
SparkCache salt-isolation change is not included. The API has no configured
authentication; restrict it to trusted clients or an authenticated gateway.

Full-context pressure, C16 multimedia, arbitrary video accuracy and long-running
store pressure require separate qualification. Retained SGLang-container restart
failures remain unresolved; fresh-container success does not establish a fix.
The retained B12X sparse-attention kernels also have unresolved upstream
program-identity assertions involving pool-size specialization. Component
compatibility checks and bounded model checks do not constitute a blanket
upstream GPU-suite pass.

## Sources and rollback

The [source manifest](sources/manifest.json) and complete patches independently
reconstruct the exact vLLM/B12X source trees. [Component provenance](components.md)
identifies inherited source, notices and licensing. Source hashes are not a claim
that every inherited foundation dependency can be rebuilt offline.

Keep the previous image, checkout/deployment plan and cache namespace when
upgrading. Stop only the replacement deployment before restoring its saved
predecessor. Existing releases and their hashes remain immutable; the
[RC1 selection](../shared-2026.09.0-rc.1/README.md) and prior R37 releases remain
available for reproduction. Do not reuse a different image's cache-source lease.
