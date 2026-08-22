# Rebuilt GLM-5.2 3.5-bpw image: clean-checkout bring-up

Status: **implemented**; not qualified.

## Conditions

The evidence covers one image built by `runtime/exl3-r7/build-image.sh` from a
clean SparkRing checkout, placed on four directly cabled NVIDIA DGX Spark
systems, and started under a generated exact 40-query-row operator profile.

| Attribute | Value |
|---|---|
| SparkRing revision | `c1a520027b502faa15ad29cd41f59dacfefea2a4` |
| Parent image manifest | `sha256:6fc26fdad81a18f0fff67ce0a05f6d90165625ea2e1cac8a6f39bfb462017028` |
| Parent image ID | `sha256:50036224411e5ef04d651730c56d794111991b37981ec76ca81b66ea7d35dae7` |
| Rebuilt image ID | `sha256:5569c4778c9561a8595ac283c7adf31e22be1d35517aa208569af5224244a2da` |
| Model | `brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78` |
| Model revision | `9ab9579774cc432df91567a36f6e9e863e0d4c9f` |
| Model config SHA-256 | `fabb73eb513ec64f3a365da396b38de8d55b3930edfb11baeecbf34ecafa6126` |
| Model index SHA-256 | `9fd852f69ed64442e31dce1cbc5fe7acd0a76bfb848e945d272fe98d00d0c9cd` |
| Rank layout | physical ranks 0–3 on one four-edge cycle; TP4/DCP4 |
| DCP policy | `ag_rs`, interleave one |
| KV representation | dynamic NVFP4 latent with FP8 RoPE |
| Speculation | fixed MTP4 |
| Traffic scope | startup health and model discovery only |
| Cache state | not recorded; no cache result is claimed |

The four hosts and their site addresses are intentionally anonymous. The
immutable image and model identities are retained because they define the
evidence boundary without identifying the site.

## Measurement

The operator performed these gates against the image ID above:

1. built the image with `runtime/exl3-r7/build-image.sh` from the recorded
   checkout;
2. ran `scripts/download_exl3_r7.py verify` against the pinned model revision,
   hashing every indexed shard against its published LFS SHA-256;
3. generated the dynamic-NVFP4, compressed-KV gather, SIRCL, and exact
   40-query-row profile layers from tracked scripts;
4. ran `scripts/preflight.py` against the resolved four-rank site;
5. launched all four ranks and required the 17-entry in-container runtime
   attestation on every rank;
6. waited for model loading, exact-state receipt generation, piecewise/full
   graph capture, application startup, `/health`, and `/v1/models`.

Worker logs supplied weight bytes and load durations. The reported duration is
the minimum-to-maximum range across four workers; no average or variability
estimate was computed. That range is a single-run startup measurement, not a
qualified performance distribution. Graph capture and startup completed before
API health was accepted. No serving throughput, request latency,
deterministic-output, concurrency, or acceptance workload was measured.

The raw site, preflight, profile, attestation, and worker-log receipts remain
outside the public repository. Their public derivatives are the hashes and
counts below. That retention boundary prevents independent recomputation and
limits this record to implemented bring-up evidence.

Raw-record location: unavailable in a public or immutable artifact archive;
only the sanitized identities, counts, and observations in this document were
retained for public review.

## Result

Checkpoint verification returned `status: pass` for 157 runtime shards and
346,218,639,128 runtime weight bytes, equal to the pinned model index total.

| Generated artifact | SHA-256 |
|---|---|
| Complete pre-exact-state profile | `4f64a798307737446553619562f9a70b87e31bbb684df2ead7d58f2e23cb92bf` |
| Exact 40-query-row serving profile | `735e2ccad4ef17e9dc9a9d21cbcc888e9d4d0cb5b80227e5ff8a1d1fce919eca` |
| Exact-state manifest | `7e1f4d553c10f154c7e3748a45637469216e4e9cd1e01eefe7a049309c6490e4` |
| Exact-state `exl3.py` overlay | `8fad5330c88f55dc57e4d8e298f2af23e16390b97153b569a2e572e0fb5065c2` |
| Exact-state `model_runner.py` overlay | `2c2b79326e00fb1ae7494a4921b13300a75f4dcf6aa7e9943db2e9e5b63f7a23` |

The resolved site passed 116 of 116 preflight checks. The in-container runtime
attestation matched 17 of 17 entries on every rank.

| Bring-up observation | Result |
|---|---|
| Weights loaded per worker | 87.54 GiB |
| Four-worker load-duration range | 217.9–238.0 seconds |
| Mixed-Trellis structure | layer-dependent three- through five-bit tiers |
| SIRCL eager sessions | ready on all four ranks for captured payload sizes |
| CUDA graph capture | 40 query-row sizes, piecewise and full |
| Exact-state receipt | written once per rank |
| `/health` | HTTP 200 |
| `/v1/models` identity | `glm-5.2-exl3-r7-3.5bpw` |
| `/v1/models` maximum length | 262,144 tokens |

## Conclusion

The clean-checkout builder, checkpoint verifier, profile generators, preflight,
runtime attestation, and four-rank launch path are **implemented** for the
recorded image and model identities. The result does not promote the rebuilt
image to qualified status and does not transfer performance or output-quality
claims from another image.

## Limitations

- The rebuilt image has not run the fixed-seed equivalence and bounded C1/C2/C8
  procedure in
  [`GLM52_35BPW_ACCEPTANCE_RUNBOOK.md`](../../../docs/GLM52_35BPW_ACCEPTANCE_RUNBOOK.md).
- Post-workload rank and transport health has not been recorded.
- Raw receipts are not public, so the public record cannot independently
  recompute the counts or duration range.
- One build and bring-up ran on one four-Spark appliance; there is no repeat or
  portable hardware claim.
- Native transport tests passed 18 of 18 CTest targets on one GB10 host, not on
  the four-rank serving path.
- No serving-throughput, request-latency, acceptance-rate, cache, or
  output-quality result is claimed.
