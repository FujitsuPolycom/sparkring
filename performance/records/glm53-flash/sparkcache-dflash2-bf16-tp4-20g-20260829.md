# GLM-5.3 Flash 20 GiB KV live observation

Status: **research-only** for the exact public image, model revisions, and
TP4/DCP1 configuration described here. The qualified 12 GiB profile remains
the public reproduction contract.

## Conditions

Four NVIDIA DGX Spark systems served the public SparkCache image
`ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:cd4045bba2a0f3dc55361560f8c3a3f171939854db28d48dfdae58eed9c44943`
with image ID
`sha256:7c007cf673c35f5818da7fea8faa343304baed00f489efdcbd027d6616b8a290`
on every rank. The service used TP4/DCP1/PP1, the pinned target and BF16
DFlash2 revisions, 20 GiB of FP8 GPU KV memory per rank, a 524,288-token model
limit, 8,192 batched tokens, 32 sequences, and the 48/40 GiB SparkCache
maximum/low-watermark policy.

## Measurement and result

| Measurement | Result |
|---|---|
| Measured KV capacity | 916,676 tokens |
| Maximum 524,288-token concurrency | 1.75× |
| Model load / graph capture | 443.994 / 49 seconds |
| Semantic canary | 0.997 seconds; correct |
| Persistent restore | 8,192 tokens; request 2.242 seconds |
| Restore time by rank | 168.8, 160.1, 190.7, 172.2 ms |
| DFlash invariant | 51 drafts; 357 draft tokens (`51 × 7`) |
| CUDA allocation | 82,479 MiB per rank |
| Host `MemAvailable` | 23.2 GiB on the limiting rank; 27.1–27.9 GiB elsewhere |
| RDMA and health | 24 RTS worker QPs/rank; zero preemptions, restarts, OOMs, cgroup OOM events, or fatal matches |

The first persistent request after startup recomputed while worker inventories
reached the scheduler. The following identical request restored the reusable
span. All target, DFlash, and DFlash2 CUDA graph captures completed.

## Conclusion

Twenty GiB per rank is operational at the 524,288-token model limit. The
measured 916,676-token KV capacity cannot hold one 1,048,576-token request.
Linear capacity scaling projects 24 GiB as a practical 1M candidate, but that
configuration is not tested by this record.

## Limitations

- The cache-disabled profile was not tested with 20 GiB.
- A 1,048,576-token model limit and KV sizes above 20 GiB were not tested.
- This is functional capacity evidence, not a throughput or soak result.
- Rank-local SparkCache storage remains configured for a 48 GiB ceiling; the
  limiting rank has insufficient ordinary disk headroom for that ceiling plus
  a comfortable filesystem reserve.

The sanitized machine-readable receipt is
[`performance/receipts/glm53-flash/sparkcache-dflash2-bf16-tp4-20g-20260829/observation.json`](../../receipts/glm53-flash/sparkcache-dflash2-bf16-tp4-20g-20260829/observation.json).
