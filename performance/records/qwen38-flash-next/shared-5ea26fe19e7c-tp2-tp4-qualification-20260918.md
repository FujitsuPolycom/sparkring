# Qwen QAD TP2/TP4 bounded qualification

Status: **qualified for the bounded workloads below**. The measured image is a local build, not a published pull selection. Profile image pins remain unchanged.

The ARM64 GB10 serving image `5ea26fe19e7c` contains the Kraken-based runtime and B12X sparse-attention changes from PR #394 at `7530ea27d7d923dbc71a76a356922bd8b6b1611a`. It is a labels-only child of GPU-tested runtime image `556f6c882e14`; the [qualification record](shared-5ea26fe19e7c-tp2-tp4-qualification-20260918.json) records complete IDs and artifact hashes.

The checkpoint is Qwen3.8-Flash-Next NVFP4-QAD revision `629bc3218833a38b475b719f34aa571666f4a03e`. Every arm uses 24 GiB KV per rank, 262,144-token context, 16 maximum sequences, 8,192-token batch budget, DCP1, MTP3, and media limits of three images and one video.

| Configuration | Startup KV tokens | Cold prefill 8K / 64K / 128K tok/s | Decode C1 / C8 tok/s |
|---|---:|---:|---:|
| TP2 + SparkCache | 2,877,721 | 3,214 / 3,045 / 2,827 | 47.6 / 223.8 |
| TP2 without SparkCache | 2,877,721 | 3,284 / 3,059 / 2,816 | 48.2 / 224.5 |
| TP4 + SparkCache | 3,131,214 | 3,863 / 3,649 / 3,328 | 68.2 / 266.9 |
| TP4 without SparkCache | 3,131,214 | 3,994 / 3,762 / 3,421 | 67.0 / 270.2 |

## Conditions and results

- Each arm independently passed two short exact-answer requests, midpoint retrieval at 70,705–70,728 and 141,725 actual prompt tokens, and a three-solid-color-image plus one-red-video request. Exact counts per arm are in the JSON.
- Prefill figures are medians of three samples per exact token count. Every request starts with a fresh nonce and has explicit zero cached-token credit. The synthetic archival-text fixture is not the historical decode-benchmark prompt.
- Decode figures are medians of three 10-second C1/C8 runs at 8K context, with a 512-token request limit, greedy sampling and loop detection. No failed or capacity-limited cells were accepted. The JSON includes acceptance length and normalized engine-step rates.
- Cache-enabled arms published two fixtures, restarted every worker without configuration changes, and restored both fixtures with exact answers and positive external restore on every rank: 5,696 tokens per fixture on TP2; 7,200 on TP4. Three additional decode runs after restart are retained separately.
- Cache-disabled arms have independent launch/output evidence, no connector in live Docker arguments, and no SparkCache metric series.

## GPU checks and limits

The unmodified upstream GPU selection is **21 passed, 4 failed**, not fully green. All four failures assert pool-independent program identity; two retained Triton validation/compression kernels still specialize physical pool size. The same four assertions also fail on the RC1 image without B12X PR #394; `gpu_rc1_program_key_control` records that control. A separate six-test supplement passes while explicitly checking the retained kernel source, rejecting unrelated program changes, and checking selection-program independence. The JSON includes SHA256 hashes for all three JUnit artifacts and the supplemental test source. This supplement does not turn the four upstream failures into passes.

These four-arm measurements do not isolate the contribution of B12X PR #394. The [matched saved-image comparison](shared-5ea26fe19e7c-tp2-tp4-comparison-20260918.md) records the whole-stack comparison and its TP4 prefill regression. First-start and post-restart observations are both retained; neither is discarded to improve the reported result.

Full 262K retrieval, C16 decode, prolonged cache pressure, arbitrary video understanding and request-level cache-salt isolation remain unqualified. The benchmark's automatic hybrid KV-capacity estimate is not used; capacity above comes from vLLM startup logs. Profile defaults remain unchanged.

## Reproduction and evidence availability

See the [exact-token cold-prefill methodology](../../methodology/exact-cold-prefill.md) and the [RC2 source description](../../../runtime/releases/shared-2026.09.0-rc.2/README.md). The portable probe reproduces the measured request/timing method; it was not the historical executed harness. Raw site artifacts are retained locally, not published with this record. The JSON preserves their hashes, numeric samples, correctness outcomes and qualification limits; hashes alone are not independently downloadable raw evidence.
