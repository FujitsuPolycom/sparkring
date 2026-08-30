# GLM-5.3 recurrent publication and exact prefix restore validation

Status: **qualified** for the bounded functional checks recorded below on the
exact local image. Restore timing is **research-only**. DFlash response quality
is **unsupported** by this record.

## Conditions

The final artifact was the local ARM64 image
`sparkring-glm53-sparkcache:dflash7-pr39-reaching-d93cb3d-arm64`, image ID
`sha256:ed60be066d6d9eadea267bc4597a0687869f3ddb95a3e5c6f86649893a838eb8`.
It had no published OCI digest. The image construction receipt had SHA-256
`2c4a02efe91df5de21c5e3c92f65710b7d41680f25c22baed43ca96c1e5a51d3`.
The builder created the image at
`2026-08-30T05:23:13.09226879-05:00` and completed that 8,690-byte receipt at
`2026-08-30T05:23:18.699254136-05:00`.

The artifact bound these source contracts:

| Component | Exact identity |
|---|---|
| SparkRing pull request | [#146](https://github.com/FujitsuPolycom/sparkring/pull/146) |
| SparkRing commit | `d93cb3d98305041081cf572521602625185112ae` |
| SparkRing tree | `867c43d0107856c3ba43500912462008ba149cc8` |
| SparkCache pull request | [#39](https://github.com/FujitsuPolycom/sparkcache/pull/39) |
| SparkCache commit | `65b6642df1afc64366430d3aef9aca01f5c5e1c3` |
| SparkCache tree | `41ad0a119ba109fd28900a2dcc9f9b4d8c293809` |
| SparkCache deployable-source SHA-256 | `a2add45a9f97446f6c2a843355161da9a5499ff7501b4750d2163591785d7345` |
| SparkCache vLLM contract SHA-256 | `8adbdfa3fd4b06b213c3aab45255a0b039f1c9940a4b1fad0efd004d263227c9` |
| SparkCache CUDA placement library SHA-256 | `d57509052b73853bcc8e3c3f47bb81748d87b9cbd8d908fc20d4c79a09aa400c` |
| vLLM retained compiled commit | `da4d7be6c97434f6942292ed8abbf4b32dc44355` |
| vLLM Python commit and tree | `0b67266a0f37d6146a8403fb8482403c62f412d5`, `ba9484ccb33aa56e90ff2f447f15ca9b9da97639` |
| B12X commit and tree | `b1d541f9e71a35f030d45fae437630fff7507c2a`, `c69cdec1c59a08e8e0e549f930fa8abcfb5134ae` |
| Recurrent-boundary patch SHA-256 | `5a6561a5bbab990dcd03bfd6a485ea26c3b5a578c2fd61b76305767b16dbfba0` |
| Recurrent-publication patch SHA-256 | `587fc332917a8ffd5a29712dc5253d51e6051eca1166ed4a165e576a84f2e300` |

Serving used four NVIDIA DGX Spark systems at TP4, DCP1, and PP1. The target
checkpoint was `local-inference-lab/GLM-5.3-Flash-NVFP4` revision
`520de24eabf507659eaef7c70f14fd584527facc`. The external DFlash checkpoint was
`incoai/GLM-5.3-Flash-DFlash2` revision
`dc77ff1c99eeb2df044ee3d4f0094eb033fee410`. The target used
fastsafetensors with queue size one; the draft used safetensors. Serving used
seven speculative tokens, FP8 KV, 256-token vLLM blocks, an 8,192-token
prefill batch limit, and 32 maximum sequences.

SparkCache used the requested `tail-cow-v1` publication schema, resolved to
`page-tail-cow-v1` for opaque GLM pages, canonical CUDA configuration, eight
CUDA restore I/O workers, and two load threads. The observation window was
2026-08-30 09:59:15 through 11:06:44 UTC.

The live restore observations used `block_pages_v1`. The C16 case restored one
exact 131,072-token prefix for requests with distinct tails. It did not use
`per_token_rows` or exercise different-root row-descriptor coalescing. That
coalescing behavior is implemented with GPU-free coverage only and receives no
live qualification from this artifact.

A second local artifact supplied corroborating evidence for the connector's
publication-target API before the scheduler explicitly limited publication
to the allocation that reaches the target. That image was
`sparkring-glm53-sparkcache:dflash7-pr39-boundary-8a887be-arm64`, image ID
`sha256:1b4e58dc0999292da34d7418688b2b7f745a5b4d06e048ceb19f06f9d63a1185`,
at SparkRing commit `8a887bebefa4bfcc0b47fc24de34a986b042fb29`. Its observations are labeled
as predecessor evidence below and do not qualify the final image.

The exact identities, raw per-rank fields, request observations, manifest
fields, scoped status, and limitations are retained in
[`validation.json`](../../receipts/glm53-flash/pr146-recurrent-publication/validation.json).

## Measurement

Image labels and construction fields came from the image receipt. A read-only
container-state snapshot checked the exact image ID, process state, restart
count, and OOM-killed flag on each rank after the clean relaunch. The
containers had no Docker healthcheck, so this record calls that check process
health rather than HTTP health.

The earliest observed replacement-container creation time was
`2026-08-30T10:46:32.079344046Z`; all four creation times and observed start
times are retained in the machine receipt. The exact construction and
source-contract receipt therefore existed before every observed replacement
container was created. No retained service-availability timeline proves that
the previous service remained continuously available, so this record does
not make that claim.

The request harness retained prompt and response digests, token counts, HTTP
outcomes, and client elapsed time. The semantic canary compared one fixed
response with its expected content. Client time is included only as a
single-run observation because the retained output does not specify an
independently audited clock or warm-up policy.

SparkCache emitted `sparkcache-restore-timing/v1` records on every rank. A
restore qualified only when every rank named the same request and span and
reported `outcome=verified`. The clean-restart case required a scheduler hit
for the persistent 8,192-token entry after all four containers restarted.

The 131,072-to-262,144 publication was inspected from the committed rank-zero
manifest. Its file SHA-256 was checked before parsing. The v2 manifest named
the committed 131,072-token context as its base and the 262,144-token context
as its result. It represented the appended 128K tail as 826,457,677 encoded bytes in 13
objects with a 64 MiB target size. The base remained a 512-object opaque page
root.

The shared exact prefix case sent 16 concurrent requests with one identical
131,072-token prefix and 16 distinct tails. All 16 prompt SHA-256 values were
different. During that cohort, each rank emitted exactly one restore shard
for request `cmpl-9ac4e6b523215233-0-9c41ccbc`, and no second external restore
or recurrent-publication warning was observed. One logical external restore
therefore supplied the shared exact prefix; its four log entries were the TP4 rank
shards of that operation, not four independent restores.

The predecessor artifact also ran an 8K prime and clean persistent replay,
one cold 128K flat restore, and identical-prefix waves at C2, C8, and C16.
Those measurements show the behavior of that exact artifact only.

## Result

| Check or observation | Result |
|---|---|
| Construction verification before container creation | exact receipt completed at `2026-08-30T05:23:18.699254136-05:00`; earliest replacement container created at `2026-08-30T10:46:32.079344046Z` |
| Final-image process state after clean relaunch | four ranks running the exact image; restart counts `0,0,0,0`; OOM-killed flags `false,false,false,false` |
| Semantic canary | semantic match; 27 prompt tokens; 45 completion tokens; 1.411 s client time |
| Clean-restart persistent 8K restore | 8,192 tokens and 103,841,965 bytes per rank; all ranks verified; 59.759-66.550 ms rank end-to-end; 1.573 s client time |
| Persistent 128K restore | 131,072 tokens and 813,068,464 bytes per rank; all ranks verified; 154.233-285.283 ms rank end-to-end |
| Tail-only 128K-to-256K publication | base 131,072; result 262,144; 826,457,677 delta bytes; 13 delta objects; 64 MiB target object size |
| Persistent 256K restore | 262,144 tokens; 1,575,821,491 page bytes; 1,024 logical chunks; all ranks verified; 6,688.887-7,206.548 ms rank end-to-end |
| Final-image C16 shared exact prefix | 16 distinct prompts; 16 HTTP 200 responses; one logical external 128K restore; 2.932669 s minimum, 4.242638 s p50, 4.244709 s p95/maximum client time |
| Predecessor clean-restart 8K | 8,192-token publication target; 59-67 ms rounded rank range; 1.744 s client time |
| Predecessor cold flat 128K | 512 opaque page objects; 3.39-4.15 s observed rank range |
| Predecessor identical-prefix C2 | 2 HTTP 200 responses; 4.667638-4.860512 s client range |
| Predecessor identical-prefix C8 | 8 HTTP 200 responses; 4.152364-4.925845 s client range |
| Predecessor identical-prefix C16 | 16 HTTP 200 responses; 4.393610-6.208569 s client range |

## Conclusion

The exact final image preserved semantic generation and completed verified
persistent restores at 8K after a clean restart, at 128K, and at 256K. The
committed 256K manifest referenced the existing 128K base and represented the
appended 128K tail with 13 delta objects, which is the expected tail-only copy-on-write
publication shape.

The exact construction and source-contract receipt preceded creation of all
four observed replacement containers. This establishes construction
verification before the observed replacement, but it does not establish
continuous availability of the previous service.

The final C16 shared exact prefix cohort completed 16 distinct requests after
one rank-sharded external restore of the common 128K prefix. This qualifies
exact prefix reuse for that one bounded cohort. It does not qualify
`per_token_rows` different-root row-descriptor coalescing and does not
establish a throughput or latency claim.

The predecessor C2, C8, and C16 waves corroborate the connector API behavior
before the reaching-allocation correction. They are not evidence for the
final image and are not used to widen its qualification.

## Limitations

- Every restore timing and concurrency value is a single observation or one
  wave. The values are research-only and do not establish variability,
  throughput, or soak behavior.
- The 128K opaque base still consists of 512 per-page objects. The appended
  128K delta is grouped into 13 objects, but flat opaque snapshots are not
  macro-grouped.
- The 256K restore read 1,024 logical chunks and took 6.689-7.207 seconds
  across ranks. This is a correctness result, not a speed claim.
- The final image has one C16 shared exact prefix wave. Different-root
  row-descriptor coalescing is implemented with GPU-free coverage only and was
  not exercised. The C2, C8, and C16
  identical-prefix matrix belongs to the predecessor artifact and cannot be
  transferred to the final image.
- The semantic canary checks one fixed expected response. No scored DFlash
  response-quality benchmark was run.
- The image has no published OCI digest and exists only on the observed
  deployment hosts.
- The retained timestamps do not establish that the previous service remained
  continuously available before or during replacement.
- `--prefill-schedule-interval 8` was not tested; the profile default remained
  in use.
- A complete command transcript and sanitized raw HTTP response bodies were
  not retained with this record.
