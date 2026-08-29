# GLM-5.3 Flash DFlash2 SparkCache restart-and-restore record

Status: **qualified** for the exact artifacts, four-rank topology, serving
geometry, cache policy, and 8,192-token reusable span recorded here.

## Conditions

- Four directly cabled NVIDIA DGX Spark systems in a switchless cycle.
- TP4, DCP1, and PP1.
- Target `local-inference-lab/GLM-5.3-Flash-NVFP4@520de24eabf507659eaef7c70f14fd584527facc`.
- BF16 drafter `incoai/GLM-5.3-Flash-DFlash2@dc77ff1c99eeb2df044ee3d4f0094eb033fee410`.
- vLLM `local-inference-lab/vllm@da4d7be6c97434f6942292ed8abbf4b32dc44355`.
- SparkCache source-tree SHA-256
  `6210f439c64e4079ed3304c9cc181174abb3e6045de740ba7b7c2546bcaf6ac2`.
- Seven-file vLLM lease-contract SHA-256
  `2e3b17fd6a34f2dbb8e91a99b83dbf18629cf0e718f9f814236da4bbfc9ae3f1`.
- Model limit 524,288; scheduler budget 8,192; 32 sequences; 12 GiB FP8
  key-value memory per rank; block size 256.
- Asynchronous scheduling, native prefix caching, chunked prefill, aligned
  recurrent cache, FlashKDA prefill, `FULL_AND_PIECEWISE` target graphs, and
  vLLM-selected DFlash FULL graphs.
- Dedicated rank-local cache roots with streaming snapshots and native direct
  restore disabled.
- Qualification request program SHA-256
  `1c3e5d2c8471173b3016e54e97e8218edff73a0ee0c910becd208e2a6d1d0c0b`;
  deployment orchestrator SHA-256
  `aa568cf7daba1ed8bf62204bc79f88dfc829a0ea495b635d3c46d718651368c8`;
  all-rank verifier SHA-256
  `f04f2dbb540f1ff063531be7e0de075a5b195909324c43040a076052a0bda69e`.

The four rank-local parent and derived image IDs were:

| Rank | Parent image ID | Derived image ID |
|---:|---|---|
| 0 | `sha256:ddd13fb1ea8ca61aaf771715dc8c5a52dfe6860f0cc62c145d155916bf381fc9` | `sha256:56f051b1b1b6f9f858ea5d21b7933b64af81c22bee2c417a3f8b4466220e37e6` |
| 1 | `sha256:7fb81337ba088a6bf0bbce71b22a5881f812a21af9ac1d6deea9533a8e9eed92` | `sha256:8506935b369bd4f0d5d73495ded9a2fcb52bbe2f310ea093818e5d3d5366ae38` |
| 2 | `sha256:9bd97e3d77de969ee0788aaac31b2888fd4c6a3d893ac5fc544ca85363927935` | `sha256:b969a49ec091157c686a3bc3f52816b6aa910e495af0c92780a321ea5fbd5324` |
| 3 | `sha256:d592c83cc04106532adf7d8d410347062ac1b80fc1b6981deca414b5335efff4` | `sha256:c9f0be4dccfd8fdcec80a3edce1ad217604fa09afee0f14d13a2839fb97eed9f` |

## Measurement

The persistent request repeated `benchmark` 8,192 times, appended
`Request 0: summarize the repeated prefix briefly.`, used temperature 1.0,
allowed 64 completion tokens, and used one client request at a time. It
produced 8,215 prompt tokens with an 8,192-token reusable aligned span. The
semantic request used a separate 27-token prompt, temperature zero, a
256-token completion limit, and disabled thinking. All ranks stored the same
deterministic reusable span in a fresh namespace. The four
serving containers were replaced without removing their rank-local cache
roots. Every rank then reported `checked=3 offered=3 rejected=0`. An identical
request was repeated until the scheduler had received a complete all-rank
inventory checkpoint and formed restore quorum. A separate uncached semantic
canary followed the restore. Process, log, speculative-decoding, and RDMA state
were inspected after generation.

The run population was one cold store request, one first request after restart
that safely recomputed, one following request that restored, and one uncached
semantic canary. The client measured end-to-end request duration with Python's
monotonic `time.perf_counter`. SparkCache rank logs measured each restore
operation. Values below are individual observations; no averaging, percentile,
or variability estimate is reported for the single-run functional gate.

The public request commands were:

```bash
python deploy/glm53_flash/qualification_request.py \
  --endpoint http://<rank-0-address>:8015 \
  --model glm-5.3-flash-nvfp4-dflash7-bf16-tp4 \
  --kind persistent --output <persistent-receipt>
python deploy/glm53_flash/qualification_request.py \
  --endpoint http://<rank-0-address>:8015 \
  --model glm-5.3-flash-nvfp4-dflash7-bf16-tp4 \
  --kind semantic --output <semantic-receipt>
```

Site-resolved addresses and host paths are intentionally omitted. The four
sanitized request receipts are in
[`performance/receipts/glm53-flash/sparkcache-dflash2-bf16-tp4-20260828/`](../../receipts/glm53-flash/sparkcache-dflash2-bf16-tp4-20260828/).

## Result

| Measurement | Result |
|---|---|
| Cold snapshot time by rank | 88.8, 125.2, 181.1, 149.0 ms |
| Cold durable-commit time by rank | 214.3, 222.5, 222.0, 209.0 ms |
| Durable cache per rank | 3 manifests, 77 chunks, 284,880,209 bytes |
| External-prefix queries | 16,457 tokens |
| External-prefix hits | 8,192 tokens |
| Restored request duration | 1.509 seconds |
| Restore time by rank | 155.6, 147.2, 194.0, 151.8 ms |
| DFlash drafts / draft tokens | 43 / 301 (`43 × 7`) |
| Accepted draft tokens | 112 |
| Semantic canary | 1.176 seconds; finish reason `stop`; suffix `SPARKCACHE_GLM53_OK` |
| Scheduler preemptions | 0 |
| RDMA state | 24 RTS `VLLM::Worker` QPs per rank |
| Process health | 0 restarts, OOMs, or fatal-log matches |
| Target artifact | ranks 0, 2, 3: strict 59/59 files; rank 1: the same 59 files plus `.cache/huggingface` metadata |

The first request after API startup recomputed because all-rank inventory had
not reached the scheduler. The following identical request formed quorum and
restored. Recompute is the configured fail-closed outcome when a restore
cannot be proved safe.

## Conclusion

The exact composition restored persistent target-model context after a
coordinated engine restart, maintained DFlash's seven-draft-token invariant,
and continued correct generation.

## Limitations

- Another parent or derived image is not covered by the four recorded pairs.
- The result covers an 8,192-token restored span and does not establish
  larger-span latency or throughput neutrality.
- The result does not cover streaming snapshots, native direct restore, MTP,
  another checkpoint, another topology, or another scheduler/cache geometry.
- Full reasoning-trace equality is not used as a semantic oracle. The gate
  requires final-answer suffix, finish reason, continued generation, all-rank
  restore, and fail-closed counters.
- The target repository does not record a base-checkpoint revision.
- Rank 1 passed verification of all 59 expected target repository files but
  failed strict no-extra-files mode because `.cache/huggingface` metadata was
  present. A no-extra-files claim applies only to ranks 0, 2, and 3.
- The loaded NCCL binary is pinned by SHA-256, but its receipt does not bind it
  to an NVIDIA NCCL source commit and complete patch-build record.
- Rank logs, manifest files, metrics snapshots, and the site-resolved command
  were not retained in this public repository. The four request receipts and
  aggregate qualification summary cannot independently reproduce every
  reported counter.

## Provenance

The full component lineage, license constraints, source hashes, verified
vLLM commits and pull requests, and adaptation-source hashes are recorded in
[`runtime/glm53-flash/pins.json`](../../../runtime/glm53-flash/pins.json).
No base-checkpoint or binary-build lineage beyond that manifest is inferred.
