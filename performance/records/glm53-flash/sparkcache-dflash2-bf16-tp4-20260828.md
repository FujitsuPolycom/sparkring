# GLM-5.3 Flash DFlash2 TP4/DCP1 functional qualification

Status: **qualified** for the immutable OCI image, model revisions, four-rank
topology, serving geometry, and 8,192-token persistent restore described here.

## Conditions

- Four NVIDIA DGX Spark systems connected as a direct RoCE cycle; TP4, DCP1,
  PP1.
- SparkCache image
  `ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:cd4045bba2a0f3dc55361560f8c3a3f171939854db28d48dfdae58eed9c44943`;
  local image ID
  `sha256:7c007cf673c35f5818da7fea8faa343304baed00f489efdcbd027d6616b8a290`
  on every rank.
- Source-built parent runtime
  `ghcr.io/fujitsupolycom/sparkring-glm53-runtime@sha256:864adfe68f458223e186a19844ac80c7adc7365e5db1f25e109b85fc19850dcd`.
- Target checkpoint
  `local-inference-lab/GLM-5.3-Flash-NVFP4@520de24eabf507659eaef7c70f14fd584527facc`.
- BF16 draft checkpoint
  `incoai/GLM-5.3-Flash-DFlash2@dc77ff1c99eeb2df044ee3d4f0094eb033fee410`;
  seven speculative tokens and TP4 draft execution.
- vLLM `local-inference-lab/vllm@da4d7be6c97434f6942292ed8abbf4b32dc44355`
  and B12X `2fcf23a0ce269be27b2e03fece73d46e90e6aeea`.
- Source-built NCCL `v2.30.7-1` commit
  `73cf112295c33aee2b895f329f592f2a9b4b0f97`, patched tree
  `abdeb053b94c3f6d472cd55ae2b79ca821299009`, and loaded-library SHA-256
  `5f1c3f10d5ace66d4ba584415bbfe42b6ac1a0a9116a3b81dcbe50516ad924b3`.
- 524,288-token model limit, 8,192-token scheduler budget, 32 sequences,
  12 GiB FP8 GPU KV memory per rank, asynchronous scheduling, chunked
  prefill, native prefix caching, Triton KDA prefill, and
  `FULL_AND_PIECEWISE` target CUDA graphs.
- SparkCache maximum 48 GiB and low watermark 40 GiB per rank; native direct
  restore and streaming snapshots disabled.

## Measurement

The deterministic persistent request produced 8,215 prompt tokens with one
8,192-token reusable span. Every rank committed digest `a6d78b05026f...`.
All serving containers were replaced while rank-local cache directories were
preserved. Each worker discovered one local manifest with zero rejections.
One request recomputed while the scheduler received worker inventories. The
following identical request restored the aligned span. A separate semantic
request then verified continued generation.

The same image was also launched with no `--kv-transfer-config` argument. The
cache-disabled run used the same target, draft, scheduler, graph, KV-memory,
and transport settings.

## Result

| Measurement | Result |
|---|---|
| Persistent cache per rank | 1 manifest, 32 chunks, 103,890,664 bytes |
| Durable commit time by rank | 271.2, 271.2, 267.6, 313.8 ms |
| External-prefix queries / hits | 16,457 / 8,192 tokens |
| Restored request duration | 1.902 seconds |
| Restore time by rank | 162.7, 171.8, 158.8, 156.8 ms |
| Cache-enabled DFlash counters | 72 drafts, 504 draft tokens, 170 accepted tokens |
| Cache-enabled semantic canary | 2.840 seconds; `stop`; `semantic_match: true` |
| Cache-disabled semantic canary | 2.703 seconds; `stop`; `semantic_match: true` |
| Cache-disabled DFlash counters | 33 drafts, 231 draft tokens, 87 accepted tokens |
| RDMA state | 24 RTS `VLLM::Worker` QPs per rank in both profiles |
| Health | zero preemptions, restarts, OOMs, or fatal-log matches |

The equality `draft_tokens = 7 × drafts` held in both profiles. The
cache-disabled profile emitted no SparkCache connector logs and recorded zero
external-cache queries.

## Conclusion

The recorded image restores persistent GLM-5.3 target context after a
coordinated TP4 restart while BF16 DFlash2 continues correct seven-token
speculation. The same image also serves correctly when the external connector
is omitted.

## Limitations

- The measurements are functional single observations, not throughput,
  latency-distribution, or soak results.
- Restore evidence covers one 8,192-token span. Larger spans are unqualified.
- MTP drafting, other draft checkpoints, other topologies, native direct
  restore, and streaming snapshots are unsupported by this record.
- The source-built image uses stock safetensors loading and Triton KDA
  prefill. InstantTensor checkpoint loading and FlashKDA prefill are
  unsupported by this image.
- The optional `deep_ep` import reports a duplicate-NCCL warning. vLLM then
  selects `/opt/sparkring/nccl/libnccl.so.2` and serves successfully.
- The target repository does not identify its unquantized base-checkpoint
  revision.

## Provenance

[`runtime/glm53-flash/pins.json`](../../../runtime/glm53-flash/pins.json)
records source commits, trees, patches, model hashes, licenses, image digests,
SBOM hashes, and build-receipt identities. The request receipts are in
[`performance/receipts/glm53-flash/sparkcache-dflash2-bf16-tp4-20260828/`](../../receipts/glm53-flash/sparkcache-dflash2-bf16-tp4-20260828/).
