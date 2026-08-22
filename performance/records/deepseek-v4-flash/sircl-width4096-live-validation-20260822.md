# DeepSeek-V4-Flash-0731 width-4096 SIRCL live validation

Status: **research-only; live-validated on one four-Spark appliance**. The
machine-readable observations are retained in
[`sircl-width4096-live-validation-20260822.json`](sircl-width4096-live-validation-20260822.json).

## Conditions

Four directly cabled NVIDIA DGX Sparks ran
`deepseek-ai/DeepSeek-V4-Flash-0731` as TP4/DCP1 using the serving-image
manifest pinned by `runtime/faststart-lock.json`. The local image ID was
`sha256:50036224411e5ef04d651730c56d794111991b37981ec76ca81b66ea7d35dae7`.
The checkpoint revision and per-file hashes were not recorded.

The serving contract used BF16 weights, a 1,048,576-token model limit, 32
maximum sequences, a 4,096-token batch budget, 16 GiB of `fp8_ds_mla` key-value
memory per rank, block size 256, asynchronous scheduling with full-input-length
reservation, and DSpark with five speculative tokens and the B12X MoE backend.
Prefix caching and external key-value caching were disabled.

The candidate contract mounts two vLLM CUDA-capture overlays, the native SIRCL
library, and three SIRCL Python modules as read-only files. The JSON record binds the
SHA-256 of every mounted file. The graph contract admitted up to 512 query rows
at hidden width 4096. Five draft rows plus one target row at 32 sequences
requires at most 192 rows.

## Measurement

The API gate required HTTP health, the served model name and context limit, and
the exact deterministic output `SIRCL_OK_4096`. Per-rank graph status was read
after the API gate and after the completed C1/C8 load. A valid native replay
required all ranks to report captured nodes, advancing native sequences,
equal published/consumed/completed sequence values, caught-up replay, zero
overflow, no fatal state, and verified submit/progress affinity.

The sustained-decode harness was version 0.4.31 at Git revision
`ca84f366edfe9dc0648a4ac035b568356949602d`; the harness file was modified and
is therefore identified by SHA-256
`0df3cc642677c88796ccdd9a3c88f6f7b14971d21c77c1fc612c4594f6b5ff18`.
Concurrency levels ran in separate invocations. Each cell used an exact 2,048-
or 8,192-token fully unique context, ignored EOS, a hidden ten-second C1
warmup, and a 90-second measurement window. C1/C8 used temperature zero; the
completed C32 diagnostic used temperature 1.0. A separate cold-prefill sweep
used exact 8K, 64K, and 128K prompts with prefix caching disabled.

## Result

The API returned HTTP 200 and the expected deterministic completion. Every
rank captured 6,749 width-4096 graph nodes. After the completed C1/C8 load,
every rank reported 763,769 published, consumed, and completed native
sequences; overflow remained zero, replay was caught up, and no rank reported a
fatal state or out-of-memory termination.

The accepted sustained-decode observations were 64.0 tokens/s at C1/2K, 94.6
tokens/s at C1/8K, and 290.0 tokens/s at C8/2K. All accepted cells had zero
errors, no queue, complete concurrency admission, and at least 40 hardware
samples. The C8/8K receipt selected a 52.6 tokens/s client headline from two
partial request samples while the server generation-token delta was 271.0
tokens/s. That cell is retained as a diagnostic and rejected for performance
comparison.

The C32 temperature-1.0 receipt admitted all 32 requests with zero errors and
no post-warmup queue. Its server generation-token deltas were 463.6 tokens/s at
2K and 455.3 tokens/s at 8K. The client counted only 12,007 of 41,729 server
tokens at 2K and 16,363 of 40,984 at 8K, so its 133.6 and 182.2 tokens/s
headlines are rejected. The server rates are retained as diagnostic
observations rather than substituted headlines.

Cold-prefill throughput was 2,445 tokens/s at 8K, 2,364 tokens/s at 64K, and
2,172 tokens/s at 128K. Independent server-counter validation reported 2,460,
2,376, and 2,181 tokens/s respectively.

Three independent C1 temperature-1.0 repetitions reported 62.9, 97.1, and 94.4
tokens/s at 2K, and 94.8, 99.3, and 113.6 tokens/s at 8K. The means were 84.8
and 102.5 tokens/s respectively. Every cell used the Prometheus fallback, had
zero errors and no queue, admitted its requested concurrency, and retained at
least 40 hardware samples. The spread is retained rather than collapsed into a
single peak claim.

The TP2-aligned temperature-1.0 C32/16K run required 224.1 seconds to admit all
32 unique prompts, then measured for 240.0 seconds with all 32 running and no
queue. The server recorded 127,247 generated tokens, or **530.1 tokens/s**, and
a 74.84% speculative acceptance rate. The cell had zero errors, no readiness
timeout, no capacity rejection, and 91 hardware samples. The client retained
one partial request sample and counted only 11,108 tokens, producing a rejected
46.3 tokens/s headline. A 602.9 tokens/s ten-second server-log sample occurred
as teardown reduced the active count to 31; it is a peak diagnostic, not the
240-second result.

## Conclusion

The hash-bound width-4096 SIRCL candidate executed both target and DSpark CUDA
graphs, served a deterministic completion, and sustained C1/C8 load with native
replay advancing equally on four ranks and no overflow. This validates the
bounded execution path. It does not establish numerical equivalence or a
performance advantage over patched NCCL.

## Limitations

The checkpoint identity is not pinned. The mounted runtime outputs are bound by
hash, but their source-build receipt is incomplete. No matched patched-NCCL
control used identical request bytes and ordering. Raw harness receipts contain
deployment-specific addresses and remain maintainer-held; their hashes are in
the JSON record. The completed C32 client/server accounting disagreement
requires comparison of server-counter rates against the same field from a
matched control receipt; no such control receipt is included here.
