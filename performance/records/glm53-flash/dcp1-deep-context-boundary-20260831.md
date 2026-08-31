# GLM-5.3 Flash TP4/DCP1 deep-context boundary

**Status: research-only.** This record binds a single-sequence synthetic
needle-retrieval walk to one exact Linux/ARM64 image and four GB10 systems. It
does not establish throughput, concurrency, or behavior for another artifact.

## Runtime

- Image ID: `sha256:77da063d1d51fa181eb39e519dda7c5ae4eb59a47e169cb4c33bd2cd42120225`
- vLLM revision: `55969c16d4da57da76ee5729f3102d4b2003833c`
- SparkCache revision: `65895c87a6d925fcd270fcea202ab847d7fbc2d1`
- Topology: four GB10 systems, TP4/DCP1/PP1
- Request limit: 1,048,576 tokens
- Prefill scheduler budget: 8,192 tokens
- KV format: FP8
- Speculation: DFlash2 with seven speculative tokens
- Publication: SparkCache `snapshot-v1`

The probe is `scripts/niah_boundary_probe.py` from SparkRing pull request 152
at commit `ee02757809eaa6d9bf3de6f33f7b65d30e4e268f`. It constructs an approximate
four-characters-per-token archive, places an eight-digit code at 50%, and
requires the model to return that code. Counter-based cancellation was disabled
because this runtime updates the global prompt-token counter at completion
rather than during each prefill chunk.

## Results

| FP8 KV per rank | Requested depth | Actual prompt tokens | Time | Result |
|---:|---:|---:|---:|---|
| 41 GiB | 350,000 | 323,575 | 141.0 s | Needle returned |
| 41 GiB | 375,000 | 347,167 | 154.5 s | Needle returned |
| 41 GiB | 400,000 | 370,753 | 103.6 s | Needle returned |
| 41 GiB | 450,000 | — | 263.1 s | Rank-0 worker OOM |
| 30 GiB | 450,000 | 418,237 | 189.0 s | Needle returned |
| 30 GiB | 600,000 | 560,292 | 266.4 s | Needle returned |
| 30 GiB | 750,000 | 703,697 | 346.6 s | Needle returned |
| 30 GiB | 900,000 | 847,470 | 430.1 s | Needle returned |
| 30 GiB | 1,000,000 | 936,960 computed | 469.8 s | Rank-0 and rank-1 workers OOM |
| **26 GiB** | **1,000,000** | **942,767** | **478.1 s** | **Needle returned** |

The 26 GiB allocator exposed 1,303,701 KV tokens, or 1.24 times the configured
maximum request length. During the successful prefill, available host memory
remained near 23.7 GiB on rank 0 and 25.3–25.6 GiB on ranks 1–3. No swap was
configured, no kernel OOM record appeared, and all four containers remained
healthy.

## SparkCache publication

The successful request published a 942,592-token snapshot with digest prefix
`28d1ab9295fd`.

| Rank | Snapshot | Background commit | Manifest prefix |
|---:|---:|---:|---|
| 0 | 16.642 s | 13.320 s | `c81f8dc59c77` |
| 1 | 16.671 s | 13.070 s | `74ccd2ec93d2` |
| 2 | 16.961 s | 12.032 s | `5c57f20173ec` |
| 3 | 17.894 s | 14.509 s | `f6b8fd12f693` |

The four rank objects consumed 23,468,277,760 bytes in aggregate. The snapshot
was not replayed after process replacement. GPU-to-host capture overlapped
unrelated inference and is tracked separately in
[`FujitsuPolycom/sparkcache#45`](https://github.com/FujitsuPolycom/sparkcache/issues/45).

## Limits

- The 26 GiB result is one synthetic prompt with one seed.
- Requested depth is approximate; the API-reported token count is authoritative.
- The walk covers DCP1 and one 8,192-token scheduler budget.
- It does not test concurrent deep prefills, restart restore of the 942,592-token
  snapshot, tail-only publication, sustained serving, or fault recovery.
