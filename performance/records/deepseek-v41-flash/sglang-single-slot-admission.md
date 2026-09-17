# Single-slot admission for DeepSeek-V4.1 SGLang

Status: bounded scheduling evidence on a four-GB10 direct cycle, measured
2026-09-17. This is not a soak, long-context benchmark or general quality test.

The comparison used combined image
`sha256:b03062b032bb147f5255463d4f068bfc476bf966c78c82ddeb66df4fb1b0b37d`,
SGLang revision `e087e662ba1ac4ef7747537e2a9141085efd4561`, Mia adapter
`e59e6eb67479aa68f6fa700c600dc90a0729b5ec`, and stock DeepSeek-V4.1-Flash
checkpoint `dba1be0a40aa45a94ad051997016db3960a90277`. The combined image includes
the #39187 memory and #39068 execution backports; it is not the original
public source-built image. Settings were TP4/EP4, context 655360, chunk 4096,
maximum running requests 8, requested KV 1500000, DSpark block 5, and packed
NVMe Engram with 96 I/O threads and no row cache.

Only `--min-free-slots-delay 1` changed between arms. In the installed SGLang
source, automatic DFlash-family admission requires two free slots at C8;
explicit value 1 disables this delay. The contributor reported this setting
in [PR #279](https://github.com/FujitsuPolycom/sparkring/pull/279).

Each arm had one excluded warmup and three measured waves. Seven requests
started together and an eighth arrived 250 ms later. Every request asked for
256 output tokens, ignoring EOS, with temperature 0, seed 267279 and thinking
disabled. Matching prompts requested a numbered list of arithmetic facts.

| Three-wave median | Automatic delay | One-slot admission |
|---|---:|---:|
| Completion tokens divided by whole-wave wall time | 241.94 tok/s | 345.83 tok/s |
| Eighth request time to first content | 5.366 s | 0.246 s |
| Worst first-content wait per wave | 5.366 s | 0.506 s |

Measured wave throughput samples were 241.94/240.86/258.91 tok/s for automatic
delay and 345.83/338.73/357.56 tok/s for one-slot admission. Eighth-request waits
were 5.366/5.392/4.789 seconds and 0.255/0.246/0.244 seconds, respectively.
All 64 requests, including warmups, completed at the requested output length.
Ten subsequent health, authentication, arithmetic and structured-JSON checks
passed. The observed last-slot stall was reduced and end-to-end throughput
increased about 43% for this workload.

There was one boot per arm and no randomized order or independent clock/cache
control. Forced-length generation does not establish semantic quality. These
are end-to-end scheduling measurements, not isolated decode-kernel throughput;
do not compare them directly with differently defined benchmark rates. No
long-context, mixed-prefill/decode or streaming soak was repeated. Context and
KV defaults in the public profile remain unchanged.
