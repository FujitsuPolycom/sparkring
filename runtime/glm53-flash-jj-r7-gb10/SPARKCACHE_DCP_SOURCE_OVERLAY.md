# SparkCache DCP2/DCP4 source-overlay evidence

Status: **implemented**. Bounded live evidence is recorded below. The source
overlay is not qualified or distributed.

This record applies to GLM-5.3 Flash on four directly connected GB10 systems.
It establishes that SparkCache can publish and restore the model's opaque
manager pages at TP4/DCP2 and TP4/DCP4 with four-token KV interleaving.

The tested SparkCache package had source digest
`40de372dda64dd25f493584b2ba3dae81c4350d424d3cf00cfea92452dac170c`.
It was injected into the runtime as an uncommitted source overlay. The public
SparkCache image digest in [`artifacts.json`](artifacts.json) does not contain
these bytes and remains restricted to DCP1.

SparkCache's model-specific record contains the corresponding machine-readable
evidence contract:
[GLM-5.3 Flash DCP2/DCP4 SparkCache validation](https://github.com/FujitsuPolycom/sparkcache/blob/codex/cuda-restore-stream-readiness/deploy/glm53_flash/GLM53_DCP2_DCP4_SPARKCACHE_VALIDATION.md).

## Runtime contract

| Component | Tested identity or setting |
|---|---|
| vLLM composition | `331573d20bd47e78327ed8d8b4d2e6d350bbb1ab` |
| B12X | `6255090a03b12c3f7d552102a02fac0b542fb8c9` |
| switchless NCCL library SHA-256 | `5f1c3f10d5ace66d4ba584415bbfe42b6ac1a0a9116a3b81dcbe50516ad924b3` |
| SparkCache CUDA placement SHA-256 | `d57509052b73853bcc8e3c3f47bb81748d87b9cbd8d908fc20d4c79a09aa400c` |
| target/draft identities | `a35e6bf2875c…` / `b33c03475ba7…` |
| serving geometry | TP4, 512K model limit, 8K batched tokens, FP8 target KV, 20 GiB KV per rank |
| speculative decoding | external BF16 DFlash2, depth 7 |
| DCP layout | four-token KV interleaving with B12X full-CKV prefill gather |
| semantic oracle | fixed 10,028-token prompt; expected visible output `red` |

## Results

| DCP | Stored span | Snapshot by rank | Commit by rank | Restart restore by rank | Result |
|---|---:|---|---|---|---|
| 2 | 9,216 tokens; digest `1b1bb36f670e` | 130.5, 132.9, 118.7, 129.2 ms | 157.3, 167.2, 165.7, 189.3 ms | SparkCache CUDA: 156.599, 174.131, 172.630, 151.086 ms | 78,751,393 bytes per rank; exact `red` on every rank after full restart |
| 4 | 8,192 tokens; digest `616760b4c69d` | 85.7, 76.7, 94.2, 62.4 ms | 171.4, 180.7, 147.2, 160.9 ms | SparkCache CUDA: 133.709, 130.430, 118.935, 118.781 ms | 62,953,633 bytes per rank; exact `red` on every rank after full restart |

DCP2 also completed a separate full-restart restore through the Python
placement path. All four ranks returned exact `red`; service times were
234.840, 264.505, 253.430, and 263.052 milliseconds.

Each restart included a scheduler-inventory request before the semantic
oracle. Every restore was reported as verified. Entries that cannot be
verified are recomputed rather than served from persistent storage.

## Distribution boundary

The source-overlay result is not a pull-only operator route. A replacement
SparkCache image must include the tested DCP-aware source, record its immutable
source and image digests, and complete a bounded pull-and-launch check before
the public launcher can permit `IMAGE_VARIANT=sparkcache` with DCP2 or DCP4.

The evidence does not establish throughput, concurrent long-context capacity,
512K-prompt behavior, sustained serving, or fault recovery.
