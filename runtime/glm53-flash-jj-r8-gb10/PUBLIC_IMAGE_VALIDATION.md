# Public GLM-5.3 R8 ARM64 image validation

**Status: implemented with bounded live evidence.** The immutable Linux/ARM64
image below was pulled from GHCR, distributed from one archive through three
direct links, and exercised on four GB10 systems.

```text
ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:380283a506aeb8f9d486a3c64cd738e44268c3cc21590913ea9e4685869f256a
```

The local Docker image ID is
`sha256:b3a13d8003e7de30d7737fd33c8307404e506ba570240819ec7eb4f5c611400f`.
The machine-readable record is
[`public-image-receipt.json`](public-image-receipt.json).

## Construction and distribution

The image binds vLLM `55969c16`, merged SparkCache main `c3887f34`, B12X
`6255090a`, switchless NCCL, the SparkCache CUDA placement library, and 15
retained native extensions. Construction verification checked source trees,
package subtrees, labels, installed files, and shared-library hashes.

One 8,467,064,540-byte archive with SHA-256
`7f5a9d97e16a984167f970d4c0a5ca1536aa49a02b9a46aec864d3c87a1bcb89`
was downloaded on rank 0 and relayed over three 200 Gb/s direct links. Every
rank verified the archive and loaded the same image ID.

## Serving results

| Profile | KV capacity | Stored span | Restart restore | Result |
|---|---:|---:|---:|---|
| DCP1 + SparkCache, 26 GiB/rank | 1,303,701 | 10,752 | 215–267 ms | Marker returned after process replacement |
| DCP1, vLLM prefix cache only | 1,303,701 | — | — | Connector absent; semantic request passed |
| DCP2 + SparkCache, 30 GiB/rank | 2,899,004 | 12,288 | 156–191 ms | Marker returned after process replacement |
| DCP4 + SparkCache, 30 GiB/rank | 5,402,023 | 12,288 | 133–139 ms | Marker returned after process replacement |

The vLLM-only profile retained `--enable-prefix-caching`, omitted
`--kv-transfer-config`, emitted no SparkCache connector logs, and returned its
expected marker.

## One-million-depth request

The DCP1 26 GiB profile completed the public boundary probe's 1,000,000-depth
case. The API reported 942,898 prompt tokens, a 473.4-second completion time,
`finish_reason=stop`, and the exact eight-digit needle. SparkCache then
published a 942,592-token snapshot. Capture took 15.2–18.0 seconds per rank;
background commits took 10.1–10.7 seconds.

The deep snapshot was not replayed after process replacement. Large capture
competing with unrelated inference is tracked in
[`FujitsuPolycom/sparkcache#45`](https://github.com/FujitsuPolycom/sparkcache/issues/45).

## Limits

This evidence does not establish concurrent deep-context requests, restart
restore of the 942,592-token object, tail-only deep publication, sustained
serving, fault recovery, or general throughput.
