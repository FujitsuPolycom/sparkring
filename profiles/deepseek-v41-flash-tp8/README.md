# DeepSeek-V4.1-Flash on eight Sparks

[DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/dba1be0a40aa45a94ad051997016db3960a90277)
([DeepSeek](https://huggingface.co/deepseek-ai)'s checkpoint with FP8
weights and MXFP4 routed experts, revision `dba1be0a40aa`) on all eight Sparks
of an eight-Spark ring, with DSpark speculative decoding and adaptive
verification, Engram tables on disk and a 1M-token context. Node A serves the
API on port 8015 as `DeepSeek-V4.1-Flash-TP8`, with no API key. Status:
Experimental.

The eight ranks reach each other through ConnectX relays, so only SIRCL ring
sessions carry the model's collectives, with NCCL off. It installs on an
image whose lock (schema v3) carries the SIRCL layer and lists
`deepseek-v41-flash-tp8`, such as the default image, release 2026.10.2's
([release record](../../runtime/releases/dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036/README.md)):

```bash
sudo sparkring install --profile deepseek-v41-flash-tp8
```

Each Spark holds the whole 475.3 GiB checkpoint; a blank Spark needs about
547 GiB free with the serving image and the compile cache allowance.
[Install SparkRing](../../docs/operations/install.md) covers requirements, the
fabric setup an eight-Spark ring needs, logs and recovery.

## Settings

| Setting | Value ([config.json](config.json)) |
|---|---|
| Parallelism | TP8/DCP1 on the eight positions of an eight-Spark ring, Node A as rank 0 |
| Transport | SIRCL ring sessions, NCCL off; the default tuning table's `cycle-8` row |
| Engram projections | Replicated on every rank. The four-Spark profile shards them (`"projection_tp": true`); at TP8 one start with sharding showed no consistent gain ([record](../../performance/records/images/dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036-release-2026.10.3-plain-ring8-20261012.md#not-adopted-engram-projection-sharding-at-tp8)) |
| Everything else | The values of [`deepseek-v41-flash-tp4`](../deepseek-v41-flash-tp4/config.json): DSpark5 with block rejection and adaptive verification, Engram tables on disk, 1048576-token context, 16 sequences, 8192-token batches, GPU memory utilization 0.83 |

## Evidence and open items

One installation of this profile ran on a ring of eight DGX Sparks with release
2026.10.2's image (research-only, one start): it served with a KV cache of
32,677,006 tokens and was measured once at temperature 1
([record](../../performance/records/images/dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036-release-2026.10.3-plain-ring8-20261012.md#not-adopted-engram-projection-sharding-at-tp8)).
The model's 64 attention heads, 8 output groups and 384 routed experts divide
by eight. Whether TP8 serves more requests than two groups of four on the same
ring is not measured.
