# DeepSeek-V4.1-Flash on eight Sparks

[DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/dba1be0a40aa45a94ad051997016db3960a90277)
([DeepSeek](https://huggingface.co/deepseek-ai)'s checkpoint with FP8
weights and MXFP4 routed experts, revision `dba1be0a40aa`) on all eight Sparks
of an eight-Spark ring, with DSpark speculative decoding and adaptive
verification, Engram tables on disk and a 1M-token context. Node A serves the
API on port 8015 as `DeepSeek-V4.1-Flash-TP8`, with no API key. Status:
Experimental.

The eight ranks reach each other through ConnectX relays, so only SIRCL ring
sessions carry the model's collectives, with NCCL off. No image lock in this
package lists the profile: install it with a development image lock
(schema v3) whose image carries the SIRCL layer and lists
`deepseek-v41-flash-tp8`. The
[`dev-20261007-kraken-csf-sircl-cuda1342-nccl2323-status034` recipe](../../runtime/releases/dev-20261007-kraken-csf-sircl-cuda1342-nccl2323-status034/README.md)
builds such an image and lock:

```bash
sudo sparkring install --profile deepseek-v41-flash-tp8 --image-lock LOCK
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
| Everything else | The values of [`deepseek-v41-flash-tp4`](../deepseek-v41-flash-tp4/config.json): DSpark5 with block rejection and adaptive verification, Engram tables on disk, 1048576-token context, 16 sequences, 8192-token batches, GPU memory utilization 0.83 |

## Evidence and open items

No installation of this profile has run on hardware. The four-Spark
profile served on SIRCL at TP4 on a four-Spark line of an eight-Spark ring
through SIRCL's own launcher ([package status](../../spark_transport/sircl/STATUS.md)).
The model's 64 attention heads, 8 output groups and 384 routed experts divide
by eight; whether TP8 decodes faster than two groups of four on the same
ring, and the KV capacity at TP8, are not measured.
