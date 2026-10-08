# Qwen3.8-Flash-Next NVFP4 QAD on eight Sparks

[Qwen3.8-Flash-Next NVFP4](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4/tree/60215d26cf5e42c2db6128774032d57fc62678da)
([Local Inference Lab](https://huggingface.co/local-inference-lab)'s QAD
checkpoint step 5500 of Qwen's [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next),
revision `60215d26cf5e`) on all eight Sparks of an eight-Spark ring, with MTP
speculative decoding and 262K context. Node A serves the API on port 8015 as
`Qwen3.8-Flash-Next-NVFP4-QAD-TP8`, with no API key. Status: Experimental.

The eight ranks reach each other through ConnectX relays, so only SIRCL ring
sessions carry the model's collectives, with NCCL off; the hyper-connection
prefill row ownership runs on the tensor-parallel session. No image lock in
this package lists the profile: install it with a development image lock
(schema v3) whose image carries the SIRCL layer and lists
`qwen38-flash-next-qad-tp8`:

```bash
sudo sparkring install --profile qwen38-flash-next-qad-tp8 --image-lock LOCK
```

Each Spark holds the whole 102.6 GiB checkpoint; a blank Spark needs about
163 GiB free with the serving image and the compile cache allowance.
[Install SparkRing](../../docs/operations/install.md) covers requirements, the
fabric setup an eight-Spark ring needs, logs and recovery.

## Settings

| Setting | Value ([config.json](config.json)) |
|---|---|
| Parallelism | TP8/DCP1 on the eight positions of an eight-Spark ring, Node A as rank 0 |
| Transport | SIRCL ring sessions, NCCL off; the default tuning table's `cycle-8` row |
| Checkpoint | Step 5500 only (branch `qad-step5500-ple1000`); the four-Spark profile's other checkpoints are not offered at TP8 |
| Everything else | The values of [`qwen38-flash-next-qad-tp4`](../qwen38-flash-next-qad-tp4/config.json): 262144-token context, 16 sequences, 24 GiB FP8 KV cache per rank, MTP3, hyper-connection token-row prefill ownership |

## Evidence and open items

No installation of this profile has run on hardware. The four-Spark
profile served on SIRCL at TP4 on a four-Spark line of an eight-Spark ring
through SIRCL's own launcher ([package status](../../spark_transport/sircl/STATUS.md)).

- Image admission requires the image to declare the profile's
  hyper-connection mode for eight ranks (`hc_supported_modes["8"]` in its
  external-software receipt). The external-image builder declares modes for
  two and four ranks only (`HC_SUPPORTED_MODES` in
  [`external_context.py`](../../runtime/images/external_context.py)), so an
  image built by it is refused until it declares eight.
- The model's 24 attention heads, 16 and 48 linear-attention key and value
  heads and 512 experts divide by eight, and its two KV heads are replicated;
  the KV capacity and decode rate at TP8 are not measured.
