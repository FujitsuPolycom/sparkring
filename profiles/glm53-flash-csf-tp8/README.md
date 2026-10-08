# GLM-5.3-Flash CSF on eight Sparks

[GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD/tree/dec48abd33efa73c3bb7c95b74eee10cad34f9be)
([Local Inference Lab](https://huggingface.co/local-inference-lab)'s QAD
quantization of Z.ai's [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16),
revision `dec48abd33ef`: routed experts in NVFP4 with losslessly compressed
(CSF) block scales, MXFP8 attention) on all eight Sparks of an eight-Spark
ring, with MTP speculative decoding and a 1M-token context. Node A serves the
API on port 8015 as `GLM-5.3-Flash-CSF-TP8`, with no API key. Status:
Experimental.

The eight ranks reach each other through ConnectX relays, so only SIRCL ring
sessions carry the model's collectives, with NCCL off. No image lock in this
package lists the profile: install it with a development image lock
(schema v3) whose image carries the SIRCL layer and lists
`glm53-flash-csf-tp8`. The
[`dev-20261007-kraken-csf-sircl-cuda1342-nccl2323-status034` recipe](../../runtime/releases/dev-20261007-kraken-csf-sircl-cuda1342-nccl2323-status034/README.md)
builds such an image and lock, with the CSF-capable vLLM below:

```bash
sudo sparkring install --profile glm53-flash-csf-tp8 --image-lock LOCK
```

[Install SparkRing](../../docs/operations/install.md) covers requirements, the
fabric setup an eight-Spark ring needs, logs and recovery.

## Checkpoint

The checkpoint needs an image whose vLLM reads CSF scales: SIRCL's pinned
vLLM build `sparkring-kraken-beta-20261007-bc9ea774`, which the image lock
lists in `sircl.vllm_pins`. The installer refuses the profile on any other
image.

Each Spark holds the whole checkpoint, 165.5 GiB. A blank Spark needs about
226 GiB free: the checkpoint, the serving image and the compile cache
allowance.

## Settings

| Setting | Value ([config.json](config.json)) |
|---|---|
| Parallelism | TP8/DCP1 on the eight positions of an eight-Spark ring, Node A as rank 0 |
| Transport | SIRCL ring sessions, NCCL off; the default tuning table's `cycle-8` row |
| Quantization and loader | `--quantization nvfp4_csf --load-format nvfp4_csf`; W4A16 decode (`VLLM_B12X_MOE_FP4_FORCE_A16=1`) |
| Context / sequences / batch | 1048576 / 16 / 8192 |
| KV allocation | 37 GiB FP8 per rank in 1,024-token blocks |
| Speculation | MTP3 with probabilistic drafts on B12X attention; draft tensor-parallel size 8; the draft's experts on the `marlin` MoE backend |
| Everything else | The values of [`glm53-flash-nvfp4-spark-tp4`](../glm53-flash-nvfp4-spark-tp4/config.json) |

## Evidence and open items

No installation of this profile has run on hardware.

- The sequence count is the four-Spark profile's; at TP8 each rank holds
  half the weights, and no larger value has been measured.
- Measured serving of this checkpoint at TP4 on SIRCL, outside this
  repository, also set `B12X_W4A16_FP32_TOPK_WEIGHTS=1`,
  `B12X_W4A16_A4_PREFILL_MIN_TOKENS=1536` and
  `B12X_W4A16_SMALL_M_OCCUPANCY=2`. This profile does not set them.
- The checkpoint's model card states that vLLM reads it with `modelopt_mixed`
  and the standard loader; that measured serving used `nvfp4_csf` for both,
  which this profile keeps.
