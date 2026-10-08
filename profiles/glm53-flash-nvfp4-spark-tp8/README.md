# GLM-5.3-Flash NVFP4-Spark on eight Sparks

[GLM-5.3-Flash NVFP4-Spark](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-Spark/tree/a608241037e4c2565356bff7ca293f2133888f88)
([Local Inference Lab](https://huggingface.co/local-inference-lab)'s NVFP4/MXFP8
quantization of Z.ai's [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16),
revision `a608241037e4`) on all eight
Sparks of an eight-Spark ring, with MTP speculative decoding and a 1M-token
context. Node A serves the API on port 8015 as `GLM-5.3-Flash-NVFP4-Spark-TP8`,
with no API key. Status: Experimental.

The eight ranks reach each other through ConnectX relays, so only SIRCL ring
sessions carry the model's collectives, with NCCL off. No image lock in this
package lists the profile: install it with a development image lock
(schema v3) whose image carries the SIRCL layer and lists
`glm53-flash-nvfp4-spark-tp8`:

```bash
sudo sparkring install --profile glm53-flash-nvfp4-spark-tp8 --image-lock LOCK
```

[Install SparkRing](../../docs/operations/install.md) covers requirements, the
fabric setup an eight-Spark ring needs, logs and recovery.

## Checkpoints

| `--checkpoint` | Checkpoint | Size per Spark | What changes |
|---|---|---|---|
| `nvfp4-spark` (default) | [GLM-5.3-Flash-NVFP4-Spark](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-Spark/tree/a608241037e4c2565356bff7ca293f2133888f88), revision `a608241037e4` | 174.8 GiB | — |
| `csf` | [GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD/tree/dec48abd33efa73c3bb7c95b74eee10cad34f9be), revision `dec48abd33ef`: QAD routed experts in NVFP4 with losslessly compressed (CSF) block scales, MXFP8 attention | 165.5 GiB | `--quantization nvfp4_csf --load-format nvfp4_csf`, 37 GiB KV cache per Spark, W4A16 decode (`VLLM_B12X_MOE_FP4_FORCE_A16=1`), the draft's experts on the `marlin` MoE backend; served as `GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD-TP8` |

The `csf` checkpoint needs an image whose vLLM reads CSF scales: SIRCL's
pinned vLLM build `sparkring-kraken-beta-20261007-bc9ea774`, which the image
lock lists in `sircl.vllm_pins`. The installer refuses `--checkpoint csf` on
any other image.

Each Spark holds the whole checkpoint. A blank Spark needs about 235 GiB free
for the default checkpoint (226 GiB for `csf`): the checkpoint, the serving
image and the compile cache allowance.

## Settings

| Setting | Value ([config.json](config.json)) |
|---|---|
| Parallelism | TP8/DCP1 on the eight positions of an eight-Spark ring, Node A as rank 0 |
| Transport | SIRCL ring sessions, NCCL off; the default tuning table's `cycle-8` row |
| Context / sequences / batch | 1048576 / 16 / 8192 |
| KV allocation | 40 GiB FP8 per rank in 1,024-token blocks |
| Speculation | MTP3 with probabilistic drafts on B12X attention; draft tensor-parallel size 8 |
| Everything else | The values of [`glm53-flash-nvfp4-spark-tp4`](../glm53-flash-nvfp4-spark-tp4/config.json) |

## Evidence and open items

No installation of this profile has run on hardware. The four-Spark
profile's settings served on SIRCL at TP4 on four-Spark lines of an
eight-Spark ring through SIRCL's own launcher
([package status](../../spark_transport/sircl/STATUS.md)).

- The KV allocation and sequence count are the four-Spark profile's; at TP8
  each rank holds half the weights, and no larger value has been measured.
- Measured serving of the CSF checkpoint at TP4 on SIRCL, outside this
  repository, also set `B12X_W4A16_FP32_TOPK_WEIGHTS=1`,
  `B12X_W4A16_A4_PREFILL_MIN_TOKENS=1536` and
  `B12X_W4A16_SMALL_M_OCCUPANCY=2`. A checkpoint entry may change only
  variables the profile sets, and this profile does not set them, so the
  `csf` entry runs without them until the profile's environment carries
  values measured for both checkpoints.
- The checkpoint's model card states that vLLM reads it with `modelopt_mixed`
  and the standard loader; that measured serving used `nvfp4_csf` for both,
  which this entry keeps.
