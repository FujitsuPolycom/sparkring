# GLM-5.3-Flash checkpoint selection

Checkpoint selection does not change a running server. Use immutable revisions
and separate model directories; never overwrite files mounted by a live model.

| Checkpoint | Pinned revision | Scope |
|---|---|---|
| [NVFP4-Spark](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-Spark/tree/a608241037e4c2565356bff7ca293f2133888f88) | `a608241037e4c2565356bff7ca293f2133888f88` | Qualified for bounded TP2/TP4 DCP1 SparkCache checks on SparkRing 2026.09.3. |
| [NVFP4 QAD](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4/tree/175ae8ce3b5af842b0d0140dbeb43e9cfc557c49) | `175ae8ce3b5af842b0d0140dbeb43e9cfc557c49` | Qualified for bounded TP2/TP4 DCP1 SparkCache checks on SparkRing 2026.09.3; not covered by retained plain-NVFP4 recipes. |
| [NVFP4-MXFP8 CSF QAD](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD/tree/dec48abd33efa73c3bb7c95b74eee10cad34f9be) | `dec48abd33efa73c3bb7c95b74eee10cad34f9be` | Research-only. Read only by vLLM `bc9ea774` and B12X `cc36aa6f` ([CSF sources](../runtime/images/compositions/kraken-csf-sources-20261007/README.md)); no installation has run on Sparks. |

## Default checkpoint by profile

The CSF checkpoint stores its routed experts' NVFP4 block scales losslessly
compressed. Only vLLM's `nvfp4_csf` quantization and loader in the sources
above read it, so a profile can make it the default only where its image
carries them: an installer image whose lock lists SIRCL's pinned vLLM build
`sparkring-kraken-beta-20261007-bc9ea774` in `sircl.vllm_pins`, such as the
image of the
[`dev-20261007-kraken-csf-sircl-cuda1342-nccl2323-status034` recipe](../runtime/releases/dev-20261007-kraken-csf-sircl-cuda1342-nccl2323-status034/README.md).

| Profiles | Default checkpoint | Reason |
|---|---|---|
| `glm53-flash-nvfp4-spark-tp2`, `glm53-flash-nvfp4-spark-tp4` | CSF (`csf`) on an image that reads it; NVFP4-Spark (`nvfp4-spark`) on every other image, the default installer image among them | Installer images without the CSF sources, 2026.10.1 among them, also list them, so each image keeps a default it can serve; `--checkpoint nvfp4-spark` and `--checkpoint nvfp4-qad` select the others ([checkpoint names](../docs/operations/install-reference.md#another-checkpoint-of-a-profile)) |
| `glm53-flash-csf-tp8` | CSF, its only checkpoint | Runs only on SIRCL ring sessions, on an image whose lock lists the pinned build |
| `glm53-flash-spark-tp2-dcp1`, `glm53-flash-spark-tp2-dcp1-nocache`, `glm53-flash-spark-tp2-dcp1-sparkcache`, `glm53-flash-spark-tp4-dcp1`, `glm53-flash-spark-tp4-dcp1-nocache`, `glm53-flash-spark-tp4-dcp1-sparkcache`, `glm53-flash-spark-tp4-dcp4`, `glm53-flash-spark-tp4-dcp4-sparkcache` | NVFP4-Spark | Their releases, `shared-2026.09.3` and `sparkring-r33-dcp4`, are published native images with neither the `nvfp4_csf` loader nor its B12X kernels |
| `glm53-flash-spark-tp4-switched` | NVFP4-Spark | Its release, `glm53-source`, builds its image from the [GLM source lock](../runtime/sparkring/source_image/glm53-tp4-lock.json), without the CSF sources |
| `glm53-flash-nvfp4-dflash2-bf16-tp4`, `sparkcache-glm53-flash-nvfp4-dflash2-bf16-sparkcache-tp4`, `glm53-mtp3-cache-checkpoints-tp4`, `glm53-spark-mtp3-managed-mesh-tp4` | Unchanged | Retired recipes whose images lack the CSF sources |

On a profile that keeps NVFP4-Spark, the CSF checkpoint is selected by
installing the installer profile of the same size with an image that reads
it:

```bash
sudo sparkring install --profile glm53-flash-nvfp4-spark-tp2 --image-lock LOCK
sudo sparkring install --profile glm53-flash-nvfp4-spark-tp4 --image-lock LOCK
```

`LOCK` is the `installer-image.json` that the recipe's build writes; until the
image is published, `sparkring images` does not list it.

## Other checkpoints

The [NVIDIA NVFP4 target](glm53-nvidia-nvfp4.md) is a Development option of the
R37 TP4 procedure.

The four-Spark installer profile `glm53-flash-nvfp4-spark-tp4` installs the
QAD revision above (`--checkpoint nvfp4-qad`) and NVIDIA revision
`da920bb0b9f4` (`--checkpoint nvidia-nvfp4`) on the installer image, each with
its own pin manifest in [`checkpoints/`](checkpoints). Both selections are
implemented, each measured on one four-Spark ring. The two-Spark installer
profile `glm53-flash-nvfp4-spark-tp2` installs the same QAD revision with
`--checkpoint nvfp4-qad`, with 5 GiB of KV cache per Spark and a
524,288-token context window to make room for its larger weights; that entry
is implemented, measured on one pair, and the pair does not offer the NVIDIA
checkpoint
([names, settings and status](../docs/operations/install-reference.md#another-checkpoint-of-a-profile)).

The [Spark target record](glm53-target-variants.json) owns its runtime revision,
metadata hashes and checkpoint identity. R35/R37 host launchers incorporate the
revision into SparkCache namespaces. Existing snapshots must not be relabeled
for another checkpoint. Frozen image recipes and benchmark receipts retain the
identities of the artifacts they reproduce.

The plain `local-inference-lab/GLM-5.3-Flash-NVFP4` repository contains a
quantization-aware distilled checkpoint: the student is trained to compensate
for quantization error. It is not the `NVFP4-Spark` checkpoint, and its updated
weights are not covered by benchmarks for plain revision `520de24`.

To download the QAD checkpoint for testing without changing a serving profile:

```bash
MODEL_DIR=/srv/models/GLM-5.3-Flash-NVFP4/175ae8c
hf download local-inference-lab/GLM-5.3-Flash-NVFP4 \
  --revision 175ae8ce3b5af842b0d0140dbeb43e9cfc557c49 \
  --local-dir "$MODEL_DIR"
```

Use `--target-model-variant nvfp4-qad` with the native-image
[TP2](glm53-flash-spark-tp2-dcp1-sparkcache/README.md) or
[TP4](glm53-flash-spark-tp4-dcp1-sparkcache/README.md) procedure. The
[qualification record](../runtime/releases/shared-2026.09.3/qualification.json)
scopes bounded text/media and restart/restore evidence. It does not qualify
full-context load, arbitrary media or cache-disabled GLM selections.
Retargeting another recipe requires matching metadata, cache identity and
serving checks; a completed download alone is not runtime qualification.
