# GLM-5.3-Flash checkpoint selection

Checkpoint selection does not change a running server. Use immutable revisions
and separate model directories; never overwrite files mounted by a live model.

| Checkpoint | Pinned revision | Scope |
|---|---|---|
| [NVFP4-Spark](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-Spark/tree/a608241037e4c2565356bff7ca293f2133888f88) | `a608241037e4c2565356bff7ca293f2133888f88` | Maintained R35/R37 host selection; serving evidence remains scoped to each recorded checkpoint. |
| [NVFP4 QAD](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4/tree/175ae8ce3b5af842b0d0140dbeb43e9cfc557c49) | `175ae8ce3b5af842b0d0140dbeb43e9cfc557c49` | Research-only checkpoint selection; not qualified by the retained plain-NVFP4 recipes. |

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

Status: **research-only** for this QAD revision in the retained SparkRing recipes.
Retargeting a recipe requires its config/index hashes, cache identity and serving
checks to agree with the selected checkpoint. A completed download alone is not
runtime qualification.
