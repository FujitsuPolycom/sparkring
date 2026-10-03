# GLM-5.3-Flash NVFP4 QAD checkpoint on two Sparks: KV cache capacity at 5 GiB per Spark

Status: **implemented; engine-reported KV pool from one model start on one half of a four-Spark ring, measured 2026-10-02; not a test of concurrent full-context requests; not serving-qualified**.

This record measures how many tokens the KV cache holds when the two-Spark GLM
installer profile `glm53-flash-nvfp4-spark-tp2` serves its `nvfp4-qad`
checkpoint, `local-inference-lab/GLM-5.3-Flash-NVFP4` revision
`175ae8ce3b5a`, with that checkpoint's 5 GiB of KV cache per Spark. The
`nvfp4-qad` entry of `glm53-flash-nvfp4-spark-tp2` in
[`profile-capacity.json`](../../profile-capacity.json) cites it.

## Conditions

- **Installation:** `sparkring install --profile glm53-flash-nvfp4-spark-tp2 --checkpoint nvfp4-qad --on 2,3`.
- **Cluster:** Sparks 2 and 3 of a four-Spark ring, which share a direct
  cable and serve the model as a two-Spark tensor-parallel deployment (TP2)
  ([two models on one ring](../../../docs/operations/install-reference.md#two-models-on-one-ring)).
- **Image:** installer image `dev-20261001-kraken-cuda1342-nccl2323-status034`
  ([release selection](../../../runtime/releases/dev-20261001-kraken-cuda1342-nccl2323-status034/release.json)).
- **Checkpoint:** `local-inference-lab/GLM-5.3-Flash-NVFP4` revision
  `175ae8ce3b5a`, the `nvfp4-qad` entry of the profile's
  [`config.json`](../../../profiles/glm53-flash-nvfp4-spark-tp2/config.json).
- **Serving settings:**

  | Setting | Value | Set by |
  |---|---|---|
  | KV cache per Spark | 5 GiB (`--kv-cache-memory-bytes 5368709120`) | `nvfp4-qad` checkpoint entry |
  | Context window | 524,288 tokens (`--max-model-len 524288`) | `nvfp4-qad` checkpoint entry |
  | KV cache data type | FP8 (`--kv-cache-dtype fp8`) | Profile |
  | Page size | 2,048 tokens (`--block-size 2048`) | Profile |
  | Requests at once | 8 (`--max-num-seqs 8`) | Profile |
  | Speculative decoding | MTP with 3 draft tokens per step | Profile |

## Measurement

- **KV capacity:** the vLLM engine's startup log line
  `GPU KV cache size: N tokens, Maximum concurrency for M tokens per request: Rx`.
  N is the number of tokens the allocated KV cache holds; R is N divided by
  the context window M.
- **Model memory:** the engine's startup log line
  `Model loading took N GiB memory`.
- The measurement covers one model start. It sent no request.

## Result

- **KV capacity:** `GPU KV cache size: 736,274 tokens, Maximum concurrency for 524,288 tokens per request: 1.40x`.
- **Model memory:** `Model loading took 93.65 GiB memory` on each Spark.
- **Comparison:** the same checkpoint settings on installer image
  `dev-20260930-spinwait-cuda1342-nccl2323-status033`, on one directly cabled
  Spark pair, reported the same 736,274 tokens and 93.65 GiB
  ([pair record](../images/dev-20260930-spinwait-glm53-flash-nvfp4-spark-tp2-nvfp4-qad-20261001.md)).

## Conclusion

With 5 GiB of FP8 KV cache per Spark in 2,048-token pages, the
`nvfp4-qad` checkpoint of `glm53-flash-nvfp4-spark-tp2` starts on two Sparks
with a KV pool of 736,274 tokens, 1.40 times its 524,288-token context window.

## Limitations

- The pool is the engine's report at startup. It is not proof that
  concurrent full-context requests fit: no request was sent, and the 1.40x
  ratio is a division, not two requests served at once.
- One start on one half of one ring. The pool depends on the KV cache size,
  page size, KV data type and speculative-decoding settings above; other
  settings give other pools.
- This record has no functional, correctness or throughput check. The
  [pair record](../images/dev-20260930-spinwait-glm53-flash-nvfp4-spark-tp2-nvfp4-qad-20261001.md)
  has those checks for the earlier image.
