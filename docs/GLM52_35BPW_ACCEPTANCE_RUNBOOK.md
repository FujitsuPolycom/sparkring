# GLM-5.2 rebuilt-image acceptance workload

Use this procedure for the fixed-seed and bounded-concurrency item in
[`GLM52_35BPW_PROMOTION_CHECKLIST.md`](GLM52_35BPW_PROMOTION_CHECKLIST.md).
It produces functional-equivalence, speculative-counter, and C1/C2/C8
receipts for one immutable rebuilt image. It does not promote an image by
itself or establish a general performance claim.

## Conditions

Record these values before sending traffic:

- rebuilt Docker image ID and immutable parent manifest digest;
- SparkRing commit and generated exact-Q40 profile SHA-256;
- model repository, revision, config SHA-256, and index SHA-256;
- four physical ranks in cycle order, TP4/DCP4, `ag_rs`, interleave one;
- KV representation and bytes per rank, model length, maximum sequences, and
  maximum batched tokens;
- cache state and every client other than the acceptance harness;
- NVIDIA driver, CUDA runtime, and whether the P2P override is active.

The MTP0 baseline and MTP4 candidate must use the same rebuilt image,
checkpoint, parallelism, KV geometry, graph settings, and transport settings.
Only the speculative depth changes.

## Fixed-seed equivalence

Start the rebuilt image with speculative decoding disabled and capture the
MTP0 baseline:

```bash
python runtime/exl3-r7/mtp4_qualification.py capture-mtp0 \
  --base-url http://<rank0-management-address>:8000 \
  --model glm-5.2-exl3-r7-3.5bpw \
  --output /path/to/receipts/mtp0.json
```

Restart the same image with its generated fixed-MTP4 exact-Q40 profile, then
qualify it against the baseline:

```bash
python runtime/exl3-r7/mtp4_qualification.py qualify-mtp4 \
  --base-url http://<rank0-management-address>:8000 \
  --model glm-5.2-exl3-r7-3.5bpw \
  --baseline /path/to/receipts/mtp0.json \
  --output /path/to/receipts/mtp4.json
```

The tracked harness fixes the prompt corpus, seed, greedy sampling,
repetitions, and 128/256-token completion lengths. It requires finite log
probabilities, repeatable completion hashes, equality with the MTP0 hashes,
and positive, coherent fixed-MTP4 counters at positions zero through three.
Either JSON document with status other than `pass` rejects promotion.

## Bounded C1/C2/C8 workload

Before the workload, copy each rank's JSON file at its configured
`SPARK_TP4_GRAPH_STATUS_PATH` to
`/path/to/receipts/transport-before-rank<RANK>.json`. Use the reviewed site
plan for container names and SSH targets; keep those site values out of the
public receipt.

Use
[`local-inference-lab/llm-inference-bench`](https://github.com/local-inference-lab/llm-inference-bench)
commit `0b4185b5b435e948b199c9077a00b084864aa963`. Install its declared
dependencies in an isolated environment, then run exactly one 16K-context
cell at C1, C2, and C8:

```bash
python llm_decode_bench.py \
  --host <rank0-management-address> \
  --port 8000 \
  --model glm-5.2-exl3-r7-3.5bpw \
  --contexts 16k \
  --concurrency 1,2,8 \
  --request-count 16 \
  --warmup-request-count 1 \
  --max-tokens 128 \
  --temperature 0 \
  --token-targeting exact \
  --kv-budget 1156864 \
  --dcp-size 4 \
  --skip-prefill \
  --no-resume \
  --display-mode plain \
  --output /path/to/receipts/c1-c2-c8.json
```

This is a finite request-count gate: each cell discards one warm-up request and
waits for exactly 16 measured requests. Promotion requires all three cells,
zero request errors, no capacity skip, no warm-up timeout, the requested
concurrency in every cell, and 128 completion tokens for every successful
request. Throughput values are retained as observations but have no promotion
threshold.

## Post-run gates and retention

After the C8 cell:

1. require `/health` HTTP 200 and the same `/v1/models` identity and model
   length;
2. require all four containers to remain running with unchanged process
   generations;
3. copy each rank's updated `SPARK_TP4_GRAPH_STATUS_PATH` JSON to
   `/path/to/receipts/transport-after-rank<RANK>.json` and run:

   ```bash
   python runtime/exl3-r7/mtp4_qualification.py audit-transport \
     --before-status /path/to/receipts/transport-before-rank0.json \
     --before-status /path/to/receipts/transport-before-rank1.json \
     --before-status /path/to/receipts/transport-before-rank2.json \
     --before-status /path/to/receipts/transport-before-rank3.json \
     --after-status /path/to/receipts/transport-after-rank0.json \
     --after-status /path/to/receipts/transport-after-rank1.json \
     --after-status /path/to/receipts/transport-after-rank2.json \
     --after-status /path/to/receipts/transport-after-rank3.json \
     --output /path/to/receipts/transport.json
   ```

   Require status `pass`; this rejects fatal, overflow, dropped-signature,
   stock TP/vocabulary fallback, and missing captured-width evidence.
4. record the SHA-256 of `mtp0.json`, `mtp4.json`, `c1-c2-c8.json`, and
   `transport.json`;
5. copy sanitized receipts into the promotion evidence record or retain them at
   an immutable public artifact location.

The acceptance record must state conditions, measurement, result, conclusion,
and limitations. Missing receipts, a changed image ID, an incomplete cell, or
failed post-run health rejects promotion rather than becoming a partial pass.
