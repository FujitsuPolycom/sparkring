# Validate a serving profile

This runbook measures a running deployment's prefill and decode speed,
long-context accuracy, concurrency and restart/cache behavior with a repeatable
workload. Run it after the model serves, whether installed with
`sudo sparkring install` ([Install SparkRing](install.md)) or a profile guide
([setup, step 6](setup.md#6-test-a-response-and-save-restart-instructions)).
Keep the results with the recipe so another operator can repeat them.

## Required test card

Fill the report with results you already have when their model, configuration,
workload and measurement method match, then run only the missing checks or
repeats. Keep different configurations in separate rows and record how many
repeats each cell has. Example: the
[GLM MTP3 report](../../performance/records/glm53-flash/spark-mtp3-validation-summary-20260905.md).

| Check | Standard workload | Repetitions | Record |
|---|---|---:|---|
| Startup and readiness | Every rank, API, scheduler, configured warmup | Initial start and one planned restart | Versions, startup time, errors, readiness |
| Prefill | C1; 8K, 16K, 32K, 64K, 128K | 3 samples per context | TTFT, actual prompt tokens, tokens/s, cached tokens |
| Sustained decode | 8K, 32K, 64K × C1, C2, C4, C8, C12, C16 | 3 complete sweeps | Aggregate tokens/s, TTFT, ITL, effective concurrency, acceptance |
| Coding peak | Built-in coding workload, temperature 1 | 3 samples | Generation tokens/s, output length, truncation |
| Estonia accuracy | Estonia v2, C1 and C8, temperature 1 | 30 requests at each concurrency | Correct/attempted, truncation, completion-token distribution |
| Long-context retrieval | Exact value, superseding revision, cross-reference; depths 5%, 50%, 95% | One sweep at 128K/256K/512K; repeat failures and matched controls | Actual tokens, answer, finish reason, seed |
| Persistent cache, when enabled | Identical long prompt before and after model restart | 3 independent prompt seeds | Correct answer, prompt hash, per-rank publication/restoration, external hits |
| Mixed traffic | C4 decode while a fresh 64K prefill arrives | 3 paired idle/loaded trials | Decode throughput/ITL change and prefill TTFT |
| Sustained operation | Representative context and concurrency | 30 minutes | Request errors, memory/cache trend, transport health |

Stay within the profile's context and concurrency limits, and mark cells beyond
them **unsupported by this profile**, not zero. For profiles above 512K
context, add a retrieval case near the limit that leaves room for the answer.
Actual tokenization, not the context label, decides whether a request fits.

For a quick tuning screen, run one prefill sweep, 32K decode at C1/C4/C8, and
the three retrieval modes at one long context. Use the full card for a
recommendation or a published comparison.

## Prepare one private results directory

Commands use Bash on Linux, run from the SparkRing checkout, against a trusted
OpenAI-compatible endpoint. Take the values from the profile's guide; the
example limits are for a TP4/DCP4 profile.

```bash
set -euo pipefail
SPARKRING_REPO=$(pwd)
ENDPOINT='http://REPLACE_WITH_RANK0_ADDRESS:8015'
MODEL='REPLACE_WITH_SERVED_MODEL_ID'
DCP=4
CONTEXT_LIMIT=1048576
CONCURRENCIES='1,2,4,8,12,16'
BUSY_C=4
RUN="$HOME/sparkring-results/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN"
git rev-parse HEAD > "$RUN/sparkring-revision.txt"
curl --fail --silent --show-error --max-time 10 "$ENDPOINT/v1/models" > "$RUN/models.json"
curl --fail --silent --show-error --max-time 10 "$ENDPOINT/health" > "$RUN/health-before.txt"
```

Record the recipe path, image digest, model revision, quantization, TP/DCP/PP,
context/sequence/batch limits, attention/MoE/linear backends, graph shapes,
transport routes, speculation settings, cache policy, request sampling,
thinking mode and chat-template settings. Keep the last two fixed between
compared runs. Keep the resolved launch configuration privately, and keep API
credentials out of receipts and public documents.

Start measuring only after the profile's all-rank readiness check and warmup
pass and unrelated traffic has stopped. Run native collective checks only in
their documented stopped-model window, never beside the model.

### Obtain the throughput/accuracy benchmark

Use [Local Inference Lab's llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench)
at revision `bd88816e9e7bcc97e1bcfd954c3053528f31af69`:

```bash
BENCH="$RUN/llm-inference-bench"
git clone https://github.com/local-inference-lab/llm-inference-bench.git "$BENCH"
git -C "$BENCH" checkout --detach bd88816e9e7bcc97e1bcfd954c3053528f31af69
python3 -m venv "$RUN/venv"
BENCH_PY="$RUN/venv/bin/python"
"$BENCH_PY" -m pip install httpx rich psutil
"$BENCH_PY" -m pip freeze > "$RUN/benchmark-dependencies.txt"
sha256sum "$BENCH/llm_decode_bench.py" > "$RUN/benchmark-source.sha256"
COMMON=("$BENCH_PY" "$BENCH/llm_decode_bench.py"
  --host "$ENDPOINT" --model "$MODEL" --no-hw-monitor
  --display-mode plain --no-resume)
```

Decline automatic source updates during a campaign; the commands redirect stdin
from `/dev/null`, and the source hash is checked at the end. For authenticated
endpoints, configure the benchmark's authentication. SparkRing's two probes read
an API key from the environment variable named by `--api-key-env`
(default `OPENAI_API_KEY`).

## Prefill: three samples per context

The [prefill probe](../../performance/harnesses/validation/prefill_probe.py)
sends temperature 1, gives each sample a unique prompt prefix, requests one
token, and measures client time to the first content or reasoning token along
with server usage.

```bash
python3 performance/harnesses/validation/prefill_probe.py \
  --endpoint "$ENDPOINT" --model "$MODEL" \
  --contexts 8k,16k,32k,64k,128k --context-limit "$CONTEXT_LIMIT" \
  --repeats 3 --temperature 1 --output "$RUN/prefill.jsonl"
```

Report median, minimum and maximum TTFT and prompt tokens/s per context, using
the server's prompt counts. A cold sample needs zero cached tokens; if the
server omits that field, confirm with its cache counters. Label cache-primed
samples separately, and do not clear unrelated persistent cache data to force
a cold result.

## Decode matrix and coding peak

Run the matrix three times. The first sweep also takes three coding-peak
samples after its decode matrix.

```bash
for repeat in 1 2 3; do
  CODING=()
  if [ "$repeat" = 1 ]; then
    CODING=(--coding-peak --coding-peak-runs 3
      --coding-peak-max-tokens 2000 --coding-peak-temperature 1)
  fi
  "${COMMON[@]}" --skip-prefill --contexts 8k,32k,64k \
    --concurrency "$CONCURRENCIES" --dcp-size "$DCP" \
    --token-targeting exact --temperature 1 --max-tokens 2048 \
    --duration 30 --decode-warmup-seconds 5 \
    --cell-warmup-timeout-seconds 900 "${CODING[@]}" \
    --output "$RUN/decode-r${repeat}.json" </dev/null
done
```

Achieved concurrency must match the request. Before aggregating, review errors,
underfilled cells, capacity limits, warmup timeouts and the token-count method.
Report each cell's median and range across the three sweeps, and keep failed
attempts with their reason next to a separately named retry.

Record speculative depth and acceptance with output throughput. Coding peak
reports median and peak generation speed, output lengths and truncation; it
measures speed, not coding accuracy.

## Estonia: long-context accuracy and consistency

Run the [Estonia benchmark](https://github.com/local-inference-lab/llm-inference-bench)
at C1 and C8, thirty sampled runs each. Lower C8 if the profile's sequence
limit requires it, and make sure the prompt plus output budget fits.

```bash
for concurrency in 1 8; do
  "${COMMON[@]}" --test-profile estonia \
    --profile-concurrency "$concurrency" --profile-runs 30 \
    --max-tokens 40000 --completion-stats-temperature 1 \
    --completion-stats-top-p 1 --completion-stats-seed 9046500 \
    --completion-stats-save-text --completion-stats-stall-timeout 600 \
    --completion-stats-request-timeout 1800 \
    --output "$RUN/estonia-c${concurrency}.json" </dev/null
done
```

Report correct/attempted, errors, output-budget hits and completion-token
distributions, and inspect wrong or unparseable answers. The test primes the
prefix cache, so its TTFT describes only this workload; use the decode matrix
for aggregate throughput. Keep full outputs private.

## Long needle hunt: retrieval, revisions, and cross-references

The [retrieval harness](../../performance/harnesses/validation/README.md)
places opaque values at set document depths. `exact` retrieves a value,
`revision` picks a superseding record over a decoy, and `join` follows an alias
between separated records. It checks tokenization and completion budget before
sending.

```bash
python3 performance/harnesses/validation/needle_hunt.py \
  --base-url "$ENDPOINT" --model "$MODEL" --context-limit "$CONTEXT_LIMIT" \
  --contexts 128k,256k,512k --positions 5,50,95 \
  --modes exact,revision,join --temperature 1 --seed 20260905 \
  --max-tokens 2048 --output "$RUN/needle.jsonl"
```

Every supported cell needs the expected answer and a normal finish. Record
actual prompt tokens, position, seed, answer and finish reason. Repeat failures
with the same fixture and a shorter-context control before changing the
runtime. For one-million-token profiles, add `--contexts 960k` if the complete
request fits.

## SparkCache publication and restoration

Mark this check **not applicable** for profiles without SparkCache. Otherwise
use the profile's publication threshold, cache namespace, all-rank stop/restart
procedure and restoration logs. Do not rename cache entries or change the
checkpoint or speculation settings between the two requests.

Run this for three seeds, with separate before and after files:

```bash
SEED=20260905  # Repeat the sequence with 20260906 and 20260907.
CACHE_PROBE=(python3 performance/harnesses/validation/needle_hunt.py
  --base-url "$ENDPOINT" --model "$MODEL" --context-limit "$CONTEXT_LIMIT"
  --contexts 128k --positions 50 --modes join --temperature 1
  --seed "$SEED" --max-tokens 2048)
"${CACHE_PROBE[@]}" --output "$RUN/cache-${SEED}-before.jsonl"
```

1. Require a correct answer and cache publication on **every rank**.
2. Save each rank's cache logs and counters and the model process identities.
3. Drain requests. Stop and restart the model with the guide's coordinated
   commands, keeping its cache directories. Do not restart NIC helpers on their
   own or guess container names.
4. Wait for all-rank readiness and warmup.
5. Send the identical fixture:

```bash
"${CACHE_PROBE[@]}" --output "$RUN/cache-${SEED}-after.jsonl"
```

Require matching prompt hashes, a correct answer, compatible cache identities,
external-hit counters and per-rank restoration in the logs. GPU prefix reuse
inside one process does not test disk restoration. For the managed MTP3
profile, the [cache procedure](../GLM53_SPARK_MTP3_MESH_QUICKSTART.md#model-output-and-persistent-cache-restoration)
gives concrete restart commands and a second recall fixture.

## Mixed traffic and sustained operation

For each of three trials, measure a 60-second idle decode control, then run the
same decode cell while one fresh 64K prefill arrives. Use the same sampling and
workload settings for both.

```bash
DECODE_CELL=("${COMMON[@]}" --skip-prefill --contexts 32k
  --concurrency "$BUSY_C" --dcp-size "$DCP" --temperature 1
  --token-targeting exact --max-tokens 2048 --duration 60
  --decode-warmup-seconds 5 --cell-warmup-timeout-seconds 900)
for repeat in 1 2 3; do
  "${DECODE_CELL[@]}" --output "$RUN/idle-r${repeat}.json" </dev/null
done
```

Each loaded trial uses two terminals on the same benchmark host. Set `repeat`
to 1, then 2, then 3. In terminal A, with the variables above:

```bash
repeat=1
"${DECODE_CELL[@]}" --display-mode live \
  --output "$RUN/mixed-r${repeat}.json" </dev/null
```

Wait until the live display shows `ready C=... ctx=32K` for the selected
concurrency and the measurement countdown is running; a warmup timeout does not
count. Then, in terminal B with the same `RUN`, `ENDPOINT`, `MODEL`,
`CONTEXT_LIMIT` and `repeat`, send the prefill:

```bash
date -u +%FT%TZ > "$RUN/mixed-prefill-r${repeat}-started.txt"
python3 performance/harnesses/validation/prefill_probe.py \
  --endpoint "$ENDPOINT" --model "$MODEL" --contexts 64k \
  --context-limit "$CONTEXT_LIMIT" --repeats 1 --temperature 1 \
  --output "$RUN/mixed-prefill-r${repeat}.jsonl"
date -u +%FT%TZ > "$RUN/mixed-prefill-r${repeat}-finished.txt"
```

Use the decode event log and timestamps to confirm the prefill overlapped the
measured window, not context preparation or warmup; if not, repeat the trial
with a longer duration in both arms. After the three pairs, run the sustained
workload:

```bash
"${COMMON[@]}" --skip-prefill --contexts 32k --concurrency "$BUSY_C" \
  --dcp-size "$DCP" --temperature 1 --token-targeting exact \
  --max-tokens 2048 --duration 1800 --decode-warmup-seconds 5 \
  --cell-warmup-timeout-seconds 900 --output "$RUN/soak.json" </dev/null
sha256sum --check "$RUN/benchmark-source.sha256"
```

Watch memory and cache trends, request failures, transport counters and rank
health during the sustained run. Report mixed-load throughput, ITL and TTFT
changes against the paired idle controls. This runbook does not kill hosts or
change networking under load; fault injection needs its own procedure.

## Additional checks for the intended application

- **Quantization accuracy:** a fixed GSM8K, MMLU-Pro or GPQA subset from the
  benchmark's dataset profiles, with the same items and seeds as the reference
  quantization.
- **Scored coding:** a pinned coding-task suite with executable unit tests, run
  only in a disposable sandbox with no secrets, host mounts or network and with
  CPU, memory and time limits. Record pass counts separately from coding peak.
- **Structured output and tool use:** schema-valid JSON, valid tool arguments
  and correct stop behavior for the served tool/template configuration.
- **Multimodal input:** image and video checks, only for profiles that support them.

## Report and operator decision

For shared-image GLM activation receipts (schema
`sparkring-r33-activation-receipt/v1`), run the offline validator:

```bash
python3 runtime/common/verify_activation.py --receipt /path/to/activation.json
```

It requires each rank exactly once and checks the declared SparkCache state and
the profile's image, source and runtime identities. It does not contact hosts.

Write the report with [the template](../PROFILE_VALIDATION_REPORT_TEMPLATE.md):
recipe and source identities, commands, raw-receipt hashes, three-run medians
and ranges, accuracy counts, failures and restart/cache results. Link detailed
receipts rather than embedding private host configuration or full model
responses.

Transport correctness, the required retrieval cases, service health and cache
restoration are pass/fail. For sampled reasoning and coding tests, and for
latency and throughput, agree on the thresholds for the intended workload
before comparing profiles. Adopting a profile is the operator's decision.
