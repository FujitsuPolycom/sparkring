# Validate a serving profile

Use this runbook to measure a deployed profile's speed, long-context
accuracy, concurrency behavior, and restart/cache behavior. Keep the results
with the recipe so another operator can repeat the same workload.

Status: **implemented**—the test tools are ready to use. Record which checks
pass and how the profile performs, then use those results to decide whether
it is ready for your workload.

## Required test card

Start by filling the report with results you already have. Reuse completed
runs when their model, configuration, workload, and measurement method fit
the check. Keep different configurations in separate rows, and record how
many repeats are actually available. Then run only the missing checks or
the repeats needed for the comparison you want to make.

The [native-MTP3 hybrid report](../performance/records/glm53-flash/spark-mtp3-validation-summary-20260905.md)
shows an example: existing decode, Estonia, needle, and cache results are
combined with three-pass prefill measurements, leaving a short gap list.

| Check | Standard workload | Repetitions | Record |
|---|---|---:|---|
| Startup and readiness | Every rank, API, scheduler, configured warmup | Initial start and one planned restart | Versions, startup time, errors, readiness |
| Prefill | C1; 8K, 16K, 32K, 64K, 128K | 3 samples per context | TTFT, actual prompt tokens, tokens/s, cache evidence |
| Sustained decode | 8K, 32K, 64K × C1, C2, C4, C8, C12, C16 | 3 complete sweeps | Aggregate tokens/s, TTFT, ITL, effective concurrency, acceptance |
| Coding peak | Built-in coding workload, temperature 1 | 3 samples | Generation tokens/s, output length, truncation |
| Estonia accuracy | Estonia v2, C1 and C8, temperature 1 | 30 requests at each concurrency | Correct/attempted, truncation, completion-token distribution |
| Long-context retrieval | Exact value, superseding revision, cross-reference; depths 5%, 50%, 95% | One sweep at 128K/256K/512K; repeat failures and matched controls | Actual tokens, answer, finish reason, seed |
| Persistent cache, when enabled | Identical long prompt before and after model restart | 3 independent prompt seeds | Correct answer, prompt hash, per-rank publication/restoration, external hits |
| Mixed traffic | C4 decode while a fresh 64K prefill arrives | 3 paired idle/loaded trials | Decode throughput/ITL change and prefill TTFT |
| Sustained operation | Representative context and concurrency | 30 minutes | Request errors, memory/cache trend, transport health |

Choose contexts and concurrency within the profile's declared limits. Record
unavailable cells as **unsupported by this profile**, not as zero throughput.
For profiles supporting more than 512K context, add a retrieval case near the
declared limit while reserving room for the answer. Actual tokenization, not
the context label, determines whether a request fits.

For a quick tuning screen, run one prefill sweep, 32K/C1/C4/C8 decode, and
three retrieval modes at one long context. Use the complete card for a
recommendation or a comparison intended for publication.

## Prepare one private results directory

Commands below use Bash on Linux and a trusted, OpenAI-compatible local
endpoint. Run them from the SparkRing checkout. Replace these values from
the profile's quickstart; the example limits describe a TP4/DCP4 profile.

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
curl --fail --silent --show-error "$ENDPOINT/v1/models" > "$RUN/models.json"
curl --fail --silent --show-error "$ENDPOINT/health" > "$RUN/health-before.txt"
```

Record the recipe path, image digest, model revision, quantization, TP/DCP/PP,
context/sequence/batch limits, attention/MoE/linear backends, graph shapes,
transport routes, speculation settings, cache policy, and request sampling.
Keep a copy of the resolved launch configuration privately. Do not include
API credentials in receipts or public documents. Record thinking mode and
chat-template settings; keep them fixed between comparison runs.

Use the profile's documented all-rank readiness check. Start measurements only
after its warmup completes and unrelated traffic is stopped. If custom native
collective checks are supplied, run them in their documented stopped-model
window before serving; do not launch GPU transport probes beside the model.

### Obtain the throughput/accuracy benchmark

Use [Local Inference Lab's llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench).
These commands use revision `bd88816e9e7bcc97e1bcfd954c3053528f31af69`.

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

Decline automatic source updates during a measurement campaign. The commands
redirect stdin from `/dev/null`; verify the source hash after the campaign.
Authenticated endpoints require the benchmark's documented authentication
configuration. SparkRing's two validation probes accept an API key through
the environment named by `--api-key-env`, default `OPENAI_API_KEY`.

## Prefill: three samples per context

The [prefill probe](../performance/harnesses/validation/prefill_probe.py)
explicitly sends temperature one, creates a unique prompt prefix for each
sample, and measures client time to the first content or reasoning token.
It requests one generated token and records server usage. The benchmark's
integrated scouts are useful displays, but this probe supplies the explicit
sampling and repeat policy for the test card.

```bash
python3 performance/harnesses/validation/prefill_probe.py \
  --endpoint "$ENDPOINT" --model "$MODEL" \
  --contexts 8k,16k,32k,64k,128k --context-limit "$CONTEXT_LIMIT" \
  --repeats 3 --temperature 1 --output "$RUN/prefill.jsonl"
```

Report median, minimum, and maximum TTFT and prompt tokens/s for each context.
Use actual server prompt counts. Require zero cached tokens for a confirmed
cold-prefix sample; if the server omits that field, corroborate with its
cache/compute counters. Keep cache-primed measurements separately labeled.
Do not clear unrelated persistent cache data to manufacture a cold result.

## Decode matrix and coding peak

Run the full matrix three times. The first sweep also collects three coding
peak samples; that benchmark runs the coding workload after its decode matrix.

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

Require the achieved concurrency to match the requested concurrency. Review
errors, underfilled cells, capacity limits, warmup timeouts, and token-count
method before aggregating. Report per-cell median and range across the three
sweeps. Keep failed attempts with their reason and a separately named retry.

Record speculative depth and acceptance alongside output throughput.
Tokens/s divided by acceptance length is a derived rate; GPU forward counts
need separate instrumentation. For coding peak, report median and peak
generation speed plus output lengths and truncation. Coding peak measures
speed; functional coding accuracy belongs to a separately scored test suite.

## Estonia: long-context accuracy and consistency

Run the [Estonia benchmark](https://github.com/local-inference-lab/llm-inference-bench)
at C1 and C8. Each setting uses thirty sampled runs of the same task. Change
C8 to a supported concurrency if the profile's sequence limit is lower.
Ensure the fixed prompt plus its output budget fits the profile before starting.

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

Report correct/attempted, errors, output-budget hits, and completion-token
distributions. Inspect wrong or unparseable final answers. The test primes
the prefix cache, so its TTFT describes that workload. Its weighted generation
rate uses summed request times; use the decode matrix for aggregate cluster
throughput. Preserve full outputs privately for review.

## Long needle hunt: retrieval, revisions, and cross-references

The [repository retrieval harness](../performance/harnesses/validation/README.md)
places opaque values at controlled document depths. `exact` retrieves a value,
`revision` selects a superseding record over a decoy, and `join` follows an
alias between separated records. Tokenization and completion-budget checks
run before submission.

```bash
python3 performance/harnesses/validation/needle_hunt.py \
  --base-url "$ENDPOINT" --model "$MODEL" --context-limit "$CONTEXT_LIMIT" \
  --contexts 128k,256k,512k --positions 5,50,95 \
  --modes exact,revision,join --temperature 1 --seed 20260905 \
  --max-tokens 2048 --output "$RUN/needle.jsonl"
```

Require the expected answer and normal completion for every supported cell.
Record actual prompt tokens, position, seed, answer, and finish reason.
Repeat failures with the same fixture and a shorter-context control before
changing the runtime. For one-million-token profiles, add a `--contexts 960k`
sweep if tokenization confirms that the complete request fits.

## SparkCache publication and restoration

For profiles without SparkCache, mark this check **not applicable**. For enabled
profiles, use the profile's documented publication threshold, cache namespace,
all-rank stop/restart procedure, and restoration logs. Do not rename cache
entries or change checkpoint/speculation identity between the two requests.

Run this sequence for three distinct seeds, using separate before/after files:

```bash
SEED=20260905  # Repeat the sequence with 20260906 and 20260907.
CACHE_PROBE=(python3 performance/harnesses/validation/needle_hunt.py
  --base-url "$ENDPOINT" --model "$MODEL" --context-limit "$CONTEXT_LIMIT"
  --contexts 128k --positions 50 --modes join --temperature 1
  --seed "$SEED" --max-tokens 2048)
"${CACHE_PROBE[@]}" --output "$RUN/cache-${SEED}-before.jsonl"
```

1. Require a correct answer and confirm compatible publication on **every rank**.
2. Save rank cache logs/counters and record the model process identities.
3. Drain requests. Use the selected quickstart's coordinated model stop and
   restart commands; preserve its cache directories. Do not restart NIC helpers
   independently or substitute guessed container names.
4. Wait for all-rank readiness and completed warmup.
5. Submit the identical fixture:

```bash
"${CACHE_PROBE[@]}" --output "$RUN/cache-${SEED}-after.jsonl"
```

Require matching prompt hashes, correct output, compatible cache identities,
external-hit counters, and explicit per-rank restoration evidence. GPU prefix
reuse within one process does not exercise disk restoration. For the managed
MTP3 profile, the [cache procedure](GLM53_SPARK_MTP3_MESH_QUICKSTART.md#model-output-and-persistent-cache-restoration)
provides concrete coordinated restart commands and a second recall fixture.

## Mixed traffic and sustained operation

For each of three trials, measure a 60-second idle decode control, then run
the same decode cell while submitting one fresh 64K prefill. Use the same
sampling and workload settings for both arms.

```bash
DECODE_CELL=("${COMMON[@]}" --skip-prefill --contexts 32k
  --concurrency "$BUSY_C" --dcp-size "$DCP" --temperature 1
  --token-targeting exact --max-tokens 2048 --duration 60
  --decode-warmup-seconds 5 --cell-warmup-timeout-seconds 900)
for repeat in 1 2 3; do
  "${DECODE_CELL[@]}" --output "$RUN/idle-r${repeat}.json" </dev/null
done
```

For each loaded trial, use two terminals on the same benchmark controller.
Set `repeat` to 1, then 2, then 3. In terminal A, with the variables above:

```bash
repeat=1
"${DECODE_CELL[@]}" --display-mode live \
  --output "$RUN/mixed-r${repeat}.json" </dev/null
```

Wait until the live display shows `ready C=... ctx=32K` for the selected
concurrency and the measurement countdown advances. A warmup timeout is not
that readiness event. Then, in terminal B with the same `RUN`, `ENDPOINT`,
`MODEL`, `CONTEXT_LIMIT`, and `repeat` values, submit the prefill:

```bash
date -u +%FT%TZ > "$RUN/mixed-prefill-r${repeat}-started.txt"
python3 performance/harnesses/validation/prefill_probe.py \
  --endpoint "$ENDPOINT" --model "$MODEL" --contexts 64k \
  --context-limit "$CONTEXT_LIMIT" --repeats 1 --temperature 1 \
  --output "$RUN/mixed-prefill-r${repeat}.jsonl"
date -u +%FT%TZ > "$RUN/mixed-prefill-r${repeat}-finished.txt"
```

Check the saved decode event log and timestamps to confirm the prefill
overlapped the measured decode window, not context preparation or warmup.
Repeat a trial with a longer duration in both arms if that condition is not
met. After the three paired trials, run the sustained workload:

```bash
"${COMMON[@]}" --skip-prefill --contexts 32k --concurrency "$BUSY_C" \
  --dcp-size "$DCP" --temperature 1 --token-targeting exact \
  --max-tokens 2048 --duration 1800 --decode-warmup-seconds 5 \
  --cell-warmup-timeout-seconds 900 --output "$RUN/soak.json" </dev/null
sha256sum --check "$RUN/benchmark-source.sha256"
```

Review memory/cache capacity trends, request failures, transport counters,
and rank health during the sustained run. Report the mixed-load throughput,
ITL, and TTFT changes relative to the paired idle controls. Fault injection
requires its own reviewed maintenance procedure; this runbook does not kill
hosts or alter networking during load.

## Additional checks for the intended application

- **Quantization accuracy:** a fixed GSM8K, MMLU-Pro, or GPQA subset using the
  benchmark's dataset profiles; preserve dataset identity and compare the same
  items/seeds against the reference quantization.
- **Scored coding:** a pinned coding-task suite with executable unit tests.
  Run generated code only in a disposable sandbox with no secrets, host mounts,
  or network and with CPU/memory/time limits. Record pass counts separately
  from coding peak speed.
- **Structured output and tool use:** schema-valid JSON, valid tool arguments,
  and correct stop behavior for the tool/template configuration actually served.
- **Multimodal input:** image/video checks only for profiles that support them.

## Report and operator decision

Use [the report template](PROFILE_VALIDATION_REPORT_TEMPLATE.md). Include the
recipe and source identities, commands, raw-receipt hashes, three-run medians
and ranges, accuracy counts, failure details, and restart/cache evidence.
Keep the report concise; link detailed receipts rather than embedding private
host configuration or complete model responses in the public repository.

Transport correctness, required retrieval cases, service health, and cache
restoration are explicit functional gates. For sampled reasoning/coding tests,
record scores and agree on the accuracy floor before comparing profiles.
Agree on latency/throughput budgets for the intended workload before declaring
a performance regression or a winner. A complete test card supplies evidence;
promotion remains an operator decision.
