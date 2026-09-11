# Long-context retrieval validation

Status: **implemented** correctness harness for the
[profile validation runbook](../../../docs/operations/profile-validation.md).
Python 3.10 or later and the standard library are sufficient.

`needle_hunt.py` checks three deterministic repository fixtures:

- `exact`: retrieve a case-sensitive opaque identifier.
- `revision`: select the identifier in a superseding decision over a decoy.
- `join`: follow a service-to-alias mapping to its change-window identifier.

The program sends `/tokenize` and OpenAI-compatible `/v1/chat/completions`
requests sequentially. Use the deployed model's actual maximum context for
`--context-limit`, and leave room for the output budget. Target sizes describe
estimated corpus sizes, not exact token counts. Every case requires successful
tokenization and rejects an actual prompt plus output budget larger than the
declared limit. Endpoints without compatible `/tokenize` support are unsupported;
there is no approximate-count fallback.

```bash
python3 performance/harnesses/validation/needle_hunt.py \
  --base-url http://192.0.2.10:8015 --model SERVED_MODEL_NAME \
  --context-limit 1m --contexts 64k,128k,256k \
  --positions 5,50,95 --modes exact,revision,join \
  --temperature 1 --seed 20260723 --max-tokens 512 \
  --output /existing-results-directory/retrieval.jsonl
```

Authentication, when required, comes from `OPENAI_API_KEY` or the variable named
by `--api-key-env`. Keys are not accepted as visible command-line arguments or
written into configuration receipts. Redirects and environment HTTP proxies are
disabled. HTTP error bodies are excluded because servers can echo credentials.
Do not place credentials in `--base-url` or chat-template options.

`--chat-template-kwargs` accepts a JSON object applied identically to tokenization
and generation. Set it only when the deployed template requires that behavior.
Sampling defaults to temperature one; the fixture and generation seed are the
base seed plus the one-based case index. Output ending with `finish_reason=length`
is an error even if it contains the answer. Increase `--max-tokens` in a separate
recorded run when a model needs a larger reasoning/output budget.

The absent output path receives durable JSONL records: configuration, each case
as it completes, and a summary. A case records prompt SHA-256, fixture seed,
expected identifier, response, actual token count, usage, finish reason, and
timing. Existing files are never overwritten. Exit zero means every planned case
passed; exit two means a failed answer or request/guard error; interruption exits
130. The fixture scoring rule accepts exactly one occurrence of the
expected identifier; `exact_output` separately records strict answer equality.
This does not prove absence of contradictory extra text.

`--repetitions 2` repeats each identical prompt with the same seed. Separate runs
with the same ordered contexts, positions, modes, and seed also reproduce prompt
hashes. These enable before/after cache checks without changing cache state.
Correlate responses with server cache counters; identical prompts alone do not
prove cache hits or restored prefixes. The harness never restarts a model,
deletes caches, or changes network configuration. Timings include serving work
and are not a cold-prefill throughput benchmark.

## Fixture provenance and offline checks

The repository fixture behavior derives from `llm_needle_hunt.py` version
`0.1.0`, SHA-256
`0f3d2ff9d47bae66657da883c978bd0074e87e592e4d1c0b177e6e58084238e4`.
This repository contains the complete required fixture implementation. Tests pin
the generated document hashes for all three modes. Fixture text is synthetic
test data, including its filenames, revision labels, and authority declarations.

```bash
python3 -m pytest performance/harnesses/validation -q
```

Tests do not contact an API or host. Records should additionally identify the
model revision, immutable image, serving settings, topology, cache state, and
concurrent traffic before supporting a profile-validation claim.

## Prefill measurement

`prefill_probe.py` measures temperature-controlled client TTFT with three
unique-prefix samples per context. It calibrates prompts using `/tokenize`,
records actual usage and cache evidence, and writes incremental JSONL receipts.
It requests one output token; its metric is prompt tokens divided by TTFT.
Samples require a stream completion terminator, a normal `stop` or `length`
finish reason, and positive integer prompt/output usage. Malformed cache counts
are rejected. Missing cache counts remain unknown, not evidence of a cold prefix.
See the shared runbook for commands and the complete measurement protocol.

## Growing-conversation soak and idle probes

Status: **implemented**. `conversation_soak.py` reuses the prefill probe's
HTTP and streaming helpers to measure concurrent conversations that grow by a
calibrated number of prompt tokens each turn. It runs sequential, unique-prefix
5,300-token probes before and immediately after the conversation workers finish.
It never changes the serving stack, speculation method, cache access mode, or
namespace. Select those externally and record their identities with `--metadata`.
The `--arm` option labels a receipt; it does not configure the server.

Print the request and output bounds without any network activity first. This
example targets the GLM-5.3 MTP3 profile:

```bash
python3 performance/harnesses/validation/conversation_soak.py \
  --plan --model glm-5.3-flash-spark --arm baseline-mtp3 --context-limit 1m \
  --concurrency 2 --start-tokens 32768 --max-turns-per-agent 4 \
  --tail-tokens 2048 --max-tokens 64 --probe-output-tokens 64 \
  --duration-seconds 300 --max-soak-prompt-tokens 1000000 \
  --reset-tokens 160000 --seed 2026090601
```

Replace `--plan` with `--endpoint http://192.0.2.10:8015` and
`--output /existing-results-directory/baseline-mtp3.jsonl` to execute. The
short shape admits at most eight conversation requests and six idle probes.
Every request is guarded by successful server tokenization and the declared
context limit. Calibration preserves prior messages and includes the same
chat-template options used for generation.

For a longer MTP3 run, use `--concurrency 4 --start-tokens 100000
--max-turns-per-agent 200 --duration-seconds 3600 --max-soak-prompt-tokens
50000000 --reset-tokens 160000 --max-tokens 512 --probe-output-tokens 300`.
Keep the remaining options explicit and run `--plan` first. The time limit
stops admission of new soak requests; calibration, probes, and requests already
in flight can extend wall time. The HTTP timeout is an inactivity timeout.
The global token ceiling counts full prompt tokens admitted across all workers,
including cached tokens, and excludes the separately bounded idle probes.
Each worker stops when its next prompt cannot fit the remaining shared token
budget; other workers may still admit smaller prompts. Token or turn limits
can end the run before the configured duration. Workers rotate to a fresh
conversation before its next prompt and output would exceed `--reset-tokens`.
Request errors stop further soak admissions; other in-flight requests can finish.

Use identical seeds, shapes, template settings, MTP3 settings and model
identities for paired runs. Provide isolated persistent namespaces externally
when cold independence matters, or explicitly record retained cache state.
The same seed produces the same initial synthetic prompts, while actual
assistant replies grow subsequent histories; receipts preserve those replies
and prompt hashes. Distinct seeds create fresh contexts but do not establish a
byte-identical paired comparison. Probe prefixes differ between the before and
after phases; reported cached-token counts remain available to check whether
either population was already cached. A run name does not isolate GPU prefix
caching or disk state.

`--image /path/to/fixture.png --image-every 10` adds the same local PNG or JPEG
to every tenth continuation. The receipt records its SHA-256. Keep the fixture
alongside the receipt when reproducing a multimodal run. Image bytes are sent
as a data URL and never fetched from another server. The image must be at most
4 MiB. Tokenization must account for it; an endpoint that cannot tokenize this
shape fails calibration instead of using an estimated count.

Each turn records the client request ID sent in `X-Request-ID`, response and
server request IDs when returned, start time, actual usage, tokenized prompt
count, prompt hash, assistant text and reasoning, finish reason, elapsed time,
TTFT, and timestamp offsets for every content or reasoning delta. TTFT runs
from client submission to the first nonempty content or reasoning delta.
The decode estimate is `(completion_tokens - 1) / (last_delta - first_delta)`;
streaming chunks may contain several tokens, so this is not a token-level
inter-token latency measurement. The full delta timestamps and usage allow
recalculation. `finish_reason=length` is accepted for the bounded load and is
recorded as output-budget exhaustion, not a correctness pass.

The summary compares continuation turns with reported cached tokens below or
above half the prompt, using the condition of a positive prompt increase below
10,000 tokens in the same conversation. Missing cached-token usage is excluded
from that classification. These are server-reported numbers: the harness does
not infer local reuse, external restore, or recomputation from latency. Correlate
request IDs with connector and vLLM logs to establish those causes. Tokenization
uses the serving host's CPU and is excluded from each chat latency; its overhead
and in-flight work affect the interval before the after probes.

Analyze a saved receipt offline with:

```bash
python3 performance/harnesses/validation/conversation_soak.py \
  --model glm-5.3-flash-spark --arm analysis --context-limit 1m \
  --analyze /existing-results-directory/baseline-mtp3.jsonl
```

JSONL records are flushed after each completion and existing output files are
never overwritten. API keys come from the environment and are omitted from
receipts; error bodies are omitted as well. Preserve model/image/source IDs,
topology, actual KV allocation, access mode, namespace, concurrent traffic and
initial store occupancy in the supplied metadata or an accompanying evidence
record. One short successful run does not qualify a long soak or a performance
improvement. Offline tests use in-memory HTTP fixtures and contact no hosts.

`analyze_conversation_reuse.py` joins a saved receipt to the compact
`sparkcache-reuse-trace/v1` records in saved rank logs. Supply the expected
physical ranks explicitly. A verified-restore classification requires a
successful completion for the same engine request, digest and token span on
every expected rank; an offer alone is insufficient. Scheduler lease-attachment
events are reported separately. API-only evidence is labeled `reported_cached`:
positive counts do not identify local versus external reuse, and zero counts
do not exclude a GPU lease.

The join accepts recorded request/response IDs and the runtime's exact
eight-hex-character engine suffix. Ambiguous joins cannot establish a restore
quorum. The output preserves per-rank queue, service and phase timings without
summing token spans across ranks. Log captures can contain requests outside the
receipt window; inspect unmatched IDs and timestamps before assigning them to a
missing-rank or request failure.
