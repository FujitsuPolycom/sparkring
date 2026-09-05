# Long-context retrieval validation

Status: **implemented** correctness harness for the
[profile validation runbook](../../../docs/PROFILE_VALIDATION.md).
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
See the shared runbook for commands and the complete measurement protocol.
