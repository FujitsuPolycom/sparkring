# Sealed `llm_decode_bench` C8 integration

SparkRing carries an additive integration patch for the exact historical
`llm_decode_bench.py` v0.4.31 source used by the published EXL3 results.  The
unmodified source SHA-256 is:

```text
8de7c32c0abae3c664226fb9c1c197d0752c0a0f3f5a87b3357326f1407f9c07
```

The patch adds `--prompt-manifest-out` and `--prompt-manifest-in`.  The public
implementation and validator live in
[`scripts/llm_decode_prompt_manifest.py`](../../scripts/llm_decode_prompt_manifest.py).
It records the exact JSON body for every prefix scout and every per-stream
decode request template, including an individual SHA-256 for every body and a
commitment over the complete ordered sequence.

This is an offline-validated evidence-harness extension.  It does not promote
SparkCache or any serving configuration by itself.

## Prepare the benchmark

Verify the input before applying the patch.  Do not apply it to a different
benchmark revision and assume the evidence remains comparable.

```powershell
$BenchRoot = "<BENCH_ROOT>"
$SparkRingRoot = "<SPARKRING_ROOT>"

(Get-FileHash -Algorithm SHA256 "$BenchRoot/llm_decode_bench.py").Hash.ToLower()
git -C $BenchRoot apply --check "$SparkRingRoot/runtime/benchmarks/llm_decode_bench-v0.4.31-prompt-manifest.patch"
git -C $BenchRoot apply "$SparkRingRoot/runtime/benchmarks/llm_decode_bench-v0.4.31-prompt-manifest.patch"
$env:PYTHONPATH = "$SparkRingRoot/scripts" + [IO.Path]::PathSeparator + $env:PYTHONPATH
```

The resulting file byte hash can differ between LF and CRLF checkouts.  The
input hash and `git apply --check` are the portable source/patch attestation;
retain the patched file itself with each private run bundle as usual.

## Create one sealed workload

Run a preparation invocation with the exact settings intended for every arm.
The preparation result is **not** one of the A/B/B/A measurements.  Restore the
chosen cold/cache state after preparation and before A-open.

```powershell
python "$BenchRoot/llm_decode_bench.py" `
  --host <RANK0_API_ADDRESS> --port 8000 `
  --model glm-5.2-exl3-tr3-3.25bpw `
  --contexts 16k --concurrency 8 `
  --duration 25 --max-tokens 2048 --temperature 0 `
  --unique-context-percent 100 --dcp-size 4 `
  --decode-warmup-seconds 3 --skip-prefill `
  --prompt-manifest-out <PRIVATE_EVIDENCE_ROOT>/sealed-c8-prompts.json `
  --output <PRIVATE_EVIDENCE_ROOT>/manifest-preparation.json
```

Manifest creation reserves the output name before the first request and updates
it atomically.  An interrupted preparation remains marked incomplete and
cannot be imported.  Reusing an existing manifest output path is an error.

## Run A/B/B/A

Run four independently named benchmark artifacts in this order:

1. cache/runtime arm A (`A-open`);
2. candidate arm B (`B-first`);
3. candidate arm B again (`B-second`);
4. original arm A again (`A-close`).

Every invocation must use the same arguments as preparation, replacing only
`--prompt-manifest-out` with:

```powershell
--prompt-manifest-in <PRIVATE_EVIDENCE_ROOT>/sealed-c8-prompts.json
```

Import bypasses the benchmark's random calibration request.  It reconstructs
the normal request settings, requires every non-message field to equal the
sealed value, substitutes the exact sealed messages, and fails if calls are
missing, reordered, or added.  Each benchmark result embeds a summary proving
that the complete manifest was consumed.

The integration intentionally accepts only sustained `16K/C8`,
`--skip-prefill`, no burst/coding probe, the pinned `llm_decode_bench.py`
v0.4.31 harness, vLLM, a 25-second requested window, and at least 256 maximum
output tokens. The public manifest summary includes an allowlisted exact
workload and ordered call-descriptor projection, but never request messages or
payloads. Unknown workload/descriptor fields and path-like model identifiers
fail closed so local hosts, paths, credentials, or operator notes cannot leak
through the sanitized report.

## Compare without exposing local paths

```powershell
python "$SparkRingRoot/scripts/compare_sealed_c8_abba.py" `
  --a-open <PRIVATE_EVIDENCE_ROOT>/a-open.json `
  --b-first <PRIVATE_EVIDENCE_ROOT>/b-first.json `
  --b-second <PRIVATE_EVIDENCE_ROOT>/b-second.json `
  --a-close <PRIVATE_EVIDENCE_ROOT>/a-close.json `
  --prompt-manifest <PRIVATE_EVIDENCE_ROOT>/sealed-c8-prompts.json `
  --output <PRIVATE_EVIDENCE_ROOT>/sealed-abba-report.json
```

The comparator emits no input paths. It independently validates and hashes the
authoritative sealed manifest, then requires all four embedded summaries to
match that manifest's exact commitments. It rejects any artifact unless all four
are valid sustained 16K/C8 cells at temperature zero, sustained all eight
streams without errors/readiness/capacity suppression, have exactly matched
workload metadata, and prove complete import of the same content and payload
sequence commitments. Each result must use `openai_continuous_usage`, report a
positive `client_output_tokens`, satisfy
`aggregate_tps = client_output_tokens / measurement_seconds`, and close within
one second of the requested 25-second window. The comparator also binds the
manifest's harness, served-model identifier, workload, and exact warmup/measured
scout/template sequence to every artifact; matching hashes alone are
insufficient. Its two-observation-per-arm delta is descriptive; it is not a
statistical-significance claim.
