# EXL3 cache-disabled teacher-forced diagnostic — 2026-08-10

## Result and evidence boundary

On four directly cabled NVIDIA DGX Sparks, the public-functional EXL3
3.25-bpw image was run at TP4/DCP4 in attribution arm
`a-mtp0-apc0-lmcache0`: MTP disabled, native prefix caching disabled, the
LMCache connector detached, and SparkCache disabled. The arm was attested on
all four ranks immediately before the requests.

Twenty temperature-zero, fixed-seed autoregressive repetitions produced five
complete token sequences with multiplicities `15/2/1/1/1`. Their earliest
divergence was zero-based generated index 112. Replaying the exact same forced
prefix 20 times kept the returned top-1 token stable at index 112, but the
returned top-1 choice split `19/1` at generated index 116.

This is a **public-functional live diagnostic observation**, not acceptance.
It demonstrates that cache reuse was not required for this same-context
returned top-1 nondeterminism observation: MTP and native prefix caching were
disabled, and LMCache was detached. It does not establish independence across
cache states or identify the responsible kernel, graph, attention path,
collective, quantization path, or other subsystem.

The raw report is private because it contains a site-specific API origin and
reversible token arrays. Its SHA-256 is
`51613c0e1260e1adfafa3bc925e9921aac36e3e518fa1a7c53efa9a1a087de2b`.
The closed, sanitized
[machine-readable receipt](configurations/glm52-exl3-teacher-forced-attribution-20260810.json)
has SHA-256
`89d96649a93e151621e0643fee3f4f601e97653317a959f5fb37600b2e0cbbac`.
It binds the private report by hash while omitting the prompt, token IDs,
outputs, API origin, host and container identifiers, and commands.

## Method

The diagnostic used `scripts/exl3_teacher_forced_margin_probe.py` in two
stages:

1. Run the same finite completion 20 times with temperature zero, a fixed
   seed, native cache isolation, and returned token IDs. Locate the first
   generated position where the sequences differ.
2. Choose one reference completion, force its exact prompt-plus-completion
   prefix, and request raw top-20 prompt logprobs 20 times at each generated
   position from 104 through 120. Compare the returned top-1 choice, top-1 to
   top-2 margin, common returned-token values, top-k overlap, and a
   common-support conditional symmetric KL diagnostic.

Teacher forcing removes context drift: every repetition at one probed
position uses the same token context. It does not make the returned top-k a
full distribution and does not identify the execution component responsible
for a changed score.

The probe implementation sets temperature to zero and reuses the selected
case's configured seed. The v1 public receipt does not independently serialize
those two request values; they are supported by the published probe behavior
and the private-report hash chain. A future receipt schema should carry and
validate them directly.

The live report was produced by the unchanged EXL3 image
`sha256:20c4099f2e7e3dd3c8ab64f7d7930bde4f372df1895aa3ffa593252ca04ae96f`
and model revision
`d7d79c2d14599dfce7a5d12b85f7ad73f40e623d`. The receipt records the pinned
vLLM, SparkInfer, and ExLlamaV3 source identities and clearly labels those
source pins as declared canonical pins, not live binary introspection.

## Key observations

| Generated index | Returned top-1 | Top-1/top-2 margin | Maximum common-token value delta | Minimum top-k Jaccard | Maximum conditional symmetric KL |
|---:|---|---:|---:|---:|---:|
| 112 | stable, `20/20` | 0.25–0.625; median 0.375 | 0.4009113 | 0.81818 | 0.018199 |
| 116 | split, `19/1` | 0–0.125; median 0.0625 | 0.6853895 | 0.81818 | 0.014295 |

The index-116 observation changes the aggregate diagnosis to
`cache-not-required-top1-nondeterminism-observed`. The raw report's aggregate
label was generated before the classifier was corrected to inspect every
probed position; the public reducer recomputes the label from the position
summaries and records that the raw aggregate label was stale. The raw
position-level measurements remain unchanged.

The service returned raw top-k prompt logprobs, not raw full-vocabulary logits.
Pairwise differences between returned raw logprobs are logit-margin
differences, but the reported symmetric KL is truncated to common returned
support and renormalized. It is **not full-vocabulary KLD**.

## Operational closeout

After the diagnostic, the launcher removed the MTP0/cache-off arm, recreated
all four LMCache servers, and restored the canonical fixed-MTP2, native-prefix-
cache-enabled, LMCache-CS512 deployment. All 36 rollback phase/rank actions
exited zero, all four cache servers were healthy, and the OpenAI-compatible API
returned HTTP 200. SparkCache remains disabled in the canonical configuration.

The private rollback report has SHA-256
`31006c184801772bca1ad7133ba1379c1b8d2df9a92e7489445990b0784f74f1`.
Its closed, sanitized
[canonical-restore receipt](configurations/glm52-exl3-teacher-forced-canonical-restore-20260810.json)
has SHA-256
`3370c5289ef96f5fe7303e9e369588b7b498097ba0a3d54bdba46f10123ac610`.
The public receipt records nine phases across four ranks, all with exit code
zero, while omitting remote commands, hosts, output, and exception text.

This closeout proves restoration of the advertised configuration and bounded
API health. It is not a new correctness or quality acceptance result.

## Reproduction and sanitization

Use an ignored, operator-local site file and case configuration. First create
and inspect a plan; `--execute` contacts the live endpoint and is appropriate
only after arm A has been activated and re-attested as described in the
[EXL3 acceptance runbook](EXL3_ACCEPTANCE_RUNBOOK.md).

```powershell
python scripts/exl3_teacher_forced_margin_probe.py `
  --config <IGNORED_CASE_CONFIG> `
  --case-id code-stable-unique `
  --site scripts/config/site.yaml `
  --profile <IGNORED_RESOLVED_EXL3_PROFILE> `
  --activation-receipt <PRIVATE_LIVE_ARM_RECEIPT> `
  --base-url http://<RANK0_MANAGEMENT_ADDRESS>:8000 `
  --model glm-5.2-exl3-tr3-3.25bpw `
  --attribution-arm a-mtp0-apc0-lmcache0 `
  --discovery-repetitions 20 `
  --discovery-max-tokens 128 `
  --window-before 8 `
  --window-after 8 `
  --teacher-forced-repetitions 20 `
  --top-logprobs 20 `
  --output .sparkring/exl3-attribution/teacher-forced-private.json
```

Add `--execute` only after reviewing that plan. Never publish the resulting raw
report. Produce the public receipt with the fail-closed reducer:

```powershell
python scripts/exl3_teacher_forced_margin_reduce.py `
  --input .sparkring/exl3-attribution/teacher-forced-private.json `
  --output docs/configurations/glm52-exl3-teacher-forced-attribution.json
```

The reducer rejects duplicate keys, non-finite values, schema drift, token/hash
inconsistency, summary tampering, an unattested or non-MTP0 arm, cache-enabled
input, and unsupported full-logit/KLD claims. It recomputes both the
per-position summaries and the aggregate classification before emitting only
allowlisted public fields.
