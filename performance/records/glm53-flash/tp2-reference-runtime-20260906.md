# Two-node GLM-5.3 Flash reference runtime

Status: research-only. Historical functional record; this is not a universal
throughput claim or qualification of other container images.

## Conditions

Two DGX Spark GB10 nodes, tensor parallel degree 2 and decode-context parallel
degree 1. Model: local-inference-lab/GLM-5.3-Flash-NVFP4-Spark at revision
df116c4fb16b1d37ae43d2cfd624de26ffbc832e. The Docker image identity is
sha256:4a4004b2d855b4cfc7f92eddbe4068676e10fd254cb37d6336f0dc463a2078c0,
with the attention overlay and two native libraries identified by hashes in
runtime/sparkring/manifest.json. This is the reference runtime, not the
published packaging image identified in runtime/sparkring/publication.json.

Settings: native adaptive multi-token prediction with at most 3 draft tokens;
5 GiB KV per rank; maximum context 524,288 tokens; maximum concurrent sequences
8; batch token budget 8,192; prefill interval 2. Images and video each have a
per-prompt limit of 1. Host guards stop opted-in containers after two
consecutive one-second readings below 4 GiB available RAM.

## Measurement

[Raw observations](tp2-reference-runtime-20260906.json) preserve three single-
request runs, one cohort of eight requests, and one long-prompt run. The
single-request fixture has 28 input and 256 output tokens; temperature 0,
seed 42, streaming output and ignored EOS. The eight-request fixture uses
2,048-token synthetic documents containing distinct codewords and up to 128
output tokens each. The long-prompt fixture uses 131,072 input tokens and
32 output tokens. Reported usage shows no cached tokens for the cold
long-prompt request. Timing uses the client monotonic clock and server
counter deltas, as recorded in the JSON. Tests are bounded smoke checks,
not repetitions sufficient for confidence intervals or hardware profiling.

## Result

Single-request decode was approximately 31–35 output tokens/s excluding first-
token latency. The eight-request cohort reached eight running engine requests
and returned all eight codewords. Its aggregate end-to-end output rate was
37.33 tokens/s, including input processing. The long prompt returned the
correct codeword with 65.33 seconds to first token. Both nodes committed
130,816-token cache captures, about 914.3 MiB per node.

Separately transcribed operator observations recorded minimum available RAM
of 4.920 GiB on rank 0 and 6.927 GiB on rank 1, and successful capture commits
on both ranks. Those memory samples and worker log excerpts are not included
in the linked request JSON; their completeness cannot be independently checked
from this record. No guard stop was reported during this bounded workload.

## Conclusion

The identified reference runtime supports the listed short concurrency and
long-prompt capture checks. These observations informed the documented profile
but do not automatically qualify the separately packaged public image.

## Limitations

No full 524,288-token request or eight-way large-context qualification is
established. Synthetic filler can provoke repetitive continuation after a
correct codeword. The API may expose reasoning in the content field; strict
JSON-only formatting and reasoning separation are unqualified. The record
does not establish a speed advantage over a different batch token budget.
The original request-generation harness is not bundled with this historical
record. The workload descriptions and raw response records are retained,
but exact prompt reconstruction and a fully reproducible timing comparison
require that harness; these observations must not be treated as such a comparison.
