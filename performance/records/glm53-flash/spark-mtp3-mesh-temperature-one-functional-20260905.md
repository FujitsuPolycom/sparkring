# Native-MTP3 mesh image with temperature-one readiness: functional checks

Status: **research-only**. This record concerns the exact sampling-warmup
image below. It does not inherit results from the
[separate image functional record](spark-mtp3-mesh-image-functional-20260905.md).
The [JSON record](spark-mtp3-mesh-temperature-one-functional-20260905.json)
contains selected observations and source-receipt hashes. Operational receipts
containing site details remain private.

## Conditions

Four DGX Spark GB10 nodes form a physical ring with ConnectX-7
hardware-forwarded diagonals. The model is
`local-inference-lab/GLM-5.3-Flash-NVFP4-Spark` revision
`df116c4fb16b1d37ae43d2cfd624de26ffbc832e`, using native MTP with three
speculative tokens and explicit B12X draft attention. TP4/DCP4/PP1, the
8,192-token scheduling limit, 16-sequence limit, and graph rows 4 through 64
in multiples of four are unchanged.

The image is
`sha256:2b650e432d4d08e1999c276bd9b55032fceb89836a0c3c045da39d98a289acaf`.
Its source receipt is
`9656542cc5b2d37f847925442733702c98d7fdf66ef162bfddcfaab345dff401`.
The transport manifest remains
`4204fabc93303226b9a120b094ef3c82ed4aadd1d7f97cfbe291204c027ed45f`.
The readiness helper has SHA-256
`f41c38eef41d15d63dcfc49cd6643357ca1a3ae18200ddbe4f8692d0b767ee79`.

The child image explicitly replaces the readiness helper and sets
`SPARKRING_WARMUP_TEMPERATURE=1`. Model packages, kernels, and transport source
are preserved. Readiness requests still disable thinking. Model checks in this
record separately enable thinking and use temperature one.

## Measurement

The native runner checks byte-exact BF16 all-reduce output at `[Q, 4096]`
for Q4, Q20, Q28, and Q64. Five cases per rank and shape include eager output,
changed graph inputs, and poisoned outputs. Two warmups and three diagnostic
timing samples accompany each shape; this record makes no speed comparison.

Startup is the operator's elapsed wall clock for the rank-zero launch command
to return ready. One observation is reported, without a variability estimate.
Arithmetic requests use requested concurrency 5, 7, and 16, temperature one,
thinking enabled, and a 512-token output limit. Requested concurrency is not
a measured sustained-concurrency guarantee.

The persistent-prefix check asks the model to recall `cobalt orchard lantern`
from a 27,274-token prompt. Its uncached response and per-rank publication
logs are separate gates. The same submitted request was repeated after a
stopped-container restart, with per-rank restoration logs and external-cache
metric deltas checked independently of response latency.

## Result

| Gate | Result | Observation |
|---|---|---|
| Image and warmup configuration | Passed | Exact image and temperature-one environment on all four ranks |
| Readiness sampling execution | Passed | Helper receipt records all 20 warmup batches at temperature one |
| Native RC output | Passed | 80 byte-exact checks across four shapes and four ranks |
| Native completion accounting | Passed | 24 origin QPs per shape; 16 direct and eight forwarded; zero completion errors |
| Initial startup | Passed | Rank-zero launch returned ready in 319.7 seconds |
| Thinking-enabled sampling arithmetic | Passed | 5/5 at requested C5, 7/7 at C7, and 16/16 at C16 |
| Uncached long-prefix recall | Passed | Expected phrase returned; 27,274 prompt tokens and zero cached tokens |
| Persistent publication | Passed | 26,624 tokens committed on every rank |
| Serving restart | Passed | Four ranks healthy; API and liveness ready after stopped-container restart |
| Bounded persistent restoration | Passed | Same request, 26,624 external-hit tokens, four restoration logs, expected phrase returned with stop finish reason |
| Post-restore transport progress | Passed | Every rank nonfatal and caught up; published and completed sequence both 8,182 |

The post-restart request restored 26,624 of 27,274 prompt tokens and computed
the remaining 650. All four ranks reported 79.1 MiB restored, taking 150.8,
175.4, 180.1, and 170.6 ms for ranks zero through three. These are per-rank
transfer observations, not end-to-end prefill throughput. The request returned
`cobalt orchard lantern`, satisfying the recall check.

This one-prompt, one-restart restoration check is **qualified** only for the
conditions above. The serving profile remains **research-only**.

The post-restore status snapshot reports 2,399 SIRCL captured nodes and 568
RoCEnante captured nodes per rank. The caught-up sequence check establishes
bounded progress at that snapshot, not sustained throughput or fault recovery.

A prompt-control observation is retained: an arithmetic request prefixed
`Check 16-7` reached its 512-token limit while reasoning about subtraction.
Removing the prefix produced 28/28 passing outputs across C5/C7/C16 without
changing the model. The prompt changed, so this is not an exact-prompt repeat
or proof of a transport fault. The JSON record retains output-check counts
for both prompt variants.

The readiness helper's output receipt was buffered and recovered from
rank-zero logs after serving stopped. Its 20 temperature-one batches are
execution evidence, not a claim that this receipt was visible during startup.
Restart readiness must check the API and liveness endpoint; a retained Docker
health status alone can briefly report healthy before readiness is re-established.

## Conclusion

The exact sampling-warmup image passes bounded native communication, startup,
temperature-one arithmetic, uncached recall, four-rank persistent publication,
stopped-container restart, and one external-restoration recall check. This is
not general cache, mixed-workload, or failure-containment qualification and does
not establish a recommendation or authorize publication.

## Limitations

Thinking-enabled model checks do not add thinking-enabled readiness warmup.
Mixed-prefill/decode, prolonged stress, and failure containment are not qualified;
these checks do not close the broader readiness coverage in
[issue #214](https://github.com/FujitsuPolycom/sparkring/issues/214).

Temperature-one output is stochastic. One canary or a fixed seed does not
establish general model quality, output equivalence, or cache correctness.
The [temperature-zero image's failed semantic gate](spark-mtp3-mesh-image-functional-20260905.md)
remains separate and unresolved; this passing recall check does not identify
its cause.
The hardware forwarding helper still has a bounded lifetime and requires
supervision. No model throughput or transport speedup is claimed.
