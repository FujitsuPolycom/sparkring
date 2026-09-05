# GLM-5.3 Spark native-MTP3 mesh image: functional checks

Status: **research-only**. Individual functional checks below passed; this
record is not a throughput benchmark or a recommendation to replace another
serving profile. The [machine-readable record](spark-mtp3-mesh-image-functional-20260905.json)
contains identities, selected measurements, harness hashes, and source-receipt
hashes. Operational receipts containing site details remain private.

## Conditions

The serving deployment uses four DGX Spark GB10 nodes in a physical ring,
with ConnectX-7 hardware-forwarded diagonal paths. It runs
`local-inference-lab/GLM-5.3-Flash-NVFP4-Spark` revision
`df116c4fb16b1d37ae43d2cfd624de26ffbc832e`, native MTP with three speculative
tokens, and explicit B12X draft attention. Tensor parallelism and decode-context
parallelism are both four; pipeline parallelism is one.

The tested image is
`sha256:8450c1a88d28359bc96748f024d1cd6140deb76818e58e099bae1015e0ef09c6`.
Its transport-bundle manifest is
`4204fabc93303226b9a120b094ef3c82ed4aadd1d7f97cfbe291204c027ed45f`.
The extracted host marker binary is
`0b92cd88be564e92a147d9bf981214151a12eb14a6bc8d9d7867ae8a63b21c9c`.

The scheduled-token limit is 8,192, the sequence limit is 16, and graph
capture sizes are every multiple of four from 4 through 64. Each rank has
24 GiB of FP8 KV cache. SparkCache uses a fresh persistent namespace and the
target checkpoint fingerprint for its native-MTP predictor identity.

Compared with the [recorded native-MTP3 mesh deployment](spark-mtp3-mesh-20260905.md), changes are limited to
the child image, canonical bundle serialization, operational names and paths,
and native-MTP cache identity and namespace. Parsed routing, target checkpoint,
model kernels, speculative settings, scheduling, and graph sizes are unchanged.

## Measurement

Image import and extracted files were checked independently on every rank.
The read-only fabric inspector checked eight installed hardware rules and
eight bounded-lifetime marker processes. The generated launch configuration
was compared with captured container arguments and environment, parsing JSON
arguments before comparison.

The repository's native communication runner exercised BF16 tensors of shape
`[Q, 4096]` at Q4, Q20, Q28, and Q64. Five output checks per rank and shape
compare byte-exact sums, including changed graph inputs and poisoned outputs.
Inputs use exactly representable integer values. Two warmups and three timing
samples accompany each cell; those timings are diagnostic only and do not
support a performance comparison.

The model-check harness sent one semantic-marker request and one 8,220-token
prefix request sequentially. Independent client traffic also occurred, so
deployment concurrency was not controlled. Publication was verified in all four rank logs,
not inferred from request latency. Startup time is the operator's elapsed wall
clock for the rank-zero launch command to return ready. There is one startup
observation and no variability estimate.

## Result

| Functional gate | Result | Evidence |
|---|---|---|
| Image identity and extracted files | Passed | Exact image on four ranks; 28 bundle files verified per rank |
| Host marker linkage | Passed | Dependencies resolved and help command exited successfully on four ranks |
| Fabric inspection | Passed | Eight hardware rules and eight marker processes accepted |
| Launch configuration | Passed | No unexpected differences across four ranks; parsed routing identical |
| Native eager and graph output | Passed | 80 byte-exact checks across four shapes and four ranks |
| Native path completion | Passed | 24 origin QPs per shape: 16 direct, eight forwarded; zero completion errors |
| Initial serving startup | Passed | Rank-zero launch returned ready in 278.6 seconds |
| Semantic response | Passed | Expected `SPARKCACHE_GLM53_OK` output |
| Long-prefix response | Passed | Expected output for an 8,220-token request |
| Persistent publication | Passed | 8,192 tokens committed on each of four ranks |
| Serving restart | Passed | All four containers restarted and the API completed a request |
| External restoration transfer | Observed | 8,192 external-hit tokens; each rank logged an 8,192-token restoration |
| Post-restore semantic output | Failed | Refusal instead of expected marker; 256-token output limit reached; cause unresolved |

Restoration logs reported 94.3, 110.1, 115.2, and 98.3 ms for ranks zero
through three, respectively, with 49.5 MiB reported per rank. These are transfer
observations, not evidence that the restored model state produced correct output.

## Conclusion

The packaged image starts and restarts across four ranks. Native communication
checks and persistent publication pass, and external restoration transfers are
observed. The post-restore semantic output gate fails. Its cause is unresolved;
persistent-cache correctness is not qualified.

## Limitations

The native runner selects RoCEnante directly. It does not independently prove
vLLM route selection; serving uses SIRCL for captured Q20 and Q28. Configuration
identity and successful startup are separate evidence from collective-output
correctness.

No throughput comparison, prolonged stress, or failure-containment qualification
is included. Independent client traffic overlapped the model checks, so their
timings are not clean performance measurements. Host forwarding remains manually supervised with bounded marker
lifetimes; automatic provisioning and hot renewal are not provided. One
semantic request and one publication request do not establish general model
quality or cache correctness for every context and concurrency.
