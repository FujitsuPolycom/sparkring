# GLM-5.3 opaque-page base-flight research evidence

Status: **research-only**.

SparkCache's persistent base-segment I/O mechanism is **implemented** and has
GPU-free regression coverage. The retained live observations do not qualify
page-delta restore for GLM-5.3 serving.

## Retained observations

| Condition | Mechanical result | Semantic result | Conclusion |
|---|---|---|---|
| Sixteen concurrent 131,072-token page-delta requests with one shared 98,304-token base | Every rank emitted one `sparkcache-page-base-restore-flight/v1` record with 16 participants, one physical base read, and 15 avoided reads | Codeword responses were not reliable | Host-I/O coalescing worked, but the workload exceeded the 20 GiB GLM hybrid-cache residency available per rank and cannot support a correctness claim |
| Two concurrent 131,072-token page-delta requests within resident capacity | One admitted restored request returned the wrong codeword; a request recomputed without admitted restore returned the correct codeword | The failure occurred with both SparkCache CUDA placement enabled and disabled | The defect is in a path common to reconstructed page-delta restore; the evidence does not isolate CUDA placement |
| One 131,072-token flat snapshot stored as 13 authenticated macro objects | Restore completed in 1.55–1.70 seconds | The exact `red` codeword was returned | Full-snapshot restore is the verified operational fallback for this evidence scope |

The mechanism evidence proves bounded host reads, authenticated manifests, and
request accounting. It does not prove that reconstructed page-delta state is
safe to admit for GLM-5.3 generation.

## Harness contract

`qualification.py` is retained as a deterministic research harness. It uses
stable single-token codeword oracles, records raw response hashes as diagnostic
data, and never starts, stops, or restarts a service. A mechanically complete
verdict has `kind=research-verdict` and `status=research-only`; the harness
cannot emit a qualification status.

Publication records the rank-0 scheduler-log offset before the base request.
It submits bounded two-token scheduler steps and reads only later log bytes
until one `KV Transfer metrics` record reports four ranks and four held
digests. This readiness check remains required before private-tail requests.

Replay collects every result and the unrelated later request before returning.
It preserves rejected receipts and exits nonzero after writing them. Manifest
inspection and bounded rank logs remain required to associate responses with
the exact page-delta roots and base-flight records.

The 16 × 131,072-token case is retained to reproduce the observed mechanical
coalescing and residency failure. It is not a supported qualification workload.
