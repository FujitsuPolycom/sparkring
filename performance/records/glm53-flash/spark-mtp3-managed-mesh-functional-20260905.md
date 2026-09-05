# Native MTP3 managed-mesh functional checks

Status: **research-only** functional evidence. The managed lifecycle is
implemented; comprehensive four-rank failure qualification is pending.
This record is not a throughput benchmark or an unattended-availability claim.

## Conditions

Four DGX Spark GB10 hosts used the physical cycle `0-1-2-3-0`, hardware-forwarded
opposite-peer paths, TP4/DCP4/PP1, and the
[`GLM-5.3-Flash-NVFP4-Spark` native-MTP3 profile](../../../runtime/glm53-spark-mtp3-mesh/README.md).
The target revision was `df116c4fb16b1d37ae43d2cfd624de26ffbc832e`.
The model used batch 8,192, 16 sequences, 24 GiB FP8 KV per rank, graph rows
4 through 64 in increments of four, and the profile's hybrid SIRCL/RoCEnante
routing. Kernels and cache settings were unchanged from that profile.

| Artifact | SHA-256 identity |
|---|---|
| Immutable child image | `26273b8e358df139ae913610a5d43084ff0fd08aafe282ef633a3bc74afefe47` |
| Parent image | `5e32aaa1bbe3559e81db7706ed4286248f18d27cfdb186f6b851bf786eb43075` |
| Transport bundle manifest | `4204fabc93303226b9a120b094ef3c82ed4aadd1d7f97cfbe291204c027ed45f` |
| Managed native marker source | `8684a6961b8e86aa474fa2310ff71e4cdf219a63a72ceb5593b2f95e54812792` |
| Extracted native marker binary | `2828c07e4255c4962c77425be2c88969e7eb7dd4b1bf9e36485bc705bb5d6d64` |
| Temperature-one readiness helper | `f41c38eef41d15d63dcfc49cd6643357ca1a3ae18200ddbe4f8692d0b767ee79` |

The managed host service owns persistent source markers and uses a shared
authenticated four-rank generation gate. It does not replace the model
runtime or run the intermediate packet data path on the CPU.

Two host-monitor policies are distinguished in this record:

- **Single-miss policy:** 1-second peer HTTP timeout with immediate failure
  after a transport miss. Helper, supervisor, and rule interruption
  observations under this policy are not stopping-time measurements for
  the connection-grace policy.
- **Connection-grace policy:** 2-second peer HTTP timeout and a 4-second grace
  after the first transport failure. Degraded health blocks model startup.
  Authentication/explicit-readiness failures, changed peer generations, and
  local marker exits trigger failure when observed without transport grace.
  The tested `managed_service.py` SHA-256 is
  `dbe8bb19255a87deaf4dd7bdec31531cd42866f5b863d8061192dcef333191ad`.

The native harness covered BF16 `[Q,4096]` at Q4/Q20/Q28/Q64 before model
startup. Each shape included five correctness cases on four ranks, two
warmups, and three diagnostic timing samples. The correctness checks include
rank-specific inputs and changed graph replay inputs. Timings are not used
for a performance claim here.

One cache-recall request used temperature 1, seed 9046500, a 512-token output
limit, and thinking enabled. Its input was the repository's generated recall
document, SHA-256
`3d2bc5228895566b1497e6f35f6c5aa051685f99438f27226134dfcfab15c277`.
The cache namespace was the profile's native-MTP namespace, not an external
draft identity.

The helper fault was one SIGKILL to an identity-checked owned rank-zero
marker during a running request. The request used temperature 1, C1,
`max_tokens=4096`, `ignore_eos=true`, streaming, and thinking disabled.
Its prompt requested consecutive positive integers. The 4,096 value is the
requested output limit, not the prompt length or achieved output count.

## Measurement

Raw receipts remain in the private `managed-service/` evidence directory.
The sibling JSON record identifies the sanitized observations and raw receipt
hashes; it does not publish addresses, keys, PIDs, or private command paths.

Native results count correctness cases and completion errors across four
ranks. Cache metrics use before/after deltas and reported prompt-token usage.
The recall receipt retains `persistent_restore_proven=false`: external-hit
metrics and the correct answer are evidence, but per-rank restore logs are
still required to close that individual receipt's full restoration gate.
A separate post-recovery snapshot supplies those logs for the connection-grace
policy's recall result; the tool's raw boolean is preserved unchanged.

The fault harnesses used controller `time.monotonic()` for container-state
polling. Its timer starts **after the injection SSH command returns**, not
at the exact marker kill. The helper-injection command took 0.438 seconds end to end.
The reported stop value is the first polling round that observed all four
containers stopped; it includes SSH polling overhead. It is not a bound on
hardware detection latency. One observation provides no variability estimate.

The stream receipt counts received lines, including framing, rather than
generated tokens. It records whether the SSE `[DONE]` terminal marker was
received. It does not retain per-token timing relative to fault injection.

## Result

| Check | Observation | Scope |
|---|---|---|
| Image construction/CPU verification | Passed | No GPU or model loaded during image verification |
| Native collective correctness | 80/80 cases passed; zero recorded completion errors | Four shapes × four ranks × five cases |
| Managed four-rank model startup | Model reached healthy serving with temperature-one warmup | Full warmup receipt publication pending |
| Recall after startup | Correct `cobalt orchard lantern` answer; finish reason `stop` | One stochastic request |
| External cache metrics | 26,624 external-hit tokens from 27,274 prompt tokens; RAM-prefix-hit delta zero | Per-rank restoration logs still required |
| Single-miss policy: owned helper SIGKILL | All four model containers observed stopped at 5.984 seconds after injection command return | One rank-zero helper fault during C1 streaming |
| Interrupted stream | HTTP 200 opened; 22 lines received; no `[DONE]` | Incomplete stream, not zero-output evidence |
| Network/service CPU contract suite | 48 tests passed | Offline software checks only |
| Combined profile/transport/qualification CPU suite | 389 passed, 19 skipped | Windows: 18 POSIX-only native CLI cases and one permission-mode case skipped; the 18 native CLI cases passed separately on DGX2 |
| Single-miss policy: supervisor-main SIGKILL | All four stopped at first all-stopped poll of 1.844 seconds | Separate pre-grace observation; orphan marker processes retained |
| Single-miss policy: intermediate-rule removal | All four stopped at first all-stopped poll of 4.016 seconds | Separate pre-grace observation; exact rule restored afterward |
| Single-miss policy under model loading | A single 1-second health timeout incorrectly stopped the stack | Retained regression; motivated connection grace |
| Connection-grace policy: supervisor SIGSTOP/SIGCONT | Approximately 3-second pause; all four model containers remained running | One deliberate management-supervisor pause |
| Connection-grace policy: supervisor-main SIGKILL | All four stopped at first all-stopped poll of 11.156 seconds | Timer starts after injection SSH return; two owned orphan marker processes retained |
| Connection-grace policy: explicit recovery | All nine coordinator phases passed on all four ranks | Includes model-stop barrier, orphan cleanup, and authenticated readiness |
| Repository first-install helper | `--apply` passed on ranks 0–3 with clean targets | Existing installation retained in a private backup; containers pre-created stopped |
| Connection-grace post-recovery readiness | Four healthy/running containers; API health and scheduler liveness HTTP 200 | Public `wait_managed_ready.py` receipt |
| Connection-grace post-recovery mesh | Four armed, non-degraded supervisors; two markers per rank; graph status caught up | Snapshot, not long-duration soak |
| Connection-grace persistent recall | Correct phrase; 26,624 external-hit tokens of 27,274; restore logs on all four ranks | One temperature-one request after model reload |
| Managed unit elapsed-runtime limit | `RuntimeMaxUSec=infinity` on both units at all four ranks | No scheduled service expiry |

At the preceding poll, 4.937 seconds after injection command return, ranks
0, 1, and 2 were observed stopped while rank 3 was still observed running.
The next poll, at 5.984 seconds, observed all four stopped.

## Connection-grace acceptance and retained regression

The single-miss transport-error policy caused a false stop during model
loading after one 1-second peer-health timeout. That failure is retained;
it is not reclassified as a successful safety test. CPU tests cover the
replacement distinction between temporary transport loss and explicit
negative authenticated state.

Under the connection-grace policy, a recorded supervisor pause lasted
3.000194 seconds. All four containers were running at the subsequent
observations. A separate supervisor-main SIGKILL resulted in all four
containers observed stopped at 11.156 seconds after injection SSH return.
The affected host retained two managed marker processes. Their process
presence establishes retained ownership, not measured packet delivery
during the fault.

The explicit recovery receipt records successful quiesce, model-unit stop,
pinned-container stop, all-rank model-stop barrier, mesh-unit stop, owned-child
cleanup, failure-latch reset, mesh-unit start, and authenticated readiness.
Each phase passed on all four ranks. This qualifies that bounded recovery
case; it does not qualify automatic recovery or model output after recovery.

## Post-recovery model and cache gate

The private readiness receipt `public-install/final-ready-public.json` was
generated by the repository's read-only `wait_managed_ready.py`. It reports
all four expected containers running and healthy, with rank-zero API health
and scheduler liveness returning HTTP 200.

The independently retained `final-snapshot.json` records four armed
supervisors with `peer_health_degraded=false`, two managed markers each,
caught-up native graph status, and `RuntimeMaxUSec=infinity` for model and
mesh units. Its per-rank model logs record restoration of 26,624 tokens.
Ranks 0 through 3 report restore durations of 152.1, 166.4, 161.1, and
173.1 milliseconds respectively, each for 79.1 MiB. These are logged restore
durations for one request, not an end-to-end latency comparison.

The `recall-final/receipt.json` request used temperature one and the same
27,274-token recall prompt. It returned the expected phrase with finish
reason `stop`, recorded 26,624 external-prefix-hit tokens, and had zero
RAM-prefix-hit delta. Taken together with the four per-rank restore logs,
these observations qualify this one persistent-restoration recall after
model reload. The request tool's `persistent_restore_proven=false` field
is retained: the tool does not itself inspect the required per-rank logs.

## Conclusion

The managed marker image completed the selected native checks and reached
model serving. One active-request helper failure triggered all-rank model
stopping, and the affected stream did not receive its normal terminal marker.
The cache-recall request returned the expected phrase with nonzero external
cache-hit evidence.

The connection-grace policy tolerated the tested short pause, stopped serving
after the tested supervisor loss, completed explicit all-rank recovery,
and subsequently reached model readiness with a passing persistent recall.
These bounded cases are qualified only for their written conditions. They do
not establish zero erroneous output after a fault, worst-case stopping time,
unattended recovery, or general model/cache correctness.

## Limitations

- Peer/network partitions, Docker failure, host reboot, and long-duration
  operation have no completed result in this record. Helper/rule fault timing has not been
  repeated under the connection-grace policy.
- Each fault timing has one observation, SSH overhead, and a controller
  time origin after the remote command returned.
- An incomplete stream can contain partial output. The receipt does not
  classify output emitted before versus after the injected failure.
- The recall check is one temperature-one sample, not exhaustive cache
  correctness. Four-rank restored-state logs support only the specifically
  identified post-recovery request.
- Existing temperature-zero inconclusive canaries and the separate
  [sampling-warmup image record](spark-mtp3-mesh-temperature-one-functional-20260905.md)
  retain their original scope and are not reclassified by these results.
- No prefill/decode performance comparison or public-image promotion follows
  from these functional checks.
