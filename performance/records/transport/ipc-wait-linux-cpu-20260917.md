# Shared-memory reader spin interval on Linux CPU

Status: **research-only**. A two-millisecond spin interval substantially reduced
reader CPU time during idle/decode-shaped gaps, with higher notification
latency. This is a local CPU mechanism test, not GB10 serving qualification or
evidence that changing a profile's default improves model performance.

## Conditions

The [probe](../../../integrations/vllm/ipc_wait/probe.py) extracted the complete
`SpinCondition` class from Local Inference Lab vLLM
`e2666d9a65f41fc376607531453cbd57c4c71016`, full module SHA-256
`7ff67c2ef6b8a33a13b11aa3cb202da7887d1d44eed27c6a02d817ea24807d61`.
The independently extracted ARM64 R37 module has an identical class AST;
surrounding queue code is different. [Source identities](../../../integrations/vllm/ipc_wait/sources.json).

Measurements used Python 3.12, pyzmq 27.2.0 and x86-64 Linux under WSL. One
writer process published synchronized shared state and called the real
`notify()` method; a separate reader called the real `wait()` and
`record_read()` methods, using POSIX `os.sched_yield` and real ZeroMQ IPC
sockets. No Torch, model, CUDA, RDMA or complete vLLM `MessageQueue` was involved.

Each message was acknowledged before the next was published. Client gaps were
150 ms for eight idle messages, 15 ms for 80 decode-shaped messages, and 0.2 ms
for 160 burst messages. These are synthetic arrival patterns, not measured
model decode intervals. There were three repetitions per pattern/policy,
alternating policy order by repetition. Subscription initialization was outside
the timed workload. Processes were not CPU-affinity pinned.

Reader CPU fraction is process CPU time divided by observed wall time, including
its ZMQ threads and shared-state checks. Wake latency spans publication under
the shared-state lock to observation by the reader. The lock and acknowledgement
protocol add harness work and limit the interpretation of absolute timing.

## Results

The table reports the median of three per-run observations. The p95 column is
the median of each run's p95, not a pooled percentile or confidence interval.

| Arrival pattern | Reader CPU, 1 s | Reader CPU, 2 ms | Wake p95, 1 s | Wake p95, 2 ms |
|---|---:|---:|---:|---:|
| Idle, 150 ms gaps | 97.46% | 1.34% | 44.32 us | 331.39 us |
| Decode-shaped, 15 ms gaps | 97.42% | 11.82% | 78.06 us | 244.64 us |
| Burst, 0.2 ms gaps | 93.71% | 88.60% | 37.07 us | 39.23 us |

All 1,488 published messages arrived in order. The one-second policy never
parked during these workloads. At two milliseconds the reader polled once per
idle/decode-shaped message and never polled during the burst cases. Cancellation
sent after a 20 ms delay woke a parked reader under both policies.

The [raw record](ipc-wait-linux-cpu-20260917.json) includes every latency sample,
CPU/wall timing, poll/wait count, cancellation timing, platform and harness hash.
No serving endpoint, credential or model data is included.
The recorded harness bytes are retained in local commit `9a53af1`; the measured
class is the unmodified source with explicit interval arguments. These timing
results do not qualify the optional patch's default-constructor import path.

## Conclusion and limits

The measurement confirms the mechanism behind
[SparkRing issue #189](https://github.com/FujitsuPolycom/sparkring/issues/189):
sub-second arrival gaps can keep the one-second policy spinning continuously.
Moving to socket waits trades CPU time for notification latency. Burst timing
does not establish an improvement; both policies remained in their spin window.

The optional [image source patch](../../../integrations/vllm/ipc_wait/README.md)
therefore retains a one-second default and exposes an explicit experiment
setting. Its constructor, source identity and real notification tests passed;
none establishes a GB10 power, prefill or decode result.

The probe passes a one-second poll timeout and checks synchronized test state.
It does not qualify indefinite queue waits, notification loss, native spinloop
extensions, many readers, actual command serialization or unattended restart.
DeepSeek's pinned full queue can wait indefinitely with warnings and timeout
disabled; R37 periodically rechecks shared state. The source patch preserves
both behaviors. An image-specific serving comparison must cover these runtime
boundaries before changing defaults.
