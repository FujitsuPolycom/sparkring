# Qwen QAD TP4 prefill attribution

Status: **research-only** measurement. The
[trace summary](qad-tp4-prefill-attribution.json) records a cold 16,384-token
request with one output token on four GB10 nodes, QAD revision
`629bc3218833a38b475b719f34aa571666f4a03e`, R37, TP4/DCP1 and MTP3.
The bounded collective policy remained enabled. Unprofiled request time was
4.828 seconds; the CPU/CUDA-profiled request took 5.078 seconds.

Rank 0 had a 5.050-second GPU-kernel span and 4.983 seconds covered by at
least one GPU kernel. Summed kernel categories were:

| Category | Seconds | Approximate share of kernel span |
|---|---:|---:|
| Hyperconnection normalization/gating and replicated projections | 1.433 | 28% |
| MoE | 0.968 | 19% |
| NCCL | 0.883 | 17% |
| Attention | 0.511 | 10% |
| MTP feedback GEMMs | 0.300 | 6% |
| Other kernels | 0.898 | 18% |

Categories sum kernel durations and can overlap; they are not a disjoint
critical-path decomposition. NCCL duration includes synchronization and must
not be interpreted as network transfer time alone. The other three ranks
showed similar hyperconnection costs, 1.44–1.45 seconds, and NCCL costs of
0.81–0.89 seconds. Large CPU launch gaps did not dominate this profiled run.

## Replicated hyperconnection work

The pinned vLLM module
`vllm/models/qwen3_8_flash_next/hyperconnection.py` creates its input projection
with `MergedColumnParallelLinear(..., disable_tp=True)` and its output
projection with `ReplicatedLinear`. Decoder layers apply the mixer before
attention and MoE on the full token rows.

All four GPU traces contain the same full-size HC projection shapes:
`[8192,10240] × [10240,336]` and `[8192,320] × [320,10240]`.
Their GPU events were associated with `aten::mm` shapes using profiler external
IDs. Together with the HC normalization and gating kernels, this establishes
a substantial replicated workload that tensor parallelism does not divide.
SparkRing's GLM-specific HC sharding switch does not select a Qwen implementation.

This evidence makes Qwen-compatible token-row sharding of HC work a concrete
optimization target. Its communication cost must be measured: keeping residual
state sharded and coordinating gathers/reduce-scatters may be necessary to avoid
exchanging more data than the compute savings justify. It is not a safe change
to a dispatch filter alone. Kernel fusion is another candidate where it preserves
the checkpoint's BF16 rounding behavior.

Faster large collectives may still help, but they address only part of the
measured time. Even removing all recorded NCCL duration would yield only about
1.2× for rank 0 under an otherwise unchanged, optimistic timing model.

This is a TP4 attribution, not a matched TP2/TP4 scaling experiment. The saved
TP2 record uses a different PTQ checkpoint revision. A direct scaling claim
requires the same weights, prompt, cache state and profiler conditions on TP2.
