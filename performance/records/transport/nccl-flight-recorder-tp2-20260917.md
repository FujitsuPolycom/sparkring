# Two-rank NCCL flight-recorder capture

Status: **qualified for healthy ProcessGroupNCCL FIFO capture and host
persistence on the identified two-Spark setup**. This is bounded evidence for
[issue #190](https://github.com/FujitsuPolycom/sparkring/issues/190), with no
timeout or fault injection.

## Runtime and binding identity

The image was
`sha256:41fc632d02352a69f59dc18be184488c1f9d51bd9ad0a29e22ae818af4f465fe`,
the cached ARM64 artifact for published DeepSeek manifest
`ghcr.io/fujitsupolycom/gb10-vllm-serving@sha256:827a8e8c5749b78529cc0015dd174e1b19a0accc116bc142282f8b75428f98bd`.
Torch was `2.12.0+cu132`, revision
`7661cd9c6b841b62b7f411aa52ec51f05457263b`.

Torch's compile-time NCCL metadata reports 2.29.7, and its bundled library is
mapped alongside the selected NCCL 2.30.7 library. The probe inspected the
**46 resolved NCCL relocation symbols in libtorch_cuda.so on each rank**.
Every target belonged to `/opt/sparkring/nccl/libnccl.so.2.30.7`, SHA256
`ccd57342449c3f680befcb379329b935746e5299dc4de5f2516146e0411bd85f`.
Calling `ncclGetVersion` through Torch's actual relocation returned `23007`.
Mapping another library does not establish which implementation Torch calls;
handle-scoped `dlsym` lookup is also insufficient for that distinction.

The [JSON record](nccl-flight-recorder-tp2-20260917.json) contains exact runtime,
environment, dump and private harness/report hashes. The companion CPU loader
check independently matched all 46 bindings with `LD_DEBUG=bindings` and
`LD_BIND_NOW=1`, with CUDA uninitialized.

## Measurement and result

Two GB10 ranks used IPv4 RoCEv2, `NCCL_IB_HCA==rocep1s0f0:1`, `NCCL_NET=IB`,
and no `NCCL_IB_GID_INDEX`. Each rank created float32 inputs filled with its
rank plus one. All-reduces of **17, 1024 and 65,536 elements** returned exactly
three everywhere on both ranks.

After synchronization, the probe wrote two bytes to each FIFO created by Torch:
`/tmp/fr_dump_pipe_<rank>.pipe`. It did not create the FIFO or call a private
dump API. Recorder settings included a 2000-entry trace buffer, monitoring,
timeout-dump enablement, and static output prefix
`/cache/nccl-fr/comm_lib_trace_rank_` on a host bind mount.

| Result | Rank 0 | Rank 1 |
|---|---:|---:|
| Correct all-reduces | 3/3 | 3/3 |
| Completed, shape-matched trace entries | 3 | 3 |
| Dump bytes | 1637 | 1637 |
| Unchanged complete dump observed | 1.003 s | 1.003 s |
| Process-group destruction and process exit | Normal / 0 | Normal / 0 |
| Host dump digest matched after container exit | Yes | Yes |

The dumps parsed as complete pickle records without trailing data. Collective
sequence IDs 1–3 aligned across ranks, with completed states and completion
timestamps. Each trace also included a synchronization entry. Probe execution
took approximately 9.2 seconds per rank, within the independent process budget.

## Limits

This validates on-demand recording of healthy **PyTorch ProcessGroupNCCL**
operations. Automatic watchdog/timeout, rank-loss and driver-failure capture
were not exercised. Direct PyNccl, SIRCL, RoCEnante and collectives launched
outside ProcessGroupNCCL are outside this evidence; missing recorder entries
do not establish that a serving model performed no communication.

Successful unpinned-GID communication applies to this two-rank IPv4 RoCEv2
setup. It does not qualify four-rank cycle routing, all GID selections, model
serving or communication performance. No model weights were used or changed.
