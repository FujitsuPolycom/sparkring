# Optional shared-memory reader spin interval

Status: the reader window is **implemented** in installer images, where
`sparkring install --save-cpu` selects it. This directory's source patch,
probe and preflight are **research-only** tools for controlled vLLM IPC
experiments; no published image is built with them.

| Path | What reads `SPARKRING_SHM_BUSY_LOOP_S` | Value handling | Status |
|---|---|---|---|
| Installer images listed in [installer-capabilities.json](../../../runtime/releases/installer-capabilities.json) (`shm_reader_window`) | A layer from [derive_spin_wait.py](../../../runtime/images/derive_spin_wait.py), or the image's vLLM branch carrying the same edit | `float()` of the variable, no range check; `--save-cpu` sets `0.002` | implemented |
| An image derived with this directory's [apply.py](apply.py) | The patched constructor for the two sources in [sources.json](sources.json) | Finite, 0 to 1 second; anything else is rejected | research-only |

In both, an unset variable keeps vLLM's one-second window, so the default is
unchanged. The installer refuses `--save-cpu` on an image without the window
([serving settings](../../../docs/operations/install-reference.md#serving-settings)).

vLLM's `SpinCondition` repeatedly yields the CPU while waiting for a shared
memory message, until its inactivity interval expires. It then waits for a
ZeroMQ notification. Shortening this interval can reduce CPU time between
messages while increasing notification latency. It is independent of NCCL,
RoCEnante and GPU collective routing.

Evidence:

- The [CPU measurement](../../../performance/records/transport/ipc-wait-linux-cpu-20260917.md)
  compares one second with two milliseconds using the actual class, separate
  processes and real IPC sockets. It does not establish GB10 power savings,
  model performance or serving reliability.
- The [two-Spark measurement](../../../performance/records/qwen38-flash-next/shm-spin-window-20260930.md)
  of `--save-cpu` on `qwen38-flash-next-tp2` freed about 1.7 CPU cores on
  Node A while decoding and lowered decode steps per second by about 1% with
  one request and 2% with eight. It covers one profile on one installer image.
- [Issue #189](https://github.com/FujitsuPolycom/sparkring/issues/189), which
  proposed a two-millisecond default, is closed. The default remains one
  second; the shorter window is opt-in.

## Source contract and selection

[sources.json](sources.json) binds two complete `shm_broadcast.py` inputs and
their patched outputs:

- DeepSeek 0731's vLLM revision `e2666d9a65f41fc376607531453cbd57c4c71016`.
- The installed source from ARM64 R37 image
  `sha256:2540686d726a28eb07784f9d2db5dc1f795404c7874fc1d6c11f018cd789adc2`.

Their `SpinCondition` classes match. Their surrounding queue code differs;
the patch changes only the constructor's default selection. All wait,
notification, cancellation, memory-fence, timeout and queue algorithms remain
unchanged. Unknown input bytes are rejected, and result bytes must match the
recorded hash. Reapplication accepts only that exact result.
The constructor imports its environment dependency locally; the DeepSeek
module has no module-level `os` import.

With the patch installed, an unset variable selects `1.0` seconds. The optional
value must be finite and between `0` and `1` seconds. `0.002` selects two
milliseconds; `0` permits immediate socket waiting. Explicit constructor
arguments override the environment. Writer behavior is unchanged. The optional
vLLM native spinloop extension is independent and is not tuned here.

## Apply while building an isolated derivative

Run [apply.py](apply.py) against a copied source tree or during image assembly,
before any vLLM process imports it. Never edit a serving container's module or
an existing read-only overlay. For example, in a derivative Containerfile:

```dockerfile
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
COPY integrations/vllm/ipc_wait/apply.py integrations/vllm/ipc_wait/sources.json /opt/sparkring/ipc-wait/
COPY integrations/vllm/ipc_wait/upstream/LICENSE /opt/sparkring/ipc-wait/LICENSE
COPY integrations/vllm/ipc_wait/preflight.py /opt/sparkring/ipc-wait/preflight.py
RUN /opt/venv/bin/python /opt/sparkring/ipc-wait/apply.py \
    --source /opt/venv/lib/python3.12/site-packages/vllm/distributed/device_communicators/shm_broadcast.py \
    --receipt /opt/sparkring/ipc-wait/installed.json
RUN /opt/venv/bin/python /opt/sparkring/ipc-wait/preflight.py
```

Supply an independently verified immutable parent image ID or registry digest.
The per-file hash gate does not attest an arbitrary parent image. If that image
retains an executable source checkout, apply the same guarded transform there
and retain its separate receipt. Use the normal image composition to bind the
parent identity, patch source and installed output; do not relabel the result
with the parent's image or source attestation. A source verifier that rejects
the altered module needs a separately reviewed derivative contract.

On an installer image with the window, use `sparkring install --save-cpu`
instead of this patch. For a derivative built with this patch, set
`SPARKRING_SHM_BUSY_LOOP_S=0.002` in each vLLM process's environment through an
explicit experimental container specification; omit it or set `1` for the
control.

## CPU checks and hardware acceptance

Install the repository's development dependencies, then run:

```bash
python -m pytest integrations/vllm/ipc_wait -q
python integrations/vllm/ipc_wait/probe.py \
  --source /path/to/vllm/distributed/device_communicators/shm_broadcast.py \
  --output /path/to/new-ipc-observations.json --repeats 3
```

The measured probe requires POSIX `os.sched_yield` and pyzmq. It extracts the
actual class without importing Torch/vLLM, checks ordered message receipt, and
records reader process CPU time, wake latency, socket polls and cancellation.
Only names declared by the source module's imports are available to the class
fixture. Constructor tests cover the DeepSeek module without a global `os`.
Its synchronized shared-state payload is a test fixture; it does not execute
the complete vLLM `MessageQueue`, serialize model commands or measure GPU work.
The tests also check unchanged methods, rejected preimages, environment
validation, explicit-call compatibility and finite poll timeout behavior.

The image [preflight](preflight.py) separately imports the actual installed
vLLM module and constructs real ZMQ readers/writers at the default and optional
intervals. It uses no injected globals or model execution. Require this check
inside the derivative image before a serving test; class extraction alone is
not a complete module-import check.

The two-Spark measurement covers reader CPU and decode steps per second.
Making the shorter window a profile's default would also need a comparison of
`1` and `0.002` on that profile's image, model, cache policy and workload:
idle-to-request and burst wakeups, prefill and decode latency tails,
cancellation, clean shutdown, concurrent readers, a restart and semantic
responses.

DeepSeek's pinned queue can wait indefinitely when its caller disables warnings
and supplies no timeout; R37 periodically rechecks the authoritative shared
state after at most five seconds. Both behaviors are preserved. This probe's
one-second poll bound cannot qualify missed-notification recovery in either
full queue implementation. Shorter spinning increases time spent on that path,
so a passing notification test is not sufficient to promote a new default.
