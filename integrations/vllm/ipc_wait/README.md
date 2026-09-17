# Optional shared-memory reader spin interval

Status: **research-only**. This build-time source patch exposes
`SPARKRING_SHM_BUSY_LOOP_S` for controlled vLLM IPC experiments. It retains the
one-second default. No published image, profile default or running deployment
is changed by this directory.

vLLM's `SpinCondition` repeatedly yields the CPU while waiting for a shared
memory message, until its inactivity interval expires. It then waits for a
ZeroMQ notification. Shortening this interval can reduce CPU time between
messages while increasing notification latency. It is independent of NCCL,
RoCEnante and GPU collective routing.

The [CPU measurement](../../../performance/records/transport/ipc-wait-linux-cpu-20260917.md)
compares one second with two milliseconds using the actual class, separate
processes and real IPC sockets. It supports a hardware experiment; it does not
establish GB10 power savings, model performance or serving reliability.
The originating [issue #189](https://github.com/FujitsuPolycom/sparkring/issues/189)
proposes a default change; this patch deliberately leaves that decision pending.

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

For a temporary test deployment, pass `SPARKRING_SHM_BUSY_LOOP_S=0.002` to each
relevant process. Omit it or set `1` for the control. Do not put the setting into
canonical profiles before testing their exact image and executor. Existing
launchers may reject an unknown ENV key; use an explicit experimental container
specification instead of bypassing their configuration validation.

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

Before enabling a profile, compare `1` and `0.002` on the same image with the
same model, cache policy and workload. Capture actual IPC-reader CPU usage,
idle-to-request and burst wakeups, bounded prefill/decode latency, cancellation
and clean shutdown. Include concurrent readers and a restart. Verify semantic
responses and report latency tails as well as CPU savings. Preserve the model
owner and rollback containers during this window.

DeepSeek's pinned queue can wait indefinitely when its caller disables warnings
and supplies no timeout; R37 periodically rechecks the authoritative shared
state after at most five seconds. Both behaviors are preserved. This probe's
one-second poll bound cannot qualify missed-notification recovery in either
full queue implementation. Shorter spinning increases time spent on that path,
so a passing notification test is not sufficient to promote a new default.
