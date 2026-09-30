"""Let SPARKRING_SHM_BUSY_LOOP_S set how long vLLM's shared-memory readers spin.

vLLM passes scheduler outputs to its worker processes through a shared-memory
message queue (vllm/distributed/device_communicators/shm_broadcast.py). Each
reader waits in `SpinCondition.wait()`: within `busy_loop_s` seconds of its
last read it yields the CPU and polls the buffer again, and after that it
sleeps until the writer's notification socket wakes it. The default window is
one second, longer than any gap between decode steps, so during decoding every
reader keeps a CPU core busy between steps (FujitsuPolycom/sparkring#189).

This layer makes the reader's window `SPARKRING_SHM_BUSY_LOOP_S` seconds when
that variable is set, and the constructor's value (the one-second default)
otherwise, so the image serves as its parent does until a profile sets it.
Writers and the queue's messages are unchanged.

The parent's shm_broadcast.py must have SHA-256 240fd6a1...; the result has
SHA-256 given below (both pinned). /opt/sparkring/receipts/derived-spin-wait.json
records the file.
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from runtime.images.derived_layer import SITE, Layer, main, swap  # noqa: E402

SHM = SITE + "vllm/distributed/device_communicators/shm_broadcast.py"
ENVIRONMENT = "SPARKRING_SHM_BUSY_LOOP_S"
INHERITED = "240fd6a148aa6729e110380ca1188aaab6bc722e7c7ad5110c8aa301388bd967"
RESULT = "9108d911ef6b3bf171bb43cbd3cab88317efb3e67f24bf6aa184f4fb87489e35"
READER_WINDOW = """\
            # Time to keep busy-looping on the shm buffer before going idle
            self.busy_loop_s = busy_loop_s
"""
CONFIGURED_WINDOW = """\
            # Time to keep busy-looping on the shm buffer before going idle.
            # SparkRing: SPARKRING_SHM_BUSY_LOOP_S, in seconds, replaces the
            # constructor's value when set (FujitsuPolycom/sparkring#189).
            self.busy_loop_s = float(os.environ.get("SPARKRING_SHM_BUSY_LOOP_S", busy_loop_s))
"""


def replace(read, receipt):
    return {SHM: swap(read(SHM).decode("utf-8"), READER_WINDOW, CONFIGURED_WINDOW).encode("utf-8")}


LAYER = Layer(
    name="spin-wait",
    purpose=("vLLM's shared-memory readers spin for SPARKRING_SHM_BUSY_LOOP_S seconds after a read, when set, "
             "instead of one second before sleeping until notified (FujitsuPolycom/sparkring#189)"),
    replace=replace,
    provenance="/opt/sparkring/receipts/derived-spin-wait.json",
    pins={SHM: (INHERITED, RESULT)},
)

if __name__ == "__main__":
    main(LAYER)
