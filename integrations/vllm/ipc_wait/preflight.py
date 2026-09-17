"""Import the installed vLLM IPC module and check reader construction on CPU."""
import hashlib
import json
import os
from pathlib import Path
import uuid

import zmq


def main():
    from vllm.distributed.device_communicators import shm_broadcast

    variable = "SPARKRING_SHM_BUSY_LOOP_S"
    previous = os.environ.get(variable)
    observations = []
    try:
        for setting, expected in ((None, 1.0), ("0.002", 0.002), ("0", 0.0), ("1", 1.0)):
            if setting is None:
                os.environ.pop(variable, None)
            else:
                os.environ[variable] = setting
            context = zmq.Context()
            writer = reader = None
            try:
                address = "inproc://" + str(uuid.uuid4())
                writer = shm_broadcast.SpinCondition(False, context, address)
                reader = shm_broadcast.SpinCondition(True, context, address)
                if reader.busy_loop_s != expected or writer.busy_loop_s != 0:
                    raise RuntimeError("Installed SpinCondition selected an incorrect spin interval")
                observations.append({"setting": setting, "seconds": reader.busy_loop_s})
            finally:
                for condition in (reader, writer):
                    if condition is not None:
                        for name in ("local_notify_socket", "read_cancel_socket", "write_cancel_socket"):
                            socket = getattr(condition, name, None)
                            if socket is not None:
                                socket.close(linger=0)
                context.term()
    finally:
        if previous is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = previous
    print(json.dumps({"schema": "sparkring-ipc-installed-constructor/v1",
                      "source_sha256": hashlib.sha256(Path(shm_broadcast.__file__).read_bytes()).hexdigest(),
                      "cases": observations,
                      "scope": "Actual installed module import and real ZMQ construction; no model/GPU execution"}))


if __name__ == "__main__":
    main()
