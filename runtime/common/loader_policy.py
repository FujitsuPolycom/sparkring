"""Container-scoped io_uring admission for the external B12X loader."""
import inspect
import json
from pathlib import Path

PROFILE = Path(__file__).with_name("loader-seccomp.json")
RELATIVE = "runtime/common/loader-seccomp.json"


def inspection_option():
    # Docker/Compose embeds the compact JSON, not the source filename, in the
    # container's SecurityOpt. Compare the complete policy, including deny rules.
    return "seccomp=" + json.dumps(json.loads(PROFILE.read_text()), separators=(",", ":"))


def probe():
    """Exercise all three allowed calls without a GPU, model or network."""
    import ctypes
    import os
    import platform
    if platform.machine() != "aarch64":
        raise ValueError("Loader syscall admission requires the ARM64 serving host")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    params = ctypes.create_string_buffer(256)
    fd = libc.syscall(425, 2, ctypes.byref(params))
    if fd < 0:
        raise ValueError("io_uring_setup unavailable: " + os.strerror(ctypes.get_errno()))
    try:
        operations = ctypes.create_string_buffer(16 + 256 * 8)
        if libc.syscall(427, fd, 8, ctypes.byref(operations), 256) < 0:
            raise ValueError("io_uring_register unavailable: " + os.strerror(ctypes.get_errno()))
        if libc.syscall(426, fd, 0, 0, 0, 0, 0) < 0:
            raise ValueError("io_uring_enter unavailable: " + os.strerror(ctypes.get_errno()))
    finally:
        os.close(fd)
    return {"io_uring": "available", "gpu_used": False}


def check(image_id, *, run):
    code = inspect.getsource(probe) + "\nimport json\nprint(json.dumps(probe()))\n"
    result = run(["docker", "run", "--rm", "--pull", "never", "--runtime", "runc", "--network", "none",
                  "--read-only", "--cap-drop", "ALL", "--ulimit", "memlock=-1:-1",
                  "--security-opt", "no-new-privileges", "--security-opt", "seccomp=" + str(PROFILE),
                  "--entrypoint", "python3", image_id, "-IB", "-c", code])
    observed = json.loads(result.stdout)
    if observed != {"io_uring": "available", "gpu_used": False}:
        raise ValueError("Loader policy admission returned an unexpected result")
    return observed
