"""vLLM general plugin ``libsircl``: vLLM's PyNccl loads libsircl.

vLLM's PyNccl loads the library ``VLLM_NCCL_SO_PATH`` names
(``vllm.utils.nccl.find_nccl_library``). The SparkRing installer image's
entrypoint (``/opt/sparkring/toolchain/toolchain.py serve``) sets that
variable to the image's NVIDIA NCCL before vLLM starts, so a container
variable cannot select another library. With ``libsircl`` in
``VLLM_PLUGINS``, vLLM calls ``register`` in each of its processes before
that process creates a PyNccl communicator (a worker loads general plugins in
``init_worker``, before ``init_device``). ``register`` requires the file that
``SPARKRING_LIBSIRCL_LIBRARY`` names to be a regular file whose SHA-256 is
``SPARKRING_LIBSIRCL_SHA256`` and then sets ``VLLM_NCCL_SO_PATH`` to it;
processes that vLLM starts later inherit the setting. Any other state fails
the process, so a deployment that asked for libsircl never falls back to
another NCCL.

The plugin changes nothing else: torch's ProcessGroupNCCL keeps the NCCL the
process loaded at start. Status: research-only.
"""
import hashlib
import os
from pathlib import Path

LIBRARY = "SPARKRING_LIBSIRCL_LIBRARY"
DIGEST = "SPARKRING_LIBSIRCL_SHA256"
TARGET = "VLLM_NCCL_SO_PATH"
NAME = "libsircl"
ENTRY_POINT = ("vllm.general_plugins", NAME, "sparkring_libsircl:register")
_checked = {}


class LibsirclSelectionError(RuntimeError):
    """The process was asked to use libsircl, and its library is missing or differs from its digest."""


def selected(environ=None):
    """The checked library path for ``environ`` (default ``os.environ``); LibsirclSelectionError otherwise."""
    environ = os.environ if environ is None else environ
    path, digest = environ.get(LIBRARY, "").strip(), environ.get(DIGEST, "").strip().lower()
    if not path or not digest:
        raise LibsirclSelectionError(f"VLLM_PLUGINS names {NAME}, which needs {LIBRARY} and {DIGEST}")
    library = Path(path)
    if not library.is_absolute() or library.is_symlink() or not library.is_file():
        raise LibsirclSelectionError(f"{LIBRARY}={path} is not an absolute path to a regular file")
    stat = library.stat()
    key = (path, digest, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    if key not in _checked:
        with library.open("rb") as stream:
            found = hashlib.file_digest(stream, "sha256").hexdigest()
        if found != digest:
            raise LibsirclSelectionError(f"{path} has SHA-256 {found}, not {DIGEST}={digest}")
        _checked[key] = True
    return path


def register():
    """Point vLLM's PyNccl at the checked libsircl library; vLLM may call this more than once."""
    os.environ[TARGET] = selected()
