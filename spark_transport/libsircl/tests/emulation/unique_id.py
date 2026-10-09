"""Hand rank 0's NCCL unique id to the other ranks of a test group.

Two channels: a file in a directory every rank sees (ranks on one host), or a TCP port on which rank 0
serves the 128 bytes to each other rank once (ranks on several hosts, for example a cabled pair of
Sparks). Used by ``library_rank.py`` and ``perf_rank.py``.
"""
from __future__ import annotations

import ctypes
import socket
import time
from pathlib import Path


class UniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_ubyte * 128)]


def share(lib, rank: int, world: int, id_file: str = "", id_server: str = "", timeout: float = 300) -> UniqueId:
    """Rank 0 creates the id with ``ncclGetUniqueId``; every rank returns the same id."""
    if bool(id_file) == bool(id_server):
        raise SystemExit("give one of --id-file and --id-server")
    uid = UniqueId()
    if rank == 0 and lib.ncclGetUniqueId(ctypes.byref(uid)) != 0:
        raise SystemExit("ncclGetUniqueId failed")
    deadline = time.monotonic() + timeout
    if id_file:
        path = Path(id_file)
        if rank == 0:
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(bytes(uid.internal))
            temporary.rename(path)
            return uid
        while not path.exists():
            if time.monotonic() > deadline:
                raise SystemExit("no unique id")
            time.sleep(0.01)
        ctypes.memmove(uid.internal, path.read_bytes(), 128)
        return uid
    host, _, port = id_server.rpartition(":")
    if rank == 0:
        with socket.create_server(("0.0.0.0", int(port)), reuse_port=False) as server:
            server.settimeout(timeout)
            for _ in range(world - 1):
                peer, _ = server.accept()
                with peer:
                    peer.sendall(bytes(uid.internal))
        return uid
    while True:
        try:
            with socket.create_connection((host, int(port)), timeout=5) as peer:
                data = b""
                while len(data) < 128:
                    part = peer.recv(128 - len(data))
                    if not part:
                        raise ConnectionError("short unique id")
                    data += part
            break
        except OSError:
            if time.monotonic() > deadline:
                raise SystemExit("no unique id from the id server")
            time.sleep(0.2)
    ctypes.memmove(uid.internal, data, 128)
    return uid
