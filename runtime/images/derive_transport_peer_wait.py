"""Install the prepared RoCEnante transport with the supervised peer wait into a serving image.

The parent carries the prepared transport under
/opt/sparkring/transports/tp2-rocenante-adaptive-prepared with the files of
transport manifest 2eef276d5403, whose kernels end a peer wait after a fixed
number of polls and poison the runtime on a live but late peer. This layer
replaces the six bundle files that the supervised peer wait changes with their
bytes in integrations/vllm/rocenante_prepared: the kernels also poll a host
abort word, and the proxy thread stops a wait only for an unreachable, stopped
or inconsistent peer or after B12X_ROCE_PEER_TIMEOUT_S, and logs every stall.
The proxy ABI version changes from 4 to 5, so ranks of the parent and of this
image refuse to connect to each other.

The installed manifest is rewritten as derive_transport_window.py does: it
keeps the parent's composition fields and records the replaced files'
hashes, and the receipt's `transport_manifest_sha256` and the derived lock
name it. Each replaced file is pinned to its SHA-256 in the parent
`dev-20260927-mimovision-cuda1342-nccl2323-status032` and in the repository
bundle; /opt/sparkring/receipts/derived-transport-peer-wait.json records them.
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from runtime.images.derive_transport_window import BUNDLE, SOURCE, update_receipt  # noqa: E402
from runtime.images.derive_transport_window import replace as replace_bundle  # noqa: E402
from runtime.images.derived_layer import Layer, main  # noqa: E402

# Bundle-relative file: (SHA-256 in the parent, SHA-256 in this layer).
FILES = {
    "roce/_allgather_cute.py": (
        "2bab616e7a0949fb83b8e933dbb9c531647659840a607bcd0118652569e389e5",
        "749b839f0db91878c964d21e3af9b921cd6273215d9599b3e35dffa4d3375f09",
    ),
    "roce/_cute_intrinsics.py": (
        "11737796a80a0966c04ee347d5ef83a662bde16dc188022134d2cbaa1567410b",
        "a65f247336a7056e8f161096021e4e9954c3142ba9e76452ccb6d6460756cf6d",
    ),
    "roce/_oneshot_cute.py": (
        "2c791413e084f343f782321e0187d5f8e863832a6ef38cff7c2a365e7b83a0ea",
        "98591c7432cf1ad1af8ec0ad7f3d24741cc9086f42f7763409126e5786b305b0",
    ),
    "roce/_proxy.py": (
        "f783d078d06953f79bcf3a26a61c1ba853d5ec56d07f3e927e5c14275852d854",
        "8ece853f5d05e2bd3b7a8c9a0f41138843a4521cda68ca317c1fba14877e4a71",
    ),
    "roce/_roce_proxy.c": (
        "1dcdf4d1a5b3bec2f64029d010a628d5ea3554c8c616c02da0d7aee9be123e35",
        "dae4d94f98650c55a4a2042836324203042b7e4fb279eb2aa6eded976e8b0456",
    ),
    "roce/roce_oneshot.py": (
        "1033895ecd87c6b7d5d0728d7ba598c5a6f50db50a4417db901a87761820bd70",
        "3aff7e4a78bcb90ddac8aa99985025b301b376aebf8775bc394209d74d315e42",
    ),
}


def replace(read, receipt, source=SOURCE):
    """Replace exactly the pinned bundle files and the installed manifest."""
    replaced = replace_bundle(read, receipt, source=source)
    expected = {f"{BUNDLE}/{name}" for name in FILES} | {BUNDLE + "/manifest.json"}
    if set(replaced) != expected:
        raise ValueError("The repository bundle differs from the parent in other files than this layer pins: "
                         + ", ".join(sorted(set(replaced) ^ expected)))
    return replaced


LAYER = Layer(
    name="transport-peer-wait",
    purpose=("A RoCEnante collective waits for a late peer while the peer's queue pairs acknowledge "
             "checks, and poisons the runtime only for an unreachable, stopped or inconsistent peer "
             "or after B12X_ROCE_PEER_TIMEOUT_S"),
    replace=replace,
    update_receipt=update_receipt,
    provenance="/opt/sparkring/receipts/derived-transport-peer-wait.json",
    pins={f"{BUNDLE}/{name}": hashes for name, hashes in FILES.items()},
)

if __name__ == "__main__":
    main(LAYER)
