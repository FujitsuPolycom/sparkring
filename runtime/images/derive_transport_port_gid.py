"""Install the prepared RoCEnante transport with one RoCE GID index per HCA into a serving image.

The parent `dev-20260928-plainstatus-cuda1342-nccl2323-status033` carries the
prepared transport under
/opt/sparkring/transports/tp2-rocenante-adaptive-prepared with the files of
transport manifest 9f2c0ae62e1e, whose runtime uses one RoCE GID index for
every HCA (B12X_ROCE_GID_INDEX, else NCCL_IB_GID_INDEX, else 3). This layer
replaces the four bundle files that the per-HCA index changes with their bytes
in integrations/vllm/rocenante_prepared: the runtime reads each device's GID
table at construction and uses the RoCE v2 GID of the device's fabric IPv4
address, with the configured index for a device whose table does not identify
that GID, and logs each device's index; the proxy publishes and routes from
each HCA's own index. A model therefore starts on whatever index each port's
address holds, as after a cabled neighbor restarted. The proxy ABI version
changes from 5 to 6, so ranks of the parent and of this image refuse to
connect to each other.

The installed manifest is rewritten as derive_transport_window.py does: it
keeps the parent's composition fields and records the replaced files'
hashes, and the receipt's `transport_manifest_sha256` and the derived lock
name it. Each replaced file is pinned to its SHA-256 in the parent, which
holds the files that derive_transport_peer_wait.py installed and the
repository bundle's other files, and in the repository bundle;
/opt/sparkring/receipts/derived-transport-port-gid.json records them.
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from runtime.images.derive_transport_window import BUNDLE, SOURCE, update_receipt  # noqa: E402
from runtime.images.derive_transport_window import replace as replace_bundle  # noqa: E402
from runtime.images.derived_layer import Layer, main  # noqa: E402

PARENT_RELEASE = "dev-20260928-plainstatus-cuda1342-nccl2323-status033"
# Bundle-relative file: (SHA-256 in the parent, SHA-256 in this layer).
FILES = {
    "roce/_preparation.py": (
        "905c3cf8311f14d7cafba6808443163297a18a1800fa95c698d06444b2e6b59d",
        "5de39406923a353f13d808242dc9aee776f552c49a0e1d520bb735bb7f54dd71",
    ),
    "roce/_proxy.py": (
        "8ece853f5d05e2bd3b7a8c9a0f41138843a4521cda68ca317c1fba14877e4a71",
        "240a92e2d0b4c5421f3046d16ddee6609e33da51824553cbaddf8c61160399a6",
    ),
    "roce/_roce_proxy.c": (
        "dae4d94f98650c55a4a2042836324203042b7e4fb279eb2aa6eded976e8b0456",
        "95a206775904232bd6bbff98bd408ae89cc06b324149b0d2923d791f703f60f8",
    ),
    "roce/roce_oneshot.py": (
        "3aff7e4a78bcb90ddac8aa99985025b301b376aebf8775bc394209d74d315e42",
        "11bd06f9826810de3bfd62a76ee3d012657c3a50e3c4b947e2102044ef39444f",
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
    name="transport-port-gid",
    purpose=("Each HCA of a RoCEnante runtime uses the RoCE GID index of its fabric address's RoCE v2 GID, "
             "read at startup, and the configured index only when its GID table does not identify that GID"),
    replace=replace,
    update_receipt=update_receipt,
    provenance="/opt/sparkring/receipts/derived-transport-port-gid.json",
    pins={f"{BUNDLE}/{name}": hashes for name, hashes in FILES.items()},
)

if __name__ == "__main__":
    main(LAYER)
