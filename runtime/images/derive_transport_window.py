"""Install this repository's prepared RoCEnante transport bundle into a serving image.

The parent image carries the prepared transport under
/opt/sparkring/transports/tp2-rocenante-adaptive-prepared. This layer replaces
each bundle file whose bytes in integrations/vllm/rocenante_prepared differ
from the parent's receipt. The installed manifest keeps the parent's
composition fields and records the replaced files' hashes; the receipt's
`transport_manifest_sha256` and the derived lock name that manifest.

With the parent `dev-20260924-cuda1342-nccl2323-status031`, the replaced file
is `roce/_roce_proxy.c`, whose proxy paces hardware-forwarded (two-hop) stripes
within a send window, and the installed manifest SHA-256 is
2eef276d54030a71c4774b92c89008dd563104b791734f6f588a3ca34a7c4943.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from runtime.images.derived_layer import Layer, canonical_json, main, sha  # noqa: E402

PROFILE = "tp2-rocenante-adaptive-prepared"
BUNDLE = "/opt/sparkring/transports/" + PROFILE
SOURCE = Path(__file__).resolve().parents[2] / "integrations/vllm/rocenante_prepared"


def replace(read, receipt, source=SOURCE):
    capabilities = receipt["capabilities"]
    if capabilities.get("transport_profile") != PROFILE:
        raise ValueError("Parent does not select the prepared RoCEnante transport")
    installed_bytes = read(BUNDLE + "/manifest.json")
    if sha(installed_bytes) != capabilities.get("transport_manifest_sha256"):
        raise ValueError("Parent transport manifest differs from its receipt capability")
    installed = json.loads(installed_bytes)
    repository = json.loads((source / "manifest.json").read_bytes())
    if installed["name"] != PROFILE or repository["name"] != PROFILE:
        raise ValueError("Transport manifest names a different profile")
    if set(installed["files"]) != set(repository["files"]):
        raise ValueError("Repository bundle and parent bundle list different files")
    replaced = {}
    for name, expected in sorted(repository["files"].items()):
        data = (source / name).read_bytes()
        if sha(data) != expected:
            raise ValueError("Repository bundle file differs from its manifest: " + name)
        if sha(data) != installed["files"][name]:
            installed["files"][name] = sha(data)
            replaced[f"{BUNDLE}/{name}"] = data
    if not replaced:
        raise ValueError("Parent already carries the repository transport bundle")
    replaced[BUNDLE + "/manifest.json"] = canonical_json(installed)
    return replaced


def update_receipt(receipt, replaced):
    manifest = sha(replaced[BUNDLE + "/manifest.json"])
    receipt["capabilities"]["transport_manifest_sha256"] = manifest
    return {"transport_manifest_sha256": manifest}


LAYER = Layer(
    name="transport-window",
    purpose=__doc__.splitlines()[0],
    replace=replace,
    update_receipt=update_receipt,
)

if __name__ == "__main__":
    main(LAYER)
