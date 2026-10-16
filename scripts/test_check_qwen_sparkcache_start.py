"""Saved-artifact startup checks fail closed on missing or contradictory evidence."""

import json

import pytest

from scripts import check_qwen_sparkcache_start as check
from runtime.common import installer_image
from runtime.images import derived_layer


@pytest.mark.parametrize("page", [2848, 1440])
def test_rank_log_requires_matching_geometry_profile_and_supported_groups(page):
    text = (f"Setting attention block size to {page} tokens\nkv lcm block sizes {page}\n"
            "sparkcache: topology profile=qwen38-flash-next-hybrid groups=5 policies=full,recurrent_align\n")
    assert check.check_log(text, page) == []
    for before, after in [(str(page), "32"), ("groups=5", "groups=65"),
                          ("qwen38-flash-next-hybrid", "glm53-flash-hybrid"),
                          ("full,recurrent_align", "full,sliding")]:
        assert check.check_log(text.replace(before, after), page)
    assert check.check_log(text + "kv lcm block sizes 32\n", page)
    assert check.check_log("", page)


@pytest.mark.parametrize("nodes,page", [(2, 2848), (4, 1440)])
def test_qwen_hw_3_only_rank_zero_requires_enginecore_lcm(nodes, page):
    worker = (f"Setting attention block size to {page} tokens\n"
              "sparkcache: topology profile=qwen38-flash-next-hybrid groups=5 policies=full,recurrent_align\n")
    assert check.check_log(worker, page, rank=0) == ["missing LCM page"]
    assert check.check_log(worker + f"kv lcm block sizes {page}\n", page, rank=0) == []
    for rank in range(1, nodes):
        assert check.check_log(worker, page, rank=rank) == []
        assert check.check_log(worker + "kv lcm block sizes 32\n", page, rank=rank)
        assert check.check_log("", page, rank=rank) == ["missing attention page", "missing SparkCache topology"]


def test_probe_binds_receipt_inventory_native_pins_and_lease_contract():
    inventory = (check.COMPOSITION / "sources.json").read_bytes()
    record = json.loads(inventory)
    files = {path: row["sha256"] for path, row in record["files"].items()}
    files.update(record["native_libraries"])
    files["/usr/local/lib/python3.12/dist-packages/sparkcache/runtime_patches/sparkring-source-3daf882e358b.json"] = derived_layer.digest(inventory)
    base = derived_layer.canonical_json({"files": files})
    parent = json.loads((check.ROOT / record["parent_lock"]).read_text())
    build = json.loads((check.COMPOSITION / "build-inputs.json").read_text())
    plan = {"parent_lock": parent, "payload_bytes": 1,
            "receipts": {derived_layer.BASE_RECEIPT: derived_layer.digest(base),
                         derived_layer.TOOLCHAIN_RECEIPT: "1" * 64}}
    lock = derived_layer.derived_lock(plan, {"Id": "sha256:" + "2" * 64, "Size": 1},
                                     build["name"], build["profiles"])
    profile = installer_image.QWEN_SPARKCACHE_VARIANTS[0]
    lease = "verified SparkCache KV-block lease contract: 11 files"
    check.verify_probe(lock, profile, base, inventory, lease)
    for changed_base, changed_inventory, changed_lease in [
        (b"{}", inventory, lease), (base, b"{}", lease), (base, inventory, "10 files")]:
        with pytest.raises(ValueError):
            check.verify_probe(lock, profile, changed_base, changed_inventory, changed_lease)
