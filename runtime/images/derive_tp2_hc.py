"""Allow Qwen hyper-connection token-row ownership on TP2 as well as TP4.

The parent image's HC prefill helper, qwen4-prefill feature hooks and vLLM
startup audit accept token-row ownership (`VLLM_QWEN3_8_HC_PREFILL_MODE=shard`)
only on four ranks. This layer edits exactly those rank-count gates in the
parent's own files:

- vllm/models/qwen4_exp/nvidia/hc_prefill.py divides token rows among the
  actual tensor-parallel group size (2 or 4) instead of a fixed four;
- the qwen4-prefill HC-fusion and MTP-GEMM hooks accept rank agreement from
  two or four ranks; the feature manifest and capability inventory are
  re-hashed and the capability lists TP 2 and 4;
- the startup audit message names TP2 or TP4.

The receipt lists the TP2 row-sharding HC mode, and
/opt/sparkring/receipts/derived-tp2-hc.json records every replaced file. Every
file under /opt/sparkring/features is written, including unchanged ones, so
that record lists the complete feature tree.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from runtime.images.derived_layer import Layer, canonical_json, main, sha, swap  # noqa: E402

SITE = "/usr/local/lib/python3.12/dist-packages/"
HC = SITE + "vllm/models/qwen4_exp/nvidia/hc_prefill.py"
AUDIT = SITE + "vllm/entrypoints/launchers/api_server/sparkring_startup_audit.py"
FEATURES = "/opt/sparkring/features/"
PREFILL = FEATURES + "qwen4-prefill/"
TP2_SHARD = {"prefill_row_ownership": "shard", "projection_tp": "0"}

HC_SWAPS = (
    ("        group.world_size != 4\n", "        group.world_size not in (2, 4)\n"),
    ('error = "Qwen HC ownership requires BF16 TP4/PP1/DP1/DCP1/PCP1"',
     'error = "Qwen HC ownership requires BF16 TP2 or TP4/PP1/DP1/DCP1/PCP1"'),
    ("    if rows < 1024 or rows % 4:\n", "    if rows < 1024 or rows % get_tp_group().world_size:\n"),
    ("    rows: int\n    rank: int\n    group: Any\n", "    rows: int\n    rank: int\n    group: Any\n    size: int = 4\n"),
    ("        count = self.rows // 4\n", "        count = self.rows // self.size\n"),
    ("        if tensor.shape[0] != self.rows // 4:\n", "        if tensor.shape[0] != self.rows // self.size:\n"),
    ("        output = source.new_empty((self.rows // 4, *source.shape[1:]))\n",
     "        output = source.new_empty((self.rows // self.size, *source.shape[1:]))\n"),
    ("    return RowOwnership(rows, group.rank_in_group, comm)\n",
     "    return RowOwnership(rows, group.rank_in_group, comm, group.world_size)\n"),
    ("            rows if owner is None else rows // 4,\n", "            rows if owner is None else rows // owner.size,\n"),
)
FEATURE_SWAPS = {
    PREFILL + "qwen4_hc_fusion.py": (
        "            if group.world_size != 4 or any(v != votes[0] for v in votes):\n",
        "            if group.world_size not in (2, 4) or any(v != votes[0] for v in votes):\n"),
    PREFILL + "qwen4_mtp_gemm.py": (
        "            if group.world_size != 4 or any(v != identity for v in votes):\n",
        "            if group.world_size not in (2, 4) or any(v != identity for v in votes):\n"),
}
AUDIT_SWAP = ('"This HC sharding implementation requires TP4."',
              '"This HC sharding implementation requires TP2 or TP4."')


def edit(data, *swaps):
    text = data.decode("utf-8")
    for old, new in swaps:
        text = swap(text, old, new)
    return text.encode("utf-8")


def replace(read, receipt):
    replaced = {HC: edit(read(HC), *HC_SWAPS), AUDIT: edit(read(AUDIT), AUDIT_SWAP)}
    features = {path: read(path) for path in sorted(receipt["files"]) if path.startswith(FEATURES)}
    for path, change in FEATURE_SWAPS.items():
        features[path] = edit(features[path], change)
    manifest = json.loads(features[PREFILL + "manifest.json"])
    for name in manifest["files"]:
        manifest["files"][name] = sha(features[PREFILL + name])
    features[PREFILL + "manifest.json"] = canonical_json(manifest)
    capabilities = json.loads(features[FEATURES + "capabilities.json"])
    feature = capabilities["features"]["qwen4-prefill"]
    feature["manifest_sha256"] = sha(features[PREFILL + "manifest.json"])
    feature["supported_tp"] = [2, 4]
    for relative in feature["files"]:
        feature["files"][relative] = sha(features[FEATURES + relative])
    features[FEATURES + "capabilities.json"] = canonical_json(capabilities)
    replaced.update(features)
    return replaced


def update_receipt(receipt, replaced):
    modes = receipt["capabilities"]["hc_supported_modes"]
    if TP2_SHARD not in modes["2"]:
        modes["2"].append(TP2_SHARD)
    return {}


LAYER = Layer(
    name="tp2-hc",
    purpose="Qwen HC token-row sharding and qwen4-prefill hooks on TP2 as well as TP4",
    replace=replace,
    provenance="/opt/sparkring/receipts/derived-tp2-hc.json",
    update_receipt=update_receipt,
)

if __name__ == "__main__":
    main(LAYER)
