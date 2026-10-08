"""What carried the collectives in one start, from container logs and SIRCL receipts.

Every arm is checked before it is measured:

- ``S`` and ``S+``: every rank's tensor-parallel receipt reads ``nccl=none`` and ``pynccl=skipped``, no
  rank's log shows an NCCL communicator, and the sircl plugin was loaded; the installed shims are listed.
  ``S+`` also needs ``fused_norm`` on in the receipts, and the runner records the fused calls the receipt
  taken after the measurement counts (:func:`fused_calls`).
- ``N``: every rank's log shows ``Init COMPLETE`` for an NCCL communicator and vLLM's ``PYNCCL`` all-reduce
  backend, and no rank loaded the sircl plugin.
- ``P``: the all-reduce backends vLLM reports, recorded without a rule.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

SHIM = re.compile(r"SIRCL shim ([a-z0-9_]+) installed")
BACKENDS = re.compile(r"Using (\[[^\]]*\]) all-reduce backends .* for group '([^']+)'")
RECEIPT = re.compile(r"SIRCL receipt group=(\S+) .*?nccl=(\S+) pynccl=(\S+)")


def scan_log(text: str) -> dict:
    receipts = {}
    for group, nccl, pynccl in RECEIPT.findall(text):
        receipts[group] = {"nccl": nccl, "pynccl": pynccl}
    return {"nccl_info_lines": text.count("NCCL INFO"),
            "nccl_init_complete": len(re.findall(r"NCCL INFO.*Init COMPLETE", text)),
            "nccl_net_ib": len(re.findall(r"NCCL INFO NET/IB : Using", text)),
            "sircl_plugin_loaded": bool(re.search(r"Loading plugin sircl|Platform plugin sircl is activated", text)),
            "shims": sorted(set(SHIM.findall(text))),
            "backends": sorted({f"{group}:{names}" for names, group in BACKENDS.findall(text)}),
            "receipts": receipts}


def fused_calls(receipt: Mapping | None) -> dict | None:
    """The fused-norm part of a receipt: the setting, its provider and the fused calls the receipt counted.

    The adapter counts every fused call as the decision row ``all_reduce`` / ``sircl`` / ``fused_rms_norm``
    (sparkring_sircl/vllm/norm_fusion.py). Only models on vLLM's DeepSeek-V3.2 code call the helper the
    fused path replaces, so a model on other code reports fused_norm on and no fused call.
    """
    if not receipt:
        return None
    detail = receipt.get("fused_norm_detail") or {}
    calls = sum(int(row.get("calls", 0)) for row in receipt.get("decisions") or []
                if row.get("method") == "fused_rms_norm")
    return {"fused_norm": receipt.get("fused_norm"), "provider": detail.get("provider"), "fused_calls": calls}


def check(arm: str, logs: Sequence[str], receipts: Sequence[Mapping | None] = ()) -> dict:
    ranks = [scan_log(text) for text in logs]
    problems = []
    if arm in ("S", "S+"):
        for r, scan in enumerate(ranks):
            tp = next((v for k, v in scan["receipts"].items() if k.startswith("tp:")), None)
            if not tp or tp["nccl"] != "none" or tp["pynccl"] != "skipped":
                problems.append(f"rank {r}: tensor-parallel receipt {tp}")
            if scan["nccl_info_lines"]:
                problems.append(f"rank {r}: {scan['nccl_info_lines']} NCCL INFO lines")
            if not scan["sircl_plugin_loaded"]:
                problems.append(f"rank {r}: the sircl plugin was not loaded")
        if arm == "S+":
            for r, receipt in enumerate(receipts):
                if not receipt or receipt.get("fused_norm") != "on":
                    problems.append(f"rank {r}: fused_norm is {receipt and receipt.get('fused_norm')!r}, not on")
    elif arm == "N":
        for r, scan in enumerate(ranks):
            if scan["nccl_init_complete"] < 1:
                problems.append(f"rank {r}: no NCCL communicator reached Init COMPLETE")
            if not any("PYNCCL" in b for b in scan["backends"]):
                problems.append(f"rank {r}: vLLM reports no PYNCCL all-reduce backend")
            if scan["sircl_plugin_loaded"]:
                problems.append(f"rank {r}: the sircl plugin was loaded")
    return {"arm": arm, "ranks": ranks, "fused": [fused_calls(r) for r in receipts], "problems": problems,
            "passed": not problems}
