"""Plan or execute bounded four-rank RC checks using a rendered mesh site."""
from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
from pathlib import Path
import re
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("mtp_mesh_profile", HERE.parent / "profile.py")
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


def make_plan(launch: Path, receipt_path: Path, rows: list[int], port: int) -> dict:
    """Reject modified rendered inputs and resolve rank-specific execution argv."""
    if not rows or any(q not in (4, 8, 12, 16, 20, 24, 28, 32, 64) for q in rows):
        raise ValueError("Rows must belong to the bounded native correctness matrix")
    if len(set(rows)) != len(rows) or not 1024 <= port <= 65535 - len(rows):
        raise ValueError("Rows must be distinct and rendezvous ports must fit the TCP range")
    site, topology, _ = profile.load_site(launch / "site.json")
    rendered = json.loads((launch / "fabric-plan.json").read_text())
    receipt = profile.load_image_receipt(receipt_path)
    if rendered.get("schema") != "sparkring-mtp3-mesh-render/v1":
        raise ValueError("Expected a rendered mesh plan")
    if rendered.get("image_receipt_sha256") != profile.sha(receipt_path):
        raise ValueError("Image receipt does not match rendered plan")
    if rendered.get("image", {}).get("image_id") != receipt["image_id"]:
        raise ValueError("Image identity disagrees with rendered plan")
    for name in ("site.json", "fabric.json"):
        if rendered.get("files", {}).get(name) != profile.sha(launch / name):
            raise ValueError(f"Rendered {name} hash mismatch")
    if rendered.get("topology_sha256") != topology.sha256:
        raise ValueError("Rendered topology identity mismatch")
    contract = json.loads((profile.ROOT / "spark_transport/experiments/glm53_rocenante_overlay/overlay_contract.json").read_text())
    runtime = contract["runtime"]
    cells = []
    for index, q in enumerate(rows):
        ranks = []
        for rank in range(4):
            node = topology.rank(rank)
            plan_rank = rendered["ranks"][rank]
            if (plan_rank["rank"] != rank or plan_rank["ssh_alias"] != node.ssh_alias
                    or plan_rank["management_netdev"] != node.management_netdev):
                raise ValueError("Rendered rank identity disagrees with topology")
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@-]*", node.ssh_alias):
                raise ValueError("SSH target must be a non-option alias or user@host")
            env = {"PYTHONPATH": "/qualification-empty", "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1",
                   "RANK": str(rank), "LOCAL_RANK": "0", "WORLD_SIZE": "4",
                   "MASTER_ADDR": site["management_addresses"][0], "MASTER_PORT": str(port + index),
                   "GLOO_SOCKET_IFNAME": node.management_netdev,
                   "B12X_ROCE_HCA": ",".join(contract["canonical_hca_order"]),
                   "B12X_ROCE_PEER_HCA_MAP": contract["peer_hca_maps"][str(rank)],
                   "B12X_ROCE_GID_INDEX": str(runtime["gid_index"]),
                   "B12X_ROCE_TWO_WAVE_THRESHOLD_BYTES": str(runtime["direct_then_diagonal_threshold_bytes"]),
                   "B12X_ROCE_WAVE_MODE": runtime["wave_mode"], "B12X_ROCE_CACHE_DIR": "/cache/roce"}
            name = f"mtp3-native-{port}-q{q}-r{rank}"
            argv = ["docker", "run", "--rm", "-i", "--name", name, "--gpus", "all", "--privileged",
                    "--network", "host", "--ipc", "host", "--ulimit", "memlock=-1:-1",
                    "-v", f"{site['cache_roots'][rank]}/native-qualification:/cache"]
            for key, value in env.items():
                argv += ["-e", key + "=" + value]
            argv += ["--entrypoint", "timeout", receipt["image_id"], "240", "python3", "-", "--bytes", str(q * 4096 * 2),
                     "--warmups", "2", "--samples", "3", "--graph-ops", "3", "--retire-timeout", "10"]
            ranks.append({"rank": rank, "host": node.ssh_alias, "container": name, "argv": argv})
        cells.append({"rows": q, "ranks": ranks})
    return {"schema": "sparkring-mtp3-native-plan/v1", "status": "research-only", "image": receipt["image_id"],
            "source_sha256": profile.sha(HERE / "native_roce.py"), "cells": cells}


def remote(host: str, argv: list[str], source: str | None = None) -> dict:
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, shlex.join(argv)],
                            input=source, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=270)
    return {"host": host, "argv": argv, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--image-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", nargs="+", type=int, default=[4, 20, 28, 64])
    parser.add_argument("--port", type=int, default=29960)
    parser.add_argument("--execute-authorized", action="store_true")
    args = parser.parse_args()
    plan = make_plan(args.launch, args.image_receipt, args.rows, args.port)
    if not args.execute_authorized:
        print(json.dumps(plan, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    # Require an exclusive test window before creating any GPU/RDMA workload.
    for rank in plan["cells"][0]["ranks"]:
        state = remote(rank["host"], ["docker", "ps", "-q"])
        image = remote(rank["host"], ["docker", "image", "inspect", plan["image"], "--format", "{{.Id}}"])
        (args.output / f"preflight-r{rank['rank']}.json").write_text(json.dumps({"containers": state, "image": image}, indent=2))
        if state["returncode"] or state["stdout"].strip() or image["returncode"] or image["stdout"].strip() != plan["image"]:
            raise RuntimeError("Require no running containers and the exact image on every rank")
    source = (HERE / "native_roce.py").read_text()
    for cell in plan["cells"]:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda rank: remote(rank["host"], rank["argv"], source), cell["ranks"]))
        (args.output / f"q{cell['rows']}.json").write_text(json.dumps(results, indent=2) + "\n")
        passed = all(row["returncode"] == 0 for row in results) and "EVIDENCE_JSON " in results[0]["stdout"]
        print(json.dumps({"rows": cell["rows"], "passed": passed}), flush=True)
        if not passed:
            raise RuntimeError("Native check failed; inspect test containers before serving")


if __name__ == "__main__":
    main()
