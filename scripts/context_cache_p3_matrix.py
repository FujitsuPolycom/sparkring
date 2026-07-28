#!/usr/bin/env python3
"""P3 sabotage matrix: damage, request, verdict — machine-readable.

Pass criteria per mode (goal G-CACHE P3):
  engine_survived  - API healthy after the request
  no_wrong_output  - completion either correct-recall or a clean error
  self_healed      - the damaged rank's entry was removed
  no_partial       - other ranks' entries untouched
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from context_cache_gate import DEFAULT_BASE_URL, build_prompt, run_request  # noqa: E402

# Ring SSH targets, one per rank. Set SPARKRING_TARGETS to a comma-separated
# user@host list (rank 0 first), as the sibling runner scripts do; the default
# below is a placeholder list so the module stays importable without it.
_DEFAULT_TARGETS = "user@spark0,user@spark1,user@spark2,user@spark3"
HOSTS = dict(
    enumerate(
        target.strip()
        for target in os.environ.get(
            "SPARKRING_TARGETS", _DEFAULT_TARGETS
        ).split(",")
        if target.strip()
    )
)
# Path to cache_manifest.py (the fail-closed engine) as deployed on each host.
# Override with SPARKRING_ENGINE for real runs.
ENGINE = os.environ.get(
    "SPARKRING_ENGINE",
    "<engine-path>/persistent_context_cache/cache_manifest.py",
)
# Name of the running vLLM serving container on each host.
CONTAINER = os.environ.get("SPARKRING_CONTAINER", "<container>")


def ssh(rank: int, command: str) -> str:
    result = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=10", HOSTS[rank], command],
        capture_output=True, text=True, timeout=180,
    )
    return result.stdout.strip()


def manifest_counts() -> dict[int, int]:
    counts = {}
    for rank in HOSTS:
        out = ssh(
            rank,
            "ls $HOME/sparkring-context-cache/store/manifests/*/ "
            "2>/dev/null | wc -l",
        )
        counts[rank] = int(out or 0)
    return counts


def healthy(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=8) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def wait_healthy(base_url: str, seconds: int = 120) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        if healthy(base_url):
            return True
        time.sleep(5)
    return False


def fire(base_url: str, words: int, seed: int) -> dict:
    prompt = build_prompt(words, seed)
    try:
        result = run_request(base_url, prompt, 48)
        result["error"] = None
        return result
    except Exception as error:  # noqa: BLE001
        return {"error": repr(error), "completion": "", "ttft_seconds": None}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--words", type=int, default=24000)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    truth = " ".join(
        build_prompt(args.words, args.seed)
        .split("LOG BEGIN\n")[1].split("\nLOG END")[0].split()[:5]
    )
    report = {"truth_recall": truth, "modes": []}

    for mode, rank in (("bitflip", 2), ("truncate", 1), ("fingerprint", 3)):
        print(f"=== {mode} on rank {rank} ===", flush=True)
        if not wait_healthy(args.base_url):
            report["modes"].append({"mode": mode, "error": "api-not-healthy"})
            break
        # Ensure a VERIFIED-healthy entry on every rank before damaging
        # anything: otherwise a mode can "pass" simply because the entry
        # was already retired and its request was an ordinary clean miss.
        before = {}
        for attempt in range(4):
            fire(args.base_url, args.words, args.seed)
            time.sleep(4)
            before = manifest_counts()
            if all(before.get(r, 0) >= 1 for r in HOSTS):
                break
            print(f"  reseed attempt {attempt + 1}: {before}", flush=True)
        if not all(before.get(r, 0) >= 1 for r in HOSTS):
            report["modes"].append(
                {"mode": mode, "error": "could-not-seed-all-ranks",
                 "counts": before}
            )
            continue
        # and prove it actually restores before we damage it
        probe = fire(args.base_url, args.words, args.seed)
        seeded_restore_ttft = probe.get("ttft_seconds")
        damage = ssh(
            rank,
            f"sudo -n python3 /tmp/context_cache_sabotage.py --store "
            f"$HOME/sparkring-context-cache/store --engine {ENGINE} "
            f"--mode {mode}",
        )
        if "damaged" not in damage:
            report["modes"].append(
                {"mode": mode, "rank": rank, "passed": False,
                 "error": "sabotage-did-not-apply", "tool_output": damage[-200:]}
            )
            print(f"  SABOTAGE FAILED TO APPLY: {damage[-200:]}", flush=True)
            continue
        result = fire(args.base_url, args.words, args.seed)
        time.sleep(3)
        # Convergence probe: after the damaged entry is retired, the very
        # next identical request must succeed (fresh prefill + republish).
        recovery = fire(args.base_url, args.words, args.seed)
        time.sleep(5)
        after = manifest_counts()
        # Miss-reason attribution evidence: connector lines from every rank.
        logs = {}
        for r in HOSTS:
            logs[r] = ssh(
                r,
                f"docker logs {CONTAINER} --since 6m 2>&1 | "
                "grep -a 'spark-context-cache' | tail -6",
            )
        engine_survived = healthy(args.base_url)
        completion = result.get("completion", "")
        no_wrong_output = (truth in completion) or bool(result.get("error"))
        others_intact = all(
            after.get(r, 0) >= 1 for r in HOSTS if r != rank
        )
        entry = {
            "mode": mode,
            "rank": rank,
            "damage": damage[-200:] if damage else "",
            "before": before,
            "after": after,
            "engine_survived": engine_survived,
            "no_wrong_output": no_wrong_output,
            "self_healed": after.get(rank, 1) == 0
            or before.get(rank, 0) > after.get(rank, 0),
            "others_intact": others_intact,
            "request_error": result.get("error"),
            "ttft_seconds": result.get("ttft_seconds"),
            "completion_head": completion[:80],
            "rank_logs": logs,
            "seeded_restore_ttft": seeded_restore_ttft,
        }
        entry["recovery_ok"] = truth in recovery.get("completion", "")
        entry["recovery_ttft"] = recovery.get("ttft_seconds")
        entry["passed"] = bool(
            entry["engine_survived"]
            and entry["no_wrong_output"]
            and entry["recovery_ok"]
        )
        print(json.dumps(entry, indent=1, sort_keys=True), flush=True)
        report["modes"].append(entry)

    report["passed"] = bool(
        report["modes"]
        and all(m.get("passed") for m in report["modes"])
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print("P3 matrix:", "PASS" if report["passed"] else "FAIL")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
