"""Check saved per-rank Qwen startup logs and authenticated image probe artifacts."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime.common import image_lock, installer_image  # noqa: E402

COMPOSITION = ROOT / "runtime/images/compositions/installer-sparkcache-3daf882e358b"


def check_log(text, page, *, rank=0):
    """Return all missing or incompatible startup facts from one rank's log."""
    problems = []
    required = [("attention page", rf"Setting attention block size to {page} tokens")]
    # EngineCore emits the LCM line on node rank zero; workers emit attention pages.
    if rank == 0:
        required.append(("LCM page", rf"kv lcm block sizes {page}\b"))
    for label, pattern in required:
        if not re.search(pattern, text):
            problems.append("missing " + label)
    rows = re.findall(r"sparkcache: topology profile=(\S+) groups=(\d+) policies=(\S+)", text)
    if not rows:
        problems.append("missing SparkCache topology")
    for profile, count, policies in rows:
        if (profile != "qwen38-flash-next-hybrid" or not 1 <= int(count) <= 64
                or set(policies.split(",")) != {"full", "recurrent_align"}):
            problems.append("incompatible SparkCache topology")
    # Refuse contradictory pages, even when a matching line also exists.
    for pattern in (r"Setting attention block size to (\d+) tokens", r"kv lcm block sizes (\d+)"):
        if any(int(value) != page for value in re.findall(pattern, text)):
            problems.append("inconsistent page geometry")
    return problems


def verify_probe(lock, profile, base_raw, inventory_raw, lease_text):
    image_lock.validate(lock, profile)
    if hashlib.sha256(base_raw).hexdigest() != lock["parent_receipt_sha256"]:
        raise ValueError("image probe receipt does not match the built lock")
    inventory = json.loads(inventory_raw)
    expected = json.loads((COMPOSITION / "sources.json").read_text(encoding="utf-8"))
    if inventory != expected:
        raise ValueError("image source inventory differs from the composition")
    files = json.loads(base_raw)["files"]
    target = "/usr/local/lib/python3.12/dist-packages/sparkcache/runtime_patches/sparkring-source-3daf882e358b.json"
    if files.get(target) != hashlib.sha256(inventory_raw).hexdigest():
        raise ValueError("source inventory is not authenticated by the image receipt")
    for target, row in expected["files"].items():
        if files.get(target) != row["sha256"]:
            raise ValueError("source receipt differs: " + target)
    for target, sha in expected["native_libraries"].items():
        if files.get(target) != sha:
            raise ValueError("native receipt differs: " + target)
    if "verified SparkCache KV-block lease contract: 11 files" not in lease_text:
        raise ValueError("missing successful eleven-file lease verification")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=installer_image.QWEN_SPARKCACHE_VARIANTS, required=True)
    parser.add_argument("--log", type=Path, action="append", required=True, help="one startup log per rank, in rank order")
    parser.add_argument("--image-lock", type=Path, required=True)
    parser.add_argument("--base-receipt", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--lease-log", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        nodes = installer_image.profile_nodes(args.profile)
        if len(args.log) != nodes or len({p.resolve() for p in args.log}) != nodes:
            raise ValueError(f"provide {nodes} distinct rank log paths")
        verify_probe(json.loads(args.image_lock.read_text()), args.profile,
                     args.base_receipt.read_bytes(), args.inventory.read_bytes(), args.lease_log.read_text())
        problems = [f"rank {rank}: {problem}" for rank, path in enumerate(args.log)
                    for problem in check_log(path.read_text(encoding="utf-8"), {2: 2848, 4: 1440}[nodes], rank=rank)]
        if problems:
            raise ValueError("; ".join(problems))
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    print("Qwen SparkCache startup artifacts agree; serving correctness remains a separate hardware gate.")


if __name__ == "__main__":
    main()
