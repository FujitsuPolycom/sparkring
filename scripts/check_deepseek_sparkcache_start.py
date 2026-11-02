"""Check saved DeepSeek V4.1 rank logs and authenticated image probe artifacts."""

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.check_qwen_sparkcache_start import verify_probe  # noqa: E402

PROFILE = "deepseek-v41-flash-sparkcache-tp4"
COMPOSITION = ROOT / "runtime/images/compositions/installer-sparkcache-4a17f1f5198e"
CACHE_ROOT = "/cache/deepseek-v41-flash-sparkcache-ab53c097d7ca-dba1be0a40aa"


def check_log(text, *, rank=0):
    """Require worker facts everywhere and scheduler facts only on node rank zero.

    The startup record carries role, storage settings, model profile and reuse
    policies. Archived config and topology records remain readable.
    The hardware inventory has 17 manager groups and a 256-token block LCM.
    """
    problems = []
    if rank == 0 and not re.search(r"kv lcm block sizes 256\b", text):
        problems.append("missing LCM page")
    for value in re.findall(r"kv lcm block sizes (\d+)", text):
        if int(value) != 256:
            problems.append("inconsistent page geometry")
    rows = re.findall(r"(?:sparkcache: config |sparkcache: startup [^\n]*? )role=(\w+) root=(\S+) dcp=(\d+) access_mode=(\S+)"
                      r" store=(\w+) restore=(\w+) cuda_restore=(\w+) max_span=(\d+)", text)
    roles = {row[0] for row in rows}
    for role in ("WORKER", "SCHEDULER") if rank == 0 else ("WORKER",):
        if role not in roles:
            problems.append("missing " + role + " connector config")
    for _, root, dcp, mode, store, restore, cuda, span in rows:
        if (root, dcp, mode, store, restore, cuda, span) != (
                CACHE_ROOT, "1", "read-write", "True", "True", "False", "524288"):
            problems.append("incompatible connector config")
    rows = re.findall(r"sparkcache: (?:topology|startup) profile=(\S+) groups=(\d+) policies=(\S+)", text)
    if not rows:
        problems.append("missing SparkCache topology")
    for profile, count, policies in rows:
        if (profile != "deepseek-v41-b12x-hma" or int(count) != 17
                or set(policies.split(",")) != {"full", "sliding", "replay", "transient"}):
            problems.append("incompatible SparkCache topology")
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, action="append", required=True,
                        help="four startup logs in node-rank order")
    parser.add_argument("--composition", type=Path, default=COMPOSITION,
                        help="source composition to authenticate (defaults to the combined source layer)")
    parser.add_argument("--image-lock", type=Path, required=True)
    parser.add_argument("--base-receipt", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--lease-log", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if len(args.log) != 4 or len({p.resolve() for p in args.log}) != 4:
            raise ValueError("provide four distinct rank log paths")
        verify_probe(json.loads(args.image_lock.read_text()), PROFILE,
                     args.base_receipt.read_bytes(), args.inventory.read_bytes(), args.lease_log.read_text(),
                     composition=args.composition)
        problems = [f"rank {rank}: {problem}" for rank, path in enumerate(args.log)
                    for problem in check_log(path.read_text(encoding="utf-8"), rank=rank)]
        if problems:
            raise ValueError("; ".join(problems))
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    print("DeepSeek V4.1 startup artifacts agree; H2-H11 remain separate hardware gates.")


if __name__ == "__main__":
    main()
