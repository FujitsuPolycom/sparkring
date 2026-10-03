"""Write the Compose builder page for this checkout.

The page (docs/operations/compose-builder.md) writes `sparkring install`
commands and `sparkring compose render` deployments for the profiles of the
checkout it is built from:

    python scripts/generate_compose_builder.py --output DIRECTORY [--verify]

--verify first compares the page's engine with compose.build under Node.js
(scripts/compose_builder/verify.py) and writes nothing if any case differs;
the page's data records the comparison's counts.
"""

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compose_builder import export, verify  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, required=True, help="directory that receives index.html")
    parser.add_argument("--tag", help="release tag the page names and pins; default: the checkout's exact tag")
    parser.add_argument("--commit", help="commit the page names; default: the checkout's HEAD")
    parser.add_argument("--repository", default=export.REPOSITORY, help="GitHub repository that serves install.sh")
    parser.add_argument("--verify", action="store_true", help="compare the engine with compose.build first (needs Node.js)")
    parser.add_argument("--cases", type=int, default=20, help="random valid sites per profile checkpoint for --verify")
    parser.add_argument("--seed", type=int, default=20261002, help="seed of the --verify cases")
    args = parser.parse_args(argv)
    data = export.export(tag=args.tag, commit=args.commit, repository=args.repository)
    summary = None
    if args.verify:
        node = shutil.which("node")
        if node is None:
            parser.error("--verify needs Node.js on PATH")
        summary = verify.run(data, per_checkpoint=args.cases, seed=args.seed, node=node)
        if summary["failures"]:
            print("The engine differs from compose.build; no page was written:", file=sys.stderr)
            for line in summary["failures"][:20]:
                print("  " + line, file=sys.stderr)
            return 1
        summary = {key: value for key, value in summary.items() if key != "failures"}
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "index.html"
    path.write_text(export.page(data, summary), encoding="utf-8", newline="\n")
    print(json.dumps({"page": str(path), "ref": data["ref"], "profiles": len(data["profiles"]), "verification": summary}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
