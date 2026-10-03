"""Write the Compose builder page for this checkout.

The page (docs/operations/compose-builder.md) writes `sparkring install`
commands and `sparkring compose render` deployments for the profiles of the
checkout it is built from:

    python scripts/generate_compose_builder.py --output DIRECTORY [--verify]

The page's directory also receives images/NAME.json for every other installer
image, which the page loads when that image is selected. --verify first
compares the page's engine with compose.build under Node.js
(scripts/compose_builder/verify.py) on every image and writes nothing if any
case differs; the page's data records the comparison's counts.
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
    parser.add_argument("--image-cases", type=int, default=1,
                        help="random valid sites per profile checkpoint for --verify on each non-default image")
    parser.add_argument("--default-image-only", action="store_true",
                        help="write no data files for the other installer images")
    args = parser.parse_args(argv)
    data = export.export(tag=args.tag, commit=args.commit, repository=args.repository)
    if args.default_image_only:
        data["images"] = [row for row in data["images"] if row["default"]]
    meta = {key: data[key] for key in ("repository", "tag", "commit", "ref", "since_tag", "commits_since")}
    images = {row["file"]: export.image_data(row["name"]) for row in data["images"] if row["file"]}
    summary = None
    if args.verify:
        node = shutil.which("node")
        if node is None:
            parser.error("--verify needs Node.js on PATH")
        runs = [(data, args.cases), *(({**meta, **image}, args.image_cases) for image in images.values())]
        failures, summary = [], {}
        for subject, count in runs:
            result = verify.run(subject, per_checkpoint=count, seed=args.seed, node=node)
            failures += [f"{subject['image'] or 'default image'}: {line}" for line in result.pop("failures")]
            for key, value in result.items():
                summary[key] = value if key == "seed" else summary.get(key, 0) + value
        if failures:
            print("The engine differs from compose.build; no page was written:", file=sys.stderr)
            for line in failures[:20]:
                print("  " + line, file=sys.stderr)
            return 1
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "index.html"
    path.write_text(export.page(data, summary), encoding="utf-8", newline="\n")
    for file, image in images.items():
        (args.output / file).parent.mkdir(parents=True, exist_ok=True)
        (args.output / file).write_text(json.dumps(image, separators=(",", ":")), encoding="utf-8", newline="\n")
    print(json.dumps({"page": str(path), "ref": data["ref"], "profiles": len(data["profiles"]), "images": len(data["images"]),
                      "verification": summary}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
