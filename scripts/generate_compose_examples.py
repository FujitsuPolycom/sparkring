"""Keep public Compose examples synchronized with canonical serving profiles."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.common import compose  # noqa: E402


def examples():
    source = ROOT / "profiles/qwen38-flash-next-tp2/compose/site.example.yaml"
    for profile in compose.SUPPORTED:
        site = compose.read_site(source)
        _, files = compose.build(profile, site)
        for rank in range(2):
            yield ROOT / "profiles" / profile / "compose" / f"compose.rank{rank}.yaml", files[
                f"rank{rank}/compose.yaml"
            ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if any generated public example differs",
    )
    args = parser.parse_args(argv)
    stale = []
    for path, expected in examples():
        if args.check:
            if not path.is_file() or path.read_bytes() != expected.encode():
                stale.append(path.relative_to(ROOT).as_posix())
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(expected, encoding="utf-8", newline="\n")
    if stale:
        print(
            "Regenerate Compose examples with python scripts/generate_compose_examples.py:\n"
            + "\n".join(stale)
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
