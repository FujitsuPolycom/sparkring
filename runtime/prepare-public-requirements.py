#!/usr/bin/env python3
"""Close the audited freeze over inputs that the public builder can fetch.

The reference freeze records several distributions by the private build
machine's local paths.  The public Containerfile installs those distributions
later from independently pinned public inputs.  This tool removes only that
reviewed set and refuses every unknown local/direct reference.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re


SOURCE_PROVIDED = {
    "b12x",
    "deep-gemm",
    "flashinfer-jit-cache",
    "flashinfer-python",
    "instanttensor",
    "vllm",
}


def normalized_name(line: str) -> str:
    name = line.split("@", 1)[0].strip()
    return re.sub(r"[-_.]+", "-", name).lower()


def close_freeze(text: str) -> tuple[str, list[str]]:
    kept: list[str] = []
    excluded: list[str] = []
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            kept.append(raw)
            continue
        if " @ file://" in line:
            name = normalized_name(line)
            if name not in SOURCE_PROVIDED:
                raise ValueError(
                    f"line {number}: unreviewed machine-local requirement {line!r}"
                )
            excluded.append(name)
            continue
        if " @ " in line:
            raise ValueError(
                f"line {number}: direct requirement is not publicly closed: {line!r}"
            )
        kept.append(line)
    return "\n".join(kept).rstrip() + "\n", sorted(excluded)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        output, excluded = close_freeze(args.freeze.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        parser.error(str(exc))
    args.output.write_text(output, encoding="utf-8", newline="\n")
    print(
        f"public requirements closed: {len(output.splitlines())} pinned line(s); "
        f"excluded {len(excluded)} source-provided distribution(s): "
        f"{', '.join(excluded)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
