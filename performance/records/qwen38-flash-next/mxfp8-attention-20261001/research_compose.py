#!/usr/bin/env python3
"""Derive a research Compose file from one rank of an installer deployment of qwen38-flash-next-tp2.

The installer writes each rank's Compose file under
``/srv/sparkring/sparkring/<deployment>/containers/<id>/compose.yaml``. This
program copies it with four exact changes, each of which must match once, so
the research rank runs the same image, command, environment, caches and
limits as the installer's rank:

- the project and container names become ``sr-research-mxattn-r<rank>``;
- the ``io.sparkring.deployment`` label becomes ``research-mxattn``, so the
  container is not taken for the installer's deployment;
- the read-only model mount points at ``--model`` instead of the pinned
  checkpoint;
- the runtime-binding file, which names the installer's container, is not
  mounted, and ``SPARKRING_RUNTIME_BINDING`` is not set.

    python3 research_compose.py --input compose.yaml --model DIR --output research.yaml
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


def swap(text, pattern, replacement, flags=0):
    new, count = re.subn(pattern, replacement, text, flags=flags)
    if count != 1:
        raise SystemExit(f"expected one match of {pattern!r}, found {count}")
    return new


def derive(text, model):
    rank = re.search(r"(?m)^name: sr-sparkring-qwen38-flash-next--[0-9a-f]+-r(\d+)$", text).group(1)
    text = swap(text, r"(?m)^name: sr-sparkring-qwen38-flash-next--[0-9a-f]+-r\d+$", f"name: sr-research-mxattn-r{rank}")
    text = swap(text, r"(?m)^    container_name: sr-sparkring-qwen38-flash-next--[0-9a-f]+-r\d+$",
                f"    container_name: sr-research-mxattn-r{rank}")
    text = swap(text, r"(?m)^      io\.sparkring\.deployment: [0-9a-f]{64}$", "      io.sparkring.deployment: research-mxattn")
    text = swap(text, r"(?m)^      source: /srv/sparkring/sparkring/checkpoints/[^\n]+\n      target: /models/target$",
                f"      source: {model}\n      target: /models/target")
    text = swap(text, r"(?m)^      SPARKRING_RUNTIME_BINDING: /run/sparkring/runtime-binding\.json\n", "")
    text = swap(text, r"(?m)^    - type: bind\n      source: [^\n]+/runtime-binding\.json\n      target: /run/sparkring/runtime-binding\.json\n"
                r"      read_only: true\n      bind:\n        create_host_path: false\n", "")
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--model", required=True, help="absolute checkpoint directory on this Spark")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    args.output.write_text(derive(args.input.read_text(), args.model))


if __name__ == "__main__":
    main()
