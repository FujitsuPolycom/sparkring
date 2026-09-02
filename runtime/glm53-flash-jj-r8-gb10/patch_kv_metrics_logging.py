#!/usr/bin/env python3
"""Allow KV connectors to provide short multi-line log summaries."""

from __future__ import annotations

import argparse
from pathlib import Path


_ORIGINAL = '''            xfer_metrics = self.transfer_stats_accumulator.reduce()
            xfer_metrics_str = ", ".join(f"{k}={v}" for k, v in xfer_metrics.items())
            log_fn("KV Transfer metrics: %s", xfer_metrics_str)
'''

_REPLACEMENT = '''            formatter = getattr(
                self.transfer_stats_accumulator, "format_log_lines", None
            )
            if callable(formatter):
                for line in formatter():
                    log_fn("%s", line)
            else:
                xfer_metrics = self.transfer_stats_accumulator.reduce()
                xfer_metrics_str = ", ".join(
                    f"{k}={v}" for k, v in xfer_metrics.items()
                )
                log_fn("KV Transfer metrics: %s", xfer_metrics_str)
'''


def apply_patch(path: Path) -> None:
    """Replace the pinned logger block or reject an unexpected source tree."""

    source = path.read_text(encoding="utf-8")
    if _ORIGINAL not in source:
        raise RuntimeError("pinned KV metrics logger block differs")
    path.write_text(
        source.replace(_ORIGINAL, _REPLACEMENT, 1),
        encoding="utf-8",
        newline="\n",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    apply_patch(args.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
