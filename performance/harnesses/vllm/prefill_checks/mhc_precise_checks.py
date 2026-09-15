"""CLI for semantic and cold-prefill checks on an explicitly selected deployment."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from performance.harnesses.vllm.prefill_checks.harness import Harness


def main():
    Harness(require_prefix=True, clock_name="perf_counter").main()


if __name__ == "__main__":
    main()
