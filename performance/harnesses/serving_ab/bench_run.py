"""Run llm-inference-bench's llm_decode_bench.py with its self-update check turned off.

python -m performance.harnesses.serving_ab.bench_run BENCH_DIR ARGS...

The benchmark's main() asks GitHub for a newer version and offers to pull it; a campaign keeps one
benchmark revision, so this wrapper loads the script as a module, replaces check_for_update with a function
that returns False, and calls main() with ARGS.
"""
import importlib.util
import sys
from pathlib import Path


def main() -> None:
    bench = Path(sys.argv[1]) / "llm_decode_bench.py"
    spec = importlib.util.spec_from_file_location("llm_decode_bench_campaign", bench)
    module = importlib.util.module_from_spec(spec)
    sys.argv = [str(bench)] + sys.argv[2:]
    spec.loader.exec_module(module)
    module.check_for_update = lambda console: False
    module.main()


if __name__ == "__main__":
    main()
