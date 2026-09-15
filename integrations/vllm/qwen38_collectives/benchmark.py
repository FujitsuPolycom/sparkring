"""Compatibility entry point for the maintained Qwen collective benchmark."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parents[3] / "performance/harnesses/qwen_collectives.py"),
        run_name="__main__",
    )
