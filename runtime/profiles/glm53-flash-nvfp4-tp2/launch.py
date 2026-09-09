"""Compatibility entry point for the canonical NVFP4-Spark TP2 profile."""
import importlib.util
from pathlib import Path


def main():
    path = Path(__file__).resolve().parent.parent / "glm53-flash-spark-tp2/launch.py"
    spec = importlib.util.spec_from_file_location("canonical_spark_tp2_launcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


if __name__ == "__main__":
    main()
