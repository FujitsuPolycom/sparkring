"""Verify the selected communication source before starting the vLLM CLI."""

import runpy
import sys

from sparkring_transport_selector import install_from_environment, require_active


def main():
    install_from_environment()
    require_active()
    sys.argv[0] = "vllm"
    runpy.run_module("vllm.entrypoints.cli.main", run_name="__main__")


if __name__ == "__main__":
    main()
