"""Load one vLLM extension and prove that CPU import does not initialize CUDA."""

import argparse
import importlib.util
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    path = args.path.resolve(strict=True)
    assert not torch.cuda.is_initialized()
    spec = importlib.util.spec_from_file_location(args.module, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert not torch.cuda.is_initialized()
    print(
        json.dumps(
            {
                "status": "qualified-cpu-import-only",
                "module": args.module,
                "path": str(path),
                "cuda_initialized": torch.cuda.is_initialized(),
            }
        )
    )


if __name__ == "__main__":
    main()
