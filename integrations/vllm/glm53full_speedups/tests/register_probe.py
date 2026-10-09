"""One probe process for the speedups ``test_register.py``: register and report.

Prints one JSON line with the plugin's registration state. Any refusal exits
non-zero with its message. The image's modules are not imported here: the
speedups patches run on their first import in the serving process, and the
pin replay in ``_env`` covers the file checks.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _env
import glm53full_speedups


def main() -> int:
    _env.stub_packages()
    glm53full_speedups.register()
    state = glm53full_speedups.status()
    print(json.dumps({
        "registered_flags": state["registered_flags"],
        "pending_modules": state["pending_modules"],
        "layers": state["layers"],
    }))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except glm53full_speedups.PatchRefused as error:
        print(f"PatchRefused: {error}", file=sys.stderr)
        sys.exit(1)
