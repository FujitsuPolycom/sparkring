#!/usr/bin/env python3
"""Read fleet credentials from a mounted file, then run the external Mia launcher.

The image provides /opt/dsv41/boot.py under its upstream licence. No credentials are included
in docker arguments, the launcher receipt, or this source file.
"""
import os
from pathlib import Path
import runpy
import sys

keys = [line.strip() for line in Path(os.environ["API_KEY_FILE"]).read_text().splitlines() if line.strip()]
if not keys or len(set(keys)) != len(keys) or any("," in key or any(not 33 <= ord(c) <= 126 for c in key) for key in keys):
    raise SystemExit("API key file must contain distinct, nonempty, whitespace-free keys")
os.umask(0o077)
os.environ["API_KEY"] = ",".join(keys)
print(f"Fleet authentication enabled: {len(keys)} keys", flush=True)
sys.argv = ["/opt/dsv41/boot.py", *(sys.argv[1:] or ["run"])]
runpy.run_path(sys.argv[0], run_name="__main__")
