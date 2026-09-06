"""Verify immutable runtime files before executing model warmup and serving."""

import os
import subprocess
import sys

subprocess.run(
    [sys.executable, "-S", "-B", "/opt/sparkring/bin/verify-performance.py"], check=True
)
os.execv(
    "/opt/sparkring/bin/serve-with-warmup.py",
    ["/opt/sparkring/bin/serve-with-warmup.py", *sys.argv[1:]],
)
