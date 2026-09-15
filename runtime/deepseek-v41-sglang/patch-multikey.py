#!/usr/bin/env python3
"""Accept line-separated operator keys through the pinned SGLang auth module.

Only ordinary API keys are split. A separately configured admin key stays exact.
The entrypoint joins the file's lines with commas for the internal API_KEY value.
Run with python -S in a CPU-only container; stdout is the patched source.
"""
from pathlib import Path
import sys

source = Path(sys.argv[1]).read_text()
old = "        return secrets.compare_digest(parts[1], expected_token)"
new = """        # Operator multi-key authentication.
        # Keep the raw value valid for SGLang's internal authenticated callers.
        candidates = [expected_token]
        if expected_token == api_key and expected_token != admin_api_key:
            candidates += [key.strip() for key in expected_token.split(",") if key.strip()]
        return any(secrets.compare_digest(parts[1], key) for key in candidates)"""
if source.count(old) != 1:
    raise SystemExit("auth source changed; refusing an unverified patch")
print(source.replace(old, new, 1), end="")
