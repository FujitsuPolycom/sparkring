"""Make the flat serving-adapter modules importable by performance tests."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
INTEGRATION = ROOT / "spark_transport" / "integrations" / "vllm"
sys.path.insert(0, str(INTEGRATION))
