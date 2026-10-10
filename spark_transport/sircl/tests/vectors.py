"""Reference vectors of the route derivation and the wire arithmetic.

``tests/data/routes.json`` lists layouts (cables, rank positions, lane
count) with every rank's expected route map and lane paths;
``tests/data/numeric.json`` lists stripes, chunks, flag lines, posting orders,
op words, descriptors, Swing schedules and algorithm choices with their
expected values. Both are plain data shared by the Python tests and the proxy
simulator's cross-checks.
"""

from __future__ import annotations

import json
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"


def vector_dir() -> Path:
    return DATA


def load(name: str) -> dict:
    return json.loads((DATA / name).read_text(encoding="utf-8"))
