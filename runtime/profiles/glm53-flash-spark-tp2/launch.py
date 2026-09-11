"""Compatibility entry point for the shared guarded TP2 launcher."""
from pathlib import Path

_source = Path(__file__).resolve().parents[2] / "common/tp2.py"
__file__ = str(_source)
exec(compile(_source.read_bytes(), str(_source), "exec"), globals())
