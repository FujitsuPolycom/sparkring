"""Synthetic output checks; no CUDA allocation or native-library calls."""
import sys
from types import SimpleNamespace

import pytest

from . import bidirectional_prefill_c_api_probe as probe


def test_nonfinite_output_cannot_pass_tolerance_check(monkeypatch):
    class Scalar:
        def __init__(self, value):
            self.value = value
        def reshape(self, *_):
            return self
        def float(self):
            return self
        def to(self, *_):
            return self
        def remainder(self, value):
            return Scalar(self.value % value)
        def __add__(self, other):
            return Scalar(self.value + (other.value if isinstance(other, Scalar) else other))
        __radd__ = __add__
        def __sub__(self, other):
            return Scalar(self.value - (other.value if isinstance(other, Scalar) else other))
        def __mul__(self, other):
            return Scalar(self.value * other)
        __rmul__ = __mul__
        def __truediv__(self, other):
            return Scalar(self.value / other)
        def abs(self):
            return Scalar(abs(self.value))
        def max(self):
            return self
        def sum(self):
            return self
        def item(self):
            return self.value
        def __gt__(self, other):
            return Scalar(self.value > other.value)
    mask = SimpleNamespace(all=lambda: SimpleNamespace(item=lambda: False))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(isfinite=lambda _: mask,
        arange=lambda *args, **kwargs: Scalar(0), zeros=lambda *args, **kwargs: Scalar(0),
        int32='int32', float32='float32', bfloat16='bfloat16'))
    with pytest.raises(RuntimeError, match='non-finite'):
        probe.validate_noninteger(Scalar(float('nan')), 1024)
