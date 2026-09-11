"""Four-rank smoke test for the ctypes/PyTorch TP4 boundary."""

import os

import torch

from spark_tp4_backend import _NativeSession

if "RANK" not in os.environ and __name__ != "__main__":
    import pytest

    pytest.skip(
        "requires a launched four-rank TP4 integration environment",
        allow_module_level=True,
    )

rank = int(os.environ["RANK"])
iterations = int(os.getenv("ITERATIONS", "100"))
session = _NativeSession(rank)

for sequence in range(iterations):
    parity = sequence & 1
    input_ = torch.full(
        (1, 6144),
        rank + 1 + parity,
        dtype=torch.bfloat16,
        device="cuda",
    )
    output = session.all_reduce(input_)
    expected = 10 + 4 * parity
    if not bool(torch.all(output == expected).item()):
        raise RuntimeError(
            f"rank {rank} mismatch at sequence {sequence}"
        )

print(f"rank={rank} ctypes_tp4_correct=true iterations={iterations}")
