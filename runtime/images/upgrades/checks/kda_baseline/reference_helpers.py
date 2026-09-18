"""Oracle and contract tests for chunked lower-bounded KDA prefill.

The CPU tests pin the reference algebra and the mirror's rounding policy; the
GPU tests (added with the kernels) compare the CuTe DSL op against them.
"""

from __future__ import annotations

import math

import pytest
import torch

from b12x.sequence._shared.kda_math import kda_beta, kda_log_decay, l2_normalize
from b12x.sequence.kda_prefill.reference import (
    MirrorPolicy,
    prefill_kda,
    prefill_kda_chunk_mirror,
    recurrent_kda,
)

HEAD_DIM = 128
CPU = torch.device("cpu")
PURE_FP32 = MirrorPolicy(
    shadow=False, inv_operand="fp32", u_operand="fp32", operands="fp32"
)
FLASHKDA_LIKE = MirrorPolicy(
    state_master="bf16", single_rounding=False, scale_dtype="bf16"
)


def rmse_ratio(reference: torch.Tensor, actual: torch.Tensor) -> float:
    delta = (reference.float() - actual.float()).flatten()
    base = reference.float().flatten()
    return (delta.square().mean().sqrt() / (base.square().mean().sqrt() + 1e-8)).item()


def assert_kda_close(
    name: str,
    reference: torch.Tensor,
    actual: torch.Tensor,
    *,
    ratio: float,
    peak_ratio: float = 4e-2,
    exact_atol: float = 1e-6,
) -> None:
    assert torch.isfinite(actual.float()).all(), f"{name}: non-finite values"
    delta = (reference.float() - actual.float()).abs()
    if delta.max().item() <= exact_atol:
        return
    observed = rmse_ratio(reference, actual)
    assert observed < ratio, f"{name}: rmse ratio {observed:.3e} >= {ratio}"
    rms = reference.float().square().mean().sqrt().item()
    peak = reference.float().abs().max().item()
    assert delta.max().item() <= peak_ratio * rms + 2**-6 * peak, f"{name}: peak error"


def make_inputs(
    *,
    lengths: list[int],
    heads: int = 2,
    device: torch.device = CPU,
    seed: int = 0,
    gate_profile: str = "random",
    key_profile: str = "random",
    lower_bound: float = -5.0,
    token_capacity: int | None = None,
    state_slots: int | None = None,
    initial: list[int] | None = None,
    final: list[int] | None = None,
    checkpoint: list[tuple[int, int]] | None = None,
    null_state_index: int | None = None,
) -> dict:
    """Build packed inputs; slot assignment defaults to distinct slots."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    tokens = sum(lengths)
    count = len(lengths)
    capacity = tokens if token_capacity is None else token_capacity

    def bf16(*shape, scale=0.25):
        return (torch.randn(*shape, generator=generator) * scale).to(torch.bfloat16)

    q, k, v = (
        bf16(capacity, heads, HEAD_DIM),
        bf16(capacity, heads, HEAD_DIM),
        bf16(capacity, heads, HEAD_DIM),
    )
    raw_g = bf16(capacity, heads, HEAD_DIM, scale=1.0)
    raw_beta = bf16(capacity, heads, scale=1.0)
    if gate_profile == "long_memory":
        raw_g[:, :, :32] = -12.0
    elif gate_profile == "saturated":
        raw_g.fill_(12.0)
    elif gate_profile == "zero":
        raw_g.fill_(-12.0)
    if key_profile in ("repeated", "alternating"):
        unit = torch.randn(heads, HEAD_DIM, generator=generator)
        unit = unit / unit.norm(dim=-1, keepdim=True)
        pattern = unit[None].expand(capacity, heads, HEAD_DIM).clone()
        if key_profile == "alternating":
            pattern[1::2] *= -1.0
        k = pattern.to(torch.bfloat16)
        raw_beta.fill_(12.0)
    A_log = torch.randn(heads, generator=generator) * 0.1
    dt_bias = torch.randn(heads, HEAD_DIM, generator=generator) * 0.1
    slots = 3 * count + 2 if state_slots is None else state_slots
    pool = torch.randn(slots, heads, HEAD_DIM, HEAD_DIM, generator=generator) * 0.1
    initial = list(range(count)) if initial is None else initial
    final = list(range(count, 2 * count)) if final is None else final
    checkpoint = [(0, 0)] * count if checkpoint is None else checkpoint
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    to = lambda t: t.to(device)  # noqa: E731
    return {
        "q": to(q),
        "k": to(k),
        "v": to(v),
        "raw_g": to(raw_g),
        "raw_beta": to(raw_beta),
        "A_log": to(A_log),
        "dt_bias": to(dt_bias),
        "pool": to(pool),
        "cu_seqlens": torch.tensor(cu, dtype=torch.int32, device=device),
        "initial": torch.tensor(initial, dtype=torch.int32, device=device),
        "final": torch.tensor(final, dtype=torch.int32, device=device),
        "checkpoint_slots": torch.tensor(
            [c[1] for c in checkpoint], dtype=torch.int32, device=device
        ),
        "checkpoint_offsets": torch.tensor(
            [c[0] for c in checkpoint], dtype=torch.int32, device=device
        ),
        "num_seqs": count,
        "num_tokens": tokens,
        "lower_bound": lower_bound,
        "null_state_index": null_state_index,
    }


def run_oracle(inputs: dict, fn=prefill_kda, **extra):
    pool = inputs["pool"].clone()
    output = fn(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["raw_g"],
        inputs["raw_beta"],
        inputs["A_log"],
        inputs["dt_bias"],
        pool,
        inputs["cu_seqlens"],
        inputs["initial"],
        inputs["final"],
        inputs["checkpoint_slots"],
        inputs["checkpoint_offsets"],
        inputs["num_seqs"],
        inputs["num_tokens"],
        lower_bound=inputs["lower_bound"],
        null_state_index=inputs["null_state_index"],
        **extra,
    )
    return output, pool
