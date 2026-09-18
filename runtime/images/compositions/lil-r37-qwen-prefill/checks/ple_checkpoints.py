"""Focused R37 compatibility coverage for B12X PR386 checkpoint export.

Run from the backport source root: python -m pytest -q tests/sequence/test_ple.py
The high-slot cases allocate a mostly untouched pool slightly larger than 4 GiB.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

import b12x
from b12x.sequence import ple
from b12x.sequence.ple.reference import ple_projected_u_reference


def _binding(device, *, pool=None, order=(0, 1, 2, 3, 4, 5)):
    # History prefill, decode, fresh prefill, empty, disabled, inactive.
    lengths = [16, 2, 4, 0, 5, 0]
    starts = [0]
    for row in order:
        starts.append(starts[-1] + lengths[row])
    streams, hidden, length, speculative = 4, 2560, 9, 3
    channels = streams * hidden
    generator = torch.Generator(device=device).manual_seed(1626)

    def randn(*shape):
        return torch.randn(
            shape, dtype=torch.bfloat16, device=device, generator=generator
        )

    if pool is None:
        pool = randn(16, channels, length + speculative)
    caps = ple.Caps(
        device=device,
        mode="mixed",
        max_tokens=starts[-1],
        max_seqs=len(order),
        max_state_slots=pool.shape[0],
        max_speculative_tokens=speculative,
        streams=streams,
        hidden_size=hidden,
        kernel_size=4,
        dilation=3,
    )
    plan = ple.plan(caps)
    (spec,) = plan.scratch_specs()
    weights = [randn(channels) / 32 for _ in range(3)]
    binding = ple.bind(
        plan,
        scratch=torch.empty(spec.shape, dtype=spec.dtype, device=device),
        residual=randn(starts[-1], streams, hidden),
        key=randn(starts[-1], streams, hidden),
        value=randn(starts[-1], hidden),
        k_norm_weight=weights[0],
        q_norm_weight=weights[1],
        u_norm_weight=weights[2],
        conv_weight=randn(channels, 4),
        query_start_loc=torch.tensor(starts, dtype=torch.int32, device=device),
        state_slot_ids=torch.tensor(
            [[1, 2, 3, 4, 5, -1][i] for i in order], dtype=torch.int64, device=device
        ),
        state_is_fresh=torch.tensor(
            [[False, False, True, False, False, True][i] for i in order], device=device
        ),
        num_accepted_tokens=torch.ones(len(order), dtype=torch.int32, device=device),
        num_seqs=torch.tensor([5], dtype=torch.int32, device=device),
        num_tokens=torch.tensor([starts[-1]], dtype=torch.int32, device=device),
        request_is_prefill=torch.tensor(
            [[True, False, True, True, True, True][i] for i in order], device=device
        ),
        conv_state=pool,
        out=torch.full(
            (starts[-1], streams, hidden), 91, dtype=torch.bfloat16, device=device
        ),
    )
    return binding, starts


@pytest.mark.parametrize(
    "field,kind",
    [
        ("offsets", "shape"),
        ("offsets", "dtype"),
        ("offsets", "stride"),
        ("offsets", "device"),
        ("slots", "shape"),
        ("slots", "dtype"),
        ("slots", "stride"),
        ("slots", "device"),
    ],
)
def test_checkpoint_metadata_validation(field, kind):
    binding, _ = _binding(torch.device("cpu"))
    tensors = dict(
        offsets=torch.zeros(6, dtype=torch.int32),
        slots=torch.zeros(6, dtype=torch.int64),
    )
    if kind == "shape":
        tensors[field] = tensors[field][:-1]
    elif kind == "dtype":
        tensors[field] = tensors[field].float()
    elif kind == "device":
        tensors[field] = tensors[field].to("meta")
    else:
        tensors[field] = torch.zeros(12, dtype=tensors[field].dtype)[::2]
    with pytest.raises((TypeError, ValueError), match=f"checkpoint {field}"):
        ple.export_checkpoint(binding, **tensors)


@pytest.mark.parametrize("mode", ["prefill", "decode"])
def test_checkpoint_requires_mixed_plan(mode):
    binding, _ = _binding(torch.device("cpu"))
    binding = replace(
        binding, plan=replace(binding.plan, caps=replace(binding.plan.caps, mode=mode))
    )
    with pytest.raises(ValueError, match="mixed LayerPlan"):
        ple.export_checkpoint(
            binding,
            offsets=torch.zeros(6, dtype=torch.int32),
            slots=torch.zeros(6, dtype=torch.int64),
        )


@pytest.mark.parametrize("high_slot", [False, True])
@pytest.mark.parametrize("order", [(0, 1, 2, 3, 4, 5), (1, 4, 0, 3, 2, 5)])
@torch.inference_mode()
def test_internal_checkpoint_mixed_order_graph_replay(monkeypatch, high_slot, order):
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")
    pytest.importorskip("triton")
    from b12x.sequence.ple import _kernels

    device = torch.device("cuda", torch.cuda.current_device())
    channels, length, capacity = 4 * 2560, 9, 12
    # Padding exercises the installed hybrid-cache state-slot ABI too.
    state_stride = channels * capacity + 256
    destination = (2**31 // state_stride + 2) if high_slot else 8
    if high_slot:
        assert destination * state_stride > 2**31
    pool = torch.empty_strided(
        (destination + 8, channels, capacity),
        (state_stride, capacity, 1),
        dtype=torch.bfloat16,
        device=device,
    )
    pool[:6].normal_()
    pool[destination:].fill_(91)
    original = pool[:6].clone()
    binding, starts = _binding(device, pool=pool, order=order)
    # This eager mixed call must prepare export even when no export is requested.
    ple.run_mixed(binding, eps=1e-6)
    assert binding.error_code.item() == 0
    live_after = pool[:6].clone()
    _, reference_u = ple_projected_u_reference(
        binding.residual,
        binding.key,
        binding.value,
        k_norm_weight=binding.k_norm_weight,
        q_norm_weight=binding.q_norm_weight,
        u_norm_weight=binding.u_norm_weight,
        eps=1e-6,
    )
    torch.testing.assert_close(
        binding.normalized_u, reference_u, rtol=0.02, atol=0.0078125
    )
    histories = {}
    for kind in (0, 2):
        row = order.index(kind)
        prior = (
            original[kind + 1, :, :length]
            if kind == 0
            else torch.zeros_like(original[3, :, :length])
        )
        histories[kind] = torch.cat(
            (
                prior,
                binding.normalized_u[starts[row] : starts[row + 1]].T,
            ),
            dim=1,
        )
    # Deliberately use minimally aligned contiguous metadata slices.
    offsets = torch.tensor(
        [99] + [[2, 1, 1, 1, 2, 1][i] for i in order], dtype=torch.int32, device=device
    )[1:]
    slots = torch.tensor(
        [99] + [destination + i if i != 4 else -1 for i in order],
        dtype=torch.int64,
        device=device,
    )[1:]

    def forbid_resolution(*args, **kwargs):
        raise AssertionError("checkpoint JIT resolution occurred after mixed warmup")

    monkeypatch.setattr(_kernels._export_checkpoint_kernel, "warmup", forbid_resolution)
    was_frozen = b12x.kernel_resolution_frozen()
    b12x.freeze_kernel_resolution("PLE checkpoint graph qualification")
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            ple.export_checkpoint(binding, offsets=offsets, slots=slots)
        cache_misses = _kernels._checkpoint_program.cache_info().misses
        for offset, shift, live in (
            (2, 0, 5),
            (12, 6, 5),
            (0, 0, 5),
            (16, 0, 5),
            (-1, 0, 5),
            (17, 0, 5),
            (2, 0, 0),
        ):
            offsets[order.index(0)] = offset
            slots[order.index(0)] = destination + shift
            binding.num_seqs.fill_(live)
            pool[destination:].fill_(91)
            allocated = torch.cuda.memory_allocated(device)
            graph.replay()
            torch.cuda.synchronize(device)
            assert torch.cuda.memory_allocated(device) == allocated
            assert _kernels._checkpoint_program.cache_info().misses == cache_misses
            expected = torch.full_like(pool[destination:], 91)
            if live:
                expected[2, :, :length] = histories[2][:, 1 : 1 + length]
                expected[2, :, length:] = 0
                if 0 < offset < 16:
                    expected[shift, :, :length] = histories[0][
                        :, offset : offset + length
                    ]
                    expected[shift, :, length:] = 0
            torch.testing.assert_close(pool[destination:], expected, rtol=0, atol=0)
            torch.testing.assert_close(pool[:6], live_after, rtol=0, atol=0)
            torch.testing.assert_close(pool[0], original[0], rtol=0, atol=0)
        # R37 invalid-metadata state must also suppress stale-scratch export.
        binding.num_seqs.fill_(5)
        binding.error_code.fill_(1)
        pool[destination:].fill_(91)
        graph.replay()
        torch.cuda.synchronize(device)
        assert bool((pool[destination:] == 91).all())
    finally:
        if not was_frozen:
            b12x.unfreeze_kernel_resolution()
