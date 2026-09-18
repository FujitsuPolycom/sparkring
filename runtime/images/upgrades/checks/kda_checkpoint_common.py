"""Shared CPU checkpoint semantics executed against each selected B12X source.

The assertions are retained from the protected policy-based baseline. They cover
metadata rejection and numerical checkpoint states, not CUDA kernel execution.
"""

from pathlib import Path
import os
import pytest
import torch
import b12x
from b12x.sequence.kda_prefill.metadata import validate_metadata
from b12x.sequence.kda_prefill.reference import prefill_kda_chunk_mirror, recurrent_kda
from kda_baseline.reference_helpers import PURE_FP32, make_inputs, run_oracle

_SOURCE = Path(os.environ["SPARKRING_B12X_SOURCE_ROOT"]).resolve()
assert Path(b12x.__file__).resolve().is_relative_to(_SOURCE), (
    "B12X import must use the selected source"
)


def metadata() -> dict:
    return dict(
        cu_seqlens=[0, 64, 160],
        initial_state_indices=[1, 5],
        final_state_indices=[2, 6],
        checkpoint_state_indices=[[3, 4], [7, 8]],
        checkpoint_offsets=[[16, 48], [32, 80]],
        num_seqs=2,
        num_tokens=160,
        token_capacity=192,
        seq_capacity=2,
        state_slots=10,
        null_state_index=0,
        max_checkpoints=2,
    )


@pytest.mark.parametrize("slot", [1, 2, 3, 5, 6, 7, 8])
def test_checkpoint_destinations_cannot_alias_other_owners(slot):
    args = metadata()
    args["checkpoint_state_indices"][0][1] = slot
    with pytest.raises(ValueError):
        validate_metadata(**args)


@pytest.mark.parametrize("offset", [17, 80, 16])
def test_second_checkpoint_rejects_unaligned_past_end_or_duplicate_offset(offset):
    args = metadata()
    args["checkpoint_offsets"][0][1] = offset
    with pytest.raises(ValueError):
        validate_metadata(**args)


@pytest.mark.parametrize("slot", [-1, 10])
def test_second_checkpoint_rejects_invalid_active_slot(slot):
    args = metadata()
    args["checkpoint_state_indices"][0][1] = slot
    with pytest.raises(IndexError):
        validate_metadata(**args)


def test_null_disabled_reversed_and_final_boundary_exports():
    args = metadata()
    args["final_state_indices"][0] = 1  # Own initial/final alias is legal.
    args["checkpoint_offsets"] = [[64, 16], [96, 32]]
    assert validate_metadata(**args) == [(0, 64), (64, 160)]
    args["checkpoint_state_indices"][0][1] = 0
    args["checkpoint_offsets"][0][1] = 64  # A null writer does not own an offset.
    validate_metadata(**args)
    args["checkpoint_offsets"][0][1] = 17
    with pytest.raises(ValueError, match="unaligned"):
        validate_metadata(**args)
    args["checkpoint_state_indices"][0][1] = 1
    for offset in (0, -1):
        args["checkpoint_offsets"][0][1] = offset
        validate_metadata(**args)


def test_inactive_metadata_is_ignored_but_live_counts_are_bounded():
    args = metadata()
    args.update(num_seqs=1, num_tokens=64)
    args["checkpoint_state_indices"][1] = [-999, -999]
    args["checkpoint_offsets"][1] = [17, 999]
    assert validate_metadata(**args) == [(0, 64)]
    args["num_seqs"] = 3
    with pytest.raises(ValueError, match="capacities"):
        validate_metadata(**args)


@pytest.mark.parametrize("inplace", [False, True])
def test_two_exports_equal_independent_prefix_recurrences_and_final_state(inplace):
    inputs = make_inputs(lengths=[64], heads=1, seed=830, state_slots=6)
    if inplace:
        inputs["final"][0] = inputs["initial"][0]
    one_checkpoint_output, one_checkpoint_pool = run_oracle(inputs)
    inputs["checkpoint_slots"] = torch.tensor([[3, 4]], dtype=torch.int32)
    inputs["checkpoint_offsets"] = torch.tensor([[48, 16]], dtype=torch.int32)
    output, pool = run_oracle(inputs, max_checkpoints=2)
    torch.testing.assert_close(output, one_checkpoint_output, rtol=0, atol=0)
    torch.testing.assert_close(
        pool[int(inputs["final"][0])],
        one_checkpoint_pool[int(inputs["final"][0])],
        rtol=0,
        atol=0,
    )
    for slot, offset in ((3, 48), (4, 16)):
        _, expected, _ = recurrent_kda(
            *(inputs[name][:offset] for name in ("q", "k", "v", "raw_g", "raw_beta")),
            inputs["A_log"],
            inputs["dt_bias"],
            lower_bound=-5.0,
            initial_state=inputs["pool"][int(inputs["initial"][0])],
        )
        torch.testing.assert_close(pool[slot], expected, rtol=0, atol=0)
    _, mirror_pool = run_oracle(
        inputs,
        fn=prefill_kda_chunk_mirror,
        max_checkpoints=2,
        policy=PURE_FP32,
    )
    for slot in (int(inputs["final"][0]), 3, 4):
        torch.testing.assert_close(mirror_pool[slot], pool[slot], rtol=2e-4, atol=2e-5)


def test_reference_rejection_is_transactional_for_pool_and_output():
    inputs = make_inputs(lengths=[64], heads=1, seed=831, state_slots=6)
    inputs["checkpoint_slots"] = torch.tensor([[3, 1]], dtype=torch.int32)
    inputs["checkpoint_offsets"] = torch.tensor([[16, 48]], dtype=torch.int32)
    before = inputs["pool"].clone()
    output = torch.full_like(inputs["q"], 7)
    from b12x.sequence.kda_prefill.reference import prefill_kda

    with pytest.raises(ValueError, match="duplicate"):
        prefill_kda(
            *(inputs[name] for name in ("q", "k", "v", "raw_g", "raw_beta")),
            inputs["A_log"],
            inputs["dt_bias"],
            inputs["pool"],
            inputs["cu_seqlens"],
            inputs["initial"],
            inputs["final"],
            inputs["checkpoint_slots"],
            inputs["checkpoint_offsets"],
            1,
            64,
            max_checkpoints=2,
            output=output,
        )
    assert torch.equal(inputs["pool"], before)
    assert torch.all(output == 7)


def test_plural_reference_zero_offset_returns_an_independent_initial_snapshot():
    inputs = make_inputs(lengths=[32], heads=1, seed=832)
    initial = inputs["pool"][0]
    _, _, snapshots = recurrent_kda(
        *(inputs[name] for name in ("q", "k", "v", "raw_g", "raw_beta")),
        inputs["A_log"],
        inputs["dt_bias"],
        lower_bound=-5.0,
        initial_state=initial,
        checkpoint_offsets=(0, 16),
    )
    assert isinstance(snapshots, dict) and set(snapshots) == {0, 16}
    assert torch.equal(snapshots[0], initial)
    assert snapshots[0].data_ptr() != initial.data_ptr()


@pytest.mark.parametrize("inplace", [False, True])
def test_four_exports_equal_independent_prefix_recurrences(inplace):
    inputs = make_inputs(lengths=[80], heads=1, seed=842, state_slots=8)
    if inplace:
        inputs["final"][0] = inputs["initial"][0]
    baseline_output, baseline_pool = run_oracle(inputs)
    inputs["checkpoint_slots"] = torch.tensor([[2, 3, 4, 5]], dtype=torch.int32)
    inputs["checkpoint_offsets"] = torch.tensor([[64, 16, 48, 32]], dtype=torch.int32)
    output, pool = run_oracle(inputs, max_checkpoints=4)
    torch.testing.assert_close(output, baseline_output, rtol=0, atol=0)
    torch.testing.assert_close(
        pool[int(inputs["final"][0])],
        baseline_pool[int(inputs["final"][0])],
        rtol=0,
        atol=0,
    )
    for slot, offset in zip((2, 3, 4, 5), (64, 16, 48, 32), strict=True):
        _, expected, _ = recurrent_kda(
            *(inputs[name][:offset] for name in ("q", "k", "v", "raw_g", "raw_beta")),
            inputs["A_log"],
            inputs["dt_bias"],
            lower_bound=-5.0,
            initial_state=inputs["pool"][int(inputs["initial"][0])],
        )
        torch.testing.assert_close(pool[slot], expected, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("fault", ["offset", "slot", "initial"])
def test_four_checkpoint_nonadjacent_aliases_are_rejected(fault):
    args = metadata()
    args.update(
        cu_seqlens=[0, 80],
        initial_state_indices=[0],
        final_state_indices=[1],
        checkpoint_state_indices=[[2, 3, 4, 5]],
        checkpoint_offsets=[[16, 32, 48, 64]],
        num_seqs=1,
        num_tokens=80,
        null_state_index=None,
        max_checkpoints=4,
    )
    if fault == "offset":
        args["checkpoint_offsets"][0][3] = 16
    elif fault == "slot":
        args["checkpoint_state_indices"][0][3] = 2
    else:
        args["checkpoint_state_indices"][0][3] = 0
    with pytest.raises(ValueError):
        validate_metadata(**args)
