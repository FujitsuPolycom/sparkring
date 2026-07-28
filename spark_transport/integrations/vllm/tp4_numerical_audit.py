"""Compare direct-cable TP4 and NCCL BF16 sums against FP32 ground truth."""

from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist

from spark_tp4_backend import _NativeSession


ELEMENTS = 6144
WORLD_SIZE = 4


def make_rank_input(sequence: int, rank: int) -> torch.Tensor:
    """Generate broad-scale BF16 inputs, including cancellation-heavy cases."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0x5A17 + sequence * WORLD_SIZE + rank)
    independent = torch.randn(ELEMENTS, generator=generator)

    indices = torch.arange(ELEMENTS)
    exponents = ((indices + sequence) % 12) - 6
    scale = torch.pow(2.0, exponents)

    if sequence & 1:
        shared_generator = torch.Generator(device="cpu")
        shared_generator.manual_seed(0xC011A + sequence)
        shared = torch.randn(ELEMENTS, generator=shared_generator) * scale
        coefficient = (1.0, -1.0, 0.5, -0.5)[rank]
        value = shared * coefficient + independent * scale * 0.001
    else:
        value = independent * scale
    return value.to(torch.bfloat16)


def main() -> None:
    rank = int(os.environ["RANK"])
    iterations = int(os.getenv("ITERATIONS", "1000"))
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=WORLD_SIZE,
    )
    session = _NativeSession(rank)

    compared = 0
    candidate_absolute_sum = 0.0
    reference_absolute_sum = 0.0
    candidate_squared_sum = 0.0
    reference_squared_sum = 0.0
    candidate_maximum = 0.0
    reference_maximum = 0.0
    candidate_closer = 0
    reference_closer = 0
    tied = 0
    candidate_exact = 0
    reference_exact = 0
    candidate_reference_mismatches = 0

    for sequence in range(iterations):
        cpu_inputs = [
            make_rank_input(sequence, source_rank)
            for source_rank in range(WORLD_SIZE)
        ]
        local = cpu_inputs[rank].to(device=device)
        truth = (
            torch.stack([tensor.float() for tensor in cpu_inputs])
            .sum(dim=0)
            .to(device=device)
        )

        candidate = session.all_reduce(local)
        reference = local.clone()
        dist.all_reduce(reference, op=dist.ReduceOp.SUM)

        candidate_error = (candidate.float() - truth).abs()
        reference_error = (reference.float() - truth).abs()
        candidate_error_squared = candidate_error.square()
        reference_error_squared = reference_error.square()

        compared += ELEMENTS
        candidate_absolute_sum += float(candidate_error.sum().item())
        reference_absolute_sum += float(reference_error.sum().item())
        candidate_squared_sum += float(candidate_error_squared.sum().item())
        reference_squared_sum += float(reference_error_squared.sum().item())
        candidate_maximum = max(
            candidate_maximum, float(candidate_error.max().item())
        )
        reference_maximum = max(
            reference_maximum, float(reference_error.max().item())
        )
        candidate_closer += int(
            torch.count_nonzero(candidate_error < reference_error).item()
        )
        reference_closer += int(
            torch.count_nonzero(reference_error < candidate_error).item()
        )
        tied += int(
            torch.count_nonzero(candidate_error == reference_error).item()
        )
        rounded_truth = truth.to(torch.bfloat16)
        candidate_exact += int(
            torch.count_nonzero(candidate == rounded_truth).item()
        )
        reference_exact += int(
            torch.count_nonzero(reference == rounded_truth).item()
        )
        candidate_reference_mismatches += int(
            torch.count_nonzero(
                candidate.view(torch.int16) != reference.view(torch.int16)
            ).item()
        )

    dist.barrier()
    if rank == 0:
        print(
            "TP4_NUMERICAL "
            + json.dumps(
                {
                    "iterations": iterations,
                    "elements": compared,
                    "candidate_mae": candidate_absolute_sum / compared,
                    "nccl_mae": reference_absolute_sum / compared,
                    "candidate_rmse": (
                        candidate_squared_sum / compared
                    )
                    ** 0.5,
                    "nccl_rmse": (
                        reference_squared_sum / compared
                    )
                    ** 0.5,
                    "candidate_max_abs": candidate_maximum,
                    "nccl_max_abs": reference_maximum,
                    "candidate_closer": candidate_closer,
                    "nccl_closer": reference_closer,
                    "tied": tied,
                    "candidate_exact_rounded_fp32": candidate_exact,
                    "nccl_exact_rounded_fp32": reference_exact,
                    "candidate_nccl_bit_mismatches": (
                        candidate_reference_mismatches
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
