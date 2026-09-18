"""Bounded GB10 qualification of the explicit reconciled B12X source.

Run on an idle GPU in the candidate image. This file does not modify serving
containers, source, driver configuration, or model files.
"""

from contextlib import ExitStack
from pathlib import Path
import gc
import os
import sys

SOURCE = Path(os.environ["B12X_GATE_SOURCE_ROOT"]).resolve()
sys.path.insert(0, str(SOURCE))

# The explicitly selected source must precede imports of B12X and its dependencies.
import pytest  # noqa: E402
import torch  # noqa: E402
import b12x  # noqa: E402
from b12x._lib.runtime_control import kernel_resolution_guard  # noqa: E402

assert Path(b12x.__file__).resolve().is_relative_to(SOURCE), (
    "wrong B12X source imported"
)

# Image qualification imports the installed package first, then exposes only
# the external reference-test namespace. Its production package path is fixed.
if os.environ.get("B12X_GATE_TEST_ROOT"):
    test_root = Path(os.environ["B12X_GATE_TEST_ROOT"]).resolve()
    import types

    tests_package = types.ModuleType("tests")
    tests_package.__path__ = [str(test_root / "tests")]
    sys.modules["tests"] = tests_package


def gpu():
    assert torch.cuda.is_available(), "GPU gate requires CUDA; CPU is not a pass"
    device = torch.device("cuda", 0)
    assert torch.cuda.get_device_capability(device) == (12, 1), "requires GB10/SM121"
    return device


@pytest.fixture(autouse=True)
def scopes(monkeypatch):
    from tests.sequence import test_kda_prefill as kda_cases
    from tests.sequence import test_kda_prefill_two_checkpoints_gpu as checkpoints
    from tests.sequence import test_ple as ple_cases
    from tests.attention import test_qsa_contract as qsa_cases

    monkeypatch.setattr(checkpoints, "require_gb10", gpu)
    with ExitStack() as stack:
        qsa_token = qsa_cases._test_resources.set(stack)
        ple_token = ple_cases._case_resources.set(stack)
        try:
            yield
        finally:
            qsa_cases._test_resources.reset(qsa_token)
            ple_cases._case_resources.reset(ple_token)
            gc.collect()
            while kda_cases._PREPARATIONS:
                result, session = kda_cases._PREPARATIONS.pop()
                result.close()
                session.close()


@pytest.mark.parametrize("heads", [1, 16, 32])
@pytest.mark.parametrize("capacity", [1, 2, 4])
def test_kda_checkpoint_numerical(heads, capacity):
    from tests.sequence.test_kda_prefill import (
        make_inputs,
        make_binding,
        run_oracle,
        assert_kda_close,
    )
    from tests.sequence.test_kda_prefill_two_checkpoints_gpu import (
        checkpoint_inputs,
        assert_checkpoint_oracle,
        run_op,
    )

    if capacity > 1:
        positions = [16, 48] if capacity == 2 else [16, 32, 48, 64]
        inputs = checkpoint_inputs(
            [80],
            heads=heads,
            capacity=capacity,
            offsets=[positions],
            state_slots=16,
            device=gpu(),
        )
    else:
        inputs = make_inputs(
            lengths=[80],
            heads=heads,
            state_slots=16,
            checkpoint=[(32, 3)],
            device=gpu(),
        )
    binding, tensors = make_binding(
        inputs,
        max_tokens=128,
        max_seqs=2,
        checkpoint_export=True,
        max_checkpoints=capacity,
    )
    run_op(binding, inputs)
    torch.cuda.synchronize()
    if capacity > 1:
        assert_checkpoint_oracle(binding, tensors, inputs)
    else:
        output, pool = run_oracle(inputs)
        assert binding.error_code.item() == 0
        assert_kda_close("output", output[:80], binding.output[:80], ratio=1e-2)
        for slot in (1, 3):
            assert_kda_close(
                "checkpoint", pool[slot], tensors["recurrent_state"][slot], ratio=5e-3
            )


@pytest.mark.parametrize("capacity", [2, 4])
def test_kda_frozen_dynamic_graph(capacity):
    from tests.sequence import test_kda_prefill_two_checkpoints_gpu as cases
    from b12x.sequence._shared.delta_prefill import _cute_kernels as kernels
    from b12x._lib.runtime_control import (
        KernelResolutionFrozenError,
        kernel_resolution_frozen,
        raise_if_kernel_resolution_frozen,
    )

    device = gpu()
    options = dict(capacity=capacity, state_slots=32 if capacity == 4 else 16)
    initial_offsets = [[16, 32, 64, 80]] if capacity == 4 else [[16, 96]]
    first = cases.checkpoint_inputs(
        [96], offsets=initial_offsets, device=device, **options
    )
    binding, tensors = cases.make_binding(
        first,
        max_tokens=256,
        max_seqs=4,
        checkpoint_export=True,
        max_checkpoints=capacity,
    )
    cases.run_op(binding, first)
    torch.cuda.synchronize(device)
    launchers = (
        kernels._PROLOGUE_CACHE[kernels._prologue_key(binding)],
        kernels._PREPARE_CACHE[kernels._prepare_key(binding)],
        kernels._RECURRENCE_CACHE[kernels._recurrence_key(binding)],
    )
    addresses = tuple(t.data_ptr() for t in (*tensors.values(), binding.scratch))
    variations = (
        (
            ([128], [[112, 32, 64, 16]]),
            ([64, 80, 96], [[16, 32, 48, 64], [64, 32, 16, 80], [32, 80, 16, 64]]),
            ([48], [[32, 0, 16, 0]]),
        )
        if capacity == 4
        else (
            ([128], [[112, 32]]),
            ([64, 80, 96], [[16, 48], [64, 32], [32, 80]]),
            ([48], [[32, 0]]),
        )
    )
    graph = torch.cuda.CUDAGraph()
    prior_frozen = kernel_resolution_frozen()
    try:
        with kernel_resolution_guard(f"{capacity}-checkpoint GB10 qualification"):
            assert kernel_resolution_frozen()
            with pytest.raises(KernelResolutionFrozenError):
                raise_if_kernel_resolution_frozen("qualification guard probe")
            with torch.cuda.graph(graph):
                cases.run_op(binding, first)
            for lengths, offsets in variations:
                live = cases.checkpoint_inputs(
                    lengths,
                    offsets=offsets,
                    seed=sum(lengths),
                    device=device,
                    **options,
                )
                cases.copy_live(tensors, live)
                binding.output.fill_(float("nan"))
                binding.scratch.fill_(0xFF)
                torch.cuda.synchronize(device)
                before = torch.cuda.memory_stats(device)["allocation.all.allocated"]
                assert kernel_resolution_frozen()
                graph.replay()
                torch.cuda.synchronize(device)
                assert (
                    torch.cuda.memory_stats(device)["allocation.all.allocated"]
                    == before
                )
                assert addresses == tuple(
                    t.data_ptr() for t in (*tensors.values(), binding.scratch)
                )
                cases.assert_checkpoint_oracle(binding, tensors, live)
                assert torch.isnan(binding.output[live["num_tokens"] :].float()).all()
    finally:
        graph.reset()
        assert kernel_resolution_frozen() == prior_frozen
    assert launchers == (
        kernels._PROLOGUE_CACHE[kernels._prologue_key(binding)],
        kernels._PREPARE_CACHE[kernels._prepare_key(binding)],
        kernels._RECURRENCE_CACHE[kernels._recurrence_key(binding)],
    )


@pytest.mark.parametrize(
    "fault",
    [
        "duplicate-slot",
        "initial-alias",
        "duplicate-offset",
        "unaligned",
        "out-of-range",
    ],
)
def test_kda_invalid_metadata_is_transactional(fault):
    from tests.sequence.test_kda_prefill_two_checkpoints_gpu import (
        test_two_checkpoint_gpu_invalid_metadata_preserves_state,
    )

    test_two_checkpoint_gpu_invalid_metadata_preserves_state(fault)


@pytest.mark.large_pool
def test_kda_high_pool_offset(monkeypatch):
    from tests.sequence.test_kda_prefill_two_checkpoints_gpu import (
        test_two_checkpoint_gpu_high_pool_offsets,
    )

    # The inherited helper temporarily clones the pool during preparation.
    free, _ = torch.cuda.mem_get_info(gpu())
    assert free >= 32 << 30, "large-pool gate needs at least 32 GiB free"
    monkeypatch.setenv("B12X_RUN_LARGE_POOL_TESTS", "1")
    test_two_checkpoint_gpu_high_pool_offsets()


def test_qsa_prepared_full_transaction_reference_and_graph(monkeypatch):
    from tests.attention import test_qsa_contract as cases
    from b12x.attention import qsa
    from b12x.preparation import PreparationSession, PreparedCall

    original_dynamic = cases._dynamic_inputs

    def dynamic_with_declared_rope_layout(binding, **options):
        # The test declares column-major mRoPE metadata. Its one-row priming
        # input must preserve the declared axis stride, not become contiguous.
        dynamic = original_dynamic(binding, **options)
        descriptor = binding.state.abi["operands"]["rope_positions"]
        value = dynamic["rope_positions"]
        prepared_value = torch.empty_strided(
            value.shape,
            tuple(descriptor["strides"]),
            dtype=getattr(torch, descriptor["dtype"]),
            device=value.device,
        )
        prepared_value.copy_(value)
        dynamic["rope_positions"] = prepared_value
        return dynamic

    def reprepare_rope_cache(binding, **changes):
        # BF16 interleaved RoPE caches have a different static native ABI from
        # the fixture's FP32 contiguous caches. Rebinding alone is not legal.
        assert set(changes) == {"rope_cos", "rope_sin"}
        caps = binding.state.caps
        names = (
            "main_k_cache",
            "main_v_cache",
            "k_descale",
            "v_descale",
            "main_block_table",
            "compressed_k_cache",
            "compressed_block_table",
            "raw_k_ring",
            "raw_logical_positions",
            "raw_rope_positions",
            "raw_interval_start_positions",
            "raw_state_slot_ids",
            "index_q_norm_weight",
            "index_k_norm_weight",
            "rope_cos",
            "rope_sin",
            "output",
            "selected_positions",
        )
        operands = {name: getattr(binding, name) for name in names}
        operands.update(changes)
        abi = dict(binding.state.abi["operands"])
        for name, tensor in changes.items():
            abi[name] = dict(
                dtype=str(tensor.dtype).removeprefix("torch."),
                strides=tuple(tensor.stride()),
            )
        declaration = qsa.plan(
            caps,
            invocation=qsa.invocation_from_descriptors(caps, operands=abi),
        )

        def prepare_call(state):
            scratch = torch.empty_like(binding.scratch)
            trial = state.bind_for_preparation(scratch=scratch, **operands)
            mutable = (
                trial.compressed_k_cache,
                trial.raw_k_ring,
                trial.raw_logical_positions,
                trial.raw_rope_positions,
                trial.raw_interval_start_positions,
                trial.output,
                trial.selected_positions,
            )
            saved = tuple(tensor.clone() for tensor in mutable)
            dynamic = dynamic_with_declared_rope_layout(
                trial,
                positions=(0,),
                request_ids=(0,),
            )

            def restore():
                for tensor, snapshot in zip(mutable, saved, strict=True):
                    tensor.copy_(snapshot)

            return PreparedCall(
                run=lambda: state.run_for_preparation(trial, **dynamic),
                reset=restore,
                restore=restore,
                owners=(scratch, trial, saved, dynamic),
            )

        resources = cases._resources()
        session = resources.enter_context(
            PreparationSession(device=caps.device, autotune=False)
        )
        result = session.prepare(
            (
                declaration.request(
                    name="qsa-test-bf16-interleaved-rope",
                    prepare_call=prepare_call,
                ),
            )
        )
        resources.callback(result.close)
        return qsa.bind(declaration, scratch=binding.scratch, **operands)

    monkeypatch.setattr(cases, "_dynamic_inputs", dynamic_with_declared_rope_layout)
    monkeypatch.setattr(cases, "_rebind", reprepare_rope_cache)
    cases.test_qsa_tp4_geometry_matches_reference_under_cuda_graph_replay()


@pytest.mark.parametrize("fault", ["request", "page"])
def test_qsa_invalid_metadata_prevents_mutation_and_poisons(fault):
    from tests.attention.test_qsa_contract import (
        _caps,
        _allocate_binding,
        _dynamic_inputs,
    )
    from b12x.attention import qsa

    caps = _caps(gpu(), max_batch=1, max_q_rows=1, max_raw_state_slots=1, index_heads=4)
    binding = _allocate_binding(caps)
    binding.main_block_table[0, 0] = 0
    binding.compressed_block_table[0, 0] = 0
    binding.main_k_cache.normal_()
    binding.main_v_cache.normal_()
    binding.compressed_k_cache.zero_()
    binding.raw_k_ring.zero_()
    dynamic = _dynamic_inputs(binding, positions=(0,), request_ids=(0,))
    if fault == "request":
        dynamic["request_ids"][0] = caps.max_batch
    else:
        binding.main_block_table[0, 0] = caps.num_main_cache_pages
    state = [
        binding.compressed_k_cache,
        binding.raw_k_ring,
        binding.raw_logical_positions,
        binding.raw_rope_positions,
        binding.raw_interval_start_positions,
    ]
    before = [t.clone() for t in state]
    with kernel_resolution_guard("prepared invalid QSA request"):
        output = qsa.run(binding, **dynamic)
    torch.cuda.synchronize()
    assert binding.state_errors[0].item() != 0
    assert torch.isnan(output.float()).all()
    for actual, expected in zip(state, before, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("large", [False, True])
def test_qsa_paired_score_bounds_and_frozen_replay(large):
    from types import SimpleNamespace
    from b12x.attention.qsa._score_cute import (
        compile_score_representatives,
        launch_score_representatives,
    )

    device = gpu()
    rows, heads, dim, page_size, groups = 129, 4, 128, 512, 64
    high = (1 << 31) // (page_size * dim) + 1 if large else 0
    bytes_needed = (high + 1) * page_size * dim * 2
    free, _ = torch.cuda.mem_get_info(device)
    assert free >= bytes_needed + (2 << 30), (
        "insufficient memory for high-page qualification"
    )
    cache = torch.empty((high + 1, page_size, dim), dtype=torch.bfloat16, device=device)
    torch.manual_seed(487)
    cache[high].normal_()
    query = torch.randn((rows, heads, dim), dtype=torch.bfloat16, device=device)
    table = torch.full((1, 1), high, dtype=torch.int32, device=device)
    positions = torch.full((rows,), groups * 4 - 1, dtype=torch.int64, device=device)
    requests = torch.zeros(rows, dtype=torch.int32, device=device)
    lengths = torch.tensor([groups * 4], dtype=torch.int32, device=device)
    errors = torch.zeros(rows, dtype=torch.int32, device=device)
    scores = torch.empty((rows, groups), dtype=torch.float32, device=device)
    counts = torch.empty(rows, dtype=torch.int32, device=device)
    merges = torch.empty_like(counts)
    caps = SimpleNamespace(
        index_heads=heads,
        index_head_dim=dim,
        compress_ratio=4,
        compressed_page_size=page_size,
        max_groups=groups,
        group_budget=512,
    )
    args = dict(
        prepared_query=query,
        query_positions=positions,
        request_ids=requests,
        sequence_lengths=lengths,
        compressed_cache=cache,
        compressed_block_table=table,
        state_errors=errors,
        scores=scores,
        eligible_counts=counts,
        merge_lengths=merges,
        caps=caps,
    )
    program = compile_score_representatives(**args)
    reference = (
        torch.einsum("rhd,gd->rhg", query.float(), cache[high, :groups].float())
        .clamp_min(0)
        .sum(1)
        / dim**0.5
    )
    for live_rows in (127, 128, 129):
        live = {
            name: value[:live_rows]
            if name
            in (
                "prepared_query",
                "query_positions",
                "request_ids",
                "state_errors",
                "scores",
                "eligible_counts",
                "merge_lengths",
            )
            else value
            for name, value in args.items()
        }

        def launch():
            return launch_score_representatives(
                **live, group_offset=0, group_count=groups, _prepared=program
            )

        launch()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        try:
            with kernel_resolution_guard("QSA paired score must retain one callable"):
                with torch.cuda.graph(graph):
                    launch()
                errors.zero_()
                graph.replay()
                torch.cuda.synchronize()
                assert torch.equal(errors, torch.zeros_like(errors))
                torch.testing.assert_close(
                    scores[:live_rows], reference[:live_rows], rtol=2e-5, atol=2e-5
                )
                assert torch.isfinite(scores[:live_rows]).all()
                table[0, 0] = high + 1
                graph.replay()
                torch.cuda.synchronize()
                assert (errors[:live_rows] & 512).ne(0).all()
                assert torch.isneginf(scores[:live_rows]).all()
        finally:
            graph.reset()
        table.fill_(high)
        errors.zero_()
    if large:
        assert high * cache.stride(0) > 2**31


def test_ple_checkpoint_exact_window_frozen_graph():
    from tests.sequence.test_ple import _bind_cuda_layer, _cuda_projected_inputs
    from b12x.sequence import ple

    device = gpu()
    tokens, streams, hidden, length, speculative = 16, 2, 32, 3, 3
    residual, key, value, weights, generator = _cuda_projected_inputs(
        tokens, streams, hidden, device=device, seed=718
    )
    conv_weight = torch.randn(
        (streams * hidden, 4), generator=generator, dtype=torch.bfloat16, device=device
    )
    pool = torch.randn(
        (3, streams * hidden, length + speculative),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    starts = torch.tensor([0, tokens], dtype=torch.int32, device=device)
    _, binding = _bind_cuda_layer(
        mode="mixed",
        residual=residual,
        key=key,
        value=value,
        weights=weights,
        conv_weight=conv_weight,
        query_start_loc=starts,
        state_slot_ids=torch.tensor([0], dtype=torch.int64, device=device),
        state_is_fresh=torch.tensor([False], dtype=torch.bool, device=device),
        num_accepted_tokens=torch.tensor([1], dtype=torch.int32, device=device),
        num_seqs=1,
        num_tokens=tokens,
        conv_state=pool,
        max_speculative_tokens=speculative,
        dilation=1,
        request_is_prefill=torch.tensor([True], dtype=torch.bool, device=device),
    )
    ple.run_mixed(binding, eps=1e-6)
    torch.cuda.synchronize()
    offsets = torch.tensor([1], dtype=torch.int32, device=device)
    slots = torch.tensor([1], dtype=torch.int64, device=device)
    baseline = pool.clone()
    history = torch.cat(
        (binding.gathered_state[0], binding.normalized_u[:tokens].T), dim=1
    )
    graph = torch.cuda.CUDAGraph()
    try:
        with kernel_resolution_guard("prepared PLE checkpoint export"):
            with torch.cuda.graph(graph):
                ple.export_checkpoint(binding, offsets=offsets, slots=slots)
            for offset in (1, 2, 8, 15, 0, tokens, tokens + 1):
                pool.copy_(baseline)
                offsets.fill_(offset)
                graph.replay()
                torch.cuda.synchronize()
                expected = baseline.clone()
                if 0 < offset < tokens:
                    expected[1, :, :length] = history[:, offset : offset + length]
                    expected[1, :, length:] = 0
                torch.testing.assert_close(pool, expected, rtol=0, atol=0)
            slots.fill_(pool.shape[0])
            offsets.fill_(8)
            pool.copy_(baseline)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(pool, baseline, rtol=0, atol=0)
    finally:
        graph.reset()
