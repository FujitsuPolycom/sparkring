"""Compare PR387 paired QSA scoring with the exact R37 scalar-query kernel.

The oracle compiles the preserved scalar class directly, bypassing dispatch.
Its AST and page-error helper are pinned to the installed R37 source snapshot.
High-page cases allocate slightly over 4 GiB without initializing the pool.
"""

from __future__ import annotations

import ast
from contextlib import contextmanager
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from b12x._lib import runtime_control
from b12x.attention.qsa import _score_cute as scorer


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

_ORACLE_CACHE = {}
_R37_AST = {
    "_RepresentativeScoreKernel": "715942813c8851df9b42ae81db395a3f986b4f7efc673f7a7988ef2de8f56c91",
    "_or_error": "90d4aba1218ec356b28c90aa2e38c48e698076ac3cd737d810e394fb2aff44d6",
}


@pytest.fixture(scope="module", autouse=True)
def _verify_scalar_oracle():
    tree = ast.parse(Path(scorer.__file__).read_text(encoding="utf-8"))
    nodes = {getattr(node, "name", ""): node for node in tree.body}
    for name, expected in _R37_AST.items():
        actual = hashlib.sha256(ast.dump(nodes[name], include_attributes=False).encode()).hexdigest()
        assert actual == expected, f"R37 oracle source changed: {name}"


@contextmanager
def _frozen_resolution():
    assert not runtime_control.kernel_resolution_frozen()
    runtime_control.freeze_kernel_resolution("PR387 bounded R37 graph equivalence")
    try:
        yield
    finally:
        runtime_control.unfreeze_kernel_resolution()


def _scalar_oracle(**args):
    """Launch the preserved R37 class, never the paired dispatch function."""
    tensors = tuple(args[name] for name in (
        "prepared_query", "query_positions", "request_ids", "sequence_lengths",
        "compressed_cache", "compressed_block_table", "state_errors", "scores",
        "eligible_counts", "merge_lengths",
    ))
    dtypes = {
        torch.bfloat16: scorer.BFloat16, torch.float32: scorer.Float32,
        torch.int32: scorer.Int32, torch.int64: scorer.Int64,
    }
    types = tuple(dtypes[t.dtype] for t in tensors)
    caps = args["caps"]
    geometry = tuple(int(getattr(caps, name)) for name in (
        "index_heads", "index_head_dim", "compress_ratio", "compressed_page_size",
        "max_groups", "group_budget",
    ))
    device = tensors[0].device
    key = (device.index, geometry, tuple(t.dtype for t in tensors))
    with torch.cuda.device(device):
        raw = _ORACLE_CACHE.get(key)
        if raw is None:
            kernel = scorer._RepresentativeScoreKernel(*geometry)
            scorer.raise_if_kernel_resolution_frozen("R37 scalar oracle", target=kernel, cache_key=key)
            fake = tuple(scorer.make_ptr(
                dtype, 16, scorer.cute.AddressSpace.gmem, assumed_align=dtype.width // 8,
            ) for dtype in types)
            raw = scorer.b12x_compile(
                kernel, fake, (scorer.Int64(1),) * 4, scorer.Int32(1),
                scorer.Int64(1), scorer.Int32(0), scorer.Int32(1),
                scorer.current_cuda_stream(),
                compile_spec=scorer.KernelCompileSpec.from_key(
                    "validation.qsa.r37_scalar_oracle", 1, key,
                ),
            )
            _ORACLE_CACHE[key] = raw
        cache, table, scores = tensors[4], tensors[5], tensors[7]
        raw(
            tuple(scorer._pointer(t, dtype) for t, dtype in zip(tensors, types, strict=True)),
            tuple(scorer.Int64(value) for value in (
                cache.stride(0), cache.stride(1), table.stride(0), scores.stride(0),
            )),
            scorer.Int32(tensors[0].shape[0]), scorer.Int64(cache.shape[0]),
            scorer.Int32(args["group_offset"]), scorer.Int32(args["group_count"]),
            scorer.current_cuda_stream(),
        )


def _score_case(*, high_page=False, odd_alignment=False):
    torch.manual_seed(387)
    device = torch.device("cuda", torch.cuda.current_device())
    page, dim, groups, rows = 188, 128, 1536, 129
    stride = dim + int(odd_alignment)
    first = 2**31 // (page * stride) + 1 if high_page else 3
    logical_pages = (groups + page - 1) // page
    pages = first + logical_pages * 2
    required = (pages * page * stride + 1) * 2 + 256 * 1024**2
    free, _ = torch.cuda.mem_get_info(device)
    if free < required:
        pytest.skip(f"Requires {required / 1024**3:.2f} GiB free for high-page pool")
    storage = torch.empty(pages * page * stride + 1, device=device, dtype=torch.bfloat16)
    cache = storage[int(odd_alignment):].as_strided(
        (pages, page, dim), (page * stride, stride, 1),
    )
    cache[first:].normal_()
    table = torch.arange(first, pages, device=device, dtype=torch.int32)
    table = table.reshape(2, logical_pages).flip(1).contiguous()
    query_storage = torch.empty(rows * 4 * dim + 1, device=device, dtype=torch.bfloat16)
    query = query_storage[int(odd_alignment):int(odd_alignment) + rows * 4 * dim].view(rows, 4, dim)
    query.normal_()
    positions = torch.arange(rows, device=device, dtype=torch.int64) * 13 + 3700
    requests = torch.zeros(rows, device=device, dtype=torch.int32)
    requests[65:] = 1  # Pair 64/65 crosses request ownership.
    requests[63] = -1  # Pair 62/63 includes inactive padding.
    lengths = torch.tensor([4096, 6144], device=device, dtype=torch.int32)
    errors = torch.zeros(rows + 2, device=device, dtype=torch.int32)
    scores = torch.empty((rows + 2, groups + 512 + 7), device=device)
    counts = torch.empty(rows + 2, device=device, dtype=torch.int32)
    merges = torch.empty_like(counts)
    caps = SimpleNamespace(
        index_heads=4, index_head_dim=dim, compress_ratio=4,
        compressed_page_size=page, max_groups=groups, group_budget=512,
    )
    if high_page:
        assert first * cache.stride(0) > 2**31
    return SimpleNamespace(**locals())


def _score_args(case, live, offset, count):
    return dict(
        prepared_query=case.query[:live], query_positions=case.positions[:live],
        request_ids=case.requests[:live], sequence_lengths=case.lengths,
        compressed_cache=case.cache, compressed_block_table=case.table,
        state_errors=case.errors[:live], scores=case.scores[:live],
        eligible_counts=case.counts[:live], merge_lengths=case.merges[:live],
        group_offset=offset, group_count=count, caps=case.caps,
    )


def _reset_scores(case, initial_errors):
    case.errors.copy_(initial_errors)
    case.scores.fill_(123.0)
    case.counts.fill_(-777)
    case.merges.fill_(-777)


def _snapshot_scores(case):
    return tuple(t.clone() for t in (case.scores, case.counts, case.merges, case.errors))


def _assert_scores(case, expected):
    for actual, reference in zip(
        (case.scores, case.counts, case.merges, case.errors), expected, strict=True,
    ):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.parametrize("high_page", [False, True])
@pytest.mark.parametrize("odd_alignment", [False, True])
def test_paired_scores_threshold_strides_ownership_and_graph_replay(high_page, odd_alignment):
    case = _score_case(high_page=high_page, odd_alignment=odd_alignment)
    initial_errors = case.errors.clone()
    initial_errors[20] = 8
    initial_errors[21] = 16
    calls = ((127, 0, 65), (128, 0, 1536), (129, 512, 1024))
    candidate = scorer.launch_score_representatives
    for live, offset, count in calls:
        args = _score_args(case, live, offset, count)
        _reset_scores(case, initial_errors)
        _scalar_oracle(**args)
        _reset_scores(case, initial_errors)
        candidate(**args)
    torch.cuda.synchronize()
    cache_before = {key: id(value) for key, value in scorer._CACHE.items()}
    pointers = tuple(t.data_ptr() for t in (
        case.query, case.cache, case.table, case.errors, case.scores, case.counts, case.merges,
    ))
    with _frozen_resolution():
        for live, offset, count in calls:
            args = _score_args(case, live, offset, count)
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph):
                    candidate(**args)
                for shift in (0, 4):
                    case.positions.add_(shift)
                    _reset_scores(case, initial_errors)
                    _scalar_oracle(**args)
                    expected = _snapshot_scores(case)
                    _reset_scores(case, initial_errors)
                    allocated = torch.cuda.memory_allocated()
                    graph.replay()
                    torch.cuda.synchronize()
                    assert torch.cuda.memory_allocated() == allocated
                    _assert_scores(case, expected)
                    carry = min((int(case.positions[0]) + 1) // 4, offset, 512)
                    assert torch.isfinite(case.scores[0, carry:carry + 8]).all()
                    case.positions.sub_(shift)
            finally:
                graph.reset()
    assert cache_before == {key: id(value) for key, value in scorer._CACHE.items()}
    assert pointers == tuple(t.data_ptr() for t in (
        case.query, case.cache, case.table, case.errors, case.scores, case.counts, case.merges,
    ))


def test_paired_scores_invalid_pages_match_scalar_error_and_row_masks():
    case = _score_case(odd_alignment=True)
    case.table[0, 1] = -1
    case.table[1, 1] = case.pages  # First out-of-range physical page.
    case.positions.fill_(1023)
    case.positions[0] = 751  # Eligible groups stop before invalid logical page 1.
    initial_errors = case.errors.clone()
    initial_errors[20] = 8
    args = _score_args(case, 129, 180, 64)
    # A single group block isolates error propagation from pre-existing R37
    # cross-CTA error-word races; the ordinary valid-page test spans many CTAs.
    _reset_scores(case, initial_errors)
    _scalar_oracle(**args)
    expected = _snapshot_scores(case)
    _reset_scores(case, initial_errors)
    scorer.launch_score_representatives(**args)
    torch.cuda.synchronize()
    _assert_scores(case, expected)
    assert case.errors[0].item() == 0
    assert case.errors[1].item() == 512
    assert case.errors[20].item() == 8
    assert case.errors[63].item() == 0
    assert case.errors[65].item() == 512
    assert torch.isneginf(case.scores[1, 188:244]).all()
    graph = torch.cuda.CUDAGraph()
    try:
        with _frozen_resolution():
            _reset_scores(case, initial_errors)
            with torch.cuda.graph(graph):
                scorer.launch_score_representatives(**args)
            _reset_scores(case, initial_errors)
            allocated = torch.cuda.memory_allocated()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == allocated
            _assert_scores(case, expected)
    finally:
        graph.reset()


def _transaction(kv_dtype):
    from b12x.attention import qsa

    torch.manual_seed(1387)
    device = torch.device("cuda", torch.cuda.current_device())
    rows, start, pages, main_page = 129, 4096, 34, 128
    caps = qsa.Caps(
        device=device, max_batch=1, max_raw_state_slots=1, max_q_rows=rows,
        max_seq_len=pages * main_page, num_main_cache_pages=pages,
        num_compressed_cache_pages=pages, main_page_size=main_page,
        compressed_page_size=main_page // 4, q_heads=6, kv_heads=1,
        index_heads=4, kv_dtype=kv_dtype,
    )
    plan = qsa.plan(caps)
    spec, = plan.scratch_specs()
    def rand(shape, dtype=torch.bfloat16):
        return torch.randn(shape, device=device, dtype=torch.bfloat16).to(dtype)
    def full(shape, value, dtype=torch.int64):
        return torch.full(shape, value, device=device, dtype=dtype)
    table = torch.arange(pages, device=device, dtype=torch.int32).view(1, -1)
    main_k = rand((pages, main_page, 1, 256), kv_dtype)
    main_v = rand((pages, main_page, 1, 256), kv_dtype)
    binding = qsa.bind(
        plan, scratch=torch.empty(spec.shape, device=device, dtype=spec.dtype),
        main_k_cache=main_k, main_v_cache=main_v, main_block_table=table,
        k_descale=torch.ones(1, device=device) if kv_dtype == torch.float8_e4m3fn else None,
        v_descale=torch.ones(1, device=device) if kv_dtype == torch.float8_e4m3fn else None,
        compressed_k_cache=rand((pages, main_page // 4, 128)),
        compressed_block_table=table.clone(),
        raw_k_ring=rand((1, caps.raw_ring_capacity, 128)),
        raw_logical_positions=full((1, caps.raw_ring_capacity), -1),
        raw_rope_positions=full((1, caps.raw_ring_capacity, 1), -1),
        raw_interval_start_positions=full((1,), start - 1),
        raw_state_slot_ids=full((1,), 0),
        index_q_norm_weight=torch.zeros(128, device=device),
        index_k_norm_weight=torch.zeros(128, device=device),
        rope_cos=torch.ones((caps.max_seq_len, 32), device=device),
        rope_sin=torch.zeros((caps.max_seq_len, 32), device=device),
        output=torch.empty((rows, 6, 256), device=device, dtype=torch.bfloat16),
        selected_positions=torch.empty((rows, caps.selection_width), device=device, dtype=torch.int32),
    )
    position = torch.arange(start, start + rows, device=device, dtype=torch.int64)
    args = dict(
        query=rand((rows, 6, 256)), index_query=rand((rows, 4, 128)),
        raw_index_key=rand((rows, 128)), request_ids=full((rows,), 0),
        query_positions=position, rope_positions=position[:, None],
        sequence_lengths=full((1,), start + rows, torch.int32),
        query_start_loc=torch.tensor([0, rows], device=device, dtype=torch.int32),
        num_accepted_tokens=full((1,), 1, torch.int32),
        is_prefilling=full((1,), True, torch.bool),
    )
    mutable = tuple(getattr(binding, name) for name in (
        "compressed_k_cache", "raw_k_ring", "raw_logical_positions",
        "raw_rope_positions", "raw_interval_start_positions",
    ))
    originals = tuple(t.clone() for t in mutable)
    def reset():
        for target, original in zip(mutable, originals, strict=True):
            target.copy_(original)
        binding.output.fill_(float("nan"))
        binding.selected_positions.fill_(-999)
    return qsa, binding, args, reset, mutable


@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_complete_qsa_transaction_selection_state_and_graph_replay(kv_dtype):
    qsa, binding, args, reset, mutable = _transaction(kv_dtype)
    main_k = binding.main_k_cache.view(torch.uint8).clone()
    main_v = binding.main_v_cache.view(torch.uint8).clone()
    reset()
    with patch.object(scorer, "launch_score_representatives", _scalar_oracle):
        qsa.run(binding, **args)
    torch.cuda.synchronize()
    assert torch.isfinite(binding.output).all()
    assert binding.output.abs().max() > 0
    assert not binding.state_errors.any()
    expected = tuple(t.clone() for t in (
        binding.output, binding.selected_positions, binding.state_errors, *mutable,
    ))
    reset()
    qsa.run(binding, **args)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with _frozen_resolution():
            reset()
            with torch.cuda.graph(graph):
                qsa.run(binding, **args)
            for _ in range(2):
                reset()
                allocated = torch.cuda.memory_allocated()
                graph.replay()
                torch.cuda.synchronize()
                assert torch.cuda.memory_allocated() == allocated
                for actual, reference in zip((
                    binding.output, binding.selected_positions, binding.state_errors, *mutable,
                ), expected, strict=True):
                    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                assert torch.equal(binding.main_k_cache.view(torch.uint8), main_k)
                assert torch.equal(binding.main_v_cache.view(torch.uint8), main_v)
    finally:
        graph.reset()
