"""The image's B12X DSA indexer on thread-emulated TP ranks with CPU reference kernels.

What is the image's code here: ``B12xSparseIndexer.forward`` and ``._plan``,
``_merge_dcp_topk`` (``vllm/v1/attention/backends/mla/b12x_indexer.py``), the
chunk splitters ``_split_indexer_prefill_chunks`` and
``_split_prefill_chunk_rows``, and the metadata dataclasses. What stands in:

* ``reference_paged_topk`` replaces ``_run_paged_topk`` (the b12x kernel): per
  row, logits = sum over heads of weight x ReLU(q . k) over the row's causal
  local keys read through the page table, top-k by (score descending, index
  ascending), indices local (``logical``) or physical slots, scores optional.
  Inputs are multiples of 1/8 and 1/16, so every logit is exact in FP32 and a
  row's result cannot depend on the other rows of its launch. ``jitter=True``
  emulates the GPU selector: each call returns the selected entries in a
  random order and resolves exact ties at the row's lowest selected score at
  random (``tests/gpu_local`` shows both on the real kernel).
* ``PackKernel`` and ``stable_topk_from_gathered`` replace the Triton pack and
  CuTe merge kernels that ``_merge_dcp_topk`` calls, with their semantics:
  (score, global position) pairs, top-k by (score descending, position
  ascending), selected candidates kept in input order.
* ``Group`` replaces a ``GroupCoordinator``: ``all_gather`` meets the group's
  other threads at a barrier and returns the rank-order concatenation.
* The forward context, ``get_tp_group`` and ``get_dcp_group`` are thread-local.

Each rank (thread) holds its own KV shard (DCP interleave 1: rank ``r`` of a
DCP group stores global positions ``p`` with ``p % d == r`` at local position
``p // d``), its own metadata (local row lengths) and the same queries and
weights (the indexer is replicated over TP).
"""

from __future__ import annotations

import random
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from types import ModuleType, SimpleNamespace

import split_image_env  # noqa: F401  (must precede every vllm import)
import torch
import vllm.distributed.parallel_state as parallel_state
import vllm.v1.attention.backends.mla.b12x_indexer as bi
import vllm.v1.attention.backends.mla.indexer as ix

TP = 8
TOPK = 512  # the smallest top-k _merge_dcp_topk accepts
HEADS = 4
DIM = 8
PAGE = 64
SENTINEL = -7
PREFIX = "model.layers.0.self_attn.indexer.k_cache"
TIMEOUT_S = 60.0

CONTEXT = threading.local()


# --------------------------------------------------------------------------- groups


class Exchange:
    """Barrier-based all-gather among ``world`` threads."""

    def __init__(self, world: int):
        self.world = world
        self.barrier = threading.Barrier(world, timeout=TIMEOUT_S)
        self.slots: list[torch.Tensor | None] = [None] * world

    def all_gather(self, rank: int, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        self.slots[rank] = tensor.detach().clone()
        self.barrier.wait()
        result = torch.cat(list(self.slots), dim=dim)
        self.barrier.wait()
        return result


class Group:
    """The parts of a ``GroupCoordinator`` the indexer and the plugin use."""

    def __init__(self, exchange: Exchange, rank: int, ranks: list[int]):
        self.exchange = exchange
        self.world_size = exchange.world
        self.rank_in_group = rank
        self.ranks = list(ranks)
        self.calls: list[tuple] = []

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        self.calls.append(("all_gather", tuple(input_.shape), str(input_.dtype), dim))
        return self.exchange.all_gather(self.rank_in_group, input_, dim)


# --------------------------------------------------------------------------- reference kernels


@dataclass(frozen=True)
class Plan:
    """Stand-in for a prepared b12x plan: what the reference kernel needs from it."""

    mode: str
    rows: int
    physical: bool


@dataclass
class KernelLog:
    calls: list = field(default_factory=list)  # (mode rows, launch rows)


def _rng() -> random.Random:
    rng = getattr(CONTEXT, "rng", None)
    if rng is None:
        rng = CONTEXT.rng = random.Random(1234 + getattr(CONTEXT, "rank", 0))
    return rng


def reference_paged_topk(*, module, plan, q, weights, kv_cache, seq_lens, block_table, active_width, output,
                         scores) -> None:
    rows = int(q.shape[0])
    assert q.dim() == 3 and q.is_contiguous() and weights.is_contiguous(), "the image passes contiguous inputs"
    assert tuple(output.shape) == (rows, TOPK) and output.is_contiguous()
    assert scores is None or (tuple(scores.shape) == (rows, TOPK) and scores.is_contiguous())
    assert int(seq_lens.shape[0]) == rows and int(block_table.shape[0]) == rows
    assert rows <= plan.rows, "launch above the plan's capacity"
    log: KernelLog | None = getattr(CONTEXT, "kernel_log", None)
    if log is not None:
        log.calls.append((plan.mode, rows))
    jitter = getattr(CONTEXT, "jitter", False)
    lengths = [int(v) for v in seq_lens.tolist()]
    width = max(lengths + [1])
    pages = (width + PAGE - 1) // PAGE
    neg = float("-inf")
    out_index = torch.full((rows, TOPK), -1, dtype=torch.int32)
    out_score = torch.full((rows, TOPK), neg, dtype=torch.float32)
    shared = bool((block_table[:, :pages] == block_table[:1, :pages]).all()) if pages <= block_table.shape[1] else False
    keys_shared = None
    if shared:
        positions = torch.arange(width)
        blocks = block_table[0, positions // PAGE].to(torch.long)
        keys_shared = kv_cache[blocks, positions % PAGE]
    for r in range(rows):
        length = lengths[r]
        if length <= 0:
            continue
        positions = torch.arange(length)
        if shared:
            keys = keys_shared[:length]
        else:
            keys = kv_cache[block_table[r, positions // PAGE].to(torch.long), positions % PAGE]
        logits = (torch.relu(q[r].float() @ keys.float().T) * weights[r].float()[:, None]).sum(0)
        if jitter:
            tiebreak = torch.tensor([_rng().random() for _ in range(length)])
            order = torch.sort(tiebreak, stable=True).indices
            order = order[torch.sort(-logits[order], stable=True).indices]
        else:
            order = torch.sort(-logits, stable=True).indices
        chosen = order[:TOPK]
        if jitter:
            chosen = chosen[torch.tensor(_rng().sample(range(len(chosen)), len(chosen)))]
        if plan.physical:
            slots = block_table[r, chosen // PAGE].to(torch.long) * PAGE + chosen % PAGE
            out_index[r, :len(chosen)] = slots.to(torch.int32)
        else:
            out_index[r, :len(chosen)] = chosen.to(torch.int32)
        out_score[r, :len(chosen)] = logits[chosen]
    output.copy_(out_index)
    if scores is not None:
        scores.copy_(out_score)


class PackKernel:
    """``_pack_dcp_candidates_kernel[grid](...)``: (score, global position) pairs per candidate."""

    def __getitem__(self, grid):
        def launch(indices, scores, packed, index_stride, score_stride, packed_row_stride, packed_col_stride,
                   dcp_rank, dcp_world_size, interleave, topk, block, num_warps=8):
            local = indices[:, :topk].to(torch.long)
            valid = local >= 0
            safe = local.clamp(min=0)
            global_idx = (safe // interleave) * (dcp_world_size * interleave) + dcp_rank * interleave + safe % interleave
            packed[:, :topk, 0] = torch.where(valid, scores[:, :topk], torch.full_like(scores[:, :topk], float("-inf")))
            packed[:, :topk, 1] = torch.where(valid, global_idx, torch.full_like(global_idx, -1)).to(torch.float32)
        return launch


def stable_topk_from_gathered(gathered: torch.Tensor, topk: int, out: torch.Tensor | None = None) -> torch.Tensor:
    rows = int(gathered.shape[0])
    if out is None:
        out = torch.empty((rows, topk), dtype=torch.int32)
    out.fill_(-1)
    for r in range(rows):
        score, ident = gathered[r, :, 0], gathered[r, :, 1].to(torch.long)
        valid = (ident >= 0).nonzero().flatten()
        if valid.numel() == 0:
            continue
        by_id = valid[torch.sort(ident[valid], stable=True).indices]
        ranked = by_id[torch.sort(-score[by_id], stable=True).indices]
        chosen = torch.sort(ranked[:topk]).values  # input order
        out[r, :chosen.numel()] = ident[chosen].to(torch.int32)
    return out


def install_stubs() -> None:
    """Route the image module's kernels, groups and forward context to the thread-local stand-ins."""
    if getattr(install_stubs, "done", False):
        return
    bi._run_paged_topk = reference_paged_topk
    bi._pack_dcp_candidates_kernel = PackKernel()
    bi.get_dcp_group = lambda: CONTEXT.dcp
    bi.get_forward_context = lambda: CONTEXT.forward
    parallel_state.get_tp_group = lambda: CONTEXT.tp
    parallel_state.get_dcp_group = lambda: CONTEXT.dcp
    merge_module = ModuleType("vllm.model_executor.kernels.attention.dsa.dcp_indexer_cutedsl")
    merge_module.stable_topk_from_gathered_candidates_cutedsl = stable_topk_from_gathered
    sys.modules[merge_module.__name__] = merge_module
    install_stubs.done = True


# --------------------------------------------------------------------------- problems


@dataclass(frozen=True)
class Request:
    context: int  # tokens cached before this step
    rows: int  # tokens of this step
    decode: bool = False

    @property
    def seq_len(self) -> int:
        return self.context + self.rows


@dataclass(frozen=True)
class Case:
    """One step: decode requests first (as vLLM orders them), then prefill requests."""

    dcp: int
    requests: tuple[Request, ...]
    cap: int = 96  # prefill plan rows (the image's 4,096, scaled down)
    logits_budget: int = 4 * 96 * 2048  # bytes (VLLM_SPARSE_INDEXER_MAX_LOGITS_MB, scaled down)
    seed: int = 0
    tie_keys: float = 0.0  # share of keys set to zero: ties at score 0

    @property
    def decode_rows(self) -> int:
        return sum(r.rows for r in self.requests if r.decode)

    @property
    def prefill_rows(self) -> int:
        return sum(r.rows for r in self.requests if not r.decode)

    @property
    def rows(self) -> int:
        return self.decode_rows + self.prefill_rows


def count_local(length: int, dcp: int, rank: int) -> int:
    """Positions below ``length`` that DCP rank ``rank`` stores (interleave 1)."""
    return length // dcp + (1 if length % dcp > rank else 0)


@dataclass
class Problem:
    case: Case
    keys: list  # per request: [seq_len, DIM] global index keys
    q: torch.Tensor  # [rows, HEADS, DIM]
    weights: torch.Tensor  # [rows, HEADS]
    block_table: torch.Tensor  # [requests, width], the same on every rank
    num_blocks: int


def build_problem(case: Case) -> Problem:
    generator = torch.Generator().manual_seed(case.seed)

    def eighths(*shape):
        return torch.randint(-7, 8, shape, generator=generator).float() / 8

    keys = []
    for request in case.requests:
        k = eighths(request.seq_len, DIM)
        if case.tie_keys:
            k[torch.rand(request.seq_len, generator=generator) < case.tie_keys] = 0.0
        keys.append(k)
    q = eighths(case.rows, HEADS, DIM)
    weights = torch.randint(1, 16, (case.rows, HEADS), generator=generator).float() / 16
    width = max((r.seq_len + PAGE - 1) // PAGE for r in case.requests) + 1
    num_blocks = len(case.requests) * width + 3
    order = torch.randperm(num_blocks, generator=generator)
    table = order[: len(case.requests) * width].reshape(len(case.requests), width).to(torch.int32)
    return Problem(case=case, keys=keys, q=q, weights=weights, block_table=table, num_blocks=num_blocks)


def rank_cache(problem: Problem, dcp_rank: int) -> torch.Tensor:
    """This rank's index-key cache [blocks, PAGE, DIM]: global position p at local p // d (p % d == rank)."""
    d = problem.case.dcp
    cache = torch.zeros(problem.num_blocks, PAGE, DIM)
    for index, request in enumerate(problem.case.requests):
        global_positions = torch.arange(dcp_rank, request.seq_len, d)
        local = torch.arange(global_positions.numel())
        blocks = problem.block_table[index, local // PAGE].to(torch.long)
        cache[blocks, local % PAGE] = problem.keys[index][global_positions]
    return cache


def chunk_specs(problem: Problem) -> list[tuple[slice, slice]]:
    """The image's B12X prefill launches of the step: B12xIndexerMetadataBuilder._split_prefill_chunks."""
    case = problem.case
    decodes = sum(1 for r in case.requests if r.decode)
    specs = []
    for offset, request in enumerate([r for r in case.requests if not r.decode]):
        specs += ix.DeepseekV32IndexerMetadataBuilder._split_indexer_prefill_chunks(
            torch.tensor([request.seq_len], dtype=torch.int32), torch.tensor([request.rows], dtype=torch.int32),
            1 << 40, case.logits_budget, request_offset=decodes + offset)
    return bi._split_prefill_chunk_rows(specs, case.cap)


def rank_metadata(problem: Problem, dcp_rank: int, specs: list[tuple[slice, slice]] | None = None):
    """``DeepseekV32IndexerMetadata`` of the step as this rank's builder fills it."""
    case = problem.case
    d = case.dcp
    starts = [0]
    for request in case.requests:
        starts.append(starts[-1] + request.rows)
    chunks = []
    for req_slice, query_slice in specs if specs is not None else chunk_specs(problem):
        index = req_slice.start
        request = case.requests[index]
        positions = torch.arange(request.context + query_slice.start, request.context + query_slice.stop)
        if d > 1:
            per_row = torch.tensor([count_local(int(p) + 1, d, dcp_rank) for p in positions], dtype=torch.int32)
            local_total = count_local(request.seq_len, d, dcp_rank)
        else:
            per_row = (positions + 1).to(torch.int32)
            local_total = request.seq_len
        chunks.append(ix.DeepseekV32IndexerPrefillChunkMetadata(
            block_table=problem.block_table[index:index + 1],
            cu_seqlen_ks=torch.zeros(len(positions), dtype=torch.int32),
            cu_seqlen_ke=per_row,
            cu_seq_lens=torch.tensor([0, request.seq_len], dtype=torch.int32),
            token_to_seq=torch.zeros(request.seq_len, dtype=torch.int32),
            total_seq_lens=request.seq_len,
            token_start=starts[index] + query_slice.start,
            token_end=starts[index] + query_slice.stop,
            num_reqs=1,
            skip_kv_gather=query_slice.start > 0,
            local_cu_seq_lens=torch.tensor([0, local_total], dtype=torch.int32),
            local_total_seq_lens=local_total,
            max_local_total_seq_lens=count_local(request.seq_len, d, 0) if d > 1 else request.seq_len,
        ))
    decode = None
    decode_requests = [r for r in case.requests if r.decode]
    if decode_requests:
        lengths = []
        for request in decode_requests:
            for p in range(request.context, request.seq_len):
                lengths.append(count_local(p + 1, d, dcp_rank) if d > 1 else p + 1)
        decode = bi.B12xIndexerDecodeMetadata(
            block_table=problem.block_table[:len(decode_requests)],
            seq_lens=torch.tensor(lengths, dtype=torch.int32),
            decode_lens=torch.tensor([r.rows for r in decode_requests], dtype=torch.int32),
            requires_padding=False, schedule_metadata=None,
            active_width=torch.full((1,), max(lengths), dtype=torch.int32))
    prefill = ix.DeepseekV32IndexerPrefillMetadata(chunks, max_prefill_seq_len=max(
        (r.seq_len for r in case.requests if not r.decode), default=0)) if chunks else None
    return ix.DeepseekV32IndexerMetadata(
        seq_lens=torch.tensor([r.seq_len for r in case.requests], dtype=torch.int32),
        max_seq_len=max(r.seq_len for r in case.requests), slot_mapping=torch.zeros(case.rows, dtype=torch.long),
        num_decodes=len(decode_requests), num_decode_tokens=case.decode_rows,
        num_prefills=len(case.requests) - len(decode_requests), num_prefill_tokens=case.prefill_rows,
        decode=decode, prefill=prefill)


def make_indexer(problem: Problem, dcp_rank: int, *, buffer_rows: int | None = None,
                 prepared: dict | None = None):
    """A ``B12xSparseIndexer`` with the attributes its constructor sets (no b12x, no CUDA)."""
    case = problem.case
    indexer = object.__new__(bi.B12xSparseIndexer)
    torch.nn.Module.__init__(indexer)
    indexer._module = SimpleNamespace(name="b12x stand-in")
    indexer.k_cache = SimpleNamespace(prefix=PREFIX, kv_cache=rank_cache(problem, dcp_rank))
    indexer.topk_tokens = TOPK
    indexer.max_model_len = 1 << 20
    rows = buffer_rows if buffer_rows is not None else max(case.rows + TP, 64)
    indexer.topk_indices_buffer = torch.full((rows, TOPK), SENTINEL, dtype=torch.int32)
    indexer.output_physical_slots = case.dcp == 1
    indexer.num_q_heads = HEADS
    indexer.active_width_cap = torch.full((1,), indexer.max_model_len, dtype=torch.int32)
    indexer.dcp_world_size = case.dcp
    indexer.dcp_rank = dcp_rank if case.dcp > 1 else 0
    indexer.cp_kv_cache_interleave_size = 1
    indexer.dcp_key_gather = False
    physical = case.dcp == 1
    indexer._prepared_plans = prepared if prepared is not None else {
        ("prefill", case.cap): Plan("prefill", case.cap, physical),
        ("decode", 64): Plan("decode", 64, physical),
    }
    return indexer


# --------------------------------------------------------------------------- running ranks


@dataclass
class World:
    tp: list
    dcp: list


def make_world(dcp: int) -> World:
    tp_exchange = Exchange(TP)
    tp = [Group(tp_exchange, rank, list(range(TP))) for rank in range(TP)]
    groups = []
    for start in range(0, TP, dcp):
        exchange = Exchange(dcp)
        groups.extend(Group(exchange, rank - start, list(range(start, start + dcp))) for rank in range(start, start + dcp))
    return World(tp=tp, dcp=groups)


def run_ranks(world: World, body: Callable[[int], object]) -> list:
    """Run ``body(rank)`` on every TP rank in its own thread with that rank's groups; re-raise failures."""
    results: list = [None] * TP
    errors: list = [None] * TP

    def target(rank: int) -> None:
        CONTEXT.rank = rank
        CONTEXT.tp = world.tp[rank]
        CONTEXT.dcp = world.dcp[rank]
        CONTEXT.rng = random.Random(1234 + rank)
        try:
            results[rank] = body(rank)
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors[rank] = exc
            for group in (world.tp[rank], world.dcp[rank]):
                group.exchange.barrier.abort()

    threads = [threading.Thread(target=target, args=(rank,), daemon=True) for rank in range(TP)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(TIMEOUT_S + 30)
    failures = [(rank, error) for rank, error in enumerate(errors)
                if error is not None and not isinstance(error, threading.BrokenBarrierError)]
    if failures:
        rank, error = failures[0]
        raise error
    broken = [rank for rank, error in enumerate(errors) if error is not None]
    if broken:
        raise AssertionError(f"ranks {broken} hit a broken barrier")
    if any(thread.is_alive() for thread in threads):
        raise AssertionError("a rank thread did not finish")
    return results


def set_forward(metadata) -> None:
    CONTEXT.forward = SimpleNamespace(attn_metadata={PREFIX: metadata})


def expected_rows(problem: Problem) -> torch.Tensor:
    """Independent expectation: each prefill row's top-k global positions (DCP 1: physical slots), sorted."""
    case = problem.case
    out = []
    row = 0
    for index, request in enumerate(case.requests):
        for i in range(request.rows):
            position = request.context + i
            if request.decode:
                row += 1
                continue
            keys = problem.keys[index][: position + 1]
            logits = (torch.relu(problem.q[row].float() @ keys.T) * problem.weights[row][:, None]).sum(0)
            by_id = torch.arange(position + 1)
            ranked = by_id[torch.sort(-logits, stable=True).indices][:TOPK]
            if case.dcp == 1:
                ranked = problem.block_table[index, ranked // PAGE].to(torch.long) * PAGE + ranked % PAGE
            values = torch.full((TOPK,), -1, dtype=torch.int64)
            values[: ranked.numel()] = ranked
            out.append(values.sort().values)
            row += 1
    return torch.stack(out) if out else torch.empty(0, TOPK, dtype=torch.int64)
