"""Runtime of glm_dsa_indexer_split: the prefill row split, its fallbacks, counters and verify mode.

``forward`` replaces the body of ``B12xSparseIndexer.forward``
(``vllm/v1/attention/backends/mla/b12x_indexer.py``, image 816c6d6a7e96) for
prefill steps; every other call runs the image's method. ``layout.py`` holds
the row partition; ``__init__.py`` installs the wrappers and pins the sources.

The split path restates the image's per-launch statements of
``B12xSparseIndexer.forward``: the same ``_run_paged_topk`` call with the
same plan (``self._plan("prefill", rows)``), query, weights, KV cache, page
table, per-row causal lengths and active width, followed by the same
``_merge_dcp_topk`` call. Both functions are looked up in the image module
when called, so the split and the image path run whatever the process has
installed there. Only the set of rows per launch differs.

Every decision that selects a path or issues a collective depends only on
values every rank of the TP group shares (the step's metadata, the settings,
the CUDA graph capture state), so all ranks issue the same collectives in the
same order.
"""

from __future__ import annotations

import atexit
import dataclasses
import logging
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from . import layout

PLUGIN_NAME = "glm_dsa_indexer_split"
INDEXER_MODULE = "vllm.v1.attention.backends.mla.b12x_indexer"
PARALLEL_STATE_MODULE = "vllm.distributed.parallel_state"
# b12x_indexer._INDEX_PAGE_SIZE: index-cache tokens per page (pinned file).
INDEX_PAGE_SIZE = 64
# Instance attribute holding the rank's topology.
STATE_ATTR = "_glm_dsa_indexer_split"
LOG_INTERVAL_S = 300.0  # counters line from an eager call at most this often, and once at exit

_LOG = logging.getLogger("vllm." + PLUGIN_NAME)

# Reasons a prefill call takes the image's path. Decode-only calls are counted
# separately as "decode": decode is out of scope, not a fallback.
FALLBACKS = (
    "capturing",    # inside a CUDA graph capture
    "no_metadata",  # no indexer metadata of this layer in the forward context
    "key_gather",   # the indexer was built with VLLM_DCP_INDEXER_KEY_GATHER=1
    "mixed",        # decode rows share the step and GLM_DSA_INDEXER_SPLIT_MIXED is 0
    "short",        # fewer prefill rows than GLM_DSA_INDEXER_SPLIT_MIN_ROWS
    "layout",       # chunks that are not single-request runs tiling the prefill rows in order
    "shape",        # query, weights or top-k buffer of an unexpected shape
    "unprepared",   # no prepared prefill plan, or an image launch above its row capacity
    "other",        # arguments the image itself refuses (non-FP8 query tuple, a K tensor)
)


class SplitRefused(RuntimeError):
    """The process's parallel layout or the installed code cannot carry the row split."""


@dataclass(frozen=True)
class Settings:
    """``GLM_DSA_INDEXER_SPLIT*`` values of this process (``__init__.settings_from_env``)."""

    min_rows: int = 512
    full_launches: bool = False
    mixed: bool = False
    verify: int = 0
    verify_min_context: int = 0


@dataclass(frozen=True)
class Topology:
    """One indexer's place in the TP group, checked when the indexer is constructed."""

    tp_size: int
    tp_rank: int
    dcp_size: int
    dcp_rank: int
    copies: int
    group: int
    tp_group: object = field(compare=False, repr=False)


@dataclass
class Step:
    """One eligible prefill step of one indexer layer."""

    metadata: object
    decode_rows: int
    rows: int
    chunks: list
    plan: layout.RowLayout
    launches: list
    cap: int
    context: int


SETTINGS = Settings()
_LOCK = threading.Lock()
_STATS: dict = {}
_LAST_LOG = 0.0
# Per TP rank, keyed by the identity of the rank's TP group object (one per
# worker process; one per thread in the CPU tests, which emulate ranks with
# threads): whether the engagement line was logged, and the verify budget and
# totals.
_ENGAGED: set = set()
_VERIFY: dict = {}
_VERIFY_KEYS = ("layer_chunks", "rows", "split_same_set", "split_different_set", "image_same_set",
                "image_different_set", "local_rows", "local_same", "local_tie_only", "local_different")


def configure(settings: Settings) -> None:
    global SETTINGS
    SETTINGS = settings
    reset_stats()


def reset_stats() -> None:
    global _LAST_LOG
    with _LOCK:
        _STATS.clear()
        _STATS.update({"split_calls": 0, "split_rows": 0, "split_launches": 0, "image_launches": 0,
                       "decode": 0, **{f"fallback_{name}": 0 for name in FALLBACKS}})
        _ENGAGED.clear()
        _VERIFY.clear()
        _LAST_LOG = time.monotonic()


def stats() -> dict:
    """Counters of this process, summed over its ranks (one rank per worker process)."""
    with _LOCK:
        out = dict(_STATS)
        out["verify"] = {key: sum(entry[key] for entry in _VERIFY.values()) for key in _VERIFY_KEYS}
        out["engaged"] = bool(_ENGAGED)
    return out


def _rank_key(indexer) -> int:
    return id(getattr(indexer, STATE_ATTR).tp_group)


def _count(name: str, amount: int = 1) -> None:
    with _LOCK:
        _STATS[name] = _STATS.get(name, 0) + amount


# --------------------------------------------------------------------------- image module access


def _module():
    """The image's b12x_indexer module, through which the image's functions are looked up at call time."""
    return sys.modules[INDEXER_MODULE]


def _tp_group():
    return sys.modules[PARALLEL_STATE_MODULE].get_tp_group()


def _dcp_group():
    return sys.modules[PARALLEL_STATE_MODULE].get_dcp_group()


# --------------------------------------------------------------------------- construction


def attach(indexer) -> Topology:
    """Check this rank's TP and DCP groups and record the topology on ``indexer``.

    The row split needs at least two DCP groups (KV copies) inside the TP
    group, each a run of ``d`` consecutive TP ranks, and this rank at position
    ``tp_rank % d`` of its group. Anything else refuses.
    """
    tp = _tp_group()
    tp_size, tp_rank = int(tp.world_size), int(tp.rank_in_group)
    dcp_size = int(indexer.dcp_world_size)
    try:
        copies = layout.check_geometry(tp_size, dcp_size)
    except ValueError as exc:
        raise SplitRefused(f"{PLUGIN_NAME}: {exc} (DCP {dcp_size} at TP {tp_size}); the prefill row split "
                           "applies to DCP 1, 2 and 4 at TP 8; launch without INDEXER_SPLIT") from None
    group = tp_rank // dcp_size
    dcp_rank = int(indexer.dcp_rank)
    if dcp_size > 1:
        dcp = _dcp_group()
        tp_ranks = list(tp.ranks)
        expected = tp_ranks[group * dcp_size:(group + 1) * dcp_size]
        if list(dcp.ranks) != expected or int(dcp.rank_in_group) != tp_rank % dcp_size or dcp_rank != tp_rank % dcp_size:
            raise SplitRefused(f"{PLUGIN_NAME}: DCP group ranks {list(dcp.ranks)} (rank {dcp.rank_in_group}) are not "
                               f"the run {expected} of consecutive TP ranks that holds TP rank {tp_rank}; refusing")
    topology = Topology(tp_size=tp_size, tp_rank=tp_rank, dcp_size=dcp_size, dcp_rank=dcp_rank, copies=copies,
                        group=group, tp_group=tp)
    setattr(indexer, STATE_ATTR, topology)
    return topology


# --------------------------------------------------------------------------- eligibility


def _request_key(chunk) -> tuple:
    """Identity of a chunk's request: the block-table row view and the request's lengths."""
    table = chunk.block_table
    return (int(table.data_ptr()), tuple(table.shape), tuple(table.stride()), int(chunk.total_seq_lens),
            int(chunk.local_total_seq_lens))


def _capturing(tensor: torch.Tensor) -> bool:
    return bool(tensor.is_cuda and torch.cuda.is_current_stream_capturing())


def plan_step(indexer, q_quant, k, weights) -> Step | str:
    """The split plan of this call, or the name of the reason it takes the image's path."""
    settings = SETTINGS
    if not isinstance(q_quant, torch.Tensor) or k is not None or not isinstance(weights, torch.Tensor):
        return "other"
    context = _module().get_forward_context()
    attn = getattr(context, "attn_metadata", None)
    if not isinstance(attn, dict):
        return "no_metadata"
    metadata = attn.get(indexer.k_cache.prefix)
    if metadata is None:
        return "no_metadata"
    prefill = getattr(metadata, "prefill", None)
    if prefill is None:
        return "decode"
    if getattr(indexer, "dcp_key_gather", False):
        # The image's gathered-key prefill path does not score rows per DCP
        # group, so the split's selections are not the image's.
        return "key_gather"
    if _capturing(q_quant):
        return "capturing"
    topology = getattr(indexer, STATE_ATTR, None) or attach(indexer)
    decode_rows = int(metadata.num_decode_tokens)
    rows = int(metadata.num_prefill_tokens)
    if (decode_rows or metadata.decode is not None) and not settings.mixed:
        return "mixed"
    if rows < max(1, settings.min_rows):
        return "short"
    chunks = list(prefill.chunks)
    position = decode_rows
    for chunk in chunks:
        if (int(chunk.num_reqs) != 1 or int(chunk.token_start) != position
                or int(chunk.token_end) <= int(chunk.token_start)
                or int(chunk.cu_seqlen_ks.shape[0]) != int(chunk.token_end) - int(chunk.token_start)):
            return "layout"
        position = int(chunk.token_end)
    if not chunks or position != decode_rows + rows:
        return "layout"
    buffer = indexer.topk_indices_buffer
    topk = int(indexer.topk_tokens)
    end = decode_rows + rows
    if (q_quant.dim() != 3 or int(q_quant.shape[0]) < end or weights.dim() != 2 or int(weights.shape[0]) < end
            or buffer.dim() != 2 or int(buffer.shape[0]) < end or int(buffer.shape[1]) != topk
            or not buffer.is_contiguous()):
        return "shape"
    cap = max((int(count) for mode, count in indexer._prepared_plans if mode == "prefill"), default=0)
    if cap < 1 or any(int(c.token_end) - int(c.token_start) > cap for c in chunks):
        return "unprepared"
    plan = layout.row_layout(rows, topology.tp_size, topology.dcp_size, topology.tp_rank)
    spans = [(int(c.token_start) - decode_rows, int(c.token_end) - decode_rows, _request_key(c)) for c in chunks]
    launches = layout.block_launches(spans, plan.block_start, plan.block_stop, full=settings.full_launches, cap=cap)
    return Step(metadata=metadata, decode_rows=decode_rows, rows=rows, chunks=chunks, plan=plan, launches=launches,
                cap=cap, context=max(int(c.total_seq_lens) for c in chunks))


# --------------------------------------------------------------------------- the split


Recorder = Callable[[int, int, torch.Tensor, "torch.Tensor | None"], None]


def score_block(indexer, step: Step, q_quant: torch.Tensor, weights: torch.Tensor,
                record: Recorder | None = None) -> None:
    """Score and merge this rank's group block into ``topk_indices_buffer`` (the image's per-launch statements)."""
    module = _module()
    plan = step.plan
    offset = step.decode_rows
    buffer = indexer.topk_indices_buffer
    topk = int(indexer.topk_tokens)
    dcp = int(indexer.dcp_world_size)
    scores = None
    if dcp > 1 and plan.block_rows > 0:
        scores = torch.empty((plan.block_rows, topk), dtype=torch.float32, device=q_quant.device)
    lengths: dict[int, torch.Tensor] = {}
    for launch in step.launches:
        parts = []
        for fragment in launch.fragments:
            chunk = step.chunks[fragment.chunk]
            per_row = lengths.get(fragment.chunk)
            if per_row is None:
                per_row = lengths[fragment.chunk] = chunk.cu_seqlen_ke - chunk.cu_seqlen_ks
            first = offset + fragment.start - int(chunk.token_start)
            parts.append(per_row[first:first + fragment.stop - fragment.start])
        seq_lens = parts[0].contiguous() if len(parts) == 1 else torch.cat(parts)
        head = step.chunks[launch.fragments[0].chunk]
        local = head.local_total_seq_lens if dcp > 1 else head.total_seq_lens
        pages = min(max(1, (int(local) + INDEX_PAGE_SIZE - 1) // INDEX_PAGE_SIZE), int(head.block_table.shape[1]))
        start, stop = offset + launch.start, offset + launch.stop
        rows = stop - start
        output = buffer[start:stop, :topk]
        score = None
        if scores is not None:
            score = scores[launch.start - plan.block_start:launch.stop - plan.block_start]
        module._run_paged_topk(
            module=indexer._module,
            plan=indexer._plan("prefill", rows),
            q=q_quant[start:stop].contiguous(),
            weights=weights[start:stop].contiguous(),
            kv_cache=indexer.k_cache.kv_cache,
            seq_lens=seq_lens,
            block_table=head.block_table[:1, :pages].expand(rows, pages),
            active_width=indexer.active_width_cap,
            output=output,
            scores=score,
        )
        if record is not None:
            record(start, stop, output, score)
        if score is not None:
            module._merge_dcp_topk(output, score, indexer.dcp_rank, dcp, indexer.cp_kv_cache_interleave_size)


def exchange(indexer, step: Step) -> None:
    """All-gather the rows over the TP group and copy the other groups' rows into ``topk_indices_buffer``."""
    plan = step.plan
    offset = step.decode_rows
    buffer = indexer.topk_indices_buffer
    topk = int(indexer.topk_tokens)
    start, stop = offset + plan.send_start, offset + plan.send_stop
    if stop <= int(buffer.shape[0]):
        send = buffer[start:stop]
    else:
        send = torch.empty((plan.per_rank, topk), dtype=buffer.dtype, device=buffer.device)
        valid = max(0, min(stop, offset + plan.rows) - start)
        if valid:
            send[:valid].copy_(buffer[start:start + valid])
    topology = getattr(indexer, STATE_ATTR)
    gathered = topology.tp_group.all_gather(send, dim=0)
    if tuple(gathered.shape) != (plan.padded, topk):
        raise RuntimeError(f"{PLUGIN_NAME}: the TP all-gather returned {tuple(gathered.shape)}, expected "
                           f"{(plan.padded, topk)}")
    if plan.block_start > 0:
        buffer[offset:offset + plan.block_start].copy_(gathered[:plan.block_start])
    if plan.block_stop < plan.rows:
        buffer[offset + plan.block_stop:offset + plan.rows].copy_(gathered[plan.block_stop:plan.rows])


def image_decode_part(indexer, image_forward, hidden_states, q_quant, k, weights, metadata) -> None:
    """The image's decode statements for the decode rows of a mixed step.

    The layer's metadata entry is replaced, for this call only, by a copy
    without its prefill part, so the image's own ``forward`` runs only its
    decode branch.
    """
    attn = _module().get_forward_context().attn_metadata
    key = indexer.k_cache.prefix
    attn[key] = dataclasses.replace(metadata, prefill=None)
    try:
        image_forward(indexer, hidden_states, q_quant, k, weights)
    finally:
        attn[key] = metadata


def run_split(indexer, image_forward, step: Step, hidden_states, q_quant, k, weights,
              record: Recorder | None = None) -> None:
    score_block(indexer, step, q_quant, weights, record)
    exchange(indexer, step)
    if step.decode_rows or step.metadata.decode is not None:
        image_decode_part(indexer, image_forward, hidden_states, q_quant, k, weights, step.metadata)


def forward(indexer, image_forward, hidden_states, q_quant, k, weights):
    """``B12xSparseIndexer.forward`` with the prefill row split."""
    step = plan_step(indexer, q_quant, k, weights)
    if isinstance(step, str):
        _count("decode" if step == "decode" else f"fallback_{step}")
        _maybe_log()
        return image_forward(indexer, hidden_states, q_quant, k, weights)
    if _take_verify(indexer, step):
        verify(indexer, image_forward, step, hidden_states, q_quant, k, weights)
    else:
        run_split(indexer, image_forward, step, hidden_states, q_quant, k, weights)
    _note_split(indexer, step)
    return indexer.topk_indices_buffer


# --------------------------------------------------------------------------- logging


def _image_launches(step: Step) -> int:
    return len(step.chunks)


def _note_split(indexer, step: Step) -> None:
    key = _rank_key(indexer)
    with _LOCK:
        first = key not in _ENGAGED
        _ENGAGED.add(key)
        _STATS["split_calls"] += 1
        _STATS["split_rows"] += step.plan.block_rows
        _STATS["split_launches"] += len(step.launches)
        _STATS["image_launches"] += _image_launches(step)
    if first:
        plan = step.plan
        topology = getattr(indexer, STATE_ATTR)
        first_rank = topology.group * topology.dcp_size
        _LOG.info(
            "%s: prefill row split engaged: group %d of %d (TP ranks %d-%d, DCP %d) on TP rank %d, rows [%d, %d) of "
            "%d prefill rows (padded to %d, %d per rank), launches of up to %d rows%s, first layer %s",
            PLUGIN_NAME, topology.group, topology.copies, first_rank, first_rank + topology.dcp_size - 1,
            topology.dcp_size, topology.tp_rank, plan.block_start, plan.block_stop, plan.rows, plan.padded,
            plan.per_rank, step.cap, " (full launches)" if SETTINGS.full_launches else " (image launch sizes)",
            indexer.k_cache.prefix)
    _maybe_log()


def describe_stats() -> str:
    data = stats()
    fallbacks = ", ".join(f"{name} {data[f'fallback_{name}']}" for name in FALLBACKS if data[f"fallback_{name}"])
    return (f"{data['split_calls']} layer-chunks split ({data['split_rows']} rows scored here in "
            f"{data['split_launches']} launches; the image would run {data['image_launches']}); image path: decode "
            f"{data['decode']}{', ' + fallbacks if fallbacks else ', no fallback'}")


def _device_name() -> str:
    try:
        return f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001 - only a log label
        return "?"


def _maybe_log(force: bool = False) -> None:
    global _LAST_LOG
    now = time.monotonic()
    with _LOCK:
        if not force and now - _LAST_LOG < LOG_INTERVAL_S:
            return
        _LAST_LOG = now
    _LOG.info("%s %s: %s", PLUGIN_NAME, _device_name(), describe_stats())


def _log_at_exit() -> None:
    try:
        data = stats()
        if data["split_calls"] or data["decode"] or any(data[f"fallback_{n}"] for n in FALLBACKS):
            _maybe_log(force=True)
    except Exception:  # noqa: BLE001 - never fail interpreter exit
        pass


atexit.register(_log_at_exit)


# --------------------------------------------------------------------------- verify mode


def _take_verify(indexer, step: Step) -> bool:
    """Whether this layer-chunk is verified; the same answer on every rank (shared metadata and settings)."""
    if SETTINGS.verify <= 0 or step.context < SETTINGS.verify_min_context:
        return False
    key = _rank_key(indexer)
    with _LOCK:
        entry = _VERIFY.setdefault(key, {"taken": 0, **dict.fromkeys(_VERIFY_KEYS, 0)})
        if entry["taken"] >= SETTINGS.verify:
            return False
        entry["taken"] += 1
        return True


class _Local:
    """Pre-merge top-k candidates (indices and scores) of rows, recorded per launch."""

    def __init__(self, buffer: torch.Tensor):
        self.buffer = buffer
        self.parts: list[tuple[int, int, torch.Tensor, torch.Tensor | None]] = []

    def add(self, start: int, stop: int, output: torch.Tensor, score: torch.Tensor | None) -> None:
        self.parts.append((start, stop, output.clone(), None if score is None else score.clone()))

    def row_of(self, output: torch.Tensor) -> int:
        stride = self.buffer.stride(0) * self.buffer.element_size()
        return (output.data_ptr() - self.buffer.data_ptr()) // stride

    def rows(self, start: int, stop: int):
        """Indices and scores of rows [start, stop), or None if a row was not recorded."""
        if stop <= start:
            return None
        width = int(self.buffer.shape[1])
        index = torch.full((stop - start, width), -1, dtype=torch.int32, device=self.buffer.device)
        score = torch.full((stop - start, width), float("nan"), dtype=torch.float32, device=self.buffer.device)
        covered = torch.zeros(stop - start, dtype=torch.bool, device=self.buffer.device)
        for lo, hi, idx, sc in self.parts:
            a, b = max(lo, start), min(hi, stop)
            if a >= b or sc is None:
                continue
            index[a - start:b - start] = idx[a - lo:b - lo]
            score[a - start:b - start] = sc[a - lo:b - lo]
            covered[a - start:b - start] = True
        if not bool(covered.all()):
            return None
        return index, score


def same_sets(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per row: both rows hold the same indices, in any order."""
    return (a.sort(dim=1).values == b.sort(dim=1).values).all(dim=1)


def compare_candidates(ia: torch.Tensor, sa: torch.Tensor, ib: torch.Tensor, sb: torch.Tensor):
    """Per row: (same index set, differs only among candidates tied at the row's lowest selected score).

    A row with different index sets counts as a tie-only difference when both
    rows select the same multiset of scores and the same indices above their
    lowest selected score: the selector chose between equal scores.
    """
    valid_a, valid_b = ia >= 0, ib >= 0
    neg = torch.tensor(float("-inf"), dtype=sa.dtype, device=sa.device)
    pos = torch.tensor(float("inf"), dtype=sa.dtype, device=sa.device)
    ordered_a = torch.where(valid_a, sa, neg).sort(dim=1, descending=True).values
    ordered_b = torch.where(valid_b, sb, neg).sort(dim=1, descending=True).values
    scores_equal = (ordered_a == ordered_b).all(dim=1)
    low_a = torch.where(valid_a, sa, pos).amin(dim=1, keepdim=True)
    low_b = torch.where(valid_b, sb, pos).amin(dim=1, keepdim=True)
    big = torch.iinfo(torch.int32).max
    above_a = torch.where(valid_a & (sa > low_a), ia, big).sort(dim=1).values
    above_b = torch.where(valid_b & (sb > low_b), ib, big).sort(dim=1).values
    same = same_sets(ia, ib)
    tie_only = ~same & scores_equal & (above_a == above_b).all(dim=1)
    return same, tie_only


_RECORD_LOCK = threading.Lock()
_RECORD_TLS = threading.local()
_RECORD_STATE: dict = {"users": 0, "original": None}


class _RecordMerges:
    """While open: the image module's ``_merge_dcp_topk`` records its inputs for this thread, then merges.

    The module function is replaced only while some thread verifies and is
    restored when the last one finishes; a call from a thread that is not
    recording merges as before. A worker runs one rank per process; the CPU
    tests run eight ranks as threads of one process.
    """

    def __init__(self, local: _Local):
        self.local = local
        self.module = _module()

    def __enter__(self):
        with _RECORD_LOCK:
            if _RECORD_STATE["users"] == 0:
                original = self.module._merge_dcp_topk

                def recording_merge(indices, scores, *args, **kwargs):
                    local = getattr(_RECORD_TLS, "local", None)
                    if local is not None:
                        start = local.row_of(indices)
                        local.add(start, start + int(indices.shape[0]), indices, scores)
                    return original(indices, scores, *args, **kwargs)

                _RECORD_STATE["original"] = original
                self.module._merge_dcp_topk = recording_merge
            _RECORD_STATE["users"] += 1
        _RECORD_TLS.local = self.local
        return self

    def __exit__(self, *exc):
        _RECORD_TLS.local = None
        with _RECORD_LOCK:
            _RECORD_STATE["users"] -= 1
            if _RECORD_STATE["users"] == 0:
                self.module._merge_dcp_topk = _RECORD_STATE["original"]
                _RECORD_STATE["original"] = None
        return False


def verify(indexer, image_forward, step: Step, hidden_states, q_quant, k, weights) -> None:
    """Compute one layer-chunk with the split and twice with the image's path; log the comparison.

    The model continues with the second image result. Compared, for the step's
    prefill rows: the final index sets (split against image, and image against
    image as the reference for the selector's own run-to-run behavior); and,
    under DCP, every pre-merge candidate list of this rank's block (indices and
    scores), split against image, tie-aware (``compare_candidates``).
    """
    buffer = indexer.topk_indices_buffer
    lo, hi = step.decode_rows, step.decode_rows + step.rows
    dcp = int(indexer.dcp_world_size)
    split_local = _Local(buffer)
    run_split(indexer, image_forward, step, hidden_states, q_quant, k, weights, record=split_local.add)
    split_rows = buffer[lo:hi].clone()
    image_local = _Local(buffer)
    if dcp > 1:
        with _RecordMerges(image_local):
            image_forward(indexer, hidden_states, q_quant, k, weights)
    else:
        image_forward(indexer, hidden_states, q_quant, k, weights)
    image_rows = buffer[lo:hi].clone()
    image_forward(indexer, hidden_states, q_quant, k, weights)
    again_rows = buffer[lo:hi]
    result = {"rows": step.rows, "split_same_set": int(same_sets(split_rows, image_rows).sum()),
              "image_same_set": int(same_sets(image_rows, again_rows).sum())}
    result["split_different_set"] = step.rows - result["split_same_set"]
    result["image_different_set"] = step.rows - result["image_same_set"]
    first_bad = None
    block = (lo + step.plan.block_start, lo + step.plan.block_stop)
    local_rows = local_same = local_tie_only = local_different = 0
    if dcp > 1 and block[1] > block[0]:
        mine, theirs = split_local.rows(*block), image_local.rows(*block)
        if mine is None or theirs is None:
            raise RuntimeError(f"{PLUGIN_NAME}: verify recorded no candidates for some rows of the block {block}")
        same, tie_only = compare_candidates(mine[0], mine[1], theirs[0], theirs[1])
        local_rows = block[1] - block[0]
        local_same, local_tie_only = int(same.sum()), int(tie_only.sum())
        local_different = local_rows - local_same - local_tie_only
        if local_different:
            bad = (~same & ~tie_only).nonzero()
            first_bad = int(bad[0, 0]) + block[0] - lo
    result.update(local_rows=local_rows, local_same=local_same, local_tie_only=local_tie_only,
                  local_different=local_different)
    with _LOCK:
        entry = _VERIFY.setdefault(_rank_key(indexer), {"taken": 1, **dict.fromkeys(_VERIFY_KEYS, 0)})
        entry["layer_chunks"] += 1
        for key in _VERIFY_KEYS[1:]:
            entry[key] += result[key]
        number = entry["layer_chunks"]
        totals = dict(entry)
    _LOG.info(
        "%s verify %s [%d/%d] %s: %d prefill rows, context up to %d tokens; final index sets: split vs image %d same, "
        "%d different; image vs image %d same, %d different; pre-merge candidates of this rank's %d block rows: %d same, "
        "%d differ only among tied scores, %d differ%s",
        PLUGIN_NAME, _device_name(), number, SETTINGS.verify, indexer.k_cache.prefix, step.rows, step.context,
        result["split_same_set"], result["split_different_set"], result["image_same_set"],
        result["image_different_set"], local_rows, local_same, local_tie_only, local_different,
        "" if first_bad is None else f" (first at prefill row {first_bad})")
    if number == SETTINGS.verify:
        _LOG.info(
            "%s verify %s summary: %d layer-chunks, %d prefill rows; final index sets: split vs image %d different, "
            "image vs image %d different; pre-merge candidates: %d rows, %d differ only among tied scores, %d differ",
            PLUGIN_NAME, _device_name(), totals["layer_chunks"], totals["rows"], totals["split_different_set"],
            totals["image_different_set"], totals["local_rows"], totals["local_tie_only"], totals["local_different"])


__all__ = ["FALLBACKS", "Settings", "SplitRefused", "Step", "Topology", "attach", "compare_candidates",
           "configure", "describe_stats", "exchange", "forward", "plan_step", "reset_stats", "run_split",
           "same_sets", "score_block", "stats", "verify"]
