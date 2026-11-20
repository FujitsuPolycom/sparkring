"""Row ownership for DeepSeek-V4.1 prefill: the model math, free of vLLM imports.

A tensor-parallel (TP) rank normally holds every token row of a forward and
repeats the four-stream hyperconnection (mHC) arithmetic on all of them; the
attention and MoE branches each end in an all-reduce of a ``(rows, hidden)``
partial sum. With row ownership, rank ``r`` of ``W`` keeps only rows
``[r * q, (r + 1) * q)`` (``q = rows / W``) of the residual stream and its
mixes, for every layer before the compressed-encoder/decoder (CED) boundary:

* the attention mHC runs on the owned rows, and one all-gather rebuilds the
  full-row attention input, which attention consumes unchanged;
* attention returns its TP partial unreduced, and a reduce-scatter leaves each
  rank the summed rows it owns;
* the FFN mHC runs on the owned rows, one all-gather rebuilds the MoE input,
  the MoE returns its partial unreduced and a reduce-scatter sums it.

The all-reduce of a partial equals its reduce-scatter followed by an
all-gather, so each layer moves the same bytes as before, while the per-row
mHC work falls by a factor of ``W``. Engram layers stay owned: their staged
table rows are all-reduced exactly as the image does, then each rank mixes its
own rows. At the CED boundary layer the attention input is gathered in full
(the global KV write needs every encoder row) and the residual stream is
compacted to the decoder's replay rows by a masked sum, so no full-row residual
is ever rebuilt. Layers after the boundary run the image's code unchanged.

Everything here takes its collectives and kernels from arguments, so the CPU
tests drive it with simulated ranks; ``__init__`` binds the image's objects.
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

MIN_OWNED_ROWS = 512
"""Smallest owned row count admitted: below it the collectives' fixed cost dominates."""

SUPPORTED_WORLDS = (2, 4, 8)


class RowShardRefused(RuntimeError):
    """A static precondition of row ownership does not hold for this model."""


@dataclass(frozen=True)
class Geometry:
    """The rows one rank owns in one forward."""

    rows: int
    world: int
    rank: int

    @property
    def owned(self) -> int:
        return self.rows // self.world

    @property
    def start(self) -> int:
        return self.rank * self.owned


def row_geometry(rows: int, world: int, rank: int, min_owned: int = MIN_OWNED_ROWS):
    """Return this rank's ``Geometry`` or ``None`` when ``rows`` cannot be owned.

    Rows must divide evenly: padding rows would write cache slots they do not
    own, so a non-dividing forward keeps the image's full-row path.
    """
    if world not in SUPPORTED_WORLDS or not 0 <= rank < world:
        return None
    if rows <= 0 or rows % world or rows // world < min_owned:
        return None
    return Geometry(rows, world, rank)


def owned_layers(decoder_start: int | None, num_layers: int, aux_layers) -> range:
    """Layer indices that run on owned rows; the last one is the CED boundary.

    Raises ``RowShardRefused`` when the model has no CED boundary, or when a
    draft model reads an auxiliary hidden state from an owned layer. The image
    reconstructs that state after layer ``idx`` when ``idx + 1`` is listed;
    for ``idx < decoder_start`` the tensors hold owned rows. After the boundary
    layer (``idx + 1 == decoder_start + 1``) they are already the compacted
    decoder rows, identical on every rank, so that entry is allowed.
    """
    if decoder_start is None or not 0 < decoder_start < num_layers:
        raise RowShardRefused("row ownership requires a CED decoder boundary")
    clash = sorted(a for a in aux_layers if 1 <= a <= decoder_start)
    if clash:
        raise RowShardRefused(
            f"auxiliary hidden states {clash} would read owned rows of layers "
            f"0..{decoder_start - 1}"
        )
    return range(decoder_start + 1)


class _State(threading.local):
    def __init__(self) -> None:
        self.ownership: Ownership | None = None
        self.defer_attention = 0
        self.route_owner: Ownership | None = None
        self.compressor_inputs = None


STATE = _State()
"""Per-thread forward state; vLLM runs one model forward per worker thread."""


@dataclass
class Ownership:
    """One forward's row ownership: collectives, accounting and helpers.

    ``group`` provides ``all_reduce(t)``, ``reduce_scatter(t, dim=0)`` and
    ``all_gather(t, dim=0)`` over the TP ranks in rank order (vLLM's
    ``GroupCoordinator``). ``gather_rows(source, indices)`` gathers first-axis
    rows and zeroes rows whose index is negative or out of range (the image's
    ``vllm::dsv41_gather_rows``).
    """

    geometry: Geometry
    group: Any
    gather_rows: Callable[[Any, Any], Any]
    layer_index: dict[int, int]
    boundary: int
    reduce_scatters: int = 0
    all_gathers: int = 0
    compactions: int = 0
    deferred_attention: int = 0
    engram_layers: int = 0
    engram_reduce_scatters: int = 0
    indexer_gathers: int = 0
    compact_routes: bool = False
    route_gathers: int = 0
    local_queries: bool = False
    local_attention_layers: int = 0
    gather_kv_latents: bool = False
    kv_latent_gathers: int = 0
    gather_compressor_latents: bool = False
    compressor_latent_gathers: int = 0
    band_engram: bool = False
    layers_seen: list[int] = field(default_factory=list)

    # -- row views ---------------------------------------------------------
    def local(self, tensor: Any) -> Any:
        """This rank's rows of a full-row tensor (a view)."""
        if tensor.shape[0] != self.geometry.rows:
            raise RuntimeError(
                f"row ownership expected {self.geometry.rows} rows, got "
                f"{tensor.shape[0]}"
            )
        return tensor.narrow(0, self.geometry.start, self.geometry.owned)

    # -- collectives -------------------------------------------------------
    def reduce_scatter(self, partial: Any) -> Any:
        """Sum a full-row TP partial and keep this rank's rows."""
        rows = self.geometry.rows
        if partial.shape[0] != rows:
            raise RuntimeError("row-ownership reduce-scatter needs a full-row partial")
        flat = partial.reshape(rows, -1)
        if not flat.is_contiguous():
            flat = flat.contiguous()
        out = self.group.reduce_scatter(flat, dim=0)
        self.reduce_scatters += 1
        return out.view(self.geometry.owned, *partial.shape[1:])

    def all_gather(self, owned: Any) -> Any:
        """Concatenate every rank's owned rows in rank order."""
        if owned.shape[0] != self.geometry.owned:
            raise RuntimeError("row-ownership all-gather needs owned rows")
        source = owned if owned.is_contiguous() else owned.contiguous()
        out = self.group.all_gather(source, dim=0)
        self.all_gathers += 1
        return out

    def compact(self, owned: Any, indices: Any) -> Any:
        """Gather global rows ``indices`` of an owned tensor onto every rank.

        Each rank contributes the indexed rows it owns and zeros elsewhere;
        a sum over the ranks then holds every row exactly once. Adding zeros
        is exact for finite values, so the result equals a gather from the
        full-row tensor; invalid (negative) indices yield zero rows, as the
        image's gather does.
        """
        start, count = self.geometry.start, self.geometry.owned
        local = indices - start
        mine = (indices >= start) & (local < count)
        local = local.masked_fill(~mine, -1)
        rows = self.gather_rows(owned, local)
        if not rows.is_contiguous():
            rows = rows.contiguous()
        self.compactions += 1
        return self.group.all_reduce(rows)

    @contextmanager
    def unreduced_attention(self):
        """Have the image's WO projection return its TP partial unreduced."""
        STATE.defer_attention += 1
        try:
            yield
        finally:
            STATE.compressor_inputs = None
            STATE.defer_attention -= 1

    # -- accounting --------------------------------------------------------
    def finish(self) -> None:
        """Check that every owned layer ran its collectives exactly once."""
        expected_layers = list(range(self.boundary + 1))
        if self.layers_seen != expected_layers:
            raise RuntimeError(
                f"row ownership saw layers {self.layers_seen}, expected "
                f"{expected_layers}"
            )
        full = self.boundary  # layers 0 .. boundary-1 run both branches owned
        if (
            self.reduce_scatters != 2 * full - self.local_attention_layers + self.engram_reduce_scatters
            or self.all_gathers != 2 * full + 1
            or self.compactions != 4
            or self.deferred_attention != full
        ):
            raise RuntimeError(
                "row-ownership collective accounting mismatch: "
                f"RS={self.reduce_scatters} AG={self.all_gathers} "
                f"compact={self.compactions} deferred={self.deferred_attention} "
                f"Engram RS={self.engram_reduce_scatters} "
                f"for {full} owned layers and one boundary"
            )


def attention_reduce(local: Any, all_reduce: Callable[[Any], Any]) -> Any:
    """The WO projection's TP reduction, skipped while ownership defers it."""
    if STATE.defer_attention:
        own = STATE.ownership
        if own is not None:
            own.deferred_attention += 1
        return local
    return all_reduce(local)


def indexer_score_offsets(attention: Any, rows: int, chunk_rows: int) -> range:
    """Keep native scoring tiles intact while assigning them to row owners."""
    own = STATE.ownership
    if (
        own is None
        or rows != own.geometry.rows
        or own.geometry.owned % chunk_rows != 0
        or attention.is_draft
        or not 0 <= attention.layer_id < own.boundary
        or attention.layer_id == attention.candidate_source_layer
    ):
        return range(0, rows, chunk_rows)
    first = own.geometry.start
    return range(first, first + own.geometry.owned, chunk_rows)


def publish_index_rows(attention: Any, rows: int, chunk_rows: int) -> None:
    """Assemble the full-row top-k table before tensor-parallel attention reads it."""
    offsets = indexer_score_offsets(attention, rows, chunk_rows)
    count = offsets.stop - offsets.start
    if count == rows:
        return
    own = STATE.ownership
    indices = attention.topk_indices_buffer[:rows]
    selected = indices.narrow(0, offsets.start, count).contiguous()
    indices.copy_(own.group.all_gather(selected, dim=0))
    own.indexer_gathers += 1


def gate_owned_routes(runner: Any, hidden_states: Any) -> Any:
    """Run the original gate on the rows that will select experts."""
    own = STATE.route_owner
    if (
        own is not None
        and hidden_states.shape[0] == own.geometry.rows
        and not runner.is_monolithic
        and not runner.do_naive_dispatch_combine
        and runner.moe_config.pcp_size == 1
    ):
        hidden_states = own.local(hidden_states)
    logits, _ = runner.gate(hidden_states)
    return logits


def select_owned_routes(
    runner: Any, *, hidden_states: Any, router_logits: Any,
    topk_indices_dtype: Any, input_ids: Any,
) -> tuple[Any, Any]:
    """Publish each owner's selected IDs and FP32 weight bits to the TP group."""
    own = STATE.route_owner
    if own is None or router_logits.shape[0] == hidden_states.shape[0]:
        return runner.router.select_experts(
            hidden_states=hidden_states, router_logits=router_logits,
            topk_indices_dtype=topk_indices_dtype, input_ids=input_ids,
        )
    if router_logits.shape[0] != own.geometry.owned:
        raise RuntimeError("owned MoE gate did not return its rank's rows")
    import torch

    weights, ids = runner.router.select_experts(
        hidden_states=own.local(hidden_states), router_logits=router_logits,
        topk_indices_dtype=topk_indices_dtype,
        input_ids=None if input_ids is None else own.local(input_ids),
    )
    if weights.dtype != torch.float32:
        raise RuntimeError("compact MoE routing requires FP32 weights")
    # Expert IDs fit in int32. Preserve the weight bits, including signed zero.
    record = torch.cat((weights.contiguous().view(torch.int32), ids.to(torch.int32)), dim=-1)
    gathered = own.group.all_gather(record, dim=0)
    own.route_gathers += 1
    width = weights.shape[-1]
    weights = gathered[:, :width].contiguous().view(torch.float32)
    ids = gathered[:, width:].to(ids.dtype).contiguous()
    return weights, ids


def moe_unreduced(ffn: Any, hidden_states: Any, input_ids: Any) -> Any:
    """``DeepseekV4MoE.forward`` without its final TP all-reduce or dtype cast.

    Mirrors the image's non-mega path (``_forward_fused_moe``) and asks the
    MoE runner to defer the reduction it would otherwise apply; the caller
    sums the partial and then casts, in the image's order.
    """
    if getattr(ffn, "use_mega_moe", False):
        raise RowShardRefused("row ownership does not support the mega-MoE path")
    gate = ffn.gate
    if getattr(gate, "tid2eid", None) is not None and input_ids is None:
        raise ValueError("DeepSeek V4 hash MoE routing requires input_ids.")
    if getattr(gate, "bias_vl", None) is not None and input_ids is None:
        raise ValueError("DeepSeek V4 vision MoE routing requires input_ids.")
    shape = hidden_states.shape
    partial = ffn.experts(
        hidden_states=hidden_states,
        router_logits=hidden_states,
        input_ids=input_ids,
        defer_tp_reduction=True,
    )
    return partial.view(shape)


def reduce_engram_lookup(own: Ownership, partial: Any, max_band_bytes: int) -> Any:
    """Reduce each feature band to its row owner in original column order."""
    import torch

    row_bytes_per_column = partial.shape[0] * partial.element_size()
    columns = partial.shape[-1]
    step = max(8, max_band_bytes // (8 * row_bytes_per_column) * 8)
    if not own.band_engram or step >= columns:
        own.engram_reduce_scatters += 1
        return own.reduce_scatter(partial)
    bands = []
    for start in range(0, columns, step):
        bands.append(own.reduce_scatter(partial[:, start:start + step]))
    own.engram_reduce_scatters += len(bands)
    return torch.cat(bands, dim=-1)


def owned_engram(
    engram: Any, residual: Any, hash_ids: Any, token_mask: Any, own: Ownership
) -> Any:
    """``Engram.forward`` for owned residual rows.

    Each staged lookup element has one table owner. With a replicated projection,
    reduce-scatter assembles the embedding rows directly on their token owners.
    A column-parallel projection gathers its output columns across ranks, so
    every rank supplies the same full rows and keeps its slice after projection.
    """
    module = sys.modules[type(engram).__module__]
    if (
        engram.embed_tokens.table_memory == "disk"
        and (
            not engram._disk_prepared
            or engram._disk_prepared_tokens < hash_ids.shape[0]
        )
    ):
        raise RuntimeError("Disk Engram output is not prepared")
    partial = engram.staged_rows[: hash_ids.shape[0]]
    if type(engram.wkv).__name__ == "ReplicatedLinear":
        rows = reduce_engram_lookup(own, partial, 80 * 1024**2)
        kv = engram.wkv(rows)
    else:
        rows = module.tensor_model_parallel_all_reduce(partial)
        kv = own.local(engram.wkv(rows))
    state = residual.flatten(1)
    out = state.new_empty(state.shape)
    mask = None if token_mask is None else own.local(token_mask)
    module.hyperconnection.run_engram_mix(
        state,
        kv,
        engram.norm_weights,
        eps=engram.eps,
        plan=engram.mix_plans[mask is not None],
        out=out,
        token_mask=mask,
    )
    own.engram_layers += 1
    return out.view_as(residual)


def owned_layer_forward(
    layer: Any,
    own: Ownership,
    index: int,
    x: Any,
    positions: Any,
    input_ids: Any,
    pre_mix: Any = None,
    post_mix: Any = None,
    res_mix: Any = None,
    residual: Any = None,
    engram_hashes: Any = None,
    engram_mask: Any = None,
    ced_indices: Any = None,
):
    """``DeepseekV4DecoderLayer.forward`` for a layer at or before the boundary.

    Inputs and outputs follow the image's layer: before the boundary the
    returned ``x``, ``residual``, mixes and ``ffn_pre`` hold owned rows; the
    boundary layer returns decoder (compacted) rows on every rank, which is
    what the image's next layer expects.
    """
    if index != len(own.layers_seen):
        raise RuntimeError(f"row ownership reached layer {index} out of order")
    own.layers_seen.append(index)
    boundary = index == own.boundary
    if boundary != (ced_indices is not None):
        raise RuntimeError("CED indices must arrive exactly at the boundary layer")

    if residual is None:
        # First layer: the embedding output holds every row on every rank.
        x = own.local(x)
        if pre_mix is not None:
            pre_mix = own.local(pre_mix)
    previous_output = x if residual is not None else None
    apply_engram = layer.engram is not None and engram_hashes is not None
    if residual is None:
        residual = x
    elif apply_engram:
        residual = layer._b12x_mhc.post(x, residual, post_mix, res_mix)
        previous_output = None
    if apply_engram:
        if residual.ndim == 2:
            residual = residual[:, None, :].expand(-1, layer.hc_mult, -1).contiguous()
        residual = owned_engram(
            layer.engram,
            residual,
            engram_hashes[:, layer.engram.layer_hash_index],
            engram_mask,
            own,
        )
    fn = layer.hc_attn_fn_broadcast if residual.ndim == 2 else layer.hc_attn_fn
    residual, post_mix, res_mix, x, attn_pre = layer._b12x_mhc.pre(
        residual,
        fn,
        layer.hc_attn_scale,
        layer.hc_attn_base,
        layer.attn_norm.weight,
        pre_mix,
        previous_output=previous_output,
        previous_post=post_mix if previous_output is not None else None,
        previous_comb=res_mix if previous_output is not None else None,
    )
    local_attention = (
        not boundary and own.local_queries
        and getattr(layer.attn, "_ds41_query_replica", None) is not None
    )
    local_input = (
        local_attention and own.gather_kv_latents
        and (
            (layer.attn.compressor is None and layer.attn.indexer is None)
            or (
                own.gather_compressor_latents and layer.attn.compressor is not None
                and layer.attn.indexer is not None
            )
        )
    )
    if not local_input:
        x = own.all_gather(x)

    if boundary:
        # The global KV write reads every encoder row; the decoder keeps only
        # the replay rows, so the residual stream is compacted, not gathered.
        global_kv_ready = layer.attn.prepare_global_kv(positions, x)
        x = own.gather_rows(x, ced_indices)
        residual = own.compact(residual, ced_indices)
        post_mix = own.compact(post_mix, ced_indices)
        res_mix = own.compact(res_mix, ced_indices)
        attn_pre = own.compact(attn_pre, ced_indices)
        positions = own.gather_rows(positions, ced_indices)
        if input_ids is not None:
            input_ids = own.gather_rows(input_ids, ced_indices)
        x = layer.attn(positions, x, None, global_kv_ready=global_kv_ready)
        residual, post_mix, res_mix, x, ffn_pre = layer._b12x_mhc.post_pre(
            x,
            residual,
            post_mix,
            res_mix,
            layer.hc_ffn_fn,
            layer.hc_ffn_scale,
            layer.hc_ffn_base,
            layer.ffn_norm.weight,
            attn_pre,
        )
        x = layer.ffn(x, input_ids)
        return x, residual, post_mix, res_mix, ffn_pre

    with own.unreduced_attention():
        partial = layer.attn(positions, x, None, global_kv_ready=None)
    if local_attention:
        x = partial
        own.local_attention_layers += 1
    else:
        x = own.reduce_scatter(partial)
    residual, post_mix, res_mix, x, ffn_pre = layer._b12x_mhc.post_pre(
        x,
        residual,
        post_mix,
        res_mix,
        layer.hc_ffn_fn,
        layer.hc_ffn_scale,
        layer.hc_ffn_base,
        layer.ffn_norm.weight,
        attn_pre,
    )
    x = own.all_gather(x)
    # The image casts the reduced MoE output to the input dtype; sum first.
    STATE.route_owner = own if own.compact_routes else None
    try:
        partial = moe_unreduced(layer.ffn, x, input_ids)
    finally:
        STATE.route_owner = None
    x = own.reduce_scatter(partial).to(x.dtype)
    return x, residual, post_mix, res_mix, ffn_pre
