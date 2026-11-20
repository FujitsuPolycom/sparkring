"""Full-head, row-owned prefill on the pinned V4.1 B12X engine."""

from __future__ import annotations

import copy
import functools
import sys
from dataclasses import replace
from types import SimpleNamespace

from .core import STATE
from .source_edit import edit_function


def _full_weight(group, tensor, axis=0):
    """Assemble checkpoint FP8 values or scales without changing their bits."""
    import torch

    source = tensor.detach().movedim(axis, 0).contiguous()
    bits = group.all_gather(source.view(torch.uint8), dim=0)
    return bits.view(tensor.dtype).movedim(0, axis).contiguous()


def _build_replica(attention, module):
    from vllm.distributed import get_tp_group

    group = get_tp_group()
    if attention.capacity % group.world_size:
        raise RuntimeError("full-head prefill requires equal per-rank capacity")
    shadow = copy.copy(attention)
    # Keep the original layer's parameters, submodules and buffers separate.
    shadow._parameters = attention._parameters.copy()
    shadow._modules = attention._modules.copy()
    shadow._buffers = attention._buffers.copy()
    shadow.prefix = attention.prefix + ".owned_prefill"
    shadow._owner = attention._owner
    shadow.n_local_heads *= group.world_size
    shadow.n_local_groups *= group.world_size
    shadow.capacity //= group.world_size
    shadow.indexer = None
    shadow.compressor = None
    shadow._wo_resolver = module.B12xPlanResolver("V4.1 row-owned WO projection")

    q = module.torch.nn.Module()
    q.weight = _full_weight(group, attention.wq_b.weight)
    q.weight_scale_inv = _full_weight(group, attention.wq_b.weight_scale_inv)
    q.quant_method = attention.wq_b.quant_method
    q.quant_method.process_weights_after_loading(q)
    from b12x.gemm import block_fp8_linear

    q.b12x_capacities = (shadow.capacity,)
    q.b12x_plans = (block_fp8_linear.plan(block_fp8_linear.Caps(
        device=q.weight.device, max_tokens=shadow.capacity,
        in_features=q.weight.shape[1], out_features=q.weight.shape[0],
        block_size=(32, 32),
    )),)
    shadow.wq_b = q
    shadow.attn_sink = module.torch.nn.Parameter(
        group.all_gather(attention.attn_sink.detach(), dim=0), requires_grad=False
    )

    def projection(linear, axis):
        return SimpleNamespace(
            weight=_full_weight(group, linear.weight, axis),
            weight_scale_inv=_full_weight(group, linear.weight_scale_inv, axis),
        )

    # The raw checkpoint WO weights are available during the model's post-load
    # hook. Pack the full matrices once, then release the gathered checkpoint views.
    shadow._modules.pop("wo_a")
    shadow._modules.pop("wo_b")
    shadow.wo_a = projection(attention.wo_a, 0)
    shadow.wo_b = projection(attention.wo_b, 1)
    module.DeepseekV4Attention._ds41_original_wo_setup(shadow)
    del shadow.wo_a, shadow.wo_b
    shadow._helpers = module._AttentionHelpers(shadow)
    return shadow


def _replica(attention):
    own = STATE.ownership
    if own is not None and own.local_queries and STATE.defer_attention:
        return getattr(attention, "_ds41_query_replica", None)
    return None


def _project(attention, qr, rows):
    shadow = _replica(attention)
    if shadow is None:
        return attention.wq_b(qr).view(rows, attention.n_local_heads, 512)
    own = STATE.ownership
    owned_qr = qr if rows == own.geometry.owned else own.local(qr)
    q = shadow.wq_b.quant_method.apply(shadow.wq_b, owned_qr)
    return q.view(own.geometry.owned, shadow.n_local_heads, 512)


def _query_positions(attention, positions):
    if _replica(attention) is None:
        return positions
    return STATE.ownership.local(positions)


def _query_rotary_plan(attention):
    owner = _replica(attention)
    return (owner if owner is not None else attention)._helper_plan("q")


def _index_rows(attention, q):
    if _replica(attention) is not None:
        return STATE.ownership.geometry.rows
    return q.shape[0]


def _gather_kv(attention, kv):
    if _replica(attention) is None:
        return kv
    own = STATE.ownership
    if own.gather_kv_latents and kv.shape[0] == own.geometry.owned:
        own.kv_latent_gathers += 1
        return own.all_gather(kv)
    return kv


def _gather_projection_records(own, tensors):
    """Gather mixed-precision projections in row order without dtype conversion."""
    import torch

    records = [tensor.contiguous().view(torch.uint8) for tensor in tensors]
    joined = own.all_gather(torch.cat(records, dim=-1))
    return tuple(
        record.contiguous().view(tensor.dtype)
        for record, tensor in zip(
            joined.split([record.shape[-1] for record in records], dim=-1),
            tensors, strict=True,
        )
    )


def _project_inputs(attention, hidden_states):
    qr_kv, _ = attention.fused_wqa_wkv(hidden_states)
    if _replica(attention) is None:
        return qr_kv
    own = STATE.ownership
    compressor = attention.compressor
    if (
        own.gather_compressor_latents and compressor is not None
        and attention.indexer is not None and hidden_states.shape[0] == own.geometry.owned
    ):
        compressor._project(hidden_states)
        buffers = (compressor._values, compressor._gates)[:compressor.compress_ratio]
        weights = attention.indexer.weights_proj(hidden_states)
        qr_kv, weights, *compressor_rows = _gather_projection_records(
            own, (qr_kv, weights, *(buffer[:own.geometry.owned] for buffer in buffers))
        )
        for buffer, values in zip(buffers, compressor_rows, strict=True):
            buffer[:own.geometry.rows].copy_(values)
        STATE.compressor_inputs = (compressor, weights, own.geometry.rows)
        own.compressor_latent_gathers += 1
    return qr_kv


def _index_weights(attention, hidden_states):
    inputs = STATE.compressor_inputs
    if inputs is not None and inputs[0] is attention.compressor:
        return inputs[1]
    return attention.indexer.weights_proj(hidden_states)


def _compress_rows(compressor, hidden_states):
    inputs = STATE.compressor_inputs
    if inputs is not None and inputs[0] is compressor:
        return inputs[2]
    compressor._project(hidden_states)
    return hidden_states.shape[0]


def _global_rows(attention, hidden_states):
    inputs = STATE.compressor_inputs
    if inputs is not None and inputs[0] is attention.compressor:
        return inputs[2]
    return hidden_states.shape[0]


def _attend(attention, q, output, swa, main, owner):
    shadow = _replica(attention)
    if shadow is None:
        return False
    from b12x.preparation import require_prepared

    module = sys.modules["vllm.models.deepseek_v4_1.attention"]
    geom = STATE.ownership.geometry
    rows = slice(geom.start, geom.start + geom.owned)
    plan = shadow._attention_plan("extend")
    state = require_prepared(plan, "attention.compressed_sparse_mla")
    specs = shadow._attention_workspace_specs["extend"]
    arena = module.current_workspace_manager().get_simultaneous(
        *specs,
        *((spec.shape, spec.dtype) for spec in state.scratch_plan.scratch_specs()),
    )
    metadata = arena[:len(specs)]
    attention._write_decode_metadata(metadata, swa, main, geom.owned, geom.start)
    swa_indices, swa_lengths, top_lengths = metadata[:3]
    indexed = {}
    if main is not None:
        indexed = dict(
            indexed_indices=owner.topk_indices_buffer[rows],
            indexed_lengths=top_lengths[:geom.owned],
            indexed_page_table=metadata[3][:geom.owned],
        )
    binding = module.mla.bind(
        plan, scratch=arena[len(specs):], q=q, swa_indices=swa_indices[:geom.owned],
        swa_lengths=swa_lengths[:geom.owned], **indexed,
    )
    module.mla.run(
        binding=binding, swa_k_cache=attention.swa_cache_layer.kv_cache,
        indexed_k_cache=attention._owner().kv_cache if main is not None else None,
        swa_page_size=attention.swa_cache_layer.block_size,
        indexed_page_size=attention._main_page, sm_scale=512**-0.5,
        attn_sink=shadow.attn_sink, out=output, cache_format="deepseek_v41",
    )
    return True


def _output(attention, output, positions, is_prefill):
    shadow = _replica(attention)
    if shadow is None:
        return attention._o_proj(output, positions, is_prefill=is_prefill)
    return shadow._o_proj(output, STATE.ownership.local(positions), is_prefill=True)


# Both paths write KV and select indices over the complete batch. The full-head
# plan then reads only the query rows owned by this rank.
MQA_EDITS = (
    ("        rows = q.shape[0]\n", "        rows = _ds41_index_rows(self, q)\n"),
    (
        "        # Reuse the shared arena only after indexing completes. Keep full-batch\n"
        "        # metadata beside, not overlapping, the native attention scratch.\n",
        "        if _ds41_attend_owned(self, q, output, swa, main, owner):\n"
        "            return\n",
    ),
)
MQA_HELPERS = {"_ds41_index_rows": _index_rows, "_ds41_attend_owned": _attend}

_FORWARD_EDITS = (
    (
        "        rows = hidden_states.shape[0]\n"
        "        qr_kv, _ = self.fused_wqa_wkv(hidden_states)\n",
        "        qr_kv = _ds41_project_inputs(self, hidden_states)\n"
        "        rows = qr_kv.shape[0]\n",
    ),
    (
        "                weights = self.indexer.weights_proj(hidden_states)\n",
        "                weights = _ds41_index_weights(self, hidden_states)\n",
    ),
    (
        "        qr, kv = self.q_norm(qr), self.kv_norm(kv)\n",
        "        qr, kv = self.q_norm(qr), self.kv_norm(_ds41_gather_kv(self, kv))\n",
    ),
    (
        "        q = self.wq_b(qr).view(rows, self.n_local_heads, 512)\n"
        "        l2_prefetch.issue(self._l2pf_wo, rows)\n"
        "        q = _rotated(\n"
        "            q,\n"
        "            positions,\n"
        "            self.rotary_emb.cos_sin_cache,\n"
        "            plan=self._helper_plan(\"q\"),\n"
        "        )\n",
        "        q = _ds41_project_query(self, qr, rows)\n"
        "        l2_prefetch.issue(self._l2pf_wo, rows)\n"
        "        q = _rotated(\n"
        "            q,\n"
        "            _ds41_query_positions(self, positions),\n"
        "            self.rotary_emb.cos_sin_cache,\n"
        "            plan=_ds41_query_rotary_plan(self),\n"
        "        )\n",
    ),
    (
        "        return self._o_proj(output, positions, is_prefill=is_prefill)\n",
        "        return _ds41_project_output(self, output, positions, is_prefill)\n",
    ),
)

_METADATA_EDITS = (
    (
        "    def _write_decode_metadata(self, buffers, swa, main, rows):\n",
        "    def _write_decode_metadata(self, buffers, swa, main, rows, row_offset=0):\n",
    ),
    (
        "            0,\n            swa.block_table.stride(0),\n",
        "            row_offset,\n            swa.block_table.stride(0),\n",
    ),
    (
        "                0,\n                main.block_table.stride(0),\n",
        "                row_offset,\n                main.block_table.stride(0),\n",
    ),
)


def install(module, digest, marker):
    cls = module.DeepseekV4Attention
    if "_ds41_original_wo_setup" in cls.__dict__:
        return
    edit_function(
        module, "DeepseekV4Attention._forward", digest, _FORWARD_EDITS,
        {"_ds41_project_query": _project, "_ds41_project_output": _output,
         "_ds41_gather_kv": _gather_kv, "_ds41_query_positions": _query_positions,
         "_ds41_query_rotary_plan": _query_rotary_plan, "_ds41_project_inputs": _project_inputs,
         "_ds41_index_weights": _index_weights}, marker,
    )
    edit_function(module, "DeepseekV4Attention._write_decode_metadata", digest,
                  _METADATA_EDITS, {}, marker)
    cls._write_decode_metadata.__defaults__ = (0,)
    setup = cls.setup_wo_projection
    provider = cls.get_b12x_preparation_units
    cls._ds41_original_wo_setup = setup

    @functools.wraps(setup)
    def setup_with_replica(self):
        setup(self)
        if not self.is_ced_decoder and not self.is_draft:
            object.__setattr__(self, "_ds41_query_replica", _build_replica(self, module))

    @functools.wraps(provider)
    def units_with_replica(self, layer, workload):
        units = provider(self, layer, workload)
        shadow = getattr(self, "_ds41_query_replica", None)
        if shadow is None:
            return units
        local_workload = replace(
            workload, max_tokens=shadow.capacity, token_counts=(shadow.capacity,),
            fixed_token_counts=(),
        )
        if workload.stage == "weights":
            q = shadow.wq_b
            return (
                *units,
                *q.quant_method.get_b12x_preparation_units(q, local_workload),
                *shadow._helpers.get_b12x_preparation_units(shadow, local_workload),
            )
        if workload.stage == "state":
            return (*units, *provider(shadow, shadow, local_workload))
        return units

    cls.setup_wo_projection = setup_with_replica
    cls.get_b12x_preparation_units = units_with_replica


def install_compressor(attention_module, compressor_module, attention_pin, compressor_pin, marker):
    edit_function(
        attention_module, "DeepseekV4Attention._prepare_global_kv", attention_pin,
        (("        rows = hidden_states.shape[0]\n", "        rows = _ds41_global_rows(self, hidden_states)\n"),),
        {"_ds41_global_rows": _global_rows}, marker, decorator="eager_break_during_capture",
    )
    edit_function(
        compressor_module, "DeepseekCompressor.forward", compressor_pin,
        (("        rows = hidden_states.shape[0]\n        self._project(hidden_states)\n",
          "        rows = _ds41_compress_rows(self, hidden_states)\n\n"),),
        {"_ds41_compress_rows": _compress_rows}, marker,
    )
