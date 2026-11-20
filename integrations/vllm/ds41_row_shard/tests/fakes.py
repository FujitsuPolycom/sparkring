"""Row-local and full-row stand-ins for the V4.1 decoder layer's components.

The reference path executes the image's own ``DeepseekV4DecoderLayer.forward``
and ``Engram.forward`` (``fixtures/``) on these components; the owned path runs
``core.owned_layer_forward`` on the same components. The stand-ins keep the
properties the real kernels have and the plugin relies on:

* mHC ``pre``/``post_pre``/``post`` are row-local (each output row depends
  only on the same input row), with ``post_pre`` equal to ``pre`` with the
  lagged ``previous_output`` fusion, as ``B12xMHC.post_pre`` is;
* attention mixes rows (a causal running sum), so it needs every row, and
  each rank holds a different head share whose partial is summed over ranks;
* the MoE is row-local per rank but returns a TP partial;
* Engram all-reduces rank-specific staged table rows, then mixes row-locally.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch

import tp_world

FIXTURES = Path(__file__).resolve().parent / "fixtures"
HC = 4


def gather_rows(source, indices):
    """``vllm::dsv41_gather_rows``: gather rows, zero negative or out-of-range ones."""
    valid = (indices >= 0) & (indices < source.shape[0])
    safe = indices.clamp(0, max(source.shape[0] - 1, 0)).long()
    out = source[safe].clone()
    out[~valid] = 0
    return out


def _load(name: str, namespace: dict):
    code = (FIXTURES / name).read_text()
    exec(compile(code, str(FIXTURES / name), "exec"), namespace)
    return namespace["forward"]


# -- Engram: a module of its own, as the plugin looks names up on it -----------
engram_module = types.ModuleType("fake_engram_module")
engram_module.torch = torch
engram_module.tensor_model_parallel_all_reduce = lambda t: tp_world.group().all_reduce(t)


def _run_engram_mix(state, kv, norm_weights, *, eps, plan, out, token_mask=None):
    mixed = state + torch.tanh(kv) * norm_weights + eps
    if token_mask is not None:
        mixed = torch.where(token_mask[:, None], mixed, state)
    out.copy_(mixed)


engram_module.hyperconnection = SimpleNamespace(run_engram_mix=_run_engram_mix)
sys.modules[engram_module.__name__] = engram_module


class ReplicatedLinear:
    """Every rank holds the whole projection."""

    def __init__(self, weight):
        self.weight = weight

    def __call__(self, rows):
        return rows @ self.weight


class ColumnParallelLinear:
    """``gather_output=True``: each rank projects its column shard of the same
    rows, then the shards are all-gathered along the last dimension."""

    gather_output = True

    def __init__(self, weight, world: int, rank: int):
        width = weight.shape[1] // world
        self.shard = weight[:, rank * width : (rank + 1) * width]

    def __call__(self, rows):
        shards = tp_world.group()._exchange(rows @ self.shard)
        return torch.cat(shards, dim=-1)


class FakeEngram:
    def __init__(self, rows: int, hidden: int, hash_index: int, generator, rank: int,
                 shared=None):
        """``generator`` draws this rank's staged table share; ``shared`` draws the
        replicated ``wkv`` and norm weights (every rank holds the same ones)."""
        width = 6
        shared = generator if shared is None else shared
        self.layer_hash_index = hash_index
        self.embed_tokens = SimpleNamespace(table_memory="disk")
        self._disk_prepared = True
        self._disk_prepared_tokens = rows
        # Each rank holds a different share of the table rows (column shards).
        self.staged_rows = torch.randn(rows + 3, width, generator=generator) * (rank + 1)
        weight = torch.randn(width, HC * hidden, generator=shared)
        self.wkv = ReplicatedLinear(weight)
        self.norm_weights = torch.rand(HC * hidden, generator=shared)
        self.eps = 1e-3
        self.mix_plans = {False: "plan-unmasked", True: "plan-masked"}


FakeEngram.__module__ = engram_module.__name__
FakeEngram.forward = _load("engram_forward.py", dict(engram_module.__dict__))
FakeEngram.__call__ = lambda self, *a, **k: self.forward(*a, **k)


# -- mHC -----------------------------------------------------------------------
class FakeMHC:
    """Row-local hyperconnection arithmetic with the B12xMHC call shapes."""

    def __init__(self, generator):
        self.mix = torch.randn(HC, HC, generator=generator) * 0.3

    def post(self, x, residual, post_mix, res_mix):
        x = x.reshape(x.shape[0], -1)
        return torch.einsum("rij,rjh->rih", res_mix, residual) + post_mix[:, :, None] * x[:, None, :]

    def pre(self, residual, fn, scale, base, norm_weight, pre_mix,
            previous_output=None, previous_post=None, previous_comb=None):
        if previous_output is not None:
            residual = self.post(previous_output, residual, previous_post, previous_comb)
        elif residual.ndim == 2:
            residual = residual[:, None, :].expand(-1, HC, -1).contiguous()
        means = residual.mean(-1)  # (rows, HC)
        lag = 0 if pre_mix is None else pre_mix.sum(-1, keepdim=True)
        post_mix = torch.sigmoid(means * scale[0] + base[:HC] + lag)
        res_mix = torch.tanh(means[:, :, None] * self.mix[None] * scale[1])
        attn_pre = torch.softmax(means * scale[2] + fn.sum(), dim=-1)
        x = (attn_pre[:, :, None] * residual).sum(1) * norm_weight
        return residual, post_mix, res_mix, x, attn_pre

    def post_pre(self, x, residual, post_mix, res_mix, fn, scale, base, norm_weight, pre_mix):
        return self.pre(residual, fn, scale, base, norm_weight, pre_mix,
                        previous_output=x, previous_post=post_mix, previous_comb=res_mix)


# -- attention and MoE -----------------------------------------------------------
class FakeAttention:
    def __init__(self, hidden: int, generator, rank: int, reduce):
        self.weight = torch.randn(hidden, hidden, generator=generator) * (0.1 * (rank + 1))
        self.reduce = reduce  # the edited _o_proj's reduction helper
        self.global_kv = []
        self.calls = []

    def prepare_global_kv(self, positions, hidden_states):
        self.global_kv.append(hidden_states.clone())
        return torch.empty(1)

    def __call__(self, positions, x, llama_4_scaling=None, *, global_kv_ready=None):
        self.calls.append((x.shape[0], global_kv_ready is not None))
        mixed = torch.cumsum(x, dim=0) / (1 + torch.arange(x.shape[0], dtype=x.dtype))[:, None]
        local = (mixed + positions[:, None].to(x.dtype) * 1e-2) @ self.weight
        # The image's WO projection returns (rows, hidden, 1).
        return self.reduce(local[:, :, None])


class FakeExperts:
    def __init__(self, hidden: int, generator, rank: int):
        self.weight = torch.randn(hidden, hidden, generator=generator) * (0.2 * (rank + 1))
        self.deferred = 0

    def __call__(self, hidden_states, router_logits, input_ids=None, *, defer_tp_reduction=False):
        partial = torch.tanh(hidden_states @ self.weight)
        if input_ids is not None:
            partial = partial * (1 + input_ids[:, None].to(partial.dtype) * 1e-3)
        if defer_tp_reduction:
            self.deferred += 1
            return partial
        return tp_world.group().all_reduce(partial)


class FakeMoE:
    use_mega_moe = False

    def __init__(self, hidden: int, generator, rank: int):
        self.gate = SimpleNamespace(tid2eid=None, bias_vl=None)
        self.experts = FakeExperts(hidden, generator, rank)

    def __call__(self, x, input_ids=None):
        shape = x.shape
        out = self.experts(hidden_states=x, router_logits=x, input_ids=input_ids)
        return out.to(x.dtype).view(shape)


# -- the decoder layer ---------------------------------------------------------------
class FakeLayer:
    hc_mult = HC

    def __init__(self, hidden, rows, generator, rank, reduce, engram_hash_index=None):
        g = generator
        self._b12x_mhc = FakeMHC(g)
        self.engram = (
            None if engram_hash_index is None
            else FakeEngram(rows, hidden, engram_hash_index, g, rank)
        )
        self.hc_attn_fn = torch.randn(3, generator=g)
        self.hc_attn_fn_broadcast = torch.randn(3, generator=g)
        self.hc_ffn_fn = torch.randn(3, generator=g)
        self.hc_attn_scale = torch.rand(3, generator=g) + 0.5
        self.hc_ffn_scale = torch.rand(3, generator=g) + 0.5
        self.hc_attn_base = torch.randn(3 * HC, generator=g)
        self.hc_ffn_base = torch.randn(3 * HC, generator=g)
        self.attn_norm = SimpleNamespace(weight=torch.rand(hidden, generator=g) + 0.5)
        self.ffn_norm = SimpleNamespace(weight=torch.rand(hidden, generator=g) + 0.5)
        self.attn = FakeAttention(hidden, g, rank, reduce)
        self.ffn = FakeMoE(hidden, g, rank)


FakeLayer.forward = _load("decoder_layer_forward.py", {"torch": torch, "gather_rows": gather_rows})
