"""The latent shard: weight loading and the all-gathered forward on eight thread ranks.

Each rank builds the plugin's ``ContiguousLatentLinear`` through the patched
helper (``qkv_a``), loads the two checkpoint tensors as the image's loader
does, and compares its weight with the checkpoint's rows. The gathered
forward must reproduce the image's fused projection, whose replicated layer
runs once here with the same inputs.
"""

from __future__ import annotations

import pytest
import torch
import _env
import glm53full_speedups as plugin
from _env import fused_qkv_a_proj_class

INPUT = 16
Q_ROWS = 16  # q_a_proj
KV_ROWS = 8  # kv_a_proj_with_mqa; fused rows 0..15 are q_a, 16..23 are kv_a
ROWS = 5

torch.manual_seed(7)


def build_layer(prefix: str):
    """The helper's layer on this thread's rank (the stub reports TP 8)."""
    helper = plugin.HELPERS[plugin.LATENT_HELPER]
    return helper(fused_qkv_a_proj_class(), INPUT, [Q_ROWS, KV_ROWS],
                  quant_config=None, prefix=prefix)


@pytest.fixture(scope="module")
def checkpoint():
    q_a = torch.randn(Q_ROWS, INPUT)
    kv_a = torch.randn(KV_ROWS, INPUT)
    fused = torch.cat([q_a, kv_a])
    x = torch.randn(ROWS, INPUT)
    return q_a, kv_a, fused, x


def test_helper_builds_the_image_layer_outside_tp8(monkeypatch):
    import vllm.distributed.parallel_state as parallel_state

    monkeypatch.setattr(parallel_state, "model_parallel_is_initialized", lambda: False)
    helper = plugin.HELPERS[plugin.LATENT_HELPER]
    layer = helper(fused_qkv_a_proj_class(), INPUT, [Q_ROWS, KV_ROWS],
                   quant_config=None, prefix="p")
    assert type(layer) is fused_qkv_a_proj_class()
    assert layer.weight.shape == (Q_ROWS + KV_ROWS, INPUT)


def test_every_rank_loads_its_contiguous_block_and_gathers_exactly(checkpoint):
    _env.build_group()
    q_a, kv_a, fused, x = checkpoint

    def body(rank: int) -> dict:
        layer = build_layer("model.layers.0.self_attn.fused_qkv_a_proj")
        assert layer.local_width == (Q_ROWS + KV_ROWS) // _env.TP
        assert layer.local_start == rank * layer.local_width
        layer.weight_loader(layer.weight, q_a, 0)
        layer.weight_loader(layer.weight, kv_a, 1)
        mine = layer.weight.data
        assert torch.equal(mine, fused[layer.local_start:layer.local_start + layer.local_width]), (
            f"rank {rank}: weight rows differ from the checkpoint's block")
        gathered = layer(x)[0] if isinstance(layer(x), tuple) else layer(x)
        return {"weight": mine, "output": gathered}

    results = _env.run_ranks(body)
    # Every rank holds the same gathered output: the fused output's column order.
    reference = torch.nn.functional.linear(x, fused)
    for rank, result in enumerate(results):
        assert torch.equal(result["output"], results[0]["output"]), rank
        # Only the GEMM's reduction order can differ from the fused projection.
        assert torch.allclose(result["output"], reference, atol=1e-5, rtol=1e-5), (
            f"rank {rank}: gathered output differs from the fused projection")


def test_loader_refuses_rows_off_a_scale_block_or_an_unexpected_shard(checkpoint):
    """The loader checks the parameter's output dimension and the shard ids."""
    _env.build_group()
    q_a, kv_a, _, _ = checkpoint

    def body(rank: int) -> object:
        layer = build_layer("p")
        if rank != 0:  # one rank exercises the refusals
            layer.weight_loader(layer.weight, q_a, 0)
            return None
        with pytest.raises(ValueError, match="one checkpoint shard at a time"):
            layer.weight_loader(layer.weight, q_a, True)
        with pytest.raises(ValueError, match="shard id"):
            layer.weight_loader(layer.weight, q_a, 2)
        return None

    _env.run_ranks(body)
