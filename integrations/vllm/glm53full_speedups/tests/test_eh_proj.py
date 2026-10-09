"""The MTP ``eh_proj``: row-parallel weight loading and the all-reduced forward.

Each rank builds the plugin's row-parallel layer through the patched helper
(``eh_proj``), loads the checkpoint tensor, and compares its weight columns
with the checkpoint's. The all-reduced forward must reproduce the image's
replicated ``nn.Linear`` (the reduction order can differ; the loading cannot).
"""

from __future__ import annotations

import pytest
import torch
import speedups_image_env
import glm53full_speedups as plugin

HIDDEN = 8
INPUT = 2 * HIDDEN  # the MTP input is the concatenated hidden states
ROWS = 4

torch.manual_seed(11)


@pytest.fixture(scope="module")
def checkpoint():
    weight = torch.randn(HIDDEN, INPUT)
    x = torch.randn(ROWS, INPUT)
    return weight, x


def test_helper_builds_the_replicated_linear_outside_tp8(monkeypatch):
    import vllm.distributed.parallel_state as parallel_state

    monkeypatch.setattr(parallel_state, "model_parallel_is_initialized", lambda: False)
    helper = plugin.HELPERS[plugin.EH_PROJ_HELPER]
    layer = helper(HIDDEN, "model.mtp", torch.nn.Linear)
    assert type(layer) is torch.nn.Linear
    assert layer.weight.shape == (HIDDEN, INPUT)


def test_every_rank_loads_its_columns_and_reduces_exactly(checkpoint):
    speedups_image_env.build_group()
    weight, x = checkpoint

    def body(rank: int) -> dict:
        helper = plugin.HELPERS[plugin.EH_PROJ_HELPER]
        layer = helper(HIDDEN, "model.mtp", torch.nn.Linear)
        assert layer.weight.shape == (HIDDEN, INPUT // speedups_image_env.TP)
        layer.weight_loader(layer.weight, weight)
        assert torch.equal(layer.weight.data, weight[:, rank * (INPUT // speedups_image_env.TP):
                                                    (rank + 1) * (INPUT // speedups_image_env.TP)]), (
            f"rank {rank}: weight columns differ from the checkpoint's block")
        output = layer(x)
        return {"output": output}

    results = speedups_image_env.run_ranks(body)
    reference = torch.nn.functional.linear(x, weight)
    for rank, result in enumerate(results):
        assert torch.allclose(result["output"], reference, rtol=1e-5, atol=1e-5), (
            f"rank {rank}: reduced output differs from the replicated linear")
