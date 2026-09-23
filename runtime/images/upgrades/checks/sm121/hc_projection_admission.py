"""Reject simultaneous token-row and projection ownership for Qwen HC.

Runs the production admission function with a deterministic rank vote. It does
not test GPU numerical behavior or collective execution.
"""

import ast
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


@pytest.mark.parametrize("partitions", [1, 4])
@pytest.mark.parametrize("mode", ["off", "control", "shard"])
def test_hc_uses_one_tensor_parallel_ownership_scheme(partitions, mode):
    source = (
        Path(os.environ["SPARKRING_TEST_SOURCE_ROOT"])
        / "vllm/models/qwen4_exp/nvidia/hc_prefill.py"
    )
    function = next(
        node for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "configure"
    )
    dtype = object()

    def vote(values, local, **kwargs):
        values[:] = [local] * len(values)

    namespace = {
        "torch": NS(bfloat16=dtype, distributed=NS(
            is_initialized=lambda: True, all_gather_object=vote)),
        "get_tp_group": lambda: NS(world_size=4, cpu_group=object()),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    model = NS(hyper_connection_workspace=NS(tp_size=partitions))
    config = NS(
        model_config=NS(dtype=dtype),
        parallel_config=NS(
            pipeline_parallel_size=1, data_parallel_size=1,
            decode_context_parallel_size=1, prefill_context_parallel_size=1,
            enable_expert_parallel=False, enable_eplb=False,
            use_sequence_parallel_moe=False, enable_dbo=False,
        ),
    )
    if mode == "shard" and partitions > 1:
        with pytest.raises(RuntimeError, match="projection sharding"):
            namespace["configure"](model, config, mode)
    else:
        namespace["configure"](model, config, mode)
        assert model.hc_prefill_mode == mode
