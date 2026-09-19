# SPDX-License-Identifier: Apache-2.0
"""CPU execution of KDA preparation ownership and fused-beta layout contracts.

The selected adapter methods execute with CPU tensors. Kernel dispatch and
device admission are stubs; these checks do not establish GPU arithmetic or
asynchronous-stream safety.
"""

import __future__
import ast
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import weakref

import pytest
import torch


@pytest.fixture
def layer_type(monkeypatch):
    source = Path(os.environ["SPARKRING_TEST_SOURCE_ROOT"])
    path = source / "vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                 and node.name == "KimiGatedDeltaNetAttention")
    names = {
        "_initialize_b12x_kda_decode", "_initialize_b12x_kda_prefill",
        "_b12x_kda_decode_probes", "_b12x_kda_decode_call",
        "_b12x_kda_prefill_call", "_scratch_spec", "_benchmark_values",
        "_b12x_trial_tensor",
    }
    methods = [node for node in owner.body if isinstance(node, ast.FunctionDef)
               and node.name in names]
    # The probe factory is absent in the failing retained implementation.
    assert {node.name for node in methods} >= names - {"_b12x_kda_decode_probes"}
    cls = ast.ClassDef(name="Layer", bases=[ast.Attribute(value=ast.Name(id="torch", ctx=ast.Load()),
                       attr="nn", ctx=ast.Load())], keywords=[], body=methods, decorator_list=[])
    cls.bases = [ast.Attribute(value=cls.bases[0], attr="Module", ctx=ast.Load())]
    namespace = dict(torch=torch, NULL_BLOCK_ID=0,
                     current_platform=NS(current_device=lambda: "cpu", is_cuda=lambda: True),
                     get_b12x_gdn_decode=lambda: NS(bind_kda=None, run_kda=None,
                                                   is_supported=lambda _: True),
                     get_b12x_kda_prefill=lambda: object())
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    exec(compile(module, str(path), "exec", flags=__future__.annotations.compiler_flag), namespace)
    checkpoint = ModuleType("vllm.v1.core.recurrent_prefill_checkpoint")
    checkpoint.COALESCED_CHECKPOINT_CAPACITY = 4
    checkpoint.validate_coalescing_config = lambda _: False
    monkeypatch.setitem(sys.modules, checkpoint.__name__, checkpoint)
    preparation = ModuleType("b12x.preparation")
    preparation.PreparedCall = lambda **kwargs: NS(**({"owners": ()} | kwargs))
    monkeypatch.setitem(sys.modules, preparation.__name__, preparation)
    return namespace["Layer"]


def configured(layer_type):
    layer = layer_type()
    layer.enable_b12x_kda_decode = True
    layer.kda_prefill_backend = "b12x"
    layer.head_dim, layer.local_num_heads = 128, 2
    layer.local_projection_size = 256
    layer.use_full_rank_gate = True
    layer.in_proj_qkvgfab = NS(output_size_per_partition=1154)
    layer.model_config = NS(dtype=torch.bfloat16)
    layer.num_spec = 3
    layer.gate_lower_bound = -5.0
    layer.get_state_dtype = lambda: (torch.bfloat16, torch.float32)
    layer.A_log = torch.zeros(2)
    layer.dt_bias = torch.zeros(2, 128)
    layer.o_norm = NS(weight=torch.ones(128), eps=1e-6)
    layer.kv_cache = (torch.empty(1), torch.randn(8, 2, 128, 128))
    config = NS(scheduler_config=NS(max_num_seqs=32, max_num_batched_tokens=64))
    return layer, config


@pytest.mark.parametrize("kind", ["decode", "prefill"])
@pytest.mark.parametrize("benchmark", [False, True])
def test_preparation_buffers_live_only_until_call_release(layer_type, kind, benchmark):
    layer, config = configured(layer_type)
    getattr(layer, "_initialize_b12x_kda_" + kind)(config)
    assert sum(t.numel() * t.element_size() for t in layer.buffers()) < 2048
    references = []
    observed = []

    def bind(**tensors):
        names = ("scratch", "mixed_qkv", "raw_g", "raw_beta", "z", "output") if kind == "decode" else (
            "scratch", "q", "k", "v", "raw_g", "raw_beta", "output")
        references.extend(weakref.ref(tensors[name]) for name in names)
        observed.append({name: (tuple(tensors[name].shape), tensors[name].stride()) for name in names})
        return NS(**tensors)

    def run(binding, **kwargs):
        binding.output.copy_(binding.z if kind == "decode" else binding.q)
        assert binding.num_tokens.item() == (1 if kind == "decode" else 64)

    state = NS(layout=NS(scratch_specs=lambda: [NS(shape=(128,), dtype=torch.uint8)]),
               bind=bind, bind_kda=bind, run=run)
    call = getattr(layer, "_b12x_kda_" + kind + "_call")(state, benchmark=benchmark)
    assert call.owners == (), "Published plans must not retain synthetic activation owners"
    call.produce()
    call.run()
    assert all(reference() is not None for reference in references)
    if kind == "decode":
        assert observed[0]["mixed_qkv"][0] == (128, 768)
        assert observed[0]["raw_beta"] == ((128, 2), (1154, 1))
    else:
        assert observed[0]["q"][0] == (64, 2, 128)
        assert observed[0]["raw_beta"] == ((64, 2), (2, 1))
    del call
    assert all(reference() is None for reference in references)
