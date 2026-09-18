# SPDX-License-Identifier: Apache-2.0
"""CPU admission and row-boundary checks for Qwen4Exp HC ownership.

The production methods execute without native vLLM imports. These tests do not
qualify GPU collective ordering, numerical equality, or serving performance.
"""

import __future__

import ast
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(os.environ["SPARKRING_TEST_SOURCE_ROOT"])


def load_definitions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = [node for node in tree.body if getattr(node, "name", None) in names]
    assert {node.name for node in body} == names
    exec(
        compile(
            ast.Module(body=body, type_ignores=[]),
            str(path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    return namespace


@pytest.fixture
def ownership(monkeypatch):
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    metadata = NS(
        num_prefill_tokens=1024, num_prefills=1, num_decodes=0, num_spec_decodes=0
    )
    context = NS(
        is_dummy_run=False,
        cudagraph_runtime_mode=NS(name="NONE"),
        ubatch_slices=None,
        attn_metadata={"gdn": metadata},
    )
    namespace = dict(
        torch=torch,
        dataclass=dataclass,
        is_forward_context_available=lambda: True,
        get_forward_context=lambda: context,
    )
    load_definitions(
        ROOT / "vllm/models/qwen4_exp/nvidia/hc_prefill.py",
        {"eligible", "RowOwnership"},
        namespace,
    )
    return NS(**namespace, context=context, metadata=metadata)


@pytest.mark.parametrize("rows", [1, 1023, 1025])
def test_nonpartitionable_or_small_batches_stay_replicated(ownership, rows):
    assert not ownership.eligible(NS(hc_prefill_mode="shard"), rows)


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_decodes", 1),
        ("num_spec_decodes", 1),
        ("num_prefill_tokens", 1023),
        ("num_prefills", 0),
        ("num_prefill_tokens", torch.tensor(1024)),
    ],
)
def test_any_nonpure_metadata_rejects_row_ownership(ownership, field, value):
    setattr(ownership.metadata, field, value)
    assert not ownership.eligible(NS(hc_prefill_mode="shard"), 1024)


def test_complete_pure_prefill_is_eligible_but_dummy_and_graph_are_not(ownership):
    model = NS(hc_prefill_mode="shard")
    assert ownership.eligible(model, 1024)
    ownership.context.is_dummy_run = True
    assert not ownership.eligible(model, 1024)
    ownership.context.is_dummy_run = False
    ownership.context.cudagraph_runtime_mode.name = "FULL"
    assert not ownership.eligible(model, 1024)


def test_owner_slices_disjoint_rows_and_keeps_collective_shapes(ownership):
    full = torch.arange(48).reshape(8, 6)
    slices = []
    for rank in range(4):
        comm = NS(all_gather=Mock(), reduce_scatter=Mock())
        owner = ownership.RowOwnership(8, rank, comm)
        local = owner.local(full)
        slices.append(local)
        assert owner.gather(local).shape == full.shape
        assert owner.reduce(full).shape == local.shape
        assert owner.reductions == owner.gathers == 1
        assert comm.reduce_scatter.call_args.args[1].shape == full.shape
        with pytest.raises(ValueError, match="wrong row count"):
            owner.gather(full)
    assert torch.equal(torch.cat(slices), full)


def test_decoder_preserves_ple_residual_and_uses_two_owned_reductions():
    tree = ast.parse((ROOT / "vllm/models/qwen4_exp/nvidia/model.py").read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "Qwen4ExpDecoderLayer"
    )
    method = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward"
    )
    namespace = dict(torch=torch, tensor_model_parallel_all_reduce=Mock())
    exec(
        compile(
            ast.Module(body=[method], type_ignores=[]),
            "decoder",
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    calls = []
    owner = NS(
        gather=lambda x: calls.append(("gather", x.shape[0])) or x.repeat(4, 1),
        local=lambda x: calls.append(("local", x.shape[0])) or x[:2],
        reduce=lambda x: calls.append(("reduce", x.shape[0])) or x[:2],
    )
    hc = NS(
        mix=lambda x: (x, x + 1, x), combine_and_mix=lambda x, out, inj: (x, out, inj)
    )
    layer = NS(
        ple=lambda x, *args: x + 10,
        attn_hyper_connection=hc,
        mlp_hyper_connection=hc,
        layer_type="linear_attention",
        defer_hc_reductions=True,
        linear_attn=lambda hidden_states: hidden_states * 2,
        mlp=lambda x: x + 3,
    )
    hidden, output, injection = namespace["forward"](
        layer,
        torch.ones(2, 3),
        None,
        None,
        torch.arange(8),
        input_ids=torch.arange(8),
        query_start_loc=torch.tensor([0, 8]),
        ngram_context=torch.zeros(1),
        hc_owner=owner,
    )
    assert torch.equal(hidden, torch.full((2, 3), 11.0))
    assert torch.equal(output, torch.full((2, 3), 27.0))
    assert calls == [
        ("gather", 2),
        ("local", 8),
        ("gather", 2),
        ("reduce", 8),
        ("gather", 2),
        ("reduce", 8),
    ]
    namespace["tensor_model_parallel_all_reduce"].assert_not_called()


@pytest.mark.parametrize("eligible", [True, False])
@pytest.mark.parametrize("media", [True, False])
@pytest.mark.parametrize("intermediate", [True, False])
def test_multimodal_entry_preserves_hc_dispatch_and_vision_inputs(
    ownership, eligible, media, intermediate
):
    """Exercise both production forwards, not only the HC eligibility helper."""
    path = ROOT / "vllm/models/qwen4_exp/nvidia/model.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = {}
    gate = Mock(side_effect=ownership.eligible)
    for name in ("Qwen4ExpForCausalLM", "Qwen4ExpForConditionalGeneration"):
        cls = next(n for n in tree.body if getattr(n, "name", None) == name)
        forward = next(n for n in cls.body if getattr(n, "name", None) == "forward")
        namespace = dict(
            hc_prefill=NS(eligible=gate),
            get_pp_group=lambda: NS(is_first_rank=True),
        )
        exec(
            compile(
                ast.Module(body=[forward], type_ignores=[]),
                str(path),
                "exec",
                flags=__future__.annotations.compiler_flag,
            ),
            namespace,
        )
        functions[name] = namespace["forward"]

    calls = []
    output = object()

    class Inner:
        hc_prefill_mode = "shard"

        def forward(self, *args, **kwargs):
            calls.append((args, kwargs))
            return output

        __call__ = forward

    class Language:
        model = Inner()

        def __call__(self, *args, **kwargs):
            return functions["Qwen4ExpForCausalLM"](self, *args, **kwargs)

    ownership.context.is_dummy_run = not eligible
    deepstack = object()
    wrapper = NS(
        language_model=Language(),
        _get_deepstack_input_embeds=Mock(return_value=deepstack),
        _clear_deepstack_input_embeds=Mock(),
    )
    embeds = torch.zeros(1024, 2) if media else None
    ngram = object()
    query = object()
    result = functions["Qwen4ExpForConditionalGeneration"](
        wrapper,
        torch.arange(1024),
        torch.arange(1024),
        object() if intermediate else None,
        inputs_embeds=embeds,
        query_start_loc=query,
        ngram_context=ngram,
    )
    assert result is output
    gate.assert_called_once()
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert kwargs.get("hc_prefill_eager", False) is eligible
    assert kwargs["query_start_loc"] is query
    assert kwargs["ngram_context"] is ngram
    assert kwargs["deepstack_input_embeds"] is (
        deepstack if media and not intermediate else None
    )
    assert args[3] is (None if intermediate else embeds)
    if media and not intermediate:
        wrapper._clear_deepstack_input_embeds.assert_called_once_with(1024)
    else:
        wrapper._clear_deepstack_input_embeds.assert_not_called()


def test_gdn_checkpoint_lease_refuses_non_aligned_execution():
    path = ROOT / "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "QwenGatedDeltaNetAttention"
    )
    cls.decorator_list = []
    cls.body = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "get_kv_cache_spec"
    ]
    from dataclasses import replace

    @dataclass
    class Spec:
        num_prefill_checkpoint_blocks: int = 0
        prefill_checkpoint_alignment: int | None = None

    class Base:
        def get_kv_cache_spec(self, config):
            return Spec()

    namespace = dict(GatedDeltaNetAttention=Base, MambaSpec=Spec, replace=replace)
    exec(
        compile(
            ast.Module(body=[cls], type_ignores=[]),
            str(path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    layer = namespace[cls.name]()
    layer.prefill_checkpoint_blocks = 1
    layer.gdn_prefill_backend = layer.gdn_decode_kernel = "b12x"
    config = NS(
        use_request_boundary_checkpoints=False,
        cache_config=NS(mamba_cache_mode="align"),
    )
    assert layer.get_kv_cache_spec(config).num_prefill_checkpoint_blocks == 1
    assert layer.get_kv_cache_spec(config).prefill_checkpoint_alignment == 16
    config.use_request_boundary_checkpoints = True
    with pytest.raises(ValueError, match="aligned caching"):
        layer.get_kv_cache_spec(config)
