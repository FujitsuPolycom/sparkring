# SPDX-License-Identifier: Apache-2.0
"""CPU cross-checks of allocator, GDN and PLE checkpoint boundaries.

Production metadata methods and scalar allocation predicates execute with CPU
tensors. GPU kernels and distributed cache restoration remain separate gates.
"""

import __future__

import ast
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
import triton

ROOT = Path(os.environ["SPARKRING_TEST_SOURCE_ROOT"])


def execute(nodes, namespace, label):
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
            label,
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    return namespace


def tree(relative):
    return ast.parse((ROOT / relative).read_text(encoding="utf-8"))


def method(relative, name, function):
    cls = next(
        n for n in tree(relative).body if isinstance(n, ast.ClassDef) and n.name == name
    )
    return next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == function
    )


@pytest.fixture
def metadata_api():
    namespace = dict(
        torch=torch,
        triton=triton,
        SimpleNamespace=NS,
        PIN_MEMORY=False,
        cdiv=lambda x, y: (x + y - 1) // y,
        NULL_BLOCK_ID=0,
    )
    helpers = [
        n
        for n in tree("vllm/v1/kv_cache_interface.py").body
        if getattr(n, "name", None)
        in {
            "get_mamba_prefill_checkpoint_position",
            "is_mamba_prefill_checkpoint_valid",
        }
    ]
    execute(helpers, namespace, "checkpoint geometry")
    classes = [
        n
        for n in tree("vllm/v1/attention/backends/b12x_gdn_metadata.py").body
        if isinstance(n, ast.ClassDef)
    ]
    execute(classes, namespace, "mixed GDN")
    return namespace


def allocator_needs(namespace, start, end, block_size, hash_size, eagle):
    class Spec:
        prefill_checkpoint_alignment = 16

    namespace["MambaSpec"] = Spec
    predicate = method(
        "vllm/v1/core/single_type_kv_cache_manager.py",
        "MambaManager",
        "_needs_internal_checkpoint",
    )
    execute([predicate], namespace, "allocator checkpoint admission")
    position = namespace["get_mamba_prefill_checkpoint_position"](end, hash_size, eagle)
    manager = NS(
        kv_cache_spec=Spec(),
        has_prefill_checkpoint_blocks=True,
        block_size=block_size,
        req_to_blocks={"req": []},
        block_pool=NS(hash_block_size=hash_size),
    )
    return namespace["_needs_internal_checkpoint"](
        manager, "req", start, end, position
    ), position


@pytest.mark.parametrize(
    "block_size,hash_size", [(512, 512), (2048, 512), (2048, 2048)]
)
@pytest.mark.parametrize("eagle", [False, True])
@pytest.mark.parametrize(
    "start,end",
    [(0, 8192), (0, 7870), (2048, 8194), (5120, 8192), (512, 1024), (17, 8192)],
)
def test_mixed_gdn_matches_allocator_and_ple_hash_boundary(
    metadata_api, block_size, hash_size, eagle, start, end
):
    namespace = metadata_api
    needed, position = allocator_needs(
        namespace, start, end, block_size, hash_size, eagle
    )
    common = NS(
        num_reqs=1,
        query_start_loc_cpu=torch.tensor([0, end - start]),
        query_start_loc=torch.tensor([0, end - start]),
        seq_lens=torch.tensor([end]),
        seq_lens_cpu_upper_bound=torch.tensor([end]),
        block_table_tensor=torch.arange(1, 33).reshape(1, 32),
    )
    mixed = namespace["B12xGdnMixedMetadata"](
        max_tokens=8194, max_seqs=1, state_columns=4, device=torch.device("cpu")
    )
    mixed.stage(
        common,
        torch.tensor([[31, 30, 29, 28]]),
        None,
        None,
        checkpoint_block_size=block_size,
        checkpoint_hash_block_size=hash_size,
        checkpoint_alignment=16,
        drop_eagle_checkpoint_block=eagle,
    )
    offset = int(mixed.checkpoint.checkpoint_offsets[0])
    assert (offset > 0) is needed
    if needed:
        column = (end + block_size - 1) // block_size - 2
        assert offset == position - start
        assert int(mixed.checkpoint_columns[0]) == column
        assert int(mixed.checkpoint.state_indices[0]) == column + 1

    build = method(
        "vllm/v1/attention/backends/gdn_attn.py", "GDNAttentionMetadataBuilder", "build"
    )
    checkpoint_branch = next(
        n
        for n in build.body
        if isinstance(n, ast.If)
        and "self.kv_cache_spec.num_prefill_checkpoint_blocks > 0"
        in ast.unparse(n.test)
    )
    namespace.update(
        self=NS(
            kv_cache_spec=NS(
                num_prefill_checkpoint_blocks=1,
                prefill_checkpoint_alignment=16,
                block_size=block_size,
            ),
            vllm_config=NS(cache_config=NS(mamba_cache_mode="align")),
            hash_block_size=hash_size,
            drop_eagle_checkpoint_block=eagle,
        ),
        num_prefills=1,
        num_decodes=0,
        spec_sequence_masks_cpu=None,
        m=common,
        query_start_loc_cpu=common.query_start_loc_cpu,
        query_start_loc=common.query_start_loc,
        prefill_checkpoint=None,
        GDNPrefillCheckpointMetadata=NS,
        COALESCED_CHECKPOINT_CAPACITY=4,
        async_tensor_h2d=lambda values, **kwargs: torch.tensor(values, **kwargs),
    )
    execute([checkpoint_branch], namespace, "generic GDN checkpoint builder")
    checkpoint = namespace["prefill_checkpoint"]
    assert (checkpoint is not None) is needed
    if needed:
        assert int(checkpoint.checkpoint_offsets[0]) == offset
        assert int(checkpoint.state_indices[0]) == int(
            mixed.checkpoint.state_indices[0]
        )

    class Base:
        def build(self, *args, **kwargs):
            return NS(num_reqs=1, num_decodes=0, num_prefills=1)

    cls = next(
        n
        for n in tree("vllm/models/qwen4_exp/nvidia/ple_attn.py").body
        if isinstance(n, ast.ClassDef) and n.name == "PLEAttentionMetadataBuilder"
    )
    cls.body = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name in {"build", "_refresh_checkpoints"}
    ]
    namespace.update(
        ShortConvAttentionMetadataBuilder=Base,
        PLEAttentionMetadata=NS,
        _B12X_NULL_STATE_SLOT=-1,
        replace=lambda source, **kwargs: NS(**(source.__dict__ | kwargs)),
    )
    execute([cls], namespace, "PLE builder")
    builder = namespace[cls.name]()
    builder.graph_inputs = NS(stage=lambda *args: None, num_tokens=torch.zeros(1))
    builder.checkpoint_offsets = torch.zeros(1, dtype=torch.int32)
    builder.checkpoint_slots = torch.full((1,), -1, dtype=torch.int64)
    builder.hash_block_size = hash_size
    builder.drop_eagle_checkpoint_block = eagle
    builder.kv_cache_spec = NS(
        block_size=block_size,
        num_prefill_checkpoint_blocks=1,
        prefill_checkpoint_alignment=16,
    )
    ple = builder.build(0, common)
    assert int(ple.checkpoint_offsets[0]) == offset
    assert int(ple.checkpoint_slots[0]) == (
        int(mixed.checkpoint.state_indices[0]) if needed else -1
    )


def test_qwen_model_state_forwards_scheduler_plans_with_and_without_qsa():
    prepare = method(
        "vllm/models/qwen4_exp/nvidia/model_state.py",
        "Qwen4ExpModelState",
        "prepare_attn",
    )
    assert "recurrent_prefill_checkpoint_plans" in [a.arg for a in prepare.args.args]
    forwarding = [
        n
        for n in ast.walk(prepare)
        if isinstance(n, ast.Call)
        and any(k.arg == "recurrent_prefill_checkpoint_plans" for k in n.keywords)
    ]
    assert len(forwarding) == 1
    metadata = [
        n
        for n in ast.walk(prepare)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "Qwen4ExpAttnMetadata"
    ]
    assert len(metadata) == 1
    assert any(
        k.arg == "recurrent_prefill_checkpoint_plans_cpu" for k in metadata[0].keywords
    )


def test_gdn_builder_stages_allocator_hash_and_eagle_policy():
    build = method(
        "vllm/v1/attention/backends/gdn_attn.py", "GDNAttentionMetadataBuilder", "build"
    )
    call = next(
        n
        for n in ast.walk(build)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "stage"
    )
    keywords = {k.arg: ast.unparse(k.value) for k in call.keywords}
    assert keywords["checkpoint_hash_block_size"] == "self.hash_block_size"
    assert keywords["drop_eagle_checkpoint_block"] == "self.drop_eagle_checkpoint_block"
    assert (
        keywords["checkpoint_alignment"]
        == "self.kv_cache_spec.prefill_checkpoint_alignment"
    )


@pytest.mark.parametrize("model_type", ["base", "qwen-qsa", "qwen-no-qsa", "glm"])
def test_packed_scheduler_plan_reaches_common_attention_metadata(model_type):
    namespace = dict(
        torch=torch,
        np=np,
        dataclass=dataclass,
        CUDAGraphMode=NS(FULL="FULL"),
        ModelSpecificAttnMetadata=object,
        DefaultModelState=object,
        COALESCED_CHECKPOINT_CAPACITY=4,
    )
    helpers = [
        n
        for n in tree("vllm/v1/core/recurrent_prefill_checkpoint.py").body
        if getattr(n, "name", None) in {"checkpoint_plan_rows", "validate_plan"}
    ]
    execute(helpers, namespace, "scheduler checkpoint plan validation")
    definitions = [
        n
        for n in tree("vllm/v1/worker/gpu/model_states/mamba_hybrid.py").body
        if isinstance(n, ast.ClassDef)
        and n.name in {"MambaHybridAttnMetadata", "MambaHybridModelState"}
    ]
    definitions[1].body = [
        method(
            "vllm/v1/worker/gpu/model_states/mamba_hybrid.py",
            "MambaHybridModelState",
            "prepare_attn",
        )
    ]
    execute(definitions, namespace, "base Mamba model state")
    for path, names in [
        (
            "vllm/models/qwen4_exp/nvidia/model_state.py",
            ("Qwen4ExpAttnMetadata", "Qwen4ExpModelState"),
        ),
        (
            "vllm/models/glm5next/model_state.py",
            ("Glm5NextAttnMetadata", "Glm5NextModelState"),
        ),
    ]:
        definitions = [
            n
            for n in tree(path).body
            if isinstance(n, ast.ClassDef) and n.name in names
        ]
        definitions[1].body = [method(path, names[1], "prepare_attn")]
        execute(definitions, namespace, path)
    observed = {}

    def build_attn_metadata(**kwargs):
        model = kwargs["model_specific_attn_metadata"]
        observed.update(model.get_extra_common_attn_kwargs(0, kwargs["num_reqs"]))
        return {"gdn": NS(prefill_checkpoint=NS(required_mask=[True]))}

    namespace["build_attn_metadata"] = build_attn_metadata
    cls = namespace[
        {
            "base": "MambaHybridModelState",
            "qwen-qsa": "Qwen4ExpModelState",
            "qwen-no-qsa": "Qwen4ExpModelState",
            "glm": "Glm5NextModelState",
        }[model_type]
    ]
    state = cls()
    state.uses_qsa = model_type == "qwen-qsa"
    state.vllm_config = NS(num_speculative_tokens=0)
    state._align_mode = False
    state.recoverssm = None
    values = np.zeros(1, dtype=bool)
    buffer = NS(
        np=values,
        cpu=torch.from_numpy(values),
        copy_to_gpu=lambda count: torch.from_numpy(values[:count]),
    )
    state.qsa_is_prefilling = state.selector_is_prefilling = buffer
    state._prepare_qsa_state = state._prepare_selector_state = lambda *args: (
        torch.zeros(1),
        torch.zeros(1),
        torch.ones(1),
    )
    batch = NS(
        uniform_decode_graph=False,
        cudagraph_capture=False,
        num_reqs=1,
        num_tokens=1024,
        req_ids=["request"],
        query_start_loc_np=np.array([0, 1024], dtype=np.int32),
        query_start_loc=torch.tensor([0, 1024]),
        num_scheduled_tokens=np.array([1024]),
        num_computed_tokens_np=np.array([5120]),
        seq_lens=torch.tensor([6144]),
        seq_lens_cpu_upper_bound=torch.tensor([6144]),
        is_prefilling_np=np.array([True]),
        dcp_local_seq_lens=None,
        positions=torch.arange(1024),
        prompt_lens=torch.tensor([8192]),
    )
    plan = (5120, 6144, (5632,))
    state.prepare_attn(
        batch,
        "NONE",
        (),
        torch.empty(0),
        [],
        NS(),
        recurrent_prefill_checkpoint_plans={"request": plan},
    )
    assert observed["recurrent_prefill_checkpoint_plans_cpu"] == [plan]
    assert observed["is_prefilling"].tolist() == [True]
