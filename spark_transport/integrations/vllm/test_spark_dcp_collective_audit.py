"""GPU-free tests for DCP reduce-scatter stock-family accounting."""

from __future__ import annotations

import os
import sys
import types
from unittest.mock import patch

import spark_collective_audit
import spark_dcp_collective_audit as adapter


class _FakeTensor:
    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: str = "torch.bfloat16",
        *,
        is_cuda: bool = True,
        contiguous: bool = True,
    ) -> None:
        self.shape = shape
        self.dtype = dtype
        self.is_cuda = is_cuda
        self._contiguous = contiguous

    def is_contiguous(self) -> bool:
        return self._contiguous


class _FakeCuda:
    def __init__(self) -> None:
        self.capturing = False

    def is_current_stream_capturing(self) -> bool:
        return self.capturing


def _fake_torch() -> types.ModuleType:
    module = types.ModuleType("torch")
    module.cuda = _FakeCuda()
    return module


def _group_type() -> type:
    class FakeGroupCoordinator:
        def __init__(
            self,
            *,
            unique_name: str = "dcp:0",
            world_size: int = 4,
        ) -> None:
            self.unique_name = unique_name
            self.world_size = world_size
            self.calls: list[tuple[object, int]] = []
            self.combine_calls: list[tuple[object, ...]] = []

        def _reduce_scatter_out_place(
            self,
            input_tensor: _FakeTensor,
            dim: int,
        ) -> str:
            self.calls.append((input_tensor, dim))
            return "reference"

    return FakeGroupCoordinator


def _stock_combine(
    cp_attn_out: _FakeTensor,
    cp_attn_lse: _FakeTensor,
    cp_group: object,
    ctx: object = None,
    return_lse: bool = False,
    is_lse_base_on_e: bool = True,
) -> object:
    cp_group.combine_calls.append(
        (
            cp_attn_out,
            cp_attn_lse,
            ctx,
            return_lse,
            is_lse_base_on_e,
        )
    )
    return (
        cp_attn_out,
        cp_attn_lse,
    ) if return_lse else cp_attn_out


def _modules(
    group_type: type,
    torch_module: types.ModuleType,
    *,
    include_combine: bool = True,
) -> dict[str, types.ModuleType]:
    vllm = types.ModuleType("vllm")
    distributed = types.ModuleType("vllm.distributed")
    parallel_state = types.ModuleType("vllm.distributed.parallel_state")
    parallel_state.GroupCoordinator = group_type
    modules = {
        "torch": torch_module,
        "vllm": vllm,
        "vllm.distributed": distributed,
        "vllm.distributed.parallel_state": parallel_state,
    }
    if include_combine:
        v1 = types.ModuleType("vllm.v1")
        attention = types.ModuleType("vllm.v1.attention")
        ops = types.ModuleType("vllm.v1.attention.ops")
        common = types.ModuleType("vllm.v1.attention.ops.common")
        common.cp_lse_ag_out_rs = _stock_combine
        modules.update(
            {
                "vllm.v1": v1,
                "vllm.v1.attention": attention,
                "vllm.v1.attention.ops": ops,
                adapter._COMBINE_TARGET: common,
            }
        )
    return modules


def _remove_combine_finders() -> None:
    sys.meta_path[:] = [
        finder
        for finder in sys.meta_path
        if not isinstance(finder, adapter._CombineFinder)
    ]


def test_dcp_output_reduce_scatter_is_classified_without_execution_change(
) -> None:
    group_type = _group_type()
    torch_module = _fake_torch()
    modules = _modules(group_type, torch_module)
    original = group_type._reduce_scatter_out_place
    adapter._installed = False
    spark_collective_audit._reset_for_tests()

    with (
        patch.dict(
            os.environ,
            {
                "SPARK_TP4_GRAPH_STATUS_PATH": "/tmp/status.json",
                "SPARK_TP4_DCP_COLLECTIVE_AUDIT": "1",
            },
            clear=True,
        ),
        patch.dict(sys.modules, modules),
    ):
        adapter.install()
        group = group_type()
        torch_module.cuda.capturing = True
        tensor = _FakeTensor((40, 64, 512))

        result = group._reduce_scatter_out_place(tensor, 1)

    assert result == "reference"
    assert group.calls == [(tensor, 1)]
    assert group_type._reduce_scatter_out_place is not original
    snapshot = spark_collective_audit.stock_collective_snapshot()
    assert snapshot["capture"] == {
        "dcp_output_reduce_scatter:original": 1,
    }
    assert snapshot["signatures"]["capture"] == [
        {
            "family": "dcp_output_reduce_scatter",
            "reason": "original",
            "count": 1,
            "shape": [40, 64, 512],
            "dtype": "torch.bfloat16",
            "is_cuda": True,
            "contiguous": True,
            "world_size": 4,
            "unique_name": "dcp:0",
        }
    ]


def test_non_dcp_reduce_scatter_is_visible_but_not_mislabeled() -> None:
    group_type = _group_type()
    torch_module = _fake_torch()
    modules = _modules(group_type, torch_module)
    adapter._installed = False
    spark_collective_audit._reset_for_tests()

    with (
        patch.dict(
            os.environ,
            {
                "SPARK_TP4_GRAPH_STATUS_PATH": "/tmp/status.json",
                "SPARK_TP4_DCP_COLLECTIVE_AUDIT": "1",
            },
            clear=True,
        ),
        patch.dict(sys.modules, modules),
    ):
        adapter.install()
        group = group_type(unique_name="tp:0")
        tensor = _FakeTensor((40, 64, 512))

        group._reduce_scatter_out_place(tensor, 1)

    assert spark_collective_audit.stock_collective_snapshot()["eager"] == {
        "group_reduce_scatter:original": 1,
    }


def test_stock_combine_is_audit_only_and_preserves_call_contract() -> None:
    group_type = _group_type()
    torch_module = _fake_torch()
    modules = _modules(group_type, torch_module)
    common = modules[adapter._COMBINE_TARGET]
    original = common.cp_lse_ag_out_rs
    adapter._installed = False
    spark_collective_audit._reset_for_tests()
    _remove_combine_finders()

    with (
        patch.dict(
            os.environ,
            {
                "SPARK_TP4_GRAPH_STATUS_PATH": "/tmp/status.json",
                "SPARK_TP4_DCP_COLLECTIVE_AUDIT": "1",
            },
            clear=True,
        ),
        patch.dict(sys.modules, modules),
    ):
        adapter.install()
        group = group_type()
        output = _FakeTensor((5, 64, 512))
        lse = _FakeTensor((5, 64), "torch.float32")
        ctx = object()
        torch_module.cuda.capturing = True

        result = common.cp_lse_ag_out_rs(
            output,
            lse,
            group,
            ctx=ctx,
            return_lse=True,
            is_lse_base_on_e=True,
        )

    _remove_combine_finders()
    assert common.cp_lse_ag_out_rs is not original
    assert result == (output, lse)
    assert group.combine_calls == [(output, lse, ctx, True, True)]
    assert not hasattr(group, "_spark_tp4_dcp_native")
    snapshot = spark_collective_audit.stock_collective_snapshot()
    assert snapshot["capture"] == {"dcp_combine:original": 1}
    assert snapshot["signatures"]["capture"] == [
        {
            "family": "dcp_combine",
            "reason": "original",
            "count": 1,
            "shape": [5, 64, 512],
            "dtype": "torch.bfloat16",
            "is_cuda": True,
            "contiguous": True,
            "world_size": 4,
            "unique_name": "dcp:0",
        }
    ]


def test_stock_combine_audit_repairs_preimported_alias() -> None:
    group_type = _group_type()
    torch_module = _fake_torch()
    modules = _modules(group_type, torch_module)
    common = modules[adapter._COMBINE_TARGET]
    consumer = types.ModuleType("live.mla_attention")
    consumer.cp_lse_ag_out_rs = common.cp_lse_ag_out_rs
    modules[consumer.__name__] = consumer
    adapter._installed = False
    _remove_combine_finders()

    with (
        patch.dict(
            os.environ,
            {"SPARK_TP4_DCP_COLLECTIVE_AUDIT": "1"},
            clear=True,
        ),
        patch.dict(sys.modules, modules),
    ):
        adapter.install()

    _remove_combine_finders()
    assert consumer.cp_lse_ag_out_rs is common.cp_lse_ag_out_rs


def test_stock_combine_audit_installs_deferred_import_hook() -> None:
    group_type = _group_type()
    torch_module = _fake_torch()
    modules = _modules(group_type, torch_module, include_combine=False)
    adapter._installed = False
    _remove_combine_finders()

    with (
        patch.dict(
            os.environ,
            {"SPARK_TP4_DCP_COLLECTIVE_AUDIT": "1"},
            clear=True,
        ),
        patch.dict(sys.modules, modules),
    ):
        sys.modules.pop(adapter._COMBINE_TARGET, None)
        adapter.install()
        finders = [
            finder
            for finder in sys.meta_path
            if isinstance(finder, adapter._CombineFinder)
        ]

    _remove_combine_finders()
    assert len(finders) == 1


def test_custom_dcp_mode_does_not_install_competing_combine_hook() -> None:
    group_type = _group_type()
    torch_module = _fake_torch()
    modules = _modules(group_type, torch_module)
    common = modules[adapter._COMBINE_TARGET]
    original = common.cp_lse_ag_out_rs
    adapter._installed = False
    _remove_combine_finders()

    with (
        patch.dict(
            os.environ,
            {
                "SPARK_TP4_DCP_COLLECTIVE_AUDIT": "1",
                "VLLM_SPARK_TP4_DCP_MODE": "custom",
            },
            clear=True,
        ),
        patch.dict(sys.modules, modules),
    ):
        adapter.install()
        finders = [
            finder
            for finder in sys.meta_path
            if isinstance(finder, adapter._CombineFinder)
        ]

    _remove_combine_finders()
    assert common.cp_lse_ag_out_rs is original
    assert finders == []


def test_audit_install_is_idempotent_for_both_stock_seams() -> None:
    group_type = _group_type()
    torch_module = _fake_torch()
    modules = _modules(group_type, torch_module)
    common = modules[adapter._COMBINE_TARGET]
    adapter._installed = False
    _remove_combine_finders()

    with (
        patch.dict(
            os.environ,
            {"SPARK_TP4_DCP_COLLECTIVE_AUDIT": "1"},
            clear=True,
        ),
        patch.dict(sys.modules, modules),
    ):
        adapter.install()
        reduce_scatter = group_type._reduce_scatter_out_place
        combine = common.cp_lse_ag_out_rs
        adapter.install()

    _remove_combine_finders()
    assert group_type._reduce_scatter_out_place is reduce_scatter
    assert common.cp_lse_ag_out_rs is combine
