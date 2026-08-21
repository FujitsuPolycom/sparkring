from __future__ import annotations

import pytest

from .manager_roles import (
    FailClosedRoleAssignmentAdapter,
    ManagerRole,
    ManagerRoleRegistry,
    RoleAssignmentHook,
)
from .phase_timing import PhaseKind
from .vllm_adapter import AdapterValidationError, source_sha256


class Manager:
    pass


def test_query_length_never_infers_semantic_role() -> None:
    registry = ManagerRoleRegistry()
    first_q6 = Manager()
    second_q6 = Manager()
    assert (
        registry.register(first_q6, decode_query_len=6).role
        is ManagerRole.UNKNOWN
    )
    assert (
        registry.register(second_q6, decode_query_len=6).role
        is ManagerRole.UNKNOWN
    )


def test_two_q6_managers_are_explicitly_target_and_draft() -> None:
    registry = ManagerRoleRegistry()
    verify = Manager()
    draft = Manager()
    registry.register(
        verify,
        decode_query_len=6,
        role=ManagerRole.TARGET_VERIFY,
    )
    registry.register(
        draft,
        decode_query_len=6,
        role=ManagerRole.DRAFT_BLOCK,
    )
    assert (
        registry.graph_descriptor(
            verify, graph_method="run_fullgraph"
        ).kind
        is PhaseKind.TARGET_FULL_GRAPH
    )
    assert (
        registry.graph_descriptor(
            draft, graph_method="run_fullgraph"
        ).kind
        is PhaseKind.DRAFT_MULTISTEP_GRAPH
    )


def test_explicit_mtp_step_promotes_only_q1_to_draft() -> None:
    registry = ManagerRoleRegistry()
    q1 = Manager()
    q6 = Manager()
    registry.register(
        q1, decode_query_len=5, role=ManagerRole.DRAFT_BLOCK
    )
    registry.register(
        q6, decode_query_len=6, role=ManagerRole.TARGET_VERIFY
    )
    descriptor = registry.graph_descriptor(
        q1, graph_method="run_fullgraph", draft_step=3
    )
    assert descriptor.kind is PhaseKind.DRAFT_MULTISTEP_GRAPH
    assert "step=3" in descriptor.name
    with pytest.raises(RuntimeError, match="non-draft"):
        registry.graph_descriptor(
            q6, graph_method="run_fullgraph", draft_step=0
        )


def test_registration_is_idempotent_and_bounded() -> None:
    registry = ManagerRoleRegistry(maximum_managers=1)
    manager = Manager()
    first = registry.register(
        manager,
        decode_query_len=5,
        role=ManagerRole.TARGET_VERIFY,
    )
    assert (
        registry.register(
            manager,
            decode_query_len=5,
            role=ManagerRole.TARGET_VERIFY,
        )
        == first
    )
    with pytest.raises(RuntimeError, match="capacity"):
        registry.register(Manager(), decode_query_len=1)


def test_piecewise_verify_is_other_graph_not_full() -> None:
    registry = ManagerRoleRegistry()
    manager = Manager()
    registry.register(
        manager,
        decode_query_len=5,
        role=ManagerRole.TARGET_VERIFY,
    )
    descriptor = registry.graph_descriptor(
        manager, graph_method="run_pw_graph"
    )
    assert descriptor.kind is PhaseKind.OTHER_GRAPH
    assert "run_pw_graph" in descriptor.name


class Speculator:
    def __init__(self) -> None:
        self.block_size = 6
        self.forward_cudagraph_manager: Manager | None = None

    def init_cudagraph_manager(self) -> None:
        self.forward_cudagraph_manager = Manager()


def _draft_role_hook(source_hash: str | None = None) -> RoleAssignmentHook:
    return RoleAssignmentHook(
        owner=Speculator,
        method_name="init_cudagraph_manager",
        expected_source_sha256=(
            source_hash
            or source_sha256(Speculator.init_cudagraph_manager)
        ),
        role=ManagerRole.DRAFT_BLOCK,
        manager_after_call=lambda instance, args, kwargs, result: (
            instance.forward_cudagraph_manager
        ),
        decode_query_len_after_call=(
            lambda instance, args, kwargs, result: instance.block_size
        ),
    )


def test_source_pinned_speculator_init_marks_q6_as_draft() -> None:
    registry = ManagerRoleRegistry()
    original = Speculator.init_cudagraph_manager
    adapter = FailClosedRoleAssignmentAdapter(
        registry, (_draft_role_hook(),)
    )
    adapter.install()
    try:
        speculator = Speculator()
        speculator.init_cudagraph_manager()
        assert speculator.forward_cudagraph_manager is not None
        identity = registry.identity(speculator.forward_cudagraph_manager)
        assert identity.decode_query_len == 6
        assert identity.role is ManagerRole.DRAFT_BLOCK
    finally:
        adapter.uninstall()
    assert Speculator.init_cudagraph_manager is original


def test_role_adapter_source_mismatch_mutates_nothing() -> None:
    registry = ManagerRoleRegistry()
    original = Speculator.init_cudagraph_manager
    adapter = FailClosedRoleAssignmentAdapter(
        registry, (_draft_role_hook("0" * 64),)
    )
    with pytest.raises(AdapterValidationError, match="source mismatch"):
        adapter.install()
    assert Speculator.init_cudagraph_manager is original
