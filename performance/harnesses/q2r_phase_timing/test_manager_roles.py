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


@pytest.mark.parametrize("after_assignment", [False, True])
def test_role_install_interrupt_restores_all_assigned_hooks(monkeypatch, after_assignment):
    import builtins
    from . import manager_roles

    class Owner:
        def first(self):
            return Manager()
        def second(self):
            return Manager()

    originals = [Owner.first, Owner.second]
    hooks = tuple(RoleAssignmentHook(
        Owner, name, source_sha256(getattr(Owner, name)), ManagerRole.TARGET_VERIFY,
        lambda instance, args, kwargs, result: result, lambda *args: 1,
    ) for name in ("first", "second"))
    adapter = FailClosedRoleAssignmentAdapter(ManagerRoleRegistry(), hooks)
    fired = False
    def assign(owner, name, value):
        nonlocal fired
        if owner is Owner and name == "second" and not fired:
            fired = True
            if after_assignment:
                builtins.setattr(owner, name, value)
            raise KeyboardInterrupt("role assignment interrupted")
        builtins.setattr(owner, name, value)
    monkeypatch.setattr(manager_roles, "setattr", assign, raising=False)
    with pytest.raises(KeyboardInterrupt, match="role assignment interrupted"):
        adapter.install()
    assert [Owner.first, Owner.second] == originals


@pytest.mark.parametrize("copy_marker", [False, True])
def test_uninstall_preflights_all_hooks_and_preserves_replacements(copy_marker):
    import functools

    class Owner:
        def first(self):
            return Manager()

        def second(self):
            return Manager()

    originals = [Owner.first, Owner.second]
    hooks = tuple(RoleAssignmentHook(
        Owner, name, source_sha256(getattr(Owner, name)), ManagerRole.TARGET_VERIFY,
        lambda instance, args, kwargs, result: result,
        lambda *args: 1,
    ) for name in ("first", "second"))
    adapter = FailClosedRoleAssignmentAdapter(ManagerRoleRegistry(), hooks)
    adapter.install()
    installed = [Owner.first, Owner.second]

    def replacement(self):
        return installed[0](self)

    if copy_marker:
        replacement = functools.wraps(installed[0])(replacement)
    Owner.first = replacement
    with pytest.raises(AdapterValidationError, match="changed"):
        adapter.uninstall()
    assert Owner.first is replacement
    assert Owner.second is installed[1]
    Owner.first = installed[0]
    adapter.uninstall()
    assert [Owner.first, Owner.second] == originals


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


def test_explicit_draft_role_selects_step_descriptor_independent_of_query_length() -> None:
    registry = ManagerRoleRegistry()
    draft = Manager()
    target = Manager()
    registry.register(
        draft, decode_query_len=5, role=ManagerRole.DRAFT_BLOCK
    )
    registry.register(
        target, decode_query_len=6, role=ManagerRole.TARGET_VERIFY
    )
    descriptor = registry.graph_descriptor(
        draft, graph_method="run_fullgraph", draft_step=3
    )
    assert descriptor.kind is PhaseKind.DRAFT_MULTISTEP_GRAPH
    assert "step=3" in descriptor.name
    with pytest.raises(RuntimeError, match="non-draft"):
        registry.graph_descriptor(
            target, graph_method="run_fullgraph", draft_step=0
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


@pytest.mark.parametrize("value", [True, 1.9, "6", 6])
def test_role_hook_preserves_strict_query_width_type(value):
    registry = ManagerRoleRegistry()
    adapter = FailClosedRoleAssignmentAdapter(registry, (_draft_role_hook(),))
    adapter.install()
    try:
        speculator = Speculator()
        speculator.block_size = value
        if type(value) is int:
            speculator.init_cudagraph_manager()
            assert registry.identity(speculator.forward_cudagraph_manager).decode_query_len == value
        else:
            with pytest.raises(ValueError, match="positive integer"):
                speculator.init_cudagraph_manager()
    finally:
        adapter.uninstall()
