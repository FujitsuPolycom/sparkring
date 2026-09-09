"""GPU-free execution of installed scheduler selection with allocator stubs."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_mtp3_lease_accounting import ROOT, compile_nodes, wrapper


@pytest.fixture
def selection():
    path = Path(os.environ.get("MTP3_SCHEDULER_SOURCE", str(Path(__file__).parent / "mtp3_scheduler_accounted.py")))
    tree = ast.parse(path.read_text())
    lease = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                 and ast.unparse(node.test) == "request.num_computed_tokens == 0 and self.connector is not None")
    ordinary = min((node for node in ast.walk(tree) if isinstance(node, ast.If)
                    and ast.unparse(node.test) == "request.num_computed_tokens == 0"
                    and node.lineno > lease.lineno), key=lambda node: node.lineno)
    # Execute the real selection/lease branch and the real local-lookup assignment.
    ordinary.body = ordinary.body[:2]
    ordinary.orelse = []
    setup = ast.parse("request_id = request.request_id\nlocal_lease_alternative = None\ndid_prefix_cache_lookup = False\nnum_new_local_computed_tokens = 0\nhit_diverged = False").body
    result = ast.parse("return did_prefix_cache_lookup, num_new_local_computed_tokens, hit_diverged").body
    return compile_nodes([wrapper("select", "self, request", setup + [lease, ordinary] + result)])["select"]


def case(selection, *, local, lease=1024, attach=True, stale=False, preempted=False, candidate=True):
    calls = []
    members = {"request", "other-follower"}
    current = [local]
    request = SimpleNamespace(request_id="request", num_computed_tokens=0, num_tokens=2048,
                              num_prompt_tokens=2048, prefill_stats=None,
                              num_preemptions=int(preempted), shared_prefix_boundary=0)

    def lookup(request):
        calls.append("converged_lookup")
        return ("local-blocks", current[0], current[0])

    def ordinary(request):
        calls.append("ordinary_lookup")
        return (*lookup(request), False)

    def attach_lease(*args):
        calls.append("attach")
        if stale:
            current[0] = 0
        return lease if attach else 0

    def reject(*args):
        calls.append("reject")
        members.discard("request")

    connector = SimpleNamespace(get_shared_prefix_lease_candidate=lambda request: ("lease", lease) if candidate else None,
        shared_prefix_lease_attached=lambda *args: calls.append("attached"),
        shared_prefix_lease_rejected=reject)
    scheduler = SimpleNamespace(connector=connector,
        kv_cache_manager=SimpleNamespace(get_computed_blocks=lookup, attach_shared_prefix_lease=attach_lease),
        _get_local_prefix_cache_hit=ordinary)
    result = selection(scheduler, request)
    return result, request, calls, members


def test_strictly_longer_converged_local_hit_skips_attachment_and_reuses_lookup(selection):
    result, request, calls, members = case(selection, local=1536)
    assert result == (True, 1536, False)
    assert calls == ["converged_lookup", "reject"]
    assert request.num_computed_tokens == 0  # Normal scheduler allocation owns adoption.
    assert members == {"other-follower"}


@pytest.mark.parametrize("local", [0, 512, 1024])
def test_equal_or_shorter_local_hit_keeps_lease(selection, local):
    result, request, calls, members = case(selection, local=local)
    assert not result[0]
    assert request.num_computed_tokens == 1024
    assert calls == ["converged_lookup", "attach", "attached"]
    assert members == {"request", "other-follower"}


def test_rejected_lease_returns_to_normal_lookup(selection):
    result, request, calls, members = case(selection, local=512, attach=False)
    assert result == (True, 512, False)
    assert calls == ["converged_lookup", "attach", "reject", "ordinary_lookup", "converged_lookup"]
    assert request.num_computed_tokens == 0
    assert members == {"other-follower"}


def test_failed_attachment_does_not_reuse_potentially_stale_probe_blocks(selection):
    result, _, calls, _ = case(selection, local=512, attach=False, stale=True)
    assert result == (True, 0, False)
    assert calls[-2:] == ["ordinary_lookup", "converged_lookup"]


def test_preempted_request_still_uses_normal_allocation_for_a_longer_local_hit(selection):
    result, request, calls, _ = case(selection, local=1536, preempted=True)
    assert result == (True, 1536, False)
    assert request.num_computed_tokens == 0
    assert "attach" not in calls


def test_no_lease_performs_only_the_ordinary_lookup(selection):
    result, _, calls, _ = case(selection, local=1536, candidate=False)
    assert result == (True, 1536, False)
    assert calls == ["ordinary_lookup", "converged_lookup"]


@pytest.mark.skipif(not os.environ.get("SPARKCACHE_SOURCE_ROOT"), reason="optional companion SparkCache source integration")
def test_actual_connector_decline_preserves_verified_lease_and_other_followers(selection, tmp_path):
    import sys
    sys.path.insert(0, os.environ["SPARKCACHE_SOURCE_ROOT"])
    from sparkcache import test_spark_context_cache_connector as fixtures

    helper = fixtures.AsyncRestoreTests()
    connector = helper._cohort_connector(tmp_path)
    tokens = list(range(1600))
    digest = helper._offer(connector, tokens)
    leader = SimpleNamespace(request_id="leader", prompt_token_ids=tokens)
    assert connector.get_num_new_matched_tokens(leader, 0) == (1024, True)
    connector.update_state_after_alloc(leader, helper._blocks_stub(), 1024)
    connector.build_connector_meta(fixtures._empty_scheduler_output())
    connector.update_connector_output(SimpleNamespace(invalid_block_ids=set(), finished_recving={"leader"}))
    assert connector.shared_prefix_lease_published("leader", digest)
    other = SimpleNamespace(request_id="other", prompt_token_ids=tokens)
    assert connector.get_shared_prefix_lease_candidate(other) == (digest, 1024)
    request = SimpleNamespace(request_id="prefer-local", prompt_token_ids=tokens,
        num_computed_tokens=0, num_tokens=len(tokens), num_prompt_tokens=len(tokens),
        num_preemptions=0, prefill_stats=None, shared_prefix_boundary=0)
    scheduler = SimpleNamespace(connector=connector, kv_cache_manager=SimpleNamespace(
        get_computed_blocks=lambda request: ("local-blocks", 1280, 1280),
        attach_shared_prefix_lease=lambda *args: pytest.fail("short lease was attached")),
        _get_local_prefix_cache_hit=lambda request: pytest.fail("lookup repeated"))
    try:
        assert selection(scheduler, request) == (True, 1280, False)
        assert "prefer-local" not in connector._restore_flight_followers
        assert "other" in connector._restore_flight_followers
        assert connector._restore_flights[digest].lease_published
        assert connector.get_shared_prefix_lease_candidate(other) == (digest, 1024)
    finally:
        connector.shutdown()


def test_speculative_backoff_is_applied_before_comparing_with_lease(selection):
    # Execute the installed full-attention finder with equal hash/page units.
    # In this geometry resolve_block_hashes is its identity branch.
    import itertools
    from collections.abc import Sequence
    from types import SimpleNamespace

    tree = ast.parse((ROOT / "v1/core/single_type_kv_cache_manager.py").read_text())
    original = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FullAttentionManager")
    method = next(node for node in original.body if isinstance(node, ast.FunctionDef) and node.name == "find_longest_cache_hit")
    shell = ast.parse("class FullAttentionManager:\n    supports_fine_grained_hash_lookup = True\n").body[0]
    shell.body.append(method)
    future = ast.parse("from __future__ import annotations").body[0]
    scope = compile_nodes([future, shell], dict(FullAttentionSpec=SimpleNamespace,
        ChunkedLocalAttentionSpec=type("ChunkedLocalAttentionSpec", (), {}),
        resolve_block_hashes=lambda values, *args, **kwargs: values,
        itertools=itertools, Sequence=Sequence, cdiv=lambda a, b: (a + b - 1) // b))
    pool = SimpleNamespace(hash_block_size=256, get_cached_block=lambda block_hash, groups: [block_hash])
    _, backed_off = scope["FullAttentionManager"].find_longest_cache_hit(
        [1, 2, 3, 4, 5], max_length=1280, kv_cache_group_ids=[0], block_pool=pool,
        kv_cache_spec=SimpleNamespace(block_size=256), drop_eagle_block=True, alignment_tokens=256)
    assert backed_off == 1024  # Raw 1280-token evidence is not a reusable 1280-token hit.
    _, request, calls, _ = case(selection, local=backed_off, lease=1024)
    assert request.num_computed_tokens == 1024
    assert "attached" in calls
