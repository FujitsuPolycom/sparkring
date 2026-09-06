"""Exercise extracted installed vLLM lease/statistics/API code without a GPU."""

import ast
import copy
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(os.environ["MTP3_ORIGINAL_SOURCE"])


def compile_nodes(nodes, namespace=None):
    scope = {} if namespace is None else dict(namespace)
    tree = ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes), type_ignores=[]))
    exec(compile(tree, "<extracted-installed-vllm>", "exec"), scope)
    return scope


def extract_class(path, name):
    tree = ast.parse(path.read_text())
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)


def wrapper(name, arguments, statements):
    tree = ast.parse(f"def {name}({arguments}):\n    pass\n")
    tree.body[0].body = statements
    return tree.body[0]


@pytest.fixture
def runtime():
    stats = compile_nodes([extract_class(ROOT / "v1/metrics/stats.py", "PrefillStats")],
                          {"dataclass": dataclass})["PrefillStats"]
    path = Path(os.environ.get("MTP3_SCHEDULER_SOURCE", str(ROOT / "v1/core/sched/scheduler.py")))
    tree = ast.parse(path.read_text())
    candidates = [node for node in ast.walk(tree) if isinstance(node, ast.If)
                  and ast.unparse(node.test) == "request.num_computed_tokens == 0 and self.connector is not None"]
    assert len(candidates) == 1
    setup = ast.parse("request_id = request.request_id").body
    apply_lease = compile_nodes([wrapper("apply_lease", "self, request", setup + candidates)])["apply_lease"]
    processor = ast.parse((ROOT / "v1/engine/output_processor.py").read_text())
    output = [node for node in ast.walk(processor) if isinstance(node, ast.If)
              and ast.unparse(node.test) == "req_state.is_prefilling"]
    assert len(output) == 1
    receive = compile_nodes([wrapper("receive", "req_state, engine_core_output", output)])["receive"]
    serving = ast.parse((ROOT / "entrypoints/openai/chat_completion/serving.py").read_text())
    helper = next(node for node in serving.body if isinstance(node, ast.FunctionDef)
                  and node.name == "_make_prompt_tokens_details")
    details = compile_nodes([helper], {"PromptTokenUsageInfo": SimpleNamespace})[helper.name]
    return SimpleNamespace(Stats=stats, apply=apply_lease, receive=receive, details=details)


def attach(runtime, prompt, lease, *, accepted=True, preempted=0, stats=True):
    events = []
    request = SimpleNamespace(request_id="request", num_computed_tokens=0,
                              num_tokens=prompt, num_prompt_tokens=prompt,
                              num_preemptions=preempted,
                              prefill_stats=runtime.Stats() if stats else None)
    connector = SimpleNamespace(get_shared_prefix_lease_candidate=lambda request: ("lease", lease),
                                shared_prefix_lease_attached=lambda *args: events.append("attached"),
                                shared_prefix_lease_rejected=lambda *args: events.append("rejected"))
    scheduler = SimpleNamespace(connector=connector, kv_cache_manager=SimpleNamespace(
        attach_shared_prefix_lease=lambda *args: lease if accepted else 0,
        get_computed_blocks=lambda request: (None, 0, 0)))
    runtime.apply(scheduler, request)
    return scheduler, request, events


@pytest.mark.parametrize("prompt,lease,expected", [(40000, 32768, 32768), (32768, 32768, 32767)])
def test_attached_gpu_prefix_reaches_api_total_without_external_transfer(runtime, prompt, lease, expected):
    scheduler, request, events = attach(runtime, prompt, lease)
    stats = request.prefill_stats
    assert stats.num_cached_tokens == expected
    assert stats.num_local_cached_tokens == expected
    assert stats.num_external_cached_tokens == 0
    assert stats.num_computed_tokens == prompt - expected
    assert request.num_computed_tokens == expected
    # Reentering scheduling cannot double-count the attached prefix.
    runtime.apply(scheduler, request)
    assert events == ["attached"]
    stats.finalize(prompt)
    state = SimpleNamespace(is_prefilling=True, num_cached_tokens=0, num_cache_creation_tokens=0)
    runtime.receive(state, SimpleNamespace(prefill_stats=stats))
    details = runtime.details(True, state.num_cached_tokens, state.num_cache_creation_tokens, None)
    assert details.cached_tokens == expected
    assert details.created_cache_tokens == prompt - expected


def test_rejected_lease_leaves_normal_lookup_accounting_untouched(runtime):
    _, request, events = attach(runtime, 40000, 32768, accepted=False)
    assert events == ["rejected"]
    assert request.num_computed_tokens == 0
    assert vars(request.prefill_stats) == vars(runtime.Stats())


def test_preemption_does_not_rewrite_first_prefill_stats(runtime):
    _, request, events = attach(runtime, 40000, 32768, preempted=1)
    assert events == ["attached"]
    assert vars(request.prefill_stats) == vars(runtime.Stats())


def test_absent_stats_does_not_change_lease_attachment(runtime):
    _, request, events = attach(runtime, 40000, 32768, stats=False)
    assert request.prefill_stats is None
    assert request.num_computed_tokens == 32768
    assert events == ["attached"]


def test_existing_local_plus_external_total_remains_distinct(runtime):
    stats = runtime.Stats()
    stats.set(num_prompt_tokens=40000, num_local_cached_tokens=8000, num_external_cached_tokens=12000)
    details = runtime.details(True, stats.num_cached_tokens, 0, None)
    assert (stats.num_local_cached_tokens, stats.num_external_cached_tokens) == (8000, 12000)
    assert details.cached_tokens == 20000
