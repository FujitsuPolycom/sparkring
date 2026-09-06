"""Execute scheduler observations against a companion SparkCache token ledger.

Set SPARKCACHE_SOURCE to a SparkCache checkout containing request_attribution.py.
The companion check is optional; an explicitly selected invalid checkout fails.
"""

import ast
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from test_patch_scheduler import execute_statements, process_output, request, runtime
from test_patch_scheduler import scheduler_source  # noqa: F401


@pytest.fixture
def ledger_type():
    checkout = os.environ.get("SPARKCACHE_SOURCE")
    if not checkout:
        pytest.skip("SPARKCACHE_SOURCE selects the companion token ledger")
    source = Path(checkout) / "sparkcache/request_attribution.py"
    if not source.is_file():
        pytest.fail(f"Companion token ledger is missing: {source}")
    name = "sparkcache_companion_request_attribution"
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.RequestAttribution


def coupled(ledger_type, **overrides):
    req = request(**overrides)
    ledger = ledger_type(req.num_prompt_tokens)
    cls, methods = runtime()
    obj = cls()
    obj.requests = {req.request_id: req}
    summaries = []

    def record(req, event, **fields):
        if event == "finished":
            summaries.append(ledger.summary(fields["status"]))
        else:
            ledger.record(event, **fields)

    obj.connector = SimpleNamespace(request_cache_events_enabled=True,
                                    record_request_cache_event=record)
    return obj, req, ledger, methods, summaries


def admit(obj, req, methods, local, external=0):
    statement = next(node for node in ast.walk(methods["schedule"])
                     if isinstance(node, ast.If)
                     and ast.unparse(node.test) == "did_prefix_cache_lookup or cache_trace_lease_attached")
    execute_statements([statement], dict(self=obj, request=req,
        did_prefix_cache_lookup=True, cache_trace_lease_attached=False,
        num_new_local_computed_tokens=local, num_external_computed_tokens=external))


def restore(obj, req, prefix, failed=False):
    req.num_computed_tokens = prefix
    obj.failed_recving_kv_req_ids = {req.request_id} if failed else set()
    obj.finished_recving_kv_req_ids = {req.request_id}
    obj.kv_cache_manager = SimpleNamespace(cache_blocks=lambda *args: None,
                                          free=lambda *args: None)
    obj.needs_kv_cache_zeroing = False
    obj._update_waiting_for_remote_kv(req)


def dispatch(obj, req, start, count):
    req.num_computed_tokens = start
    output = SimpleNamespace(num_scheduled_tokens={req.request_id: count})
    obj._sparkcache_capture_prompt_steps(output)
    return output


def finish(obj, req, summaries, status="FINISHED_STOPPED"):
    req.status = SimpleNamespace(name=status)
    req.is_finished = lambda: True
    obj._inflight_prefills = SimpleNamespace(discard=lambda req: None)
    obj._connector_finished = lambda req: (False, None)
    obj.ec_connector = None
    obj.encoder_cache_manager = SimpleNamespace(free=lambda req: None)
    obj.finished_req_ids = set()
    obj.finished_req_ids_dict = None
    obj._free_blocks = lambda req: None
    obj._free_request(req)
    return summaries[-1]


def test_verified_external_prefix_and_async_output_account_once(ledger_type):
    obj, req, ledger, methods, summaries = coupled(ledger_type)
    admit(obj, req, methods, 256, 744)
    restore(obj, req, 1000)
    assert req.num_computed_tokens == 999
    output = dispatch(obj, req, 999, 32)
    req.num_computed_tokens = 1100
    process_output(obj, methods, output, req)
    process_output(obj, methods, output, req)
    result = finish(obj, req, summaries)
    assert result["attribution_complete"]
    assert (result["local_tokens_reused"], result["external_tokens_reused"],
            result["prompt_tokens_computed"]) == (256, 743, 1)


def test_failed_restore_and_same_generation_local_readmission(ledger_type):
    obj, req, ledger, methods, summaries = coupled(ledger_type)
    admit(obj, req, methods, 0, 1000)
    restore(obj, req, 0, failed=True)
    admit(obj, req, methods, 512)
    output = dispatch(obj, req, 512, 488)
    process_output(obj, methods, output, req)
    result = finish(obj, req, summaries)
    assert result["attribution_complete"]
    assert (result["local_tokens_reused"], result["external_tokens_reused"],
            result["prompt_tokens_computed"]) == (512, 0, 488)


def test_preemption_preserves_accepted_work_and_discards_inflight_output(ledger_type):
    obj, req, ledger, methods, summaries = coupled(
        ledger_type, status="running", spec_token_ids=[], num_output_placeholders=0)
    admit(obj, req, methods, 256)
    output = dispatch(obj, req, 256, 256)
    process_output(obj, methods, output, req)
    stale_output = dispatch(obj, req, 512, 32)
    req.num_in_flight_tokens = 32
    obj._free_request_blocks = lambda req: None
    obj.encoder_cache_manager = SimpleNamespace(free=lambda req: None)
    obj._inflight_prefills = SimpleNamespace(discard=lambda req: None)
    obj.log_stats = False
    obj.waiting = SimpleNamespace(prepend_request=lambda req: None)
    obj.reset_preempted_req_ids = set()
    obj._preempt_request(req, 1.0)
    process_output(obj, methods, stale_output, req)
    admit(obj, req, methods, 0)
    output = dispatch(obj, req, 0, 1000)
    process_output(obj, methods, output, req)
    result = finish(obj, req, summaries)
    assert result["attribution_complete"]
    assert result["preemptions"] == 1
    assert (result["local_tokens_reused"], result["external_tokens_reused"],
            result["prompt_tokens_computed"]) == (256, 0, 1256)


def test_abort_does_not_credit_a_materialized_external_prefix(ledger_type):
    obj, req, ledger, methods, summaries = coupled(ledger_type)
    admit(obj, req, methods, 0, 1000)
    restore(obj, req, 1000)
    output = dispatch(obj, req, 999, 32)
    req.is_finished = lambda: True
    process_output(obj, methods, output, req)
    result = finish(obj, req, summaries, "FINISHED_ABORTED")
    assert not result["attribution_complete"]
    assert result["external_tokens_reused"] == result["prompt_tokens_computed"] == 0


def test_deferred_gpu_lease_admission_is_credited_only_after_execution(ledger_type):
    obj, req, ledger, methods, summaries = coupled(
        ledger_type, num_computed_tokens=0, prefill_stats=None, has_encoder_inputs=False)
    nodes = list(ast.walk(methods["schedule"]))
    initialize = next(node for node in nodes if isinstance(node, ast.Assign)
                      and ast.unparse(node.targets[0]) == "cache_trace_lease_attached")
    attach = next(node for node in nodes if isinstance(node, ast.If)
                  and ast.unparse(node.test) == "attached_tokens")
    allocate = next(node for node in nodes if isinstance(node, ast.If)
                    and ast.unparse(node.test) == "new_blocks is None")
    admission = next(node for node in nodes if isinstance(node, ast.If)
                     and ast.unparse(node.test) == "did_prefix_cache_lookup or cache_trace_lease_attached")
    values = dict(self=obj, request=req, request_id=req.request_id, lease_key="lease",
                  attached_tokens=768, did_prefix_cache_lookup=False,
                  num_new_local_computed_tokens=0, num_external_computed_tokens=0,
                  new_blocks=None)
    execute_statements([initialize, attach, allocate, admission], values)
    assert ledger.attempt is None
    values["new_blocks"] = object()
    execute_statements([initialize, allocate, admission], values)
    assert ledger.local_tokens_reused == ledger.external_tokens_reused == 0
    output = dispatch(obj, req, req.num_computed_tokens, 232)
    process_output(obj, methods, output, req)
    result = finish(obj, req, summaries)
    assert result["attribution_complete"]
    assert (result["local_tokens_reused"], result["external_tokens_reused"],
            result["prompt_tokens_computed"]) == (768, 0, 232)
