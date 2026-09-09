"""Execute attribution seams extracted from the transformed scheduler source."""

import ast
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tarfile

import pytest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("cache_attribution_patch", HERE / "patch_scheduler.py")
PATCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATCH)
SOURCE = HERE.parent / "checkpoints/payload-by-sha" / PATCH.BEFORE_SHA256 / "scheduler.py"


@pytest.fixture(autouse=True, params=["fresh_prompt", "continuation"])
def scheduler_source(request, monkeypatch, tmp_path):
    if request.param == "continuation":
        with tarfile.open(HERE.parent / "continuation/source.tar.gz") as archive:
            data = archive.extractfile("vllm/v1/core/sched/scheduler.py").read()
        source = tmp_path / "continuation-scheduler.py"
        source.write_bytes(data)
        monkeypatch.setitem(globals(), "SOURCE", source)


def runtime():
    tree = ast.parse(PATCH.transform(SOURCE.read_bytes()))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Scheduler")
    methods = {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}
    chosen = [node for name, node in methods.items() if name.startswith("_sparkcache_")]
    chosen.extend(methods[name] for name in ("_update_waiting_for_remote_kv", "_preempt_request", "_free_request"))
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                              ast.ClassDef(name="Harness", bases=[], keywords=[], body=chosen, decorator_list=[])], type_ignores=[])
    namespace = {"RequestStatus": SimpleNamespace(RUNNING="running", PREEMPTED="preempted")}
    exec(compile(ast.fix_missing_locations(module), "scheduler-seams", "exec"), namespace)
    return namespace["Harness"], methods


def execute_statements(statements, values):
    # The scheduler's continue statements retain their real control flow.
    body = [ast.For(target=ast.Name(id="_once", ctx=ast.Store()),
                    iter=ast.Tuple(elts=[ast.Constant(1)], ctx=ast.Load()),
                    body=statements, orelse=[])]
    module = ast.Module(body=body, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "scheduler-statements", "exec"), values)


def request(**overrides):
    values = dict(request_id="request", num_computed_tokens=768, num_prompt_tokens=1000,
                  num_tokens=1000, num_preemptions=0, num_in_flight_tokens=32,
                  num_stale_output_tokens=0, drop_stale_output=False,
                  is_finished=lambda: False)
    values.update(overrides)
    return SimpleNamespace(**values)


def scheduler(req):
    cls, methods = runtime()
    events = []
    obj = cls()
    obj.requests = {req.request_id: req}
    obj.connector = SimpleNamespace(request_cache_events_enabled=True,
        record_request_cache_event=lambda req, event, **fields: events.append((event, fields)))
    return obj, events, methods


def test_transform_exact_source_and_idempotence(tmp_path):
    target = tmp_path / "scheduler.py"
    target.write_bytes(SOURCE.read_bytes())
    expected = PATCH.SOURCE_TRANSFORMS[hashlib.sha256(SOURCE.read_bytes()).hexdigest()]
    PATCH.apply(target)
    assert hashlib.sha256(target.read_bytes()).hexdigest() == expected
    assert PATCH.apply(target)["after_sha256"] == expected
    target.write_bytes(target.read_bytes() + b"# unsupported\n")
    with pytest.raises(ValueError, match="preimage"):
        PATCH.apply(target)


@pytest.mark.parametrize("local,external,lease,lookup,expected", [
    (767, 0, False, True, 767),  # Retained partial tail, not rounded connector argument.
    (512, 488, False, True, 512),
    (1024, 512, False, True, 1000),
    (512, 0, True, False, 768),
    (0, 0, False, False, None),  # Resume after restore is not another admission.
])
def test_admission_uses_adopted_prefix_not_lookup_offer(local, external, lease, lookup, expected):
    req = request()
    obj, events, methods = scheduler(req)
    statement = next(node for node in ast.walk(methods["schedule"])
                     if isinstance(node, ast.If) and ast.unparse(node.test) == "did_prefix_cache_lookup or cache_trace_lease_attached")
    execute_statements([statement], dict(self=obj, request=req,
        did_prefix_cache_lookup=lookup, cache_trace_lease_attached=lease,
        num_new_local_computed_tokens=local, num_external_computed_tokens=external))
    if expected is None:
        assert events == []
    else:
        assert events == [("admitted", dict(local_tokens=expected, external_tokens=min(external,1000-expected),
            lease_attached=lease, source="gpu_lease" if lease else "prefix_lookup", preemptions=0))]


def process_output(obj, methods, output, req, failed=False):
    loop = next(node for node in ast.walk(methods["update_from_output"])
                if isinstance(node, ast.For) and ast.unparse(node.target) == "(req_id, num_tokens_scheduled)")
    # Execute the real failure, abort and stale-output checks through the hook.
    end = next(i for i, node in enumerate(loop.body)
               if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
               and isinstance(node.value.func, ast.Attribute)
               and node.value.func.attr == "_sparkcache_complete_prompt_step")
    execute_statements(loop.body[:end + 1], dict(self=obj, scheduler_output=output,
        req_id=req.request_id, num_tokens_scheduled=32,
        failed_kv_load_req_ids={req.request_id} if failed else set()))


def test_completed_ranges_survive_async_counter_advance_and_exclude_decode():
    req = request(num_computed_tokens=992)
    obj, events, methods = scheduler(req)
    output = SimpleNamespace(num_scheduled_tokens={req.request_id: 32})
    obj._sparkcache_capture_prompt_steps(output)
    req.num_computed_tokens = 1100
    process_output(obj, methods, output, req)
    assert events == [("prompt_step_completed", dict(start_token=992, end_token=1000,
                                                    preemptions=0, stale=False))]
    obj._sparkcache_complete_prompt_step(output, req, False)
    assert len(events) == 1


@pytest.mark.parametrize("failure", ["invalid_restore", "abort", "stale", "stale_drop", "preempted"])
def test_failed_aborted_and_stale_output_cannot_credit_prompt_work(failure):
    req = request()
    obj, events, methods = scheduler(req)
    output = SimpleNamespace(num_scheduled_tokens={req.request_id: 32})
    obj._sparkcache_capture_prompt_steps(output)
    if failure == "abort":
        req.is_finished = lambda: True
    if failure in ("stale", "stale_drop"):
        req.num_stale_output_tokens = 32
        req.drop_stale_output = failure == "stale_drop"
    if failure == "preempted":
        req.num_preemptions = 1
    process_output(obj, methods, output, req, failed=failure == "invalid_restore")
    assert events == []


def test_first_decode_output_can_commit_admitted_reuse_without_prompt_compute():
    req = request(num_computed_tokens=1024, num_preemptions=1)
    obj, events, methods = scheduler(req)
    output = SimpleNamespace(num_scheduled_tokens={req.request_id: 32})
    obj._sparkcache_capture_prompt_steps(output)
    process_output(obj, methods, output, req)
    assert events[0][1] == dict(start_token=1000, end_token=1000, preemptions=1, stale=False)


@pytest.mark.parametrize("failed,valid,expected", [(False, 1000, 999), (True, 0, 0), (True, 512, 512)])
def test_restore_finalization_reports_effective_prefix_after_clamp_or_failure(failed, valid, expected):
    req = request(num_computed_tokens=valid)
    obj, events, _ = scheduler(req)
    obj.failed_recving_kv_req_ids = {req.request_id} if failed else set()
    obj.finished_recving_kv_req_ids = {req.request_id}
    obj.kv_cache_manager = SimpleNamespace(cache_blocks=lambda *args: None, free=lambda *args: None)
    obj.needs_kv_cache_zeroing = False
    obj._update_waiting_for_remote_kv(req)
    assert events == [("restore_finalized", dict(success=not failed,
                                               valid_prefix_tokens=expected, preemptions=0))]
    assert not obj.finished_recving_kv_req_ids


def test_unknown_or_disabled_connector_needs_no_accounting_metadata():
    req = request()
    obj, events, _ = scheduler(req)
    obj.connector = SimpleNamespace()
    output = SimpleNamespace(num_scheduled_tokens={req.request_id: 32})
    obj._sparkcache_capture_prompt_steps(output)
    obj._sparkcache_record_event(req, "admitted")
    assert not hasattr(output, "_sparkcache_prompt_steps")
    assert events == []


def test_preemption_and_terminal_hooks_keep_attempt_generation_and_status():
    req = request(status="running", spec_token_ids=[], num_output_placeholders=0)
    obj, events, _ = scheduler(req)
    obj._free_request_blocks = lambda req: None
    obj.encoder_cache_manager = SimpleNamespace(free=lambda req: None)
    obj._inflight_prefills = SimpleNamespace(discard=lambda req: None)
    obj.log_stats = False
    obj.waiting = SimpleNamespace(prepend_request=lambda req: None)
    obj.reset_preempted_req_ids = set()
    obj._preempt_request(req, 1.0)
    assert req.num_computed_tokens == 0
    assert events == [("preempted", {"preemptions": 1})]
    req.status = SimpleNamespace(name="FINISHED_ABORTED")
    req.is_finished = lambda: True
    obj._connector_finished = lambda req: (False, None)
    obj.ec_connector = None
    obj.finished_req_ids = set()
    obj.finished_req_ids_dict = None
    obj._free_blocks = lambda req: None
    obj._free_request(req)
    assert events[-1] == ("finished", {"status": "FINISHED_ABORTED", "preemptions": 1})


def test_lease_attribution_survives_allocation_deferral_until_accepted_output():
    req = request(num_computed_tokens=0, prefill_stats=None, has_encoder_inputs=False)
    obj, events, methods = scheduler(req)
    schedule_nodes = list(ast.walk(methods["schedule"]))
    initialize = next(node for node in schedule_nodes if isinstance(node, ast.Assign)
                      and ast.unparse(node.targets[0]) == "cache_trace_lease_attached")
    attach = next(node for node in schedule_nodes if isinstance(node, ast.If)
                  and ast.unparse(node.test) == "attached_tokens")
    allocate = next(node for node in schedule_nodes if isinstance(node, ast.If)
                    and ast.unparse(node.test) == "new_blocks is None")
    admit = next(node for node in schedule_nodes if isinstance(node, ast.If)
                 and ast.unparse(node.test) == "did_prefix_cache_lookup or cache_trace_lease_attached")
    values = dict(self=obj, request=req, request_id=req.request_id, lease_key="lease",
                  attached_tokens=768, did_prefix_cache_lookup=False,
                  num_new_local_computed_tokens=0, num_external_computed_tokens=0,
                  new_blocks=None)
    execute_statements([initialize, attach, allocate, admit], values)
    assert req.num_computed_tokens == 768
    assert req._sparkcache_pending_lease_generation == 0
    assert events == []
    values["new_blocks"] = object()
    execute_statements([initialize, allocate, admit], values)
    assert events[0] == ("admitted", dict(local_tokens=768, external_tokens=0,
                                        lease_attached=True, source="gpu_lease", preemptions=0))
    assert req._sparkcache_pending_lease_generation is None
    execute_statements([initialize, allocate, admit], values)
    assert len(events) == 1
    output = SimpleNamespace(num_scheduled_tokens={req.request_id: 32})
    obj._sparkcache_capture_prompt_steps(output)
    process_output(obj, methods, output, req)
    assert events[-1] == ("prompt_step_completed", dict(start_token=768, end_token=800,
                                                       preemptions=0, stale=False))


def test_ordinary_decode_avoids_repeated_empty_prompt_events():
    req = request(num_computed_tokens=1000)
    obj, events, methods = scheduler(req)
    first = SimpleNamespace(num_scheduled_tokens={req.request_id: 32})
    queued = SimpleNamespace(num_scheduled_tokens={req.request_id: 32})
    obj._sparkcache_capture_prompt_steps(first)
    obj._sparkcache_capture_prompt_steps(queued)
    process_output(obj, methods, first, req)
    obj._sparkcache_complete_prompt_step(queued, req, False)
    assert len(events) == 1
    following = SimpleNamespace(num_scheduled_tokens={req.request_id: 32})
    obj._sparkcache_capture_prompt_steps(following)
    assert following._sparkcache_prompt_steps == {}
    req.num_preemptions = 1
    obj._sparkcache_capture_prompt_steps(following)
    assert following._sparkcache_prompt_steps[req.request_id] == (1000, 1000, 1)
