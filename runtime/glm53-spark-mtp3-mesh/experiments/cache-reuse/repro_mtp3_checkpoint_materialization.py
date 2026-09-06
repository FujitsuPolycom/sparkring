"""Trace MTP3 checkpoint scheduling and retention from exact source fixtures.

The fixture manifest identifies the image and source inputs. The checker uses
512-token recurrent/hash pages and 2,048-token DCP4 scheduling alignment.
Controlled scheduling budgets expose produced and retained checkpoint
boundaries without GPU execution or a reconstruction of a live request trace.
"""

import ast
import json
import math
from types import SimpleNamespace

from repro_mtp3_sparse_retention import ROOT, FullAttentionSpec, MambaSpec, execute, load_algorithms


def extracted_method(path, cls_name, method_name):
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == cls_name)
    return next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)


def load_scheduler(path=None):
    path = path or ROOT / "v1/core/sched/scheduler.py"
    shell = ast.parse("class Scheduler:\n    pass\n").body[0]
    shell.body = [extracted_method(path, "Scheduler", name) for name in
                  ("_mamba_block_aligned_split", "_reserve_prefill_lookahead")]
    scope = {}
    execute([shell], scope)
    return scope["Scheduler"]


def partial_hit_eligibility(scheduler, kv_cache_config, path=None):
    """Execute the source fixture's recurrent partial-tail eligibility rule."""
    tree = ast.parse((path or ROOT / "v1/core/sched/scheduler.py").read_text())
    nodes = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
             and any(ast.unparse(target) == "self.mamba_partial_cache_hit" for target in node.targets)]
    assert len(nodes) == 1
    execute(nodes, dict(self=scheduler, kv_cache_config=kv_cache_config, MambaSpec=MambaSpec))
    return scheduler.mamba_partial_cache_hit


def checkpoint_kernel_metadata():
    tree = ast.parse((ROOT / "v1/attention/backends/gdn_attn.py").read_text())
    loops = [node for node in ast.walk(tree) if isinstance(node, ast.For)
             and ast.unparse(node.target) == "row" and ast.unparse(node.iter) == "request_rows"]
    assert len(loops) == 1
    function = ast.parse("def checkpoint(start, end, block_size):\n    pass\n").body[0]
    function.body = ast.parse("all_query_lens=[end-start]\nseq_lens=[end]\nrequest_rows=[0]\ncheckpoint_offsets=[]\ncheckpoint_columns=[]").body
    function.body.append(loops[0])
    function.body.extend(ast.parse("return start+checkpoint_offsets[0] if checkpoint_offsets[0] else None").body)
    scope = {}
    execute([function], scope)
    return scope["checkpoint"]


def main():
    algorithms = load_algorithms(ROOT / "v1/core/kv_cache_utils.py")
    utils = ast.parse((ROOT / "v1/core/kv_cache_utils.py").read_text())
    resolve = next(node for node in utils.body if isinstance(node, ast.FunctionDef)
                   and node.name == "resolve_kv_cache_block_sizes")
    scope = dict(math=math, AttentionSpec=FullAttentionSpec, MambaSpec=MambaSpec)
    execute([resolve], scope)
    recurrent = MambaSpec(512)
    recurrent.mamba_cache_mode = "align"
    cache = SimpleNamespace(block_size=512, prefix_match_unit=None, enable_prefix_caching=True)
    config = SimpleNamespace(cache_config=cache,
                             parallel_config=SimpleNamespace(decode_context_parallel_size=4),
                             kv_transfer_config=object())
    groups = SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=FullAttentionSpec(512)),
                                               SimpleNamespace(kv_cache_spec=recurrent)])
    scheduler_size, hash_size = scope[resolve.name](groups, config)
    assert (scheduler_size, hash_size) == (2048, 512)
    checkpoint = checkpoint_kernel_metadata()
    Scheduler = load_scheduler()
    cases = []
    for label, budgets in (
        ("full_8192_token_budgets", [8192] * 8),
        ("reduced_budget_materializes_predecessor", [8192, 8192, 8192, 7680, 8192, 8192]),
    ):
        scheduler = Scheduler()
        scheduler.cache_config = cache
        scheduler.use_eagle = True
        scheduler.mamba_has_prefill_checkpoint_blocks = True
        scheduler.mamba_partial_cache_hit = True
        scheduler.hash_block_size = hash_size
        scheduler.max_num_scheduled_tokens = 8192
        scheduler.scheduler_config = SimpleNamespace(long_prefill_token_threshold=0)
        # One native MTP module, repeated three times, uses one prefill lookahead.
        scheduler.num_prefill_lookahead = 1
        scheduler._recurrent_publication_boundaries = lambda request: (32768,)
        request = SimpleNamespace(num_prompt_tokens=32789, num_tokens=32789,
                                  num_computed_tokens=0, shared_prefix_boundary=0)
        steps = []
        materialized = set()
        retained = set()
        cached_blocks = 0
        for budget in budgets:
            start = request.num_computed_tokens
            if start == request.num_prompt_tokens:
                break
            count = scheduler._mamba_block_aligned_split(request, min(budget, request.num_tokens-start))
            count = scheduler._reserve_prefill_lookahead(request, start, count)
            assert count > 0
            end = start + count
            internal = checkpoint(start, end, recurrent.block_size)
            produced = {end} if end % recurrent.block_size == 0 else set()
            if internal is not None:
                produced.add(internal)
            materialized.update(produced)
            num_full = end // recurrent.block_size
            mask = algorithms.MambaManager.reachable_block_mask(
                start_block=cached_blocks, end_block=num_full,
                alignment_tokens=scheduler_size, kv_cache_spec=recurrent,
                use_eagle=True, retention_interval=0,
                reachable_boundaries=(request.num_prompt_tokens-1,),
            )
            registered = {(cached_blocks + index + 1) * recurrent.block_size
                          for index, keep in enumerate(mask) if keep}
            retained.update(registered.intersection(materialized))
            steps.append(dict(start=start, end=end, internal_checkpoint=internal,
                              produced=sorted(produced), newly_retained=sorted(registered)))
            cached_blocks = num_full
            request.num_computed_tokens = end
        assert request.num_computed_tokens == request.num_prompt_tokens
        assert retained == {32768}
        cases.append(dict(case=label, steps=steps, materialized=sorted(materialized),
                          retained=sorted(retained), required_predecessor=32256,
                          predecessor_materialized=32256 in materialized,
                          predecessor_retained=32256 in retained))
    assert not cases[0]["predecessor_materialized"]
    assert cases[1]["predecessor_materialized"]
    print(json.dumps(dict(schema="mtp3-installed-checkpoint-trace/v1", gpu_executed=False,
        unique_effective_group_sizes=[2048,512], scheduler_block_size=scheduler_size,
        hash_block_size=hash_size, lookup_alignment=hash_size, retention_alignment=scheduler_size,
        retention_interval=0, num_prefill_checkpoint_blocks=1, cases=cases), sort_keys=True))


if __name__ == "__main__":
    main()
