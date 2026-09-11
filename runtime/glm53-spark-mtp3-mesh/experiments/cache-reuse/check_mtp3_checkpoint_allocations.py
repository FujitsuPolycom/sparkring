"""Reproduce MTP3 aligned checkpoint allocation and cache registration on CPU.

Execute allocation and BlockPool.cache_full_blocks from the declared source
fixtures together with the transformed split/retention methods. Synthetic block
state labels track the running endpoint and GDN internal checkpoint scheduled
for GPU writes. This checks metadata/allocator consistency; it does not execute
GPU kernels or prove those writes completed on hardware.
Both populations start with empty block tables. Fresh cases are initial
prefill; resumed cases replay requests whose total length exceeds their prompt.
"""
from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

from repro_mtp3_checkpoint_materialization import (
    checkpoint_kernel_metadata, extracted_method, load_scheduler,
    partial_hit_eligibility,
)
from repro_mtp3_sparse_retention import MambaSpec, ROOT, execute, load_algorithms

WORK = Path(__file__).resolve().parent
PATCHED = Path(os.environ.get('MTP3_RETENTION_SOURCE', str(WORK / 'mtp3-retention-patched')))
EXPECTED_SCHEDULER = '75efa57e7ff5a77c76714b85e2e4d8e1d7f456d9a9eec6c67ebb11ca382942f9'
EXPECTED_MANAGER = 'd2e35b012e0cf45ab3771f545c35ca48f2a5858549c574a352975607369124e2'
EXPECTED_ALLOCATOR = '10846c4994e7860deab8b42c8bcd3315ddc96d14a478d4012c398418cc17a04c'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extracted_classes():
    allocation = ast.parse('class Allocator:\n pass').body[0]
    allocation.body = [extracted_method(ROOT / 'v1/core/single_type_kv_cache_manager.py',
                                       'MambaManager', method)
                       for method in ('_needs_internal_checkpoint', 'allocate_new_blocks')]
    scope = {'MambaSpec': MambaSpec, 'cdiv': lambda a, b: (a + b - 1) // b}
    execute([allocation], scope)
    algorithms = load_algorithms(ROOT / 'v1/core/kv_cache_utils.py',
        manager_source=PATCHED / 'v1/core/single_type_kv_cache_manager.py')
    registration = ast.parse('class Registrar:\n pass').body[0]
    registration.body = [extracted_method(ROOT / 'v1/core/block_pool.py', 'BlockPool', 'cache_full_blocks')]
    register_scope = {'resolve_block_hashes': algorithms.resolve_block_hashes,
                      'make_block_hash_with_group_id': lambda value, group: (value, group)}
    execute([registration], register_scope)
    return scope['Allocator'], register_scope['Registrar'], algorithms


def run_case(prompt, total, shared, budget, speculative, classes, Scheduler, checkpoint):
    Allocator, Registrar, algorithms = classes
    scheduler = Scheduler()
    scheduler.cache_config = SimpleNamespace(block_size=512)
    scheduler.use_eagle = True
    scheduler.mamba_has_prefill_checkpoint_blocks = True
    scheduler.hash_block_size = 512
    scheduler.block_size = 2048
    scheduler.need_mamba_block_aligned_split = True
    scheduler.kv_cache_manager = SimpleNamespace(coordinator=SimpleNamespace(enable_partial_hash_hits=True))
    scheduler.mamba_partial_cache_hit = partial_hit_eligibility(scheduler,
        SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=MambaSpec(512))]),
        PATCHED / 'v1/core/sched/scheduler.py')
    scheduler.max_num_scheduled_tokens = 8192
    scheduler.scheduler_config = SimpleNamespace(long_prefill_token_threshold=0)
    scheduler.num_prefill_lookahead = 1
    scheduler._recurrent_publication_boundaries = lambda request: ((prompt - 1) // 2048 * 2048,)
    request = SimpleNamespace(request_id='replay', num_prompt_tokens=prompt, num_tokens=total,
                              num_computed_tokens=0, shared_prefix_boundary=shared,
                              block_hashes=list(range(512, total + 1, 512)))
    allocator = Allocator()
    allocator.kv_cache_spec = MambaSpec(512)
    allocator.kv_cache_spec.num_prefill_checkpoint_blocks = 1
    allocator.mamba_cache_mode = 'align'
    allocator.block_size = 512
    allocator.num_speculative_blocks = speculative
    allocator.req_to_blocks = {'replay': []}
    allocator._num_checkpoint_blocks = {}
    allocator._partial_hit_reqs = {}
    allocator._allocated_block_reqs = set()
    allocator.last_state_block_idx = {}
    allocator._null_block = SimpleNamespace(is_null=True, state=None, block_hash=None)
    allocator.block_pool = SimpleNamespace(get_new_blocks=lambda n: [
        SimpleNamespace(is_null=False, state=None, block_hash=None, block_hash_num_tokens=None)
        for _ in range(n)])
    failures, retained, selected_null, steps = [], set(), 0, []
    retired_state_slots = 0
    registrar = Registrar()
    registrar.hash_block_size = 512
    registrar.enable_kv_cache_events = False

    def insert(block_hash, block, *, num_tokens):
        if block.state != num_tokens:
            failures.append({'expected_tokens': num_tokens, 'state_tokens': block.state,
                             'step_start': request.num_computed_tokens})
        else:
            retained.add(num_tokens)
        block.block_hash, block.block_hash_num_tokens = block_hash, num_tokens

    def remove(block):
        block.block_hash, block.block_hash_num_tokens = None, None
        return []

    registrar._insert_block_hash = insert
    registrar._remove_cached_block_hashes = remove
    registrar._emit_block_removed_events = lambda removed: None
    cached = 0
    while request.num_computed_tokens < total:
        start = request.num_computed_tokens
        count = scheduler._mamba_block_aligned_split(request, min(budget, total - start))
        count = scheduler._reserve_prefill_lookahead(request, start, count)
        if count <= 0:
            raise AssertionError(f'zero progress: {prompt=}, {total=}, {shared=}, {budget=}, {start=}')
        end = start + count
        last = allocator.last_state_block_idx.get('replay')
        # Emulate MambaManager.remove_skipped_blocks in the fixture's
        # v1/core/single_type_kv_cache_manager.py. last_state_block_idx
        # identifies the state allocated two steps ago, not the running endpoint.
        if last is not None and last < (start + 511) // 512 - 1:
            retired_state_slots += int(not allocator.req_to_blocks['replay'][last].is_null)
            allocator.req_to_blocks['replay'][last] = allocator._null_block
        allocator._num_checkpoint_blocks['replay'] = int(
            allocator._needs_internal_checkpoint('replay', end, start))
        allocator.allocate_new_blocks('replay', end, end)
        blocks = allocator.req_to_blocks['replay']
        running = blocks[(end + 511) // 512 - 1]
        assert not running.is_null
        running.state = end
        internal = checkpoint(start, end, 512)
        if internal is not None and not blocks[internal // 512 - 1].is_null:
            blocks[internal // 512 - 1].state = internal
        mask = algorithms.MambaManager.reachable_block_mask(
            start_block=cached, end_block=end // 512, alignment_tokens=2048,
            kv_cache_spec=allocator.kv_cache_spec, use_eagle=True, retention_interval=0,
            reachable_boundaries=(prompt - 1, shared))
        selected_null += sum(keep and blocks[index].is_null for index, keep in enumerate(mask, cached))
        # Execute cache registration so its null-slot handling participates in
        # the check of every hash boundary against the planned state writes.
        registrar.cache_full_blocks(request, blocks, cached, end // 512, 512, 1, mask)
        steps.append(end)
        cached, request.num_computed_tokens = end // 512, end
        assert len(steps) < 1024
    required_prompt = ((prompt - 1) // 512 - 1) * 512
    return {'prompt_tokens': prompt, 'num_tokens': total, 'shared_boundary': shared,
            'budget': budget, 'speculative_blocks': speculative, 'steps': steps,
            'retained_boundaries': sorted(retained), 'selected_null_slots': selected_null,
            'retired_state_slots': retired_state_slots,
            'stale_or_unwritten_registered_states': failures,
            'prompt_predecessor_retained': required_prompt in retained}


def main():
    global ROOT, PATCHED
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, default=ROOT, help="Original vllm directory from this experiment's fixture composer (compose.py)")
    parser.add_argument('--candidate-root', type=Path, default=PATCHED, help="Candidate vllm directory from this experiment's fixture composer (compose.py)")
    args = parser.parse_args()
    ROOT, PATCHED = args.source_root, args.candidate_root
    # The extracted helpers resolve their unchanged support files from this root.
    import repro_mtp3_sparse_retention as sparse_helpers
    import repro_mtp3_checkpoint_materialization as checkpoint_helpers
    sparse_helpers.ROOT = ROOT
    checkpoint_helpers.ROOT = ROOT
    paths = {'scheduler': PATCHED / 'v1/core/sched/scheduler.py',
             'retention_manager': PATCHED / 'v1/core/single_type_kv_cache_manager.py',
             'allocator_manager': ROOT / 'v1/core/single_type_kv_cache_manager.py',
             'block_pool': ROOT / 'v1/core/block_pool.py',
             'gdn_metadata': ROOT / 'v1/attention/backends/gdn_attn.py',
             'hash_utils': ROOT / 'v1/core/kv_cache_utils.py',
             'coordinator': ROOT / 'v1/core/kv_cache_coordinator.py',
             'algorithm_loader': WORK / 'repro_mtp3_sparse_retention.py',
             'checkpoint_loader': WORK / 'repro_mtp3_checkpoint_materialization.py',
             'checker': Path(__file__).resolve()}
    identities = {name: {'path': str(path), 'sha256': sha256(path)} for name, path in paths.items()}
    for name, expected in [('scheduler', EXPECTED_SCHEDULER), ('retention_manager', EXPECTED_MANAGER),
                           ('allocator_manager', EXPECTED_ALLOCATOR)]:
        if identities[name]['sha256'] != expected:
            raise RuntimeError(f'unexpected {name} SHA-256: {identities[name]["sha256"]}')
    classes = extracted_classes()
    Scheduler = load_scheduler(paths['scheduler'])
    checkpoint = checkpoint_kernel_metadata()
    populations = {'fresh': [(value, value) for value in (32768, 32789, 33280, 100968)],
                   'resumed': [(32789, 33301), (32789, 40000), (32768, 33301), (32768, 40000)]}
    result = {'schema': 'mtp3-checkpoint-allocation-registration-check/v1',
              'time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'source_inputs': identities, 'gpu_executed': False, 'populations': {},
              'limits': ['Checks allocator/retention against planned endpoint/checkpoint writes; no GPU execution',
                         'Fresh and resumed requests start from empty block tables, not an injected partial local hit',
                         'Only mamba_cache_mode=align, block_size/hash_block_size=512, four-way decode context parallelism (DCP4), retention alignment=2048, num_prefill_lookahead=1',
                         'A missing prompt predecessor means no reusable entry was recorded; this check detects invalid registered state, not complete cache-hit coverage']}
    for label, pairs in populations.items():
        cases = [run_case(prompt, total, shared, budget, speculative, classes, Scheduler, checkpoint)
                 for prompt, total in pairs for shared in (8192, 12345, 15872, 16384, 16401, 20000, 30000)
                 for budget in (8192, 7680, 1024, 512) for speculative in (0, 1, 3)]
        summary = {'cases': len(cases), 'retired_state_slots': sum(case['retired_state_slots'] for case in cases), 'selected_null_slots': sum(case['selected_null_slots'] for case in cases),
                   'stale_or_unwritten_registered_states': sum(len(case['stale_or_unwritten_registered_states']) for case in cases),
                   'cases_without_prompt_predecessor_retention': sum(not case['prompt_predecessor_retained'] for case in cases)}
        result['populations'][label] = {'summary': summary, 'cases': cases}
        assert summary['cases'] == 336
    result['passed'] = all(not item['summary']['stale_or_unwritten_registered_states']
                           for item in result['populations'].values())
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'passed': result['passed'], 'output': str(args.output),
                      'summary': {name: item['summary'] for name, item in result['populations'].items()}}, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
