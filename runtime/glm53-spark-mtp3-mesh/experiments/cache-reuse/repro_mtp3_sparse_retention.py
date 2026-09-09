"""Check prefix lookup and retention using exact vLLM source fixtures on CPU.

The block pool contains explicitly described synthetic materialized checkpoints.
The result identifies lookup/retention mismatches under the supplied geometry.
It does not establish which intermediate states a live model materializes.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import itertools
import json
import os
from collections import namedtuple
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import overload


ROOT = Path(os.environ.get("MTP3_ORIGINAL_SOURCE", str(Path(__file__).resolve().parent / "mtp3-vllm-source")))


class FullAttentionSpec:
    def __init__(self, block_size, dcp_replicated=False):
        self.block_size = block_size
        self.dcp_replicated = dcp_replicated


class MambaSpec:
    def __init__(self, block_size):
        self.block_size = block_size


class ChunkedLocalAttentionSpec:
    pass


def execute(nodes, scope):
    tree = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                           *copy.deepcopy(nodes)], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), "<installed-vllm-algorithms>", "exec"), scope)


def load_algorithms(hash_utils, *, manager_source=None):
    scope = dict(FullAttentionSpec=FullAttentionSpec, MambaSpec=MambaSpec,
                 ChunkedLocalAttentionSpec=ChunkedLocalAttentionSpec, Sequence=Sequence,
                 itertools=itertools, overload=overload, cdiv=lambda a, b: (a + b - 1) // b)
    utils_tree = ast.parse(hash_utils.read_text())
    execute([node for node in utils_tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
             and node.name in {"BlockHashListWithBlockSize", "resolve_block_hashes"}], scope)
    manager_tree = ast.parse((manager_source or ROOT / "v1/core/single_type_kv_cache_manager.py").read_text())
    for name, methods in (("FullAttentionManager", {"find_longest_cache_hit"}),
                          ("MambaManager", {"find_longest_cache_hit", "reachable_block_mask"})):
        original = next(node for node in manager_tree.body if isinstance(node, ast.ClassDef) and node.name == name)
        shell = ast.parse(f"class {name}:\n    supports_fine_grained_hash_lookup = True\n").body[0]
        shell.body += [node for node in original.body if isinstance(node, ast.FunctionDef) and node.name in methods]
        execute([shell], scope)
    coordinator = ast.parse((ROOT / "v1/core/kv_cache_coordinator.py").read_text())
    original = next(node for node in coordinator.body if isinstance(node, ast.ClassDef)
                    and node.name == "HybridKVCacheCoordinator")
    shell = ast.parse("class HybridKVCacheCoordinator:\n    pass\n").body[0]
    shell.body = [node for node in original.body if isinstance(node, ast.FunctionDef)
                  and node.name in {"find_longest_cache_hit", "_cache_hit_alignment_tokens"}]
    execute([shell], scope)
    return SimpleNamespace(**scope)


class Pool:
    def __init__(self, hash_block_size, retained_mamba, shared_boundary):
        self.hash_block_size = hash_block_size
        self.null_block = SimpleNamespace(is_null=True, block_hash=None)
        self.retained_mamba = set(retained_mamba)
        self.shared_boundary = shared_boundary
        self.queries = []

    def get_cached_block(self, block_hash, groups):
        self.queries.append((block_hash, tuple(groups)))
        if block_hash > self.shared_boundary or any(
            group == 1 and block_hash not in self.retained_mamba for group in groups
        ):
            return None
        return [SimpleNamespace(is_null=False, block_hash=(block_hash, group)) for group in groups]


def run_case(algorithms, args, *, drop=True, extra_checkpoint=None):
    Full = algorithms.FullAttentionManager
    Mamba = algorithms.MambaManager
    target = FullAttentionSpec(args.page)
    state = MambaSpec(args.page)
    draft = FullAttentionSpec(args.page, dcp_replicated=True)
    mask = Mamba.reachable_block_mask(
        start_block=0, end_block=args.shared // args.page,
        alignment_tokens=args.scheduler_alignment, kv_cache_spec=state,
        use_eagle=False, retention_interval=0, reachable_boundaries=(args.prompt - 1,),
    )
    retained = [(index + 1) * args.page for index, keep in enumerate(mask) if keep]
    if extra_checkpoint is not None:
        retained.append(extra_checkpoint)
    pool = Pool(args.hash_unit, retained, args.shared)
    coordinator = algorithms.HybridKVCacheCoordinator()
    coordinator.kv_cache_config = SimpleNamespace(kv_cache_groups=(target, state, draft))
    coordinator.single_type_managers = (
        SimpleNamespace(block_size=args.page * args.dcp),
        SimpleNamespace(block_size=args.page), SimpleNamespace(block_size=args.page),
    )
    spec_group = namedtuple("SpecGroup", "spec group_ids manager_cls use_eagle")
    coordinator.attention_groups = [spec_group(target, [0], Full, False), spec_group(state, [1], Mamba, False),
                                    spec_group(draft, [2], Full, drop)]
    coordinator.block_pool = pool
    coordinator.hash_block_size = args.hash_unit
    coordinator.scheduler_block_size = args.scheduler_alignment
    coordinator.enable_partial_hash_hits = args.fine_hits
    coordinator.dcp_world_size = args.dcp
    hashes = list(range(args.hash_unit, args.continuation, args.hash_unit))
    separate = []
    for spec, groups, manager, use_eagle in coordinator.attention_groups:
        _, hit = manager.find_longest_cache_hit(
            block_hashes=hashes, max_length=args.continuation - 1,
            kv_cache_group_ids=groups, block_pool=pool, kv_cache_spec=spec,
            drop_eagle_block=use_eagle, alignment_tokens=coordinator._cache_hit_alignment_tokens,
            dcp_world_size=args.dcp if isinstance(spec, FullAttentionSpec) else 1,
        )
        separate.append(hit)
    _, reconciled, uncached = coordinator.find_longest_cache_hit(hashes, args.continuation - 1)
    return dict(drop_speculative_draft=drop, retained_mamba_tokens=retained,
                extra_materialized_checkpoint=extra_checkpoint,
                per_group_hit_tokens=separate, reconciled_hit_tokens=reconciled,
                uncached_shared_prefix_tokens=uncached)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hash-utils-source", type=Path, default=ROOT / "v1/core/kv_cache_utils.py")
    parser.add_argument("--hash-unit", type=int, default=256)
    parser.add_argument("--scheduler-alignment", type=int, default=2048)
    parser.add_argument("--page", type=int, default=512)
    parser.add_argument("--dcp", type=int, default=4)
    parser.add_argument("--prompt", type=int, default=32789)
    parser.add_argument("--continuation", type=int, default=34856)
    parser.add_argument("--shared", type=int, default=32768)
    parser.add_argument("--fine-hits", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    algorithms = load_algorithms(args.hash_utils_source)
    baseline = run_case(algorithms, args)
    controls = [run_case(algorithms, args, drop=False)]
    draft_hit = baseline["per_group_hit_tokens"][-1]
    for unit in sorted({args.page, args.scheduler_alignment}):
        checkpoint = draft_hit // unit * unit
        if checkpoint > 0:
            controls.append(run_case(algorithms, args, extra_checkpoint=checkpoint))
    paths = [ROOT / "v1/core/single_type_kv_cache_manager.py", ROOT / "v1/core/kv_cache_coordinator.py",
             args.hash_utils_source]
    print(json.dumps(dict(schema="mtp3-sparse-retention-metadata-reproducer/v1",
                          geometry={key: value for key, value in vars(args).items() if key != "hash_utils_source"},
                          synthetic_checkpoint_metadata=True, gpu_executed=False,
                          sources={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
                          baseline=baseline, controls=controls), sort_keys=True))
    if baseline["reconciled_hit_tokens"] == 0 and baseline["per_group_hit_tokens"][0] > 0:
        assert controls[0]["reconciled_hit_tokens"] > 0
        assert all(case["reconciled_hit_tokens"] > 0 for case in controls[1:])


if __name__ == "__main__":
    main()
