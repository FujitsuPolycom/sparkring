"""CPU tests of packaged checkpoint methods; no live model or CUDA qualification."""

import ast
from collections import defaultdict
from dataclasses import dataclass
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS, ModuleType
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).parent
spec = importlib.util.spec_from_file_location(
    "continuation_regression_installer", ROOT / "install.py"
)
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)
_, sources = installer.package()
helper = ModuleType("continuation_regression_helper")
exec(
    compile(
        sources["vllm/v1/core/recurrent_prefill_checkpoint.py"],
        "recurrent_prefill_checkpoint.py",
        "exec",
    ),
    helper.__dict__,
)


class MambaSpec:
    block_size = 512
    num_prefill_checkpoint_blocks = 2


def extract(path, name, methods, scope, base=None):
    cls = next(
        n
        for n in ast.parse(sources["vllm/" + path]).body
        if isinstance(n, ast.ClassDef) and n.name == name
    )
    cls.body = [
        n
        for n in cls.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in methods
    ]
    cls.bases = [ast.Name(id=base, ctx=ast.Load())] if base else []
    cls.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            cls,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(ROOT / path), "exec"), scope)
    return scope[name]


@dataclass
class Block:
    block_id: int
    is_null: bool = False
    ref_cnt: int = 1
    block_hash: object = None
    block_hash_num_tokens: int = 0
    state: object = None


class Pool:
    hash_block_size = 512

    def __init__(self):
        self.next = 1
        self.allocated = []
        self.freed = []
        self.fail = False
        self.registered = []

    def get_new_blocks(self, count):
        if self.fail:
            raise RuntimeError("injected pool failure")
        blocks = [Block(i) for i in range(self.next, self.next + count)]
        self.next += count
        self.allocated.extend(blocks)
        return blocks

    def free_blocks(self, blocks):
        for block in blocks:
            if block.is_null:
                continue
            assert block.ref_cnt > 0
            block.ref_cnt -= 1
            self.freed.append(block.block_id)

    def cache_full_blocks(
        self,
        *,
        request,
        blocks,
        num_cached_blocks,
        num_full_blocks,
        block_size,
        kv_cache_group_id,
        block_mask,
    ):
        for column in range(num_cached_blocks, num_full_blocks):
            block = blocks[column]
            if block.is_null or (
                block_mask is not None and not block_mask[column - num_cached_blocks]
            ):
                continue
            block.block_hash = (column, kv_cache_group_id)
            block.block_hash_num_tokens = (column + 1) * block_size
            self.registered.append(
                (column, block.block_id, block.block_hash_num_tokens)
            )


def manager(speculative=3):
    path = "v1/core/single_type_kv_cache_manager.py"
    scope = {
        "MambaSpec": MambaSpec,
        "cdiv": lambda a, b: (a + b - 1) // b,
        "continuation_layout": helper.continuation_layout,
        "get_group_id": lambda key: key[1],
    }
    base = extract(
        path,
        "SingleTypeKVCacheManager",
        {
            "cache_blocks",
            "remove_skipped_blocks",
            "_remove_blocks_in_range",
            "pop_blocks_for_free",
            "free",
        },
        scope,
    )
    scope["Base"] = base
    cls = extract(
        path,
        "MambaManager",
        {
            "get_num_blocks_to_allocate",
            "allocate_new_blocks",
            "remove_skipped_blocks",
            "_needs_internal_checkpoint",
            "get_num_skipped_tokens",
            "cache_blocks",
            "_cache_partial_tail_block",
            "_queue_aligned_recurrent_boundary",
            "reachable_block_mask",
            "pop_blocks_for_free",
        },
        scope,
        base="Base",
    )
    obj = cls()
    obj.kv_cache_spec = MambaSpec()
    obj.block_size = 512
    obj.num_speculative_blocks = speculative
    obj.mamba_cache_mode = "align"
    obj.req_to_blocks = defaultdict(list)
    obj._planned_recurrent_checkpoints = {}
    obj._planned_recurrent_publications = {}
    obj._allocated_block_reqs = set()
    obj._partial_hit_reqs = {}
    obj._num_checkpoint_blocks = {}
    obj.last_state_block_idx = {}
    obj._null_block = Block(0, True, 0)
    obj.block_pool = Pool()
    obj.num_cached_block = {}
    obj.cached_blocks_this_step = set()
    obj.scheduler_block_size = 2048
    obj.use_eagle = True
    obj.kv_cache_group_id = 7
    obj._pending_aligned_recurrent_boundaries = []
    obj.recurrent_publication_boundary = None
    obj._producer_partial_tail_reqs = {}
    obj._pending_partial_tail_offloads = []
    obj._has_partial_local_hit = lambda *_: False
    obj._get_num_evictable_blocks = lambda _: 0
    return obj


def seed(obj, start):
    worker = []
    for end in range(8192, start + 1, 8192):
        obj.remove_skipped_blocks("r", max(0, end - 8192))
        suffix = obj.allocate_new_blocks("r", end, end)
        worker.extend(suffix)
        obj.req_to_blocks["r"][end // 512 - 1].state = end
    return worker


def request(prompt, start=0, identifier="r"):
    return NS(
        request_id=identifier,
        num_computed_tokens=start,
        num_prompt_tokens=prompt,
        num_tokens=prompt,
        shared_prefix_boundary=0,
        has_encoder_inputs=False,
        num_preemptions=0,
        spec_token_ids=[],
        num_in_flight_tokens=0,
        status="waiting",
        resumable=False,
    )


def scheduler(obj, enabled=True, budget=8192):
    scope = {
        "fresh_prompt_plan": helper.fresh_prompt_plan,
        "final_chunk_plan": helper.final_chunk_plan,
        "continuation_layout": helper.continuation_layout,
        "MambaSpec": MambaSpec,
        "RequestStatus": NS(WAITING="waiting"),
    }
    cls = extract(
        "v1/core/sched/scheduler.py",
        "Scheduler",
        {
            "_record_continuation_origin",
            "_recurrent_checkpoint_plan",
            "_mamba_block_aligned_split",
            "_free_request_blocks",
        },
        scope,
    )
    owner = cls()
    owner._two_checkpoint_prefill_enabled = True
    owner._continuation_prefill_enabled = enabled
    owner._continuation_prefill_origins = {}
    owner.cache_config = NS(block_size=512)
    owner.max_num_scheduled_tokens = budget
    owner.use_eagle = True
    owner.mamba_has_prefill_checkpoint_blocks = True
    owner.mamba_partial_cache_hit = True
    owner.hash_block_size = 512
    owner.scheduler_config = NS(long_prefill_token_threshold=0)
    owner.kv_cache_manager = NS(coordinator=NS(single_type_managers=[obj]))
    owner._recurrent_publication_boundaries = lambda req: tuple(
        [((req.num_prompt_tokens - 1) // 2048) * 2048]
    )
    return owner


class ContinuationTests(unittest.TestCase):
    def test_fresh_8k_path_retained_when_continuation_disabled(self):
        obj = manager()
        owner = scheduler(obj, enabled=False)
        req = request(8192)
        plan = owner._recurrent_checkpoint_plan(req, 0, 8192)
        self.assertEqual(plan, (0, 8192, (6144, 7168)))
        obj._planned_recurrent_checkpoints["r"] = plan
        self.assertEqual(obj.get_num_blocks_to_allocate("r", 8192, [], 0, 0, 8192), 6)
        table = obj.allocate_new_blocks("r", 8192, 8192)
        self.assertEqual(
            [i for i, b in enumerate(table) if not b.is_null], [11, 13, 15, 16, 17, 18]
        )

    def test_request_release_keeps_an_extra_publication_pin(self):
        for end, targets in ((10240, (9216,)), (16384, (14336, 15360))):
            obj = manager()
            seed(obj, 8192)
            obj._planned_recurrent_checkpoints["r"] = (8192, end, targets)
            obj.allocate_new_blocks("r", end, end)
            published = obj.req_to_blocks["r"][targets[0] // 512 - 1]
            published.state = targets[0]
            published.ref_cnt += 1
            obj.free("r")
            self.assertEqual(published.ref_cnt, 1)
            self.assertEqual(published.state, targets[0])
            self.assertNotIn("r", obj.req_to_blocks)
            self.assertNotIn("r", obj.last_state_block_idx)
            self.assertEqual(len(obj.block_pool.freed), len(set(obj.block_pool.freed)))
            obj.block_pool.free_blocks([published])
            self.assertEqual(published.ref_cnt, 0)

    def test_real_scheduler_and_allocator_schedules_8k_and_6k(self):
        expected = {
            8192: {
                9216: [8192, 1024],
                10240: [8192, 2048],
                16384: [8192, 8192],
                32768: [8192] * 4,
            },
            6144: {
                9216: [6144, 3072],
                10240: [6144, 4096],
                16384: [6144, 6144, 4096],
                32768: [6144] * 5 + [2048],
            },
        }
        for budget, cases in expected.items():
            for prompt, want in cases.items():
                with self.subTest(budget=budget, prompt=prompt):
                    obj = manager()
                    owner = scheduler(obj, budget=budget)
                    req = request(prompt)
                    chunks = []
                    worker = []
                    while req.num_computed_tokens < prompt:
                        start = req.num_computed_tokens
                        size = owner._mamba_block_aligned_split(
                            req, min(budget, prompt - start)
                        )
                        plan = owner._recurrent_checkpoint_plan(
                            req, start, start + size
                        )
                        obj.remove_skipped_blocks("r", start)
                        if plan:
                            obj._planned_recurrent_checkpoints["r"] = plan
                        count = obj.get_num_blocks_to_allocate(
                            "r", start + size, [], start, start, start + size
                        )
                        before = len(obj.block_pool.allocated)
                        worker.extend(
                            obj.allocate_new_blocks("r", start + size, start + size)
                        )
                        self.assertEqual(len(obj.block_pool.allocated) - before, count)
                        # Simulated completion stamps only states this step actually produces.
                        if plan:
                            for token in plan[2]:
                                obj.req_to_blocks["r"][token // 512 - 1].state = token
                        obj.req_to_blocks["r"][(start + size) // 512 - 1].state = (
                            start + size
                        )
                        if start == 0:
                            owner._record_continuation_origin(req, 0, 0, 0, False)
                        obj._planned_recurrent_checkpoints.clear()
                        chunks.append(size)
                        req.num_computed_tokens += size
                    self.assertEqual(chunks, want)
                    self.assertEqual(worker[prompt // 512 - 1].state, prompt)

    def test_actual_allocator_source_and_worker_prefix(self):
        for speculative in (0, 1, 3):
            for start in (8192, 16384, 24576, 57344):
                for tail in (1536, 2048, 4096, 6144, 8192):
                    with self.subTest(speculative=speculative, start=start, tail=tail):
                        obj = manager(speculative)
                        worker = seed(obj, start)
                        old = tuple(obj.req_to_blocks["r"])
                        worker_before = tuple(worker)
                        plan = helper.final_chunk_plan(
                            start=start,
                            end=start + tail,
                            prompt=start + tail,
                            num_tokens=start + tail,
                            block_size=512,
                            publications=(((start + tail - 1) // 2048) * 2048,),
                        )
                        obj._planned_recurrent_checkpoints["r"] = plan
                        admission = obj.get_num_blocks_to_allocate(
                            "r", start + tail, [], start, start, start + tail
                        )
                        self.assertEqual(admission, len(plan[2]) + 1)
                        before = len(obj.block_pool.allocated)
                        suffix = obj.allocate_new_blocks(
                            "r", start + tail, start + tail
                        )
                        worker.extend(suffix)
                        self.assertEqual(
                            len(obj.block_pool.allocated) - before, admission
                        )
                        self.assertEqual(
                            worker[: len(worker_before)], list(worker_before)
                        )
                        table = obj.req_to_blocks["r"]
                        source = start // 512 - 1
                        self.assertIs(table[source], old[source])
                        self.assertEqual(table[source].state, start)
                        active = (
                            [source]
                            + [p // 512 - 1 for p in plan[2]]
                            + list(range((start + tail) // 512 - 1, len(table)))
                        )
                        self.assertEqual(
                            len(active), len({worker[c].block_id for c in active})
                        )
                        self.assertTrue(all(worker[c] is table[c] for c in active))
                        owned = [b.block_id for b in table if not b.is_null]
                        self.assertEqual(len(owned), len(set(owned)))
                        self.assertEqual(obj.last_state_block_idx["r"], source)
                        # An in-flight next chunk must not free its source yet.
                        obj.remove_skipped_blocks("r", start)
                        self.assertIs(obj.req_to_blocks["r"][source], old[source])
                        # After that chunk is processed, ordinary cleanup may retire it.
                        obj.remove_skipped_blocks("r", start + tail)
                        self.assertTrue(obj.req_to_blocks["r"][source].is_null)

    def test_10k_reuses_existing_checkpoint_column(self):
        obj = manager()
        worker = seed(obj, 8192)
        checkpoint = worker[17]
        plan = (8192, 10240, (9216,))
        obj._planned_recurrent_checkpoints["r"] = plan
        worker.extend(obj.allocate_new_blocks("r", 10240, 10240))
        self.assertIs(obj.req_to_blocks["r"][17], checkpoint)
        self.assertIs(worker[17], checkpoint)
        self.assertEqual(len(obj.block_pool.allocated), 6)  # initial4 +new2

    def test_failed_pool_allocation_does_not_mutate_table(self):
        obj = manager()
        seed(obj, 8192)
        before = tuple(obj.req_to_blocks["r"])
        obj.block_pool.fail = True
        obj._planned_recurrent_checkpoints["r"] = (8192, 16384, (14336, 15360))
        with self.assertRaises(RuntimeError):
            obj.allocate_new_blocks("r", 16384, 16384)
        self.assertEqual(tuple(obj.req_to_blocks["r"]), before)
        self.assertNotIn("r", obj.last_state_block_idx)

    def test_private_reserve_and_source_guards(self):
        for mutate in (
            lambda o: setattr(o.req_to_blocks["r"][16], "ref_cnt", 2),
            lambda o: setattr(o.req_to_blocks["r"][17], "block_hash", ("x", 7)),
            lambda o: o.req_to_blocks["r"].__setitem__(15, o._null_block),
            lambda o: o.req_to_blocks["r"].__setitem__(18, o.req_to_blocks["r"][17]),
            lambda o: o.req_to_blocks["r"].append(o._null_block),
        ):
            obj = manager()
            seed(obj, 8192)
            mutate(obj)
            before = tuple(obj.req_to_blocks["r"])
            with self.assertRaises(ValueError):
                helper.continuation_layout(
                    before, (8192, 16384, (14336, 15360)), 512, 3
                )
            self.assertEqual(tuple(obj.req_to_blocks["r"]), before)

    def test_scheduler_opt_in_provenance_and_fallback(self):
        obj = manager()
        owner = scheduler(obj)
        req = request(16384)
        self.assertIsNone(owner._recurrent_checkpoint_plan(req, 8192, 16384))
        owner._record_continuation_origin(req, 0, 0, 0, False)
        seed(obj, 8192)
        req.num_computed_tokens = 8192
        self.assertEqual(owner._mamba_block_aligned_split(req, 8192), 8192)
        owner._continuation_prefill_enabled = False
        self.assertEqual(owner._mamba_block_aligned_split(req, 8192), 6144)
        owner._continuation_prefill_enabled = True
        for field, value in [
            ("num_preemptions", 1),
            ("has_encoder_inputs", True),
            ("spec_token_ids", [3]),
            ("num_tokens", 16385),
            ("resumable", True),
        ]:
            old = getattr(req, field)
            setattr(req, field, value)
            self.assertIsNone(owner._recurrent_checkpoint_plan(req, 8192, 16384))
            setattr(req, field, old)
        replacement = request(16384, 8192)
        self.assertIsNone(owner._recurrent_checkpoint_plan(replacement, 8192, 16384))

    def test_origin_rejected_for_cached_or_async_admission_and_cleared_on_free(self):
        obj = manager()
        owner = scheduler(obj)
        req = request(16384)
        for args in [(6144, 0, 6144, False), (6144, 6144, 0, False), (0, 0, 0, True)]:
            owner._record_continuation_origin(req, *args)
            self.assertNotIn("r", owner._continuation_prefill_origins)
        owner._record_continuation_origin(req, 0, 0, 0, False)
        owner.defer_block_free = False
        owner.kv_cache_manager.free = lambda _: None
        owner._free_request_blocks(req)
        self.assertNotIn("r", owner._continuation_prefill_origins)

    def test_6k_budget_and_hard_cap(self):
        self.assertIsNone(
            helper.final_chunk_plan(
                start=8192,
                end=16384,
                prompt=16384,
                num_tokens=16384,
                block_size=512,
                publications=(14336,),
                max_chunk_tokens=6144,
            )
        )
        self.assertEqual(
            helper.final_chunk_plan(
                start=12288,
                end=16384,
                prompt=16384,
                num_tokens=16384,
                block_size=512,
                publications=(14336,),
                max_chunk_tokens=6144,
            ),
            (12288, 16384, (14336, 15360)),
        )
        self.assertIsNone(
            helper.final_chunk_plan(
                start=8192,
                end=24576,
                prompt=24576,
                num_tokens=24576,
                block_size=512,
                publications=(22528,),
                max_chunk_tokens=32768,
            )
        )

    def test_cache_registration_and_interior_publication(self):
        obj = manager()
        seed(obj, 8192)
        obj.num_cached_block["r"] = 16
        plan = (8192, 16384, (14336, 15360))
        obj._planned_recurrent_checkpoints["r"] = plan
        obj._planned_recurrent_publications["r"] = (14336,)
        obj.allocate_new_blocks("r", 16384, 16384)
        req = request(16384, 8192)
        obj.cache_blocks(req, 16384)
        by_position = {
            position: identifier
            for _, identifier, position in obj.block_pool.registered
        }
        self.assertEqual(by_position[14336], obj.req_to_blocks["r"][27].block_id)
        self.assertEqual(by_position[15360], obj.req_to_blocks["r"][29].block_id)
        self.assertEqual(
            [row[3] for row in obj._pending_aligned_recurrent_boundaries], [14336]
        )

    def test_wrapper_clears_plan_on_failed_admission(self):
        obj = manager()
        seed(obj, 8192)
        req = request(16384, 8192)
        cls = extract(
            "v1/core/kv_cache_manager.py", "KVCacheManager", {"allocate_slots"}, {}
        )
        owner = cls()
        owner.coordinator = NS(single_type_managers=[obj])
        observed = []

        def inner(**kwargs):
            observed.append(obj._planned_recurrent_checkpoints["r"])
            return None

        owner._allocate_slots_without_checkpoint_plan = inner
        with patch.dict(
            sys.modules, {"vllm.v1.core.recurrent_prefill_checkpoint": helper}
        ):
            self.assertIsNone(
                owner.allocate_slots(
                    req,
                    8192,
                    recurrent_prefill_checkpoint_plan=(8192, 16384, (14336, 15360)),
                    recurrent_checkpoint_publications=(14336,),
                )
            )
        self.assertEqual(len(observed), 1)
        self.assertFalse(obj._planned_recurrent_checkpoints)
        self.assertFalse(obj._planned_recurrent_publications)

        def raises(**kwargs):
            raise RuntimeError("injected inner failure")

        owner._allocate_slots_without_checkpoint_plan = raises
        with patch.dict(
            sys.modules, {"vllm.v1.core.recurrent_prefill_checkpoint": helper}
        ):
            with self.assertRaises(RuntimeError):
                owner.allocate_slots(
                    req,
                    8192,
                    recurrent_prefill_checkpoint_plan=(8192, 16384, (14336, 15360)),
                    recurrent_checkpoint_publications=(14336,),
                )
            for extra in (
                {"num_external_computed_tokens": 512},
                {"delay_cache_blocks": True},
                {"recurrent_checkpoint_publications": (8192,)},
            ):
                with self.assertRaises(ValueError):
                    owner.allocate_slots(
                        req,
                        8192,
                        recurrent_prefill_checkpoint_plan=(8192, 16384, (14336, 15360)),
                        **extra,
                    )
        self.assertFalse(obj._planned_recurrent_checkpoints)
        self.assertFalse(obj._planned_recurrent_publications)

    def test_hooks_are_in_actual_admission_and_cleanup_sites(self):
        tree = ast.parse(sources["vllm/v1/core/sched/scheduler.py"])
        methods = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        calls = [
            n
            for n in ast.walk(methods["schedule"])
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "_record_continuation_origin"
        ]
        self.assertEqual(len(calls), 1)
        self.assertIn(
            "_continuation_prefill_origins.pop",
            ast.unparse(methods["_update_requests_with_invalid_blocks"]),
        )
        for call in [
            n for n in ast.walk(methods["schedule"]) if isinstance(n, ast.Call)
        ]:
            for keyword in call.keywords:
                if keyword.arg == "recurrent_checkpoint_publications":
                    self.assertIn(
                        "checkpoint_plan[0] < p < checkpoint_plan[1]",
                        ast.unparse(keyword.value),
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
