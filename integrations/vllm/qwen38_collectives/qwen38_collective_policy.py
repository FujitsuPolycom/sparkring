"""Local Qwen collective policy experiment; install before vLLM imports."""

import importlib.abc
import importlib.machinery
import hashlib
import json
import os
import sys
from pathlib import Path

MODE = os.environ.get("QWEN_DISPATCH_MODE")
try:
    LIMIT = int(os.environ.get("QWEN_DISPATCH_AR_BYTES", "2097152"))
except ValueError as error:
    raise SystemExit("Qwen collective cutoff must be an integer byte count") from error
TRACE = os.environ.get("QWEN_DISPATCH_TRACE", "0") == "1"
if MODE not in ("both", "reduce", "nccl") or not 0 <= LIMIT <= 2097152:
    raise SystemExit("Invalid local Qwen collective policy")
seen = set()


def emit(kind, **fields):
    if not TRACE:
        return
    line = json.dumps({"kind": kind, **fields}, sort_keys=True, default=str)
    if line not in seen:
        seen.add(line)
        print("QWEN_DISPATCH " + line, flush=True)


def patch_adapter(module):
    cls = module.B12xRoceAllReduce
    vote = cls._exchange_vote

    def agreed(self, reason, limits):
        policies = [None] * self.world_size
        module.dist.all_gather_object(policies, (MODE, LIMIT), group=self.group)
        if any(p != policies[0] for p in policies):
            raise RuntimeError("Qwen dispatch policy differs across ranks")
        return vote(self, reason, limits)

    cls._exchange_vote = agreed
    ar = cls.should_custom_ar
    ag = cls.should_all_gather

    def reduce(self, inp):
        allowed = (
            MODE != "nccl"
            and inp.numel() * inp.element_size() <= LIMIT
            and ar(self, inp)
        )
        if TRACE and self.rank == 0:
            emit(
                "all_reduce",
                shape=list(inp.shape),
                dtype=str(inp.dtype),
                bytes=inp.numel() * inp.element_size(),
                capture=module.torch.cuda.is_current_stream_capturing(),
                selected="rocenante" if allowed else "fallback",
            )
        return allowed

    def gather(self, inp, dim):
        allowed = MODE == "both" and ag(self, inp, dim)
        if TRACE and self.rank == 0:
            emit(
                "all_gather",
                shape=list(inp.shape),
                dtype=str(inp.dtype),
                bytes=inp.numel() * inp.element_size(),
                dim=dim,
                capture=module.torch.cuda.is_current_stream_capturing(),
                selected="rocenante" if allowed else "fallback",
            )
        return allowed

    cls.should_custom_ar = reduce
    cls.should_all_gather = gather
    print(
        "QWEN_DISPATCH_POLICY "
        + json.dumps({"mode": MODE, "all_reduce_max_bytes": LIMIT}),
        flush=True,
    )


def patch_graph(module):
    original = module.CUDAGraphWrapper.__call__

    def call(self, *args, **kwargs):
        if TRACE and module.is_forward_context_available():
            ctx = module.get_forward_context()
            emit(
                "graph_call",
                descriptor=str(ctx.batch_descriptor),
                mode=str(ctx.cudagraph_runtime_mode),
                wrapper_mode=str(self.runtime_mode),
            )
        return original(self, *args, **kwargs)

    module.CUDAGraphWrapper.__call__ = call


PATCHES = {
    "vllm.distributed.device_communicators.b12x_roce_all_reduce": patch_adapter,
    "vllm.compilation.cuda_graph": patch_graph,
}
SOURCE_HASHES = {
    "vllm.distributed.device_communicators.b12x_roce_all_reduce": "ea8a0f8515c82b7d57f4c085287f4dc106de745fbf2dd5721403072aeaf75114",
    "vllm.compilation.cuda_graph": "79e62fb0954d921de1ee8d6e8efcd87dfc2f68b8ba601b48406a7534d6b489f9",
}
if not TRACE:
    PATCHES.pop("vllm.compilation.cuda_graph")


class Loader(importlib.abc.Loader):
    def __init__(self, original, name):
        self.original, self.name = original, name

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        PATCHES[self.name](module)


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in PATCHES:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            raise SystemExit("Required Qwen dispatch module is missing: " + fullname)
        if (
            not spec.origin
            or hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest()
            != SOURCE_HASHES[fullname]
        ):
            raise SystemExit("Qwen dispatch source identity mismatch: " + fullname)
        spec.loader = Loader(spec.loader, fullname)
        return spec


sys.meta_path.insert(0, Finder())
