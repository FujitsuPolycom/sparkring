"""Source-bound Qwen HC up-projection/gate fusion for local qualification."""

import hashlib
import importlib.abc
import importlib.machinery
import json
import os
import sys
from pathlib import Path

TARGET = "vllm.models.qwen4_exp.nvidia.hyperconnection"
EXPECTED = os.environ["QWEN_HC_SOURCE_SHA256"]
if os.environ.get("QWEN_HC_FUSION") != "1":
    raise SystemExit("Explicit Qwen HC fusion opt-in required")
voted = False
announced = set()


def patch(module):
    import torch
    import torch.distributed as dist
    from vllm.distributed import get_tp_group
    from qwen4_fused_gate_kernel import fused

    @torch.library.custom_op("sparkring::qwen_hc_up_gate", mutates_args=())
    def operation(x: torch.Tensor, w: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        m = x.shape[0]
        hidden = n.shape[1] // 4
        out = n.new_empty((m, hidden))
        if m == 0:
            return out
        category = "prefill" if m >= 128 else "decode"
        if category not in announced:
            announced.add(category)
            print(
                "QWEN_HC_EXECUTED " + json.dumps({"category": category, "rows": m}),
                flush=True,
            )
        if (
            x.shape != (m, 320)
            or hidden not in (640, 1280, 2560)
            or w.shape != (4 * hidden, 320)
            or n.shape != (m, 4 * hidden)
        ):
            raise RuntimeError("Qwen HC fusion received unsupported shapes")
        if any(
            t.dtype != torch.bfloat16 or not t.is_cuda or not t.is_contiguous()
            for t in (x, w, n, out)
        ):
            raise RuntimeError("Qwen HC fusion requires contiguous CUDA BF16 tensors")
        config = (
            (16, 32, 32, 4)
            if m <= 16
            else ((32, 64, 32, 8) if m <= 256 else (64, 64, 32, 8))
        )
        if m < 128:
            from b12x.norm.hyperconnection._kernels import _gate_mean_launch

            _gate_mean_launch(n, torch.nn.functional.linear(x, w), out, 4, hidden, 128)
        else:
            fused(x, w, n, out, *config)
        return out

    @operation.register_fake
    def fake(x, w, n):
        return n.new_empty((x.shape[0], n.shape[1] // 4))

    cls = module.GatedResidual
    original_init = cls.__init__
    original_mix_normalized = cls._mix_normalized

    def initialize(self, *args, **kwargs):
        global voted
        original_init(self, *args, **kwargs)
        partitions = getattr(self, "tp_size", 1)
        if (
            partitions not in (1, 4)
            or (self.hidden_size, self.hc_count, self.lora_rank)
            != (2560, 4, 320 // partitions)
        ):
            raise RuntimeError(
                "Qwen HC fusion requires H2560, four streams and rank320"
            )
        if not voted:
            group = get_tp_group()
            identity = (
                os.environ.get("QWEN_HC_FUSION"),
                EXPECTED,
                hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                hashlib.sha256(
                    Path(__file__).with_name("qwen4_fused_gate_kernel.py").read_bytes()
                ).hexdigest(),
            )
            votes = [None] * group.world_size
            dist.all_gather_object(votes, identity, group=group.cpu_group)
            if group.world_size != 4 or any(v != votes[0] for v in votes):
                raise RuntimeError("Qwen HC fusion rank agreement failed")
            voted = True
            print(
                "QWEN_HC_FUSION "
                + json.dumps(
                    {
                        "rank": group.rank_in_group,
                        "source": identity[2],
                        "kernel": identity[3],
                    }
                ),
                flush=True,
            )

    cls.__init__ = initialize

    def mix_sharded(self, normalized):
        api = module._hyperconnection_api()
        workspace = self.workspace
        tokens = normalized.shape[0]
        down = (
            self.input_mix_weight_down_block_inject
            if self.use_combine
            else self.input_mix_weight_down
        )
        width = down.weight.shape[0]
        projected_fp32 = workspace.local_down_fp32.view(-1)[: tokens * width].view(
            tokens, width
        )
        torch.mm(normalized, down.weight.T, out=projected_fp32, out_dtype=torch.float32)
        projected = workspace.local_down.view(-1)[: tokens * width].view(tokens, width)
        projected.copy_(projected_fp32)
        injection = (
            projected[:, self.lora_rank : self.lora_rank + self.hc_count]
            if self.use_combine
            else None
        )
        bottleneck = api.run_scaled_silu(
            projected[:, : self.lora_rank],
            binding=self._binding(normalized, "scaled_silu"),
        )
        bottleneck = module.tensor_model_parallel_all_gather(bottleneck, dim=-1)
        local_hidden = self.hidden_size // self.tp_size
        local_normalized = workspace.local_normalized[:tokens]
        local_normalized.view(tokens, self.hc_count, local_hidden).copy_(
            normalized.view(tokens, self.hc_count, self.hidden_size)[
                :, :, self.tp_rank * local_hidden : (self.tp_rank + 1) * local_hidden
            ]
        )
        output = operation(
            bottleneck, self.input_mix_weight_up.weight, local_normalized
        )
        return module.tensor_model_parallel_all_gather(output, dim=-1), injection

    def mix_normalized(self, normalized):
        # Keep KK's prepared projection path for decode and graph-sized batches.
        if normalized.shape[0] < 128:
            return original_mix_normalized(self, normalized)
        if getattr(self, "tp_size", 1) > 1:
            return mix_sharded(self, normalized)
        api = module._hyperconnection_api()
        if self.use_combine:
            down = self.input_mix_weight_down_block_inject(normalized)
            projected = down[:, : self.lora_rank]
            injection = down[:, self.lora_rank : self.lora_rank + self.hc_count]
        else:
            projected = self.input_mix_weight_down(normalized)
            injection = None
        bottleneck = api.run_scaled_silu(
            projected, binding=self._binding(normalized, "scaled_silu")
        )
        out = operation(bottleneck, self.input_mix_weight_up.weight, normalized)
        return out, injection

    cls._mix_normalized = mix_normalized


class Loader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        patch(module)


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if (
            spec is None
            or spec.loader is None
            or hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest() != EXPECTED
        ):
            raise SystemExit("Qwen HC source identity mismatch")
        spec.loader = Loader(spec.loader)
        return spec


sys.meta_path.insert(0, Finder())
