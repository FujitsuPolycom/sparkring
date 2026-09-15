"""Source-bound Qwen MTP prefill GEMMs with BF16 projection/add boundaries."""

import hashlib
import importlib.abc
import importlib.machinery
import json
import os
import sys
from pathlib import Path

TARGET = "b12x.sequence.mtp_feedback._kernels"
EXPECTED = os.environ["QWEN_MTP_SOURCE_SHA256"]
if os.environ.get("QWEN_MTP_TORCH_PREFILL") != "1":
    raise SystemExit("Explicit Qwen MTP GEMM opt-in required")
voted = False


def patch(module):
    import torch
    import torch.distributed as dist
    from vllm.distributed import get_tp_group

    original = module._qwen_cute_projections

    def projections(
        token_normalized,
        state_normalized,
        embedding_fc_weight,
        hidden_fc_weight,
        token_path,
        output,
        *,
        tokens,
        token_rows,
        state_rows,
        streams,
        hidden_size
    ):
        global voted
        if tokens < 128:
            return original(
                token_normalized,
                state_normalized,
                embedding_fc_weight,
                hidden_fc_weight,
                token_path,
                output,
                tokens=tokens,
                token_rows=token_rows,
                state_rows=state_rows,
                streams=streams,
                hidden_size=hidden_size,
            )
        if (streams, hidden_size) != (4, 2560):
            raise RuntimeError("Qwen MTP GEMMs require four streams and H2560")
        if not voted:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Qwen MTP GEMMs require warmup before capture")
            group = get_tp_group()
            identity = (
                EXPECTED,
                hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                128,
            )
            votes = [None] * group.world_size
            dist.all_gather_object(votes, identity, group=group.cpu_group)
            if group.world_size != 4 or any(v != identity for v in votes):
                raise RuntimeError("Qwen MTP GEMM rank agreement failed")
            voted = True
            print(
                "QWEN_MTP_GEMM "
                + json.dumps(
                    {
                        "rank": group.rank_in_group,
                        "tokens": tokens,
                        "source": identity[1],
                    }
                ),
                flush=True,
            )
        x = module._capacity_matrix(token_normalized, token_rows, hidden_size)[:tokens]
        h = module._capacity_matrix(state_normalized, state_rows, hidden_size)[
            : tokens * streams
        ]
        token_out = module._capacity_matrix(token_path, token_rows, hidden_size)[
            :tokens
        ]
        state_out = output.reshape(-1, hidden_size)[: tokens * streams]
        # The separate BF16 output stores preserve the original projection/add rounding.
        torch.mm(x, embedding_fc_weight.t(), out=token_out)
        torch.mm(h, hidden_fc_weight.t(), out=state_out)
        view = state_out.view(tokens, streams, hidden_size)
        torch.add(view, token_out[:, None, :], out=view)

    module._qwen_cute_projections = projections


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
            raise SystemExit("Qwen MTP source mismatch")
        spec.loader = Loader(spec.loader)
        return spec


sys.meta_path.insert(0, Finder())
