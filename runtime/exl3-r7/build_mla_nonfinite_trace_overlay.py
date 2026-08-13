#!/usr/bin/env python3
"""Build a graph-safe internal MLA nonfinite-trace vLLM overlay.

The generated ``mla.py`` records device-side nonfinite counts for the fused
Q/KV projection, Q up-projection, normalized KV latent, raw MLA output, local
output projection, and post-TP-reduction output. A paired ``deepseek_v2.py``
overlay owns the persistent count buffer and performs the only host sync.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


EXPECTED_SOURCE_SHA256 = "0cbbcc412238d27afe1e2980d14b29c973bfa5a38d90bc8e78dedba3e7a3526d"


def _replace_once(source: str, old: str, new: str, *, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"{label} preimage must occur exactly once, found {count}; "
            "the composed vLLM source has drifted"
        )
    return source.replace(old, new, 1)


def patch_source(source: str) -> str:
    source = _replace_once(
        source,
        "from vllm.config import CacheConfig, get_current_vllm_config\n",
        "from vllm.config import CacheConfig, get_current_vllm_config\n"
        "from vllm.distributed import tensor_model_parallel_all_reduce\n",
        label="TP all-reduce import",
    )
    source = _replace_once(
        source,
        '''        self.prefix = prefix

    def forward(''',
        '''        self.prefix = prefix
        self._r7_nonfinite_counts: torch.Tensor | None = None
        self._r7_nonfinite_base = -1

    def _r7_nonfinite_record(self, offset: int, value: torch.Tensor) -> None:
        counts = self._r7_nonfinite_counts
        if counts is not None:
            counts.select(0, self._r7_nonfinite_base + offset).copy_(
                torch.count_nonzero(~torch.isfinite(value))
            )

    def forward(''',
        label="MLA trace state",
    )
    source = _replace_once(
        source,
        '''            qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
            q_c, kv_lora = qkv_lora.split(''',
        '''            qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
            self._r7_nonfinite_record(2, qkv_lora)
            q_c, kv_lora = qkv_lora.split(''',
        label="fused QKV trace",
    )
    source = _replace_once(
        source,
        '''            q_c = self.q_a_layernorm(q_c)
            q = self.q_b_proj(q_c)[0]
        else:''',
        '''            q_c = self.q_a_layernorm(q_c)
            q = self.q_b_proj(q_c)[0]
            self._r7_nonfinite_record(3, q)
        else:''',
        label="Q up-projection trace",
    )
    source = _replace_once(
        source,
        '''        kv_c_normed = self.kv_a_layernorm(kv_c)
        # Normalize only the 512-D compressed latent before the cache writer.''',
        '''        kv_c_normed = self.kv_a_layernorm(kv_c)
        self._r7_nonfinite_record(4, kv_c_normed)
        # Normalize only the 512-D compressed latent before the cache writer.''',
        label="KV latent trace",
    )
    source = _replace_once(
        source,
        '''        attn_out = self.mla_attn(
            q,
            kv_c_for_cache,
            k_pe,
            output_shape=(hidden_states.shape[0], self.num_heads * self.v_head_dim),
        )

        return self.o_proj(attn_out)[0]''',
        '''        attn_out = self.mla_attn(
            q,
            kv_c_for_cache,
            k_pe,
            output_shape=(hidden_states.shape[0], self.num_heads * self.v_head_dim),
        )
        self._r7_nonfinite_record(5, attn_out)

        if self._r7_nonfinite_counts is None:
            return self.o_proj(attn_out)[0]

        # Mirror RowParallelLinear.forward exactly so the diagnostic can see
        # the local projection before its normal tensor-parallel reduction.
        if not self.o_proj.input_is_parallel:
            raise RuntimeError(
                "R7 MLA nonfinite trace requires an input-parallel o_proj"
            )
        bias = (
            None
            if self.o_proj.tp_rank > 0 or self.o_proj.skip_bias_add
            else self.o_proj.bias
        )
        output_parallel = self.o_proj.quant_method.apply(self.o_proj, attn_out, bias)
        self._r7_nonfinite_record(6, output_parallel)
        if self.o_proj.reduce_results and self.o_proj.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output = output_parallel
        self._r7_nonfinite_record(7, output)
        return output''',
        label="MLA core and output projection trace",
    )
    return source


def build(source_path: Path, output_path: Path, expected_sha256: str) -> None:
    source_bytes = source_path.read_bytes()
    actual_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "mla.py SHA-256 mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    patched = patch_source(source_bytes.decode("utf-8"))
    compile(patched, str(output_path), "exec")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(patched, encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--expected-sha256",
        default=EXPECTED_SOURCE_SHA256,
        help="fail-closed SHA-256 for the composed mla.py preimage",
    )
    args = parser.parse_args()
    build(args.source, args.output, args.expected_sha256.lower())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
