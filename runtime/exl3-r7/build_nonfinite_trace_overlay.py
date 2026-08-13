#!/usr/bin/env python3
"""Build a fail-closed GLM-5.2 nonfinite-trace vLLM overlay.

The generated ``deepseek_v2.py`` records one device-side nonfinite count at
each model boundary. Counts are reset and updated inside the compiled model,
so FULL_AND_PIECEWISE CUDA-graph replay remains supported. ``compute_logits``
performs the only host synchronization, reports the first failing rank/layer
boundary, and raises before sampling can serialize NaN logits.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


EXPECTED_SOURCE_SHA256 = "27cf047c09c829174025c7372403ae2607315acbb063463582dbb43607eebf04"
DEBUG_TAG = "DEBUG-r7nf-7c1d"


def _replace_once(source: str, old: str, new: str, *, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"{label} preimage must occur exactly once, found {count}; "
            "the composed vLLM source has drifted"
        )
    return source.replace(old, new, 1)


def patch_source(source: str, *, detailed: bool = True) -> str:
    source = _replace_once(
        source,
        "import operator\nimport re\n",
        "import operator\nimport os\nimport re\n",
        label="os import",
    )
    source = _replace_once(
        source,
        "logger = init_logger(__name__)\n\n\ndef _get_moe_router_dtype(",
        '''logger = init_logger(__name__)

_R7_NONFINITE_TRACE_ENV = "VLLM_SPARK_R7_NONFINITE_TRACE"
_R7_NONFINITE_LAYER_STAGES = (
    "attention_input",
    "residual_before_attention",
    "mla_fused_qkv_a_output",
    "mla_q_b_proj_output",
    "mla_kv_a_norm_output",
    "mla_attention_core_output",
    "mla_o_proj_local_output",
    "attention_output",
    "mlp_input",
    "residual_before_mlp",
    "mlp_output",
)


def _r7_nonfinite_trace_enabled() -> bool:
    raw = os.environ.get(_R7_NONFINITE_TRACE_ENV, "0").strip()
    if raw not in ("0", "1"):
        raise ValueError(f"{_R7_NONFINITE_TRACE_ENV} must be 0 or 1, got {raw!r}")
    return raw == "1"


def _r7_nonfinite_trace_record(
    counts: torch.Tensor | None,
    slot: int,
    value: torch.Tensor | None,
) -> None:
    if counts is None or value is None:
        return
    counts.select(0, slot).copy_(torch.count_nonzero(~torch.isfinite(value)))


def _r7_nonfinite_trace_report(model, hidden_states, logits) -> None:
    counts = model._r7_nonfinite_counts
    if counts is None:
        return
    if logits is not None:
        _r7_nonfinite_trace_record(counts, model._r7_nonfinite_logits_slot, logits)

    # This is the probe's only device-to-host synchronization. Every layer
    # boundary above records into persistent graph-owned storage.
    host_counts = counts.detach().cpu().tolist()
    failures = [
        (model._r7_nonfinite_names[index], int(count))
        for index, count in enumerate(host_counts)
        if count
    ]
    rank = os.environ.get("RANK", "unknown")
    if failures:
        first_name, first_count = failures[0]
        summary = ",".join(
            f"{name}:{count}" for name, count in failures[:12]
        )
        if len(failures) > 12:
            summary += f",...(+{len(failures) - 12})"
        logger.error(
            "[DEBUG-r7nf-7c1d] rank=%s first_nonfinite=%s count=%d "
            "failed=%s hidden_shape=%s hidden_dtype=%s logits_shape=%s",
            rank,
            first_name,
            first_count,
            summary,
            tuple(hidden_states.shape),
            hidden_states.dtype,
            None if logits is None else tuple(logits.shape),
        )
        raise RuntimeError(
            "[DEBUG-r7nf-7c1d] nonfinite model state: "
            f"rank={rank}, first={first_name}, count={first_count}"
        )

    if not model._r7_nonfinite_reported_finite:
        logger.warning(
            "[DEBUG-r7nf-7c1d] rank=%s all %d model boundaries finite; "
            "hidden_shape=%s logits_shape=%s",
            rank,
            len(host_counts),
            tuple(hidden_states.shape),
            None if logits is None else tuple(logits.shape),
        )
        model._r7_nonfinite_reported_finite = True


def _get_moe_router_dtype(''',
        label="trace helpers",
    )
    source = _replace_once(
        source,
        '''        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)

    def forward(''',
        '''        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self._r7_nonfinite_counts: torch.Tensor | None = None
        self._r7_nonfinite_base = -1

    def forward(''',
        label="decoder trace state",
    )
    source = _replace_once(
        source,
        '''        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        if input_is_sequence_parallel:''',
        '''        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        trace_counts = self._r7_nonfinite_counts
        if trace_counts is not None:
            trace_base = self._r7_nonfinite_base
            _r7_nonfinite_trace_record(trace_counts, trace_base, hidden_states)
            _r7_nonfinite_trace_record(trace_counts, trace_base + 1, residual)

        if input_is_sequence_parallel:''',
        label="attention input trace",
    )
    source = _replace_once(
        source,
        '''        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        if self.use_sequence_parallel_moe:''',
        '''        if trace_counts is not None:
            _r7_nonfinite_trace_record(trace_counts, trace_base + 7, hidden_states)

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        if trace_counts is not None:
            _r7_nonfinite_trace_record(trace_counts, trace_base + 8, hidden_states)
            _r7_nonfinite_trace_record(trace_counts, trace_base + 9, residual)
        if self.use_sequence_parallel_moe:''',
        label="attention and MLP input trace",
    )
    source = _replace_once(
        source,
        '''            hidden_states *= 1.0 / self.routed_scaling_factor

        return hidden_states, residual


@support_torch_compile''',
        '''            hidden_states *= 1.0 / self.routed_scaling_factor

        if trace_counts is not None:
            _r7_nonfinite_trace_record(trace_counts, trace_base + 10, hidden_states)

        return hidden_states, residual


@support_torch_compile''',
        label="MLP output trace",
    )
    source = _replace_once(
        source,
        '''        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], self.hidden_size
        )

        self.aux_hidden_state_layers = tuple[int, ...]()''',
        '''        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], self.hidden_size
        )

        self._r7_nonfinite_names: tuple[str, ...] = ()
        self._r7_nonfinite_logits_slot = -1
        self._r7_nonfinite_reported_finite = False
        if _r7_nonfinite_trace_enabled():
            names = ["model_input"]
            for layer_idx in range(int(config.num_hidden_layers)):
                names.extend(
                    f"layers.{layer_idx}.{stage}"
                    for stage in _R7_NONFINITE_LAYER_STAGES
                )
            names.extend(("final_norm", "logits"))
            self._r7_nonfinite_names = tuple(names)
            self._r7_nonfinite_logits_slot = len(names) - 1
            self.register_buffer(
                "_r7_nonfinite_counts",
                torch.zeros(len(names), dtype=torch.int64, device=self.device),
                persistent=False,
            )
            for layer_idx, layer in enumerate(self.layers):
                if isinstance(layer, DeepseekV2DecoderLayer):
                    object.__setattr__(
                        layer, "_r7_nonfinite_counts", self._r7_nonfinite_counts
                    )
                    layer._r7_nonfinite_base = 1 + layer_idx * len(
                        _R7_NONFINITE_LAYER_STAGES
                    )
                    mla_wrapper = layer.self_attn.mla_attn
                    object.__setattr__(
                        mla_wrapper,
                        "_r7_nonfinite_counts",
                        self._r7_nonfinite_counts,
                    )
                    mla_wrapper._r7_nonfinite_base = layer._r7_nonfinite_base
        else:
            self._r7_nonfinite_counts = None

        self.aux_hidden_state_layers = tuple[int, ...]()''',
        label="model trace allocation",
    )
    source = _replace_once(
        source,
        '''            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        # Compute llama 4 scaling once per forward pass if enabled''',
        '''            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        trace_counts = self._r7_nonfinite_counts
        if trace_counts is not None:
            trace_counts.zero_()
            _r7_nonfinite_trace_record(trace_counts, 0, hidden_states)

        # Compute llama 4 scaling once per forward pass if enabled''',
        label="model input trace",
    )
    source = _replace_once(
        source,
        '''        hidden_states, _ = self.norm(hidden_states, residual)
        if len(aux_hidden_states) > 0:''',
        '''        hidden_states, _ = self.norm(hidden_states, residual)
        if trace_counts is not None:
            final_norm_slot = 1 + int(self.config.num_hidden_layers) * len(
                _R7_NONFINITE_LAYER_STAGES
            )
            _r7_nonfinite_trace_record(trace_counts, final_norm_slot, hidden_states)
        if len(aux_hidden_states) > 0:''',
        label="final norm trace",
    )
    source = _replace_once(
        source,
        '''        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits''',
        '''        logits = self.logits_processor(self.lm_head, hidden_states)
        _r7_nonfinite_trace_report(self.model, hidden_states, logits)
        return logits''',
        label="logits report",
    )
    if not detailed:
        source = _replace_once(
            source,
            '''_R7_NONFINITE_LAYER_STAGES = (
    "attention_input",
    "residual_before_attention",
    "mla_fused_qkv_a_output",
    "mla_q_b_proj_output",
    "mla_kv_a_norm_output",
    "mla_attention_core_output",
    "mla_o_proj_local_output",
    "attention_output",
    "mlp_input",
    "residual_before_mlp",
    "mlp_output",
)''',
            '''_R7_NONFINITE_LAYER_STAGES = (
    "attention_input",
    "residual_before_attention",
    "attention_output",
    "mlp_input",
    "residual_before_mlp",
    "mlp_output",
)''',
            label="basic trace stage set",
        )
        source = _replace_once(
            source,
            "_r7_nonfinite_trace_record(trace_counts, trace_base + 7, hidden_states)",
            "_r7_nonfinite_trace_record(trace_counts, trace_base + 2, hidden_states)",
            label="basic attention output slot",
        )
        source = _replace_once(
            source,
            "_r7_nonfinite_trace_record(trace_counts, trace_base + 8, hidden_states)",
            "_r7_nonfinite_trace_record(trace_counts, trace_base + 3, hidden_states)",
            label="basic MLP input slot",
        )
        source = _replace_once(
            source,
            "_r7_nonfinite_trace_record(trace_counts, trace_base + 9, residual)",
            "_r7_nonfinite_trace_record(trace_counts, trace_base + 4, residual)",
            label="basic MLP residual slot",
        )
        source = _replace_once(
            source,
            "_r7_nonfinite_trace_record(trace_counts, trace_base + 10, hidden_states)",
            "_r7_nonfinite_trace_record(trace_counts, trace_base + 5, hidden_states)",
            label="basic MLP output slot",
        )
        source = _replace_once(
            source,
            '''                    mla_wrapper = layer.self_attn.mla_attn
                    object.__setattr__(
                        mla_wrapper,
                        "_r7_nonfinite_counts",
                        self._r7_nonfinite_counts,
                    )
                    mla_wrapper._r7_nonfinite_base = layer._r7_nonfinite_base
''',
            "",
            label="basic trace MLA attachment",
        )
    return source


def build(
    source_path: Path,
    output_path: Path,
    expected_sha256: str,
    *,
    detailed: bool = True,
) -> None:
    source_bytes = source_path.read_bytes()
    actual_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "deepseek_v2.py SHA-256 mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    patched = patch_source(source_bytes.decode("utf-8"), detailed=detailed)
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
        help="fail-closed SHA-256 for the composed deepseek_v2.py preimage",
    )
    parser.add_argument(
        "--basic",
        action="store_true",
        help="emit the original decoder-boundary-only trace overlay",
    )
    args = parser.parse_args()
    build(
        args.source,
        args.output,
        args.expected_sha256.lower(),
        detailed=not args.basic,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
