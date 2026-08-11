#!/usr/bin/env python3
"""Build the R7 vLLM shared CUDA-capture-stream overlay.

Spark TP4 graph sessions require one stable caller stream. vLLM captures the
target model and each speculative-draft manager in separate graph-capture
contexts, so the stock context constructor creates a different stream for each
manager. The generated overlay retains one dedicated stream per process and
CUDA device while constructing a fresh ``GraphCaptureContext`` for every
manager. Distinct channel identities and manager-owned graph pools therefore
remain independent.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


EXPECTED_SOURCE_SHA256 = "df8560a9568eb45d9b52939db847854f3ed67cd2ee1cbd50ccccf278366a710f"


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
        "import contextlib\nimport gc\nimport pickle\nimport weakref\n",
        "import contextlib\nimport gc\nimport os\nimport pickle\nimport threading\nimport weakref\n",
        label="import",
    )
    source = _replace_once(
        source,
        "\n\n@contextmanager\ndef graph_capture(\n",
        '''

_SPARK_SHARED_CAPTURE_STREAMS: dict[tuple[int, int], torch.cuda.Stream] = {}
_SPARK_ACTIVE_CAPTURE_STREAMS: set[tuple[int, int]] = set()
_SPARK_CAPTURE_STREAM_LOCK = threading.Lock()


@contextmanager
def _spark_graph_capture_context(
    device: torch.device,
    graph_capture_context: GraphCaptureContext | None,
    channel_id: str | None,
):
    """Return one fresh capture context on the process/device shared stream."""
    shared_key: tuple[int, int] | None = None
    shared_guard_acquired = False
    try:
        if graph_capture_context is None:
            canonical_device = torch.device(device)
            shared = os.getenv("VLLM_SPARK_SHARED_CAPTURE_STREAM", "0") == "1"
            if shared:
                device_index = canonical_device.index
                if device_index is None:
                    device_index = torch.cuda.current_device()
                shared_key = (os.getpid(), int(device_index))
                with _SPARK_CAPTURE_STREAM_LOCK:
                    if shared_key in _SPARK_ACTIVE_CAPTURE_STREAMS:
                        raise RuntimeError(
                            "overlapping Spark shared CUDA graph capture is unsupported"
                        )
                    _SPARK_ACTIVE_CAPTURE_STREAMS.add(shared_key)
                    shared_guard_acquired = True
                    stream = _SPARK_SHARED_CAPTURE_STREAMS.get(shared_key)
                    if stream is None:
                        stream = torch.cuda.Stream(device=canonical_device)
                        _SPARK_SHARED_CAPTURE_STREAMS[shared_key] = stream
            else:
                stream = torch.cuda.Stream(device=canonical_device)
            context = GraphCaptureContext(stream, channel_id=channel_id)
        else:
            context = graph_capture_context
            if channel_id is not None:
                if context.channel_id is not None and context.channel_id != channel_id:
                    raise ValueError(
                        "graph capture context and argument specify different "
                        "semantic channel IDs"
                    )
                if context.channel_id is None:
                    context = GraphCaptureContext(context.stream, channel_id=channel_id)
        yield context
    finally:
        if shared_guard_acquired:
            assert shared_key is not None
            with _SPARK_CAPTURE_STREAM_LOCK:
                _SPARK_ACTIVE_CAPTURE_STREAMS.discard(shared_key)


@contextmanager
def graph_capture(
''',
        label="shared context insertion",
    )
    old_body = '''    if graph_capture_context is None:
        context = GraphCaptureContext(
            torch.cuda.Stream(device=device),
            channel_id=channel_id,
        )
    else:
        context = graph_capture_context
        if channel_id is not None:
            if context.channel_id is not None and context.channel_id != channel_id:
                raise ValueError(
                    "graph capture context and argument specify different "
                    "semantic channel IDs"
                )
            if context.channel_id is None:
                context = GraphCaptureContext(context.stream, channel_id=channel_id)
    maybe_dcp_capture = (
        get_dcp_group().graph_capture(context)
        if _DCP is not None and get_dcp_group().world_size > 1
        else nullcontext()
    )
    maybe_b12x_dcp_capture: contextlib.AbstractContextManager[Any]
    if _DCP is not None and get_dcp_group().world_size > 1:
        # Import locally to avoid making distributed initialization depend on
        # attention modules. The helper is a no-op until DCP warmup creates a
        # B12X pool for this process group.
        from vllm.v1.attention.ops.dcp_alltoall import capture_b12x_dcp_a2a

        maybe_b12x_dcp_capture = capture_b12x_dcp_a2a(
            get_dcp_group(),
            context.stream,
            channel_id=context.channel_id,
        )
    else:
        maybe_b12x_dcp_capture = nullcontext()
    with (
        get_tp_group().graph_capture(context),
        get_pp_group().graph_capture(context),
        maybe_dcp_capture,
        maybe_b12x_dcp_capture,
    ):
        yield context
'''
    new_body = '''    with _spark_graph_capture_context(
        device,
        graph_capture_context,
        channel_id,
    ) as context:
        maybe_dcp_capture = (
            get_dcp_group().graph_capture(context)
            if _DCP is not None and get_dcp_group().world_size > 1
            else nullcontext()
        )
        maybe_b12x_dcp_capture: contextlib.AbstractContextManager[Any]
        if _DCP is not None and get_dcp_group().world_size > 1:
            # Import locally to avoid making distributed initialization depend on
            # attention modules. The helper is a no-op until DCP warmup creates a
            # B12X pool for this process group.
            from vllm.v1.attention.ops.dcp_alltoall import capture_b12x_dcp_a2a

            maybe_b12x_dcp_capture = capture_b12x_dcp_a2a(
                get_dcp_group(),
                context.stream,
                channel_id=context.channel_id,
            )
        else:
            maybe_b12x_dcp_capture = nullcontext()
        with (
            get_tp_group().graph_capture(context),
            get_pp_group().graph_capture(context),
            maybe_dcp_capture,
            maybe_b12x_dcp_capture,
        ):
            yield context
'''
    return _replace_once(
        source,
        old_body,
        new_body,
        label="graph_capture body",
    )


def build(source_path: Path, output_path: Path) -> str:
    source_bytes = source_path.read_bytes()
    actual = hashlib.sha256(source_bytes).hexdigest()
    if actual != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            "parallel_state.py SHA-256 mismatch: "
            f"expected {EXPECTED_SOURCE_SHA256}, got {actual}"
        )
    patched = patch_source(source_bytes.decode("utf-8"))
    compile(patched, str(output_path), "exec")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(patched, encoding="utf-8", newline="")
    return hashlib.sha256(patched.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.source, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
