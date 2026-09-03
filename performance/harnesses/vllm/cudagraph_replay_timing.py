"""Test-harness import surface for the runtime replay timer."""

from spark_transport.integrations.vllm.spark_cudagraph_replay_timing import (
    ReplayTimingCollector,
    _descriptor_key,
    graph_replay_timing_snapshot,
    install,
)

__all__ = (
    "ReplayTimingCollector",
    "_descriptor_key",
    "graph_replay_timing_snapshot",
    "install",
)
