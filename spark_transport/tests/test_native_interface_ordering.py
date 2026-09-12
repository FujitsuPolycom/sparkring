"""Source contracts for the shared GPU doorbell acquire boundary.

These assertions verify emitted instruction intent and its vocabulary caller;
they do not simulate CUDA visibility or qualify RDMA hardware ordering.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_shared_doorbell_wait_acquires_before_cta_handoff():
    source = (ROOT / "include/spark_transport/gpu_graph_command.cuh").read_text()
    wait = source[source.index("void wait_for_sequence_block("):]
    assert 'asm volatile("ld.acquire.sys.global.u64 %0, [%1];"' in wait
    assert ': "memory"' in wait
    assert wait.index("ld.acquire.sys.global.u64") < wait.index("__syncthreads()")
    assert "reinterpret_cast<const volatile" not in wait
    assert "__nanosleep(64)" in wait
    assert "publish_overflow(graph_commands, graph_sequence)" in wait


def test_vocabulary_consumer_uses_shared_acquire_for_each_remote_payload():
    source = (ROOT / "src/gpu_tp4_vocab_allgather.cu").read_text()
    assert '#include "spark_transport/gpu_graph_command.cuh"' in source
    for slot in (0, 1):
        assert f"&control{slot}->remote_sequence, doorbell_sequence, graph_commands" in source
    assert source.count("gpu_graph_command::wait_for_sequence_block(") == 4
