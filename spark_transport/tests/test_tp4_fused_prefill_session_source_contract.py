from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_fused_session_is_separate_exact_and_enqueue_only() -> None:
    h=(ROOT/'include/spark_transport/tp4_fused_prefill_session.hpp').read_text()
    c=(ROOT/'include/spark_transport/tp4_fused_prefill_c_api.h').read_text()
    s=(ROOT/'src/tp4_fused_prefill_session.cpp').read_text()
    assert 'class Tp4FusedPrefillSession' in h
    assert 'spark_tp4_fused_prefill_all_reduce' in c
    assert 'requires Q8192 width4096 rail2' in s
    assert 'worker_->wait_slot_idle(slot)' in s
    assert 'cudaMemcpyAsync' in s
    assert 'worker_->enqueue(sequence_, rail_bytes, slot)' in s
    assert 'spark_tp4_fused_prefill_all_reduce_rows' in c
    assert 'payload_bytes' in s
    assert 'launch_fused_prefill_q8192_n4' in s
    assert 'cudaStreamSynchronize(stream)' not in s
    assert 'std::_Exit(70)' in s
    assert 'kFusedPrefillArenaBytes' in s
    assert 'kOperationSlots = 2' in s
    assert 'sequence_ % kOperationSlots' in s
    assert 'cudaEventCreateWithFlags(&event' in s
    wait_idle = s.index('worker_->wait_slot_idle(slot)')
    prior_event = s.index('cudaEventSynchronize(kernel_done_[slot])', wait_idle)
    descriptor_upload = s.index('cudaMemcpyAsync', prior_event)
    launch = s.index('launch_fused_prefill_q8192_n4', descriptor_upload)
    event_record = s.index('cudaEventRecord(kernel_done_[slot], stream)', launch)
    assert wait_idle < prior_event < descriptor_upload < launch < event_record
    assert 'stable caller stream' not in s
    destructor = s.index('~Impl()')
    assert s.index('worker_.reset()', destructor) < s.index(
        'cudaEventSynchronize(kernel_done_[slot])', destructor
    )
