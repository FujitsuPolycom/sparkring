"""GLM-shaped patched fused-indexer graph correctness stress on one CUDA device.

The recorded device name identifies the hardware; the script does not gate on
it. ``wall_seconds`` spans 500 replays including the per-replay host-side
correctness checks and synchronizations, so it bounds but does not measure
kernel time.
"""
import hashlib
import inspect
import json
import time
from pathlib import Path

import torch
from test_fused_indexer import _build_case, _golden_topk
from b12x.attention.dsa_indexer.fused_indexer import (
    run_fused_paged_indexer, fused_indexer_scratch_capacity,
)


def check(idx, val, gold_values, gold_sets):
    assert torch.allclose(torch.sort(val, dim=1, descending=True).values,
                          gold_values, atol=1e-2, rtol=0)
    for row, expected in enumerate(gold_sets):
        assert set(idx[row].tolist()) == expected, row


def main():
    source = Path(inspect.getsourcefile(run_fused_paged_indexer))
    print(json.dumps({"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                      "gpu": torch.cuda.get_device_name(), "torch": torch.__version__}), flush=True)
    for rows, max_len in ((3, 65536), (4, 200000), (8, 65536), (16, 65536)):
        q, w, k, scales, pages, lengths = _build_case(
            rows, 32, max_len, 512, seed=224 + rows, device=torch.device("cuda"))
        capacity, state_words = fused_indexer_scratch_capacity(rows, 512, 48)
        pack_v = torch.empty(capacity, dtype=torch.float32, device="cuda")
        pack_i = torch.empty(capacity, dtype=torch.int32, device="cuda")
        state = torch.zeros(state_words, dtype=torch.int32, device="cuda")
        idx = torch.empty((rows, 512), dtype=torch.int32, device="cuda")
        val = torch.empty((rows, 512), dtype=torch.float32, device="cuda")
        kwargs = dict(q_bytes=q.view(torch.uint8), weights=w,
                      k_quant_bytes=k.view(torch.uint8), k_scales=scales,
                      real_page_table=pages, seqlens=lengths, num_heads=32, topk=512,
                      out_indices=idx, out_values=val, merge_threshold=0,
                      pack_values=pack_v, pack_indices=pack_i, merge_state=state,
                      merge_state_preinitialized=True)
        gold = {}
        for length in (4097, max_len):
            lengths.fill_(length)
            gold[length] = _golden_topk(q, w, k, scales, pages, lengths, 512)
        lengths.fill_(max_len)
        run_fused_paged_indexer(**kwargs)
        torch.cuda.synchronize()
        check(idx, val, *gold[max_len])
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run_fused_paged_indexer(**kwargs)
        copy_src = torch.ones(16 * 1024 * 1024, dtype=torch.float32, device="cuda")
        copy_dst = torch.empty_like(copy_src)
        copy_stream = torch.cuda.Stream()
        copy_stream.wait_stream(torch.cuda.current_stream())
        started = time.monotonic()
        for iteration in range(500):
            length = (4097, max_len)[iteration % 2]
            lengths.fill_(length)
            with torch.cuda.stream(copy_stream):
                copy_dst.copy_(copy_src)
            graph.replay()
            torch.cuda.synchronize()
            check(idx, val, *gold[length])
            assert int(state.abs().sum()) == 0
        print(json.dumps({"rows": rows, "heads": 32, "topk": 512,
                          "lengths": [4097, max_len], "graph_replays": 500,
                          "concurrent_copy_bytes": copy_src.numel()*4,
                          "wall_seconds": time.monotonic()-started,
                          "wall_seconds_scope": "replays plus host checks and synchronizations",
                          "result": "passed"}), flush=True)
        del graph, kwargs, q, w, k, scales, pages, lengths, pack_v, pack_i, state
        del idx, val, gold, copy_src, copy_dst, copy_stream
        torch.cuda.empty_cache()
    print("PASS: 2000 GLM-shaped graph replays with numerical parity and concurrent copies", flush=True)


if __name__ == "__main__":
    main()
