"""CPU check of vLLM LogitsProcessor.get_top_tokens against a full-vocabulary argmax.

Runs the installed method with its head projection and all-gather replaced by
precomputed two-rank shards, on bf16 logits rounded to a coarse grid so that
ties across and within shards are frequent.
"""
import types
import torch
import vllm.model_executor.layers.logits_processor as lpm

VOCAB, TP, ROWS, TRIALS = 154880, 2, 64, 200
shard = VOCAB // TP
processor = object.__new__(lpm.LogitsProcessor)
processor.scale, processor.soft_cap = 1.0, None
mismatches = ties = 0
generator = torch.Generator().manual_seed(0)
for trial in range(TRIALS):
    grid = [0.125, 0.25, 1.0, 4.0][trial % 4]
    logits = (torch.randn(ROWS, VOCAB, generator=generator) * 3).div(grid).round().mul(grid).to(torch.bfloat16)
    # Force exact cross-shard ties at the maximum for some rows.
    top = logits.float().max(dim=-1).values
    for row in range(0, ROWS, 4):
        logits[row, shard + int(torch.randint(shard, (1,), generator=generator))] = top[row].to(torch.bfloat16)
    expected = logits.argmax(dim=-1)
    ties += int(((logits == logits.max(dim=-1, keepdim=True).values).sum(dim=-1) > 1).sum())
    pairs = []
    for rank in range(TP):
        head = types.SimpleNamespace(tp_size=TP, shard_indices=types.SimpleNamespace(
            num_org_vocab_padding=0, org_vocab_start_index=rank * shard))
        processor._apply_head = lambda *_a, rank=rank: logits[:, rank * shard:(rank + 1) * shard].clone()
        captured = {}
        def record(pair, dim=-1, captured=captured):
            captured["pair"] = pair
            return torch.cat([pair] * TP, dim=-1)
        lpm.tensor_model_parallel_all_gather = record
        processor.get_top_tokens(head, torch.empty(ROWS, 8))
        pairs.append(captured["pair"])
    lpm.tensor_model_parallel_all_gather = lambda pair, dim=-1: torch.cat(pairs, dim=-1)
    results = []
    for rank in range(TP):
        head = types.SimpleNamespace(tp_size=TP, shard_indices=types.SimpleNamespace(
            num_org_vocab_padding=0, org_vocab_start_index=rank * shard))
        processor._apply_head = lambda *_a, rank=rank: logits[:, rank * shard:(rank + 1) * shard].clone()
        results.append(processor.get_top_tokens(head, torch.empty(ROWS, 8)))
    assert torch.equal(results[0], results[1])
    mismatches += int((results[0] != expected).sum())
print(f"rows={TRIALS * ROWS} rows_with_tied_maximum={ties} mismatches={mismatches}")
