# Packed recurrent-state external-prefix admission

Status: **implemented**, with GPU-free regression coverage. GPU restore and
performance qualification are required before deployment.

The vLLM Mamba allocator represents skipped recurrent-state positions with null
block-table entries. A fresh external prefix does not yet have that padded table
when full-sequence admission runs. Counting its logical positions as physical
blocks can reject a cache load indefinitely even while the GPU cache is empty.

The [patch](fix.patch) includes those skipped positions when computing covered
logical slots. It preserves the full-sequence reservation check, local-hit
accounting, running-request tables, and speculative-state reservations. It does
not change the allocator's physical layout or the cache-ownership methods.

Conditions reproduced on CPU: a 30,720-token external prefix, 31,124-token prompt,
256-token recurrent blocks, four checkpoint slots, six recurrent groups and a
584-block pool. Before correction, zero-speculation admission requests 748 blocks
including attention pages; the corrected accounting fits. Tests also cover local,
running and empty prefixes with zero, one and three speculative blocks.

[Source identities](source-identities.json) pin the expected integrated-source
preimages and results. Apply only to matching source; do not use fuzzy patching.
The regression lives in the existing vLLM recovery-source suite:

```bash
.venv/bin/python -m pytest --noconftest -p no:cacheprovider \
  tests/v1/kv_connector/unit/test_hybrid_recovery_source.py -q
```

The recorded CPU result is 38 passed. These tests execute source methods without
running the full scheduler or CUDA copies. Cache identity fields, salts and chunk
geometry remain unchanged. A changed runtime-source identity still requires its
own serving/cache qualification; retain the reference image for rollback.
