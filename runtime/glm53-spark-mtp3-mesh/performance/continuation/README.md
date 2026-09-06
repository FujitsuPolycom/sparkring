# Final-chunk recurrent checkpoints

Status: **research-only**. The source package preserves the four continuation
files used by the serving image identified below. CPU tests verify package
integrity and ownership updates; they do not qualify the combined rebuilt image.

`source.tar.gz` contains exact vLLM source bytes for:

- `vllm/v1/core/kv_cache_manager.py`
- `vllm/v1/core/recurrent_prefill_checkpoint.py`
- `vllm/v1/core/sched/scheduler.py`
- `vllm/v1/core/single_type_kv_cache_manager.py`

The [manifest](manifest.json) binds the archive hash, each source preimage and
replacement, and the originating overlay manifest. The archived files preserve
their upstream notices; [LICENSE](LICENSE) contains the Apache 2.0 license and
[NOTICE](NOTICE) identifies the local modifications. Compression prevents text
checkout conversion and formatters from changing source bytes.

## Behavior

With `SPARK_GDN_PREFILL_CHECKPOINTS=2` and
`SPARK_GDN_CONTINUATION_CHECKPOINTS=1`, the scheduler can retain explicit
recurrent checkpoints inside the final continuation chunk of a cold text
prefill. The chunk is capped at 8,192 tokens. Eligibility starts only after
successful uncached admission and is removed on cleanup or invalidation.
Preempted, resumed, encoder, speculative, and incompatible request states do
not use this path.

The allocator validates the recurrent source and private speculative reserve,
preserves worker-visible block IDs, and allocates checkpoint destinations
without duplicate ownership. Publication boundaries must lie strictly inside
the selected chunk; an existing boundary at the chunk start is not a new export.

## Provenance and composition

The four files match the continuation serving image with config ID
`sha256:489d1975619e9083d14f14bcd1c6cbb4a96c41e3ab3978af048dc3ed2bb452a8`.
Its parent was the published cache/checkpoint image with config ID
`sha256:6921a6c163ea40b603e19a0332330efe3dbccbf4dce9f6cbbf6b756c9231835a`.
These identities establish source provenance, not qualification of another
image containing additional changes.

The performance image installer validates all checkpoint ownership dependencies
before applying these four source replacements. `install.py` verifies the
manifest, archive, source preimages, and corresponding ownership entries before
writing any source file. It updates only those four ownership hashes and
preserves every symbol requirement and unrelated dependency.

[Request-attribution instrumentation](../attribution/README.md) then applies its
separately pinned scheduler transform. The builder generates a complete installed
file inventory after both transforms. It does not copy the originating image's
performance or ownership receipts over the combined installation. Published
image contracts retain their immutable identities.

## Offline checks

```bash
python -m pytest runtime/glm53-spark-mtp3-mesh/performance/continuation -q
```

The tests reject altered packages, unexpected runtime preimages, and ownership
drift before any source write. Attribution tests execute scheduler boundaries
against both the fresh-prompt and continuation source variants.
