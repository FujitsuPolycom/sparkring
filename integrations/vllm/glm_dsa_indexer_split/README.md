# glm_dsa_indexer_split

A vLLM general plugin for GLM-5.3 (`glm_moe_dsa`) at tensor-parallel size 8
with decode context parallelism (DCP) 1, 2 or 4 on image
`sparkring-dev/kraken:csf-sircl-libsircl-20261008` (`816c6d6a7e96`; vLLM
`0.1.dev21553+gab86b7073`, Python sources of vLLM `bc9ea774`, B12X sources
`cc36aa6f`). Status: research-only; the CPU tests in
[`tests/`](tests/) run against the image's own sources, and the verify mode
below is the qualification step on the ring.

The B12X DSA indexer scores every prefill row of a step on every TP rank,
while each of the `8 / d` DCP groups holds one full KV copy and repeats the
same work. With `GLM_DSA_INDEXER_SPLIT=1`, group `j` scores and merges only
its block of the step's prefill rows, and one all-gather over the TP group
fills `topk_indices_buffer` for every row on every rank. Each row's selection
comes from the image's own `_run_paged_topk` with the same plan and per-row
inputs the image's launch of that row uses, merged by the image's
`_merge_dcp_topk`; on the earlier eight-Spark build the prefill gain measured
+8-9% at 16K, 64K and 128K with decode unchanged. The selections must equal
the image's, up to candidates tied at a row's lowest selected score.

Settings (each off by default; a malformed value refuses):

| Variable | Default | Meaning |
|---|---|---|
| `GLM_DSA_INDEXER_SPLIT` | `0` | `1` installs the wrappers. |
| `GLM_DSA_INDEXER_SPLIT_MIN_ROWS` | `512` | Smallest step, in prefill rows, that takes the split. |
| `GLM_DSA_INDEXER_SPLIT_FULL_LAUNCHES` | `0` | Cut each request's rows of the group's block into launches of the prepared prefill plan's row capacity (4,096) instead of the image's logits-budget launches. |
| `GLM_DSA_INDEXER_SPLIT_MIXED` | `0` | Also split steps that carry decode rows (their decode rows run the image's decode statements). |
| `GLM_DSA_INDEXER_SPLIT_VERIFY` | `0` | `N > 0` computes the first N eligible layer-chunks with the split and twice with the image's path, keeps the image's result and logs the comparison. Extra collectives and syncs: not for timing. |
| `GLM_DSA_INDEXER_SPLIT_VERIFY_MIN_CONTEXT` | `0` | Verify only layer-chunks whose request context reaches this many tokens. |

Every prefill step the split cannot handle as the image does takes the
image's `forward` unchanged and is counted (`runtime.FALLBACKS`): CUDA graph
captures, steps without this layer's metadata, the image's gathered-key
prefill path (`VLLM_DCP_INDEXER_KEY_GATHER=1`), mixed steps with
`..._MIXED=0`, short steps, chunks that are not single-request runs tiling
the rows in order, unexpected shapes, unprepared plans and arguments the
image itself refuses. DCP 8 at TP 8 leaves one KV copy and refuses when the
first indexer is constructed.

## Pins

`register` requires the SHA-256 of every image file whose behavior the plugin
relies on (`FILE_CHECKS` in `__init__.py`, including the two files it wraps in
`vllm/v1/attention/backends/mla/b12x_indexer.py`) to equal the recorded value,
and of the package's own other modules (`PACKAGE_SHA256`). Any other file
version refuses with `PatchRefused`; the plugin never falls back silently and
never writes a file.

## Tests

```bash
SPARKRING_GLM53_IMAGE_SOURCES=<image-sources> python -m pytest tests/ -q
```

`<image-sources>` is a directory with `vllm/` and `b12x/` subtrees holding the
target image's Python sources, extracted read-only from the image (one file
per path):

```bash
ssh SPARK 'sudo -n docker run --rm --network none --entrypoint cat 816c6d6a7e96 \
  /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/b12x_indexer.py \
  > image/vllm/v1/attention/backends/mla/b12x_indexer.py'
```

The suite covers the pins, the settings and their refusals, the row
partition and launch cutting, the split against the image's own `forward` on
eight thread-emulated ranks (exact index sets, the all-gather layout, the
counters), the fallbacks, the verify mode and registration in fresh
interpreters.
