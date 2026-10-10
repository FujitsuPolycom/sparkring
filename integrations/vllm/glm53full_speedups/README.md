# glm53full_speedups

A vLLM general plugin for GLM-5.3-NVFP4 at tensor-parallel size 8 on image
`sparkring-dev/kraken:csf-sircl-libsircl-20261008` (`816c6d6a7e96`; vLLM
`0.1.dev21553+gab86b7073`, Python sources of vLLM `bc9ea774`, B12X sources
`cc36aa6f`). Status: research-only; the CPU tests in [`tests/`](tests/) run
against the image's own sources, and no item has run on a GPU.

The DSA model in `vllm/models/deepseek_v32` replicates two BF16 projections
on every TP rank. Each item has its own environment flag (`1` selects it,
`0` or unset leaves the image's code untouched), and both keep the image's
layer at any tensor-parallel size other than 8:

- `GLM53FULL_LATENT_SHARD`: the fused `q_a`/`kv_a` latent projection
  (`fused_qkv_a_proj`, 2624 x 6144, built `disable_tp=True`) becomes
  column-parallel: rank `r` computes fused rows `[328 r, 328 r + 328)` and
  one all-gather concatenates the blocks in rank order, reproducing the
  fused output's column order. The layer's loader copies only the rows that
  intersect the rank's block, so the b12x checkpoint loader reads exactly
  those bytes. Loading is byte-exact; only the GEMM's reduction order can
  differ.
- `GLM53FULL_EH_PROJ_TP`: the MTP draft input projection `eh_proj`
  (12288 -> 6144) becomes `RowParallelLinear(input_is_parallel=False)`:
  rank `r` keeps input columns `[1536 r, 1536 (r + 1))` and the
  tensor-parallel all-reduce sums the partial outputs. Only draft numerics
  change; rejection sampling keeps the target distribution.

## Pins

`register` requires the SHA-256 of the two edited files
(`vllm/models/deepseek_v32/attention.py`,
`vllm/models/deepseek_v32/nvidia/mtp.py`) and of every file whose behavior an
item relies on (`FILE_CHECKS`) to equal the recorded value. Any other file
version refuses with `PatchRefused`; the plugin never falls back silently and
never writes a file.

## Tests

```bash
SPARKRING_GLM53_IMAGE_SOURCES=<image-sources> python -m pytest tests/ -q
```

`<image-sources>` is a directory with `vllm/` and `b12x/` subtrees holding
the target image's Python sources, extracted read-only from the image (see
[the split plugin's README](../glm_dsa_indexer_split/README.md#tests) for the
extraction command). The suite covers the pins, the patch mechanics (each
edit applies exactly once and the compiled original equals the whole-file
compile), installation and its refusals, the latent shard's weight loading
(byte-exact per rank) and all-gathered forward, the row-parallel `eh_proj`
loading and reduced forward, and registration in fresh interpreters.
