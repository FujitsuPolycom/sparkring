# Fused indexer barrier checks

`gpu_barrier_probe.py` uses the installed B12X helper with a delayed publishing
warp. It runs one barrier to expose incomplete publication without intentionally
launching the divergent second round. Run against the unpatched B12X source to
compare the original helper with the additional entry synchronization.

`stress_indexer.py` checks the patched full indexer against its numerical oracle
over 2,000 CUDA graph replays with concurrent copies. It imports the existing
test helpers from `tests/attention/test_fused_indexer.py` in B12X commit
`9ae41c5cb9935d740456479954b0089f80bd2ef2`. Put that directory on `PYTHONPATH` and
install pytest 8.4.2. Use an isolated container and compilation cache.

```bash
python3 gpu_barrier_probe.py
PYTHONPATH=/path/to/b12x/tests/attention python3 stress_indexer.py
```

See [the GPU evidence](../../records/glm53-flash/issue224-dgx4-validation.md)
for tested image/source hashes and limits. These are kernel checks, not a full
model or SparkCache serving soak.
