# Fused indexer barrier checks

`gpu_barrier_probe.py` uses the installed B12X helper with a delayed publishing
warp. Each kernel runs one barrier; it cannot enter a second barrier with only
some blocks participating. Run against the unpatched B12X source to
compare the original helper with the additional entry synchronization.

`stress_indexer.py` checks the patched full indexer against its numerical oracle
over 2,000 CUDA graph replays with concurrent copies. It imports the existing
test helpers from `tests/attention/test_fused_indexer.py` in B12X commit
`9ae41c5cb9935d740456479954b0089f80bd2ef2`. Put that directory on `PYTHONPATH` and
install pytest 8.4.2. Apply the
[checked barrier transform](../../../runtime/glm53-flash-jj-r8-gb10/patch_indexer_barrier.py)
to the source used by the stress check. The
[validation record](../../records/glm53-flash/issue224-dgx4-validation.md)
identifies the unpatched image and both patched source hashes. Isolate the
container and compilation cache so these tests do not replace serving code or
reuse compiled kernels from the other source state.

The publication probe requires the original helper to expose incomplete data
within 10 eager calls and 10 graph replays. If it does not, the run is inconclusive
about the race; it does not establish that the original helper is correct.

```bash
python3 gpu_barrier_probe.py
PYTHONPATH=/path/to/b12x/tests/attention python3 stress_indexer.py
```

See [the GPU evidence](../../records/glm53-flash/issue224-dgx4-validation.md)
for tested image/source hashes and limits. These are kernel checks, not a full
model or SparkCache serving soak.
