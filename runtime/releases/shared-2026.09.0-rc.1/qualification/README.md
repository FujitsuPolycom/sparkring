# Bounded B12X GPU qualification inputs

The [manifest](manifest.json) pins the [harness](test_b12x_gpu_gates.py) and
reference fixtures used for prepared QSA, PLE and KDA checks. Fixture bytes match
the [accepted source reconstruction](../sources/README.md); their B12X
[license](B12X-LICENSE) is retained. This input bundle is not a test-result receipt.

Exact harness sources identified by existing receipts are retained in the
manifest's archives. Formatting the maintained harness does not transfer prior
GPU evidence to its different file hash; the fixture bytes remain unchanged.

## Frozen CPU oracle inputs

[execution-oracles-aaec6a5b.tar.gz](execution-oracles-aaec6a5b.tar.gz) preserves the
17 hash-bound vLLM/B12X gate inputs and `oracle-gates.json`, including the suite
JSON files' CRLF bytes. Archive SHA256:
`d2396570ae239d752b2d819c17a81540c0c30096a656a942f50d37ef1870c2c8`.
Its inventory SHA256 is
`aaec6a5b5fad1eebd609da9d87c8c2350c8e00d1a197650cff64da0a2c86c4d3`.
Use these archived bytes when reproducing receipts that name those hashes;
Git-normalized maintained copies can have different hashes. The archive contains
no operator policy, host configuration or wheels and does not assert test success.

## GPU harness

Run on an idle GB10 with the candidate's CUDA/PyTorch/Triton/CuTe environment,
an executable temporary directory for native compilation, and at least 32 GiB
free GPU-addressable memory for the large-pool case. Point `B12X_GATE_SOURCE_ROOT`
at the reconciled source root, or at the verified installed package's parent
directory. The harness checks the imported production package location.

```bash
QUALIFICATION="$PWD/runtime/releases/shared-2026.09.0-rc.1/qualification"
export B12X_GATE_SOURCE_ROOT=/opt/venv/lib/python3.12/site-packages
export B12X_GATE_TEST_ROOT="$QUALIFICATION/fixtures"
python -m pytest -q -p no:cacheprovider "$QUALIFICATION/test_b12x_gpu_gates.py" \
  --junitxml=/writable/results/b12x-components.xml
```

The 23 cases cover checkpoint capacities 1/2/4, TP2/TP4 KDA head geometry,
numerical references, invalid-input rejection, unchanged protected state,
64-bit pool addressing, paired QSA scoring and frozen-kernel graph replay.
Any skip leaves that case unqualified. Record image/source identities and all
manifest hashes with results; changes to the optional installed-package test
path must not be attributed to a preceding run with a different harness hash.

Run this harness, not an unrestricted pytest sweep of `fixtures/`. It scopes
test-only adaptations for retired freeze APIs and prepared RoPE layouts while
retaining the original numerical and mutation assertions. Other upstream tests
may require separate API adaptation. A passing component suite does not qualify
model loading, distributed serving, SparkCache restore, throughput or soak.

The separate [prepared RoCEnante record](../../../../performance/records/transport/rocenante-prepared-35cf12b2-20260918.md)
covers bounded two/four-rank collective and graph checks on image `35cf12b2d644`.
It identifies its own manifest, harness, per-rank receipts and NCCL reference;
the B12X component inputs above are not a substitute for that transport evidence.
