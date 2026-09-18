# Bounded B12X GPU qualification inputs

The [manifest](manifest.json) pins the [harness](test_b12x_gpu_gates.py) and
reference fixtures used for prepared QSA, PLE and KDA checks. Fixture bytes match
the [accepted source reconstruction](../sources/README.md); their B12X
[license](B12X-LICENSE) is retained. This input bundle is not a test-result receipt.

Exact harness sources identified by existing receipts are retained in the
manifest's archives. Formatting the maintained harness does not transfer prior
GPU evidence to its different file hash; the fixture bytes remain unchanged.

## Frozen CPU oracle inputs

[execution-oracles-1210ba2b.tar.gz](execution-oracles-1210ba2b.tar.gz) preserves
the CPU source oracles and installed-contract gate for runtime build
`768620d56aae`, including connector-driven checkpoint publication retention.
Its 21 hash-bound inputs match the executed policy and source-gate receipts;
the archive also contains input/identity declarations and the license
(25 members). SHA256:
`49572dd098a985317065511a0a89eda865636af2a8cb6e245ddae9cd60b7a451`.
The source-oracle inventory SHA256 is
`1210ba2b24007fa699317a9e1a0019e4e72d2bb1e10b535b9c7d6c65b26018fc`.
Original CRLF suite bytes are retained. The
[22da81ca validation record](../../../../performance/records/images/shared-22da81cae057-validation-20260918.md)
owns the results; no private operator policy, host configuration or wheels are
included, and packaging these inputs does not rerun their checks.

[execution-oracles-c7847ecb.tar.gz](execution-oracles-c7847ecb.tar.gz) preserves
the exact inputs used for runtime build `cc4f6c8f7665`: 19 unique source-oracle
files, `oracle-gates.json`, the installed-contract image gate, and its separate
input declaration (22 members). Archive SHA256:
`4ed1dadd7d7d687d2942e163eef8a2549e41dde46a08003753703dc8b46db5db`.
The source-oracle inventory SHA256 is
`c7847ecb8f45bec78690256d832bcb24002cd064047b84d8a3d71698f5b7c7df`.
It includes the executed CRLF suite bytes, not a Git-normalized reconstruction.
Only hash-bound gate inputs are included; no operator policy, hosts or wheels.
The [candidate validation record](../../../../performance/records/images/shared-a75bd02ffc1d-validation-20260918.md)
owns the results; the archive alone is not a success claim.

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

The [a75bd02f transport record](../../../../performance/records/transport/rocenante-prepared-a75bd02f-20260918.md)
uses the separate [15e522b0 probe archive](transport-probe-15e522b0.tar.gz), SHA256
`5aefa700c021547c24cbc1f6ee97ab0e68d0a6f94d23c69c58cab054bd338e08`.
Its executed `probe.py` hash is
`15e522b076afb4885c0d6eda1d80eeacc2a1a79dba6164ccc24f5d3c5ba4ed4f`;
the `be8d5f7d` archive remains tied to its preceding image-specific receipt.
