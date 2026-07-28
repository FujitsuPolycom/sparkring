# Contributing to SparkRing

SparkRing is pre-release research. Issues and discussion are welcome — bug reports, measurement questions, reproduction attempts, and design feedback all help.

## Pull requests

- Match the existing style of the surrounding code.
- All tests must be green:
  - Python: `python -m pytest spark_transport` from the repo root.
  - C++/CUDA: the CMake (CTest) suite, run in-container.
- No copied code without provenance: any code copied or adapted from another project must carry a provenance note and a corresponding license entry in `THIRD_PARTY_NOTICES.md`.
- Contributions are accepted under the project license, Apache-2.0 (see `LICENSE`).
