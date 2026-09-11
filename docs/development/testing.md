# Local checks and CI

No Spark hardware is required to contribute. Install the development dependencies
with `python -m pip install -r requirements-dev.txt`. Use a virtual environment
if your machine's Python installation is shared with other projects.

For profile, layout or documentation changes:

```bash
python scripts/check_repository_layout.py
python -m pytest runtime/common -q
python scripts/check_markdown_links.py .
python scripts/check_release_safety.py .
```

The structural check validates profiles, generated exports, references, frozen
release bytes, image-builder paths, managed-service source dependencies and
maintained imports. It does not need Docker, a GPU, network access or model files.
The link and secret scanners inspect tracked files; stage intended additions
before running them. Prose warnings are advisory.

For implementation changes, run tests beside the affected component. The
[CI workflow](../../.github/workflows/ci.yml) lists the broader suite and pinned
CPU torch dependency. The migration retains all existing test coverage and adds
canonical integration and shared-runtime tests. Some tests require POSIX modes,
Bash, the pinned LIL companion or optional dependencies; report skips accurately.

CPU tests cover configuration, packaging and lifecycle contracts. They do not
validate CUDA kernels, RDMA behavior, live serving, GPU memory stability or
performance. Hardware checks before deployment promotion are the maintainer's
responsibility; a contributor can state what they could not run.
