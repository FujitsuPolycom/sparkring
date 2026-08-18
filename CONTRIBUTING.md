# Contributing to SparkRing

SparkRing is pre-release research. Issues and discussion are welcome — bug reports, measurement questions, reproduction attempts, and design feedback all help.

Most of this repository targets hardware that most contributors will not have:
four NVIDIA DGX Spark units cabled directly to each other. That is a real
constraint, not a brush-off — so this guide is organised around **what you can
contribute with the hardware you actually have**, and it is explicit about
which claims each level of access earns you.

- [Pull requests](#pull-requests)
- [Four contributor paths](#four-contributor-paths)
- [Test suites and where they run](#test-suites-and-where-they-run)
- [How to add things](#how-to-add-things)
- [The two-lane claim policy](#the-two-lane-claim-policy)
- [Private identifiers in a public repository](#private-identifiers-in-a-public-repository)
- [What CI does and does not check](#what-ci-does-and-does-not-check)
- [Good first issues](#good-first-issues)

## Pull requests

- Match the existing style of the surrounding code.
- All prose — documentation, comments, docstrings, commit and PR text — must
  follow [Write Without Hidden Context](docs/WRITING_STANDARD.md): it must make
  sense to a reader who has the repository but none of the development history.
- All tests must be green:
  - Python: `python -m pytest spark_transport sparkcache runtime scripts` from the repo root.
  - C++/CUDA: the CMake (CTest) suite, run in-container.
- No copied code without provenance: any code copied or adapted from another project must carry a provenance note and a corresponding license entry in `THIRD_PARTY_NOTICES.md`.
- Contributions are accepted under the project license, Apache-2.0 (see `LICENSE`).

The pull request template walks through each of these plus the lane statement,
the evidence requirement for any number, and the private-identifier
attestation. Fill it in honestly; "n/a, because ..." is always an acceptable
answer, and an unticked box with an explanation is far better than a ticked box
that is not true.

Open an issue or a proposal before a large change. Disagreeing about scope in
an issue is much cheaper than in a 3,000-line diff.

## Four contributor paths

### Path 1 — Docs only

**You need:** a text editor.

Documentation is a first-class contribution here. Start with the
[documentation map](docs/README.md) to distinguish present-state
specifications, runnable runbooks, evidence records, and historical reference
material before editing a claim.

```bash
git clone https://github.com/FujitsuPolycom/sparkring.git
cd sparkring
```

There is nothing to install. CI checks that every **repo-relative** Markdown
link and heading anchor resolves. External `http(s)` links are deliberately not
fetched by CI, so check those by hand.

Rules specific to docs:

- If you change a number, section 4 of the pull request template applies. A
  number in a doc is a claim.
- Keep the `[DOCUMENTED: ...]` / `[INFERRED: ...]` markers in `docs/SETUP.md`
  accurate. Do not promote an inference to documented without a source.
- Do not remove a caveat because it makes a result look weaker. The caveats are
  the point.

### Path 2 — Python and SparkCache

**You need:** Python 3.12 and a CPU. No GPU, no cluster, no container.

This is the largest surface that is fully testable offline: the SparkCache
connector, codec, manifest and store; the vLLM overlay adapters and their
contracts; the runtime lock and verification tooling; the experiment packages.
The suites stub vLLM and use CPU tensors, and the tests that genuinely need
hardware skip themselves with a stated reason.

```bash
# Install the pinned GPU-free development and CI dependencies:
python -m pip install -r requirements-dev.txt

# Torch is separate so CPU-only contributors do not accidentally install a
# CUDA wheel. The tests use it only for CPU tensor construction/comparison.
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.11.0"

# Do NOT install runtime/pip-freeze.txt. That is an aarch64 / CUDA-13.2
# serving freeze for the container, not a development environment.

# The full offline suite:
python -m pytest spark_transport sparkcache runtime scripts -q

# Check that recipes, status roles, evidence paths, and public summary
# identities still agree:
python scripts/validate_publication_consistency.py

# Lint (the exact rule selection CI uses):
ruff check --select E,F,W --ignore E501 .
```

`torch` is needed only for CPU tensor construction and comparison; nothing in
the offline suites touches a device.

### Path 3 — Transport C++ / CUDA

**You need:** a CUDA toolkit, `libibverbs` development headers, and — for the
full suite — a GB10 GPU. One Spark is enough to build and to run most of the
suite; the pair and four-rank probes need more.

Part of the native tree builds with nothing but a C++17 compiler:

```bash
# CPU-only, no CUDA required. This is the native check CI runs.
cmake -S sparkcache/native -B build/native \
  -DCMAKE_BUILD_TYPE=Release \
  -DSPARK_CACHE_PLACEMENT_ENABLE_CUDA=OFF
cmake --build build/native --parallel
ctest --test-dir build/native --output-on-failure
```

The transport library needs the full toolchain. It is compiled with
`-Wall -Wextra -Wpedantic -Werror`, so a new warning is a build failure:

```bash
cmake -S spark_transport -B build/transport \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=121
cmake --build build/transport --parallel
ctest --test-dir build/transport --output-on-failure
```

The gate is the **whole** CTest suite: every test executable declared in
`spark_transport/CMakeLists.txt` must pass. Paste that output into the pull
request. If you cannot run it, say so in the template and ask for a maintainer
gate — that is a supported outcome, not a failure.

SparkCache's CUDA half (`spark_cache_placement`, `spark_cache_snapshot`, the
`.cu` probes) builds only when `nvcc` is present; it is compiled by the default
configure on a CUDA machine.

### Path 4 — Full four-Spark integration

**You need:** four DGX Sparks, four cage-matched direct cables, and the
checkpoint.

Follow `docs/SETUP.md` in order. Do not skip the gates; they exist because
skipping them produced failures that were expensive to diagnose.

The order that matters:

1. **Cable first, model never first.** Qualify every edge before any model
   work — `spark_transport/scripts/qualify_direct_cable.py`, one run per edge,
   documented in `spark_transport/CABLE_QUALIFICATION.md`. Link-up and ping are
   not qualification. A good median with any new CRC, drop, or PHY counter is an
   integrity failure.
2. **Model-down probes before serving.** The four-rank collective probes in
   `spark_transport/experiments/nccl_switchless_ring/` run with no model
   loaded. Run them after any NCCL or transport change.
3. **Then, and only then, serving.**

Report what you find with the **Hardware / cable result** issue form —
including negative results. "This transceiver never reached 200 Gb/s on both
ports" is a result worth recording.

## Test suites and where they run

The blocking `offline-tests` CI job runs the Python suites with Python 3.12 and
CPU-only torch on a machine with no GPU and no fabric. Test totals are emitted
by each run rather than copied into this present-state specification.

| Suite | Command | Needs | Validation |
|---|---|---|---|
| `spark_transport` Python | `python -m pytest spark_transport -q` | CPU only | Included in blocking CI |
| `sparkcache` Python | `python -m pytest sparkcache -q` | CPU only | Included in blocking CI |
| `runtime` Python | `python -m pytest runtime -q` | CPU only | Included in blocking CI |
| Public tooling | `python -m pytest scripts -q` | CPU only | Included in blocking CI |
| **All four (what CI runs)** | `python -m pytest spark_transport sparkcache runtime scripts -q` | CPU only | **Blocking CI gate** |
| SparkCache native, host half | `cmake -S sparkcache/native ... -DSPARK_CACHE_PLACEMENT_ENABLE_CUDA=OFF` + `ctest` | CPU only, C++17 compiler | runs in CI |
| SparkCache native, CUDA half | default `cmake -S sparkcache/native` + `ctest` | 1 GPU + CUDA toolkit | **manual gate** |
| `spark_transport` CTest | `cmake -S spark_transport ...` + `ctest` | CUDA toolkit + `libibverbs` to configure at all; GPU for the CUDA cases | **manual gate** |
| Pair transport probe | `spark_transport_probe --server` / `--client` | 2 Sparks + 1 cable | **manual gate** |
| Per-edge cable qualification | `qualify_direct_cable.py` | 2 nodes over SSH + the cable | **manual gate** |
| Four-rank collective probes | `experiments/nccl_switchless_ring/probe_dcp4_collectives.py` | 4 Sparks, ring cabled, patched NCCL | **manual gate** |
| Public acceptance dry-run | `python scripts/acceptance_gate.py --site ... --gate-config ...` | CPU only; complete local configuration | **dry-run by default** |
| Serving / performance windows | acceptance `--execute` or the private reference orchestrator | 4 Sparks + a complete runtime and launcher | **manual gate; stops serving** |

Hardware- or environment-dependent cases self-skip with explicit reasons.
CI uses pytest's `-rs` output so reviewers can inspect every skip reason.

**Nothing in the "manual gate" rows runs in CI.** A green CI run never means
the native build passed or the transport was verified.

## How to add things

### A transport shape

The TP4 backend dispatches on tensor shape and dtype; anything it does not
recognise falls back to the stock vLLM/NCCL path. That fallback is the safety
property — preserve it.

1. Widen the eligibility predicate in
   `spark_transport/integrations/vllm/spark_tp4_backend.py` (see
   `_target_shape_eligible`) and the corresponding session/plan sizing.
2. Add a **GPU-free contract test** next to the existing ones
   (`test_spark_tp4_backend_dispatch.py`, `test_spark_decode_payload_contract.py`)
   asserting both that the new shape is now selected and that a neighbouring
   shape still falls back. A shape change with no fallback test will be sent
   back.
3. If the shape must be CUDA-graph capturable, say so explicitly and cover the
   bucket contract (`test_spark_cudagraph_bucket_contract.py`).
4. On hardware: run the relevant probe, then the numerical audit
   (`scripts/run_tp4_numerical_audit.ps1`), which compares both BF16 reduction
   orders against an FP32 ground-truth sum. Paste the result.
5. State the byte size and rank count in the pull request. "It works" is not a
   shape specification.

### A runtime patch

Patches under `runtime/patches/` are **fail-closed and preimage-pinned**. The
overlay verifies the SHA-256 of the exact upstream source it expects before
installing, and refuses to start on a mismatch rather than guessing. A patch
that applies with fuzz defeats the entire mechanism.

1. Pin to the exact upstream commit recorded in `runtime/runtime-lock.json`.
   Do not float it.
2. Record the **SHA-256 of the exact upstream preimage file** in
   `runtime/patches/vllm/preimages.json`. One entry per file the patch touches.
3. `git apply --check` must be clean against that pinned tree with **zero fuzz
   and zero offsets**.
4. Describe the patch and its semantics in `runtime/patches/vllm/README.md`.
5. **Provenance.** If any hunk is derived from another project, name the
   project, the commit, and the license, and add the entry to
   `THIRD_PARTY_NOTICES.md`. This is why part of the reference overlay is
   withheld — see `docs/RUNTIME_GAPS.md`. Unattributed lines are the single
   most common reason a patch cannot ship.
6. Do not "fix" a preimage mismatch by regenerating the hash from whatever tree
   you happen to have. Find out why it differs.

### A benchmark

A benchmark is only useful if someone else can tell what it does not cover.

1. Put the harness where its siblings live (`scripts/`, or the relevant
   `experiments/` package) and give it a machine-readable output — JSON, not
   console prose.
2. Emit a **complete configuration label**: TP and DCP degree, concurrency,
   context length, KV bytes per rank, window length, cell count, graph mode,
   and whether contexts are unique or share a prefix.
3. Emit pass/fail gates alongside the number: request error count, graph
   census, transport counter audit. A number without its gates is not a result.
4. Add a GPU-free test for the harness's own parsing and gate logic — several
   already exist (for example `test_b12x_floor_benchmark.py`).
5. Follow the claim policy below when you write the number down.

### A hardware result

Use the **Hardware / cable result** issue form. Attach the qualification JSON
rather than a screenshot; reviewers read `cable_qualified`,
`latency_target_met`, `model_path_ready`, `failure_domain`, and
`counter_deltas`. Redact site identity first (see below).

Results from topologies other than the four-node ring are welcome and useful,
including "this does not work on my hardware".

### A runtime profile (vLLM-style model)

The generic runtime launcher (`scripts/sparkring_generic_launcher.py`) accepts
any profile that conforms to the `sparkring-runtime-profile/v1` schema. You
can add a compatible vLLM-style model without four Sparks or a GPU.

1. Copy `scripts/config/native-profile.template.json` and fill in your
   model's image, identity pins, and vLLM arguments.
2. Validate structurally (no site needed):

   ```bash
   python scripts/sparkring_generic_launcher.py --profile your-profile.json validate
   ```

3. Validate with plan-build (catches profile/site incompatibilities):

   ```bash
   python scripts/sparkring_generic_launcher.py \
     --site scripts/config/site.example.yaml \
     --profile your-profile.json validate
   ```

4. Inspect the profile structure:

   ```bash
   python scripts/sparkring_generic_launcher.py \
     --site scripts/config/site.example.yaml \
     --profile your-profile.json explain
   ```

5. Generate an offline plan:

   ```bash
   python scripts/sparkring_generic_launcher.py \
     --site scripts/config/site.example.yaml \
     --profile your-profile.json plan
   ```

6. Compare two profiles or plans:

   ```bash
   # Profile-only (no site)
   python scripts/sparkring_generic_launcher.py \
     --profile-a a.json --profile-b b.json diff

   # Plan-level (with site: topology, per-rank actions, identity, labels, mounts)
   python scripts/sparkring_generic_launcher.py \
     --site scripts/config/site.example.yaml \
     --profile-a a.json --profile-b b.json diff

   # Plan-level with independent sites (compare same profile across sites)
   python scripts/sparkring_generic_launcher.py \
     --site-a site-a.yaml --site-b site-b.yaml \
     --profile-a a.json --profile-b b.json diff
   ```

   Exit codes: `0` = identical, `1` = different, `2` = invalid input.

7. Run the conformance test suite:

   ```bash
   python -m pytest scripts/test_runtime_conformance.py -q
   ```

A generic profile is outside the current EXL3 and accepted NF3 configurations
until it is named and gated. The `validate` and `explain` commands never claim
model correctness or live acceptance. The `diff` command never normalizes away
image IDs, model identity, topology, labels, mounts, hooks, or command
changes.

Conformance fixtures live in `scripts/config/fixtures/` and assert
semantically important plan fields for generic, EXL3 bridge, and NF3 bridge
behavior.

## The two-lane claim policy

SparkRing has two support lanes, described in `README.md` and
`runtime/README.md`. The distinction is not bureaucratic — the published
numbers were measured on one of them and not the other.

| | Reference lane | Public-functional lane |
|---|---|---|
| What it is | the exact pinned runtime the published numbers were measured on | what this tree builds today |
| Status | validated | candidate |
| End-to-end serving performance | the source of every headline number | **not** yet performance-equivalent, and not supported until a full acceptance gate passes |

The rules:

1. **Never present a public-lane measurement as a reference-lane
   measurement.** If you measured it on a build from this tree, say so, in the
   same sentence as the number.
2. **Never claim reproducibility that the acceptance gate has not produced.**
   "The transport library, probes, and test suites build clean-room from this
   tree" is a claim the gate supports. "You can reproduce the published
   throughput from this tree" is not.
3. **Label every number with its configuration**, and label what it excludes.
   Shared-prefix contexts are a concurrency baseline, not a unique-context
   capacity result. A comparison across two separate measurement windows is
   indicative, not a sealed A/B — say which one you have.
4. **A fresh build is a different artifact.** New binaries produce new hashes,
   and the fail-closed launcher must have those re-pinned before it will run.
   A rebuild does not inherit the previous build's validation.
5. **Do not soften a caveat to make a result read better.** If a reviewer
   removes a caveat you wrote, push back.

Every pull request states its lane. If you are unsure, pick
public-functional — it claims less.

## Private identifiers in a public repository

This repository is public and was published from a live cluster. Two tiers,
enforced by two separate CI jobs.

### Tier 1 — hard stop (`release-safety`, blocking)

These fail the build. Do not commit:

- **Site IP addresses.** RFC 1918 private ranges and any other real address.
  Use RFC 5737 documentation ranges instead: `192.0.2.0/24`,
  `198.51.100.0/24`, `203.0.113.0/24`.
- **SSH accounts, hostnames, and domain names.** Use `<USER>`, `<NODE0>`, and
  the placeholder table in `docs/SETUP.md` Section 1.
- **Absolute host filesystem paths**, including Windows user directories. Use
  repo-relative paths or `<PATH>`.
- **Private image or repository names.**
- **Credentials of any kind** — private keys, SSH public keys, tokens,
  passwords, API keys. If you push one by accident, say so and rotate it.
  Force-pushing does not un-leak it.

The blocking job is currently clean across the whole tracked tree, so any hit
is a genuine regression rather than pre-existing debt. It is a backstop for
the attestation in the pull request template, not a replacement for it: it
only knows the shapes it was taught.

### Tier 2 — advisory (`hardware-identifiers`, never blocks)

Network interface names such as the ConnectX-7 fabric ports, their RoCE device
names, and the management NIC name are **deterministic Linux
predictable-interface names derived from the GB10 PCI topology**. Every DGX
Spark produces the same ones. They identify the hardware model that this
repository openly targets — not this particular cluster — and they appear
throughout the setup and qualification docs on purpose, because an operator
needs to know what to type.

The advisory job reports them so that authors of **new** code think twice
before hardcoding one. Prefer a command-line flag or an environment variable,
so someone with a different enumeration can run the same code. An advisory hit
is a style prompt. It is not a merge blocker, and no one should "fix" the
existing documentation because of it.

## What CI does and does not check

Every job runs on a stock GitHub-hosted `ubuntu-latest` runner: no GPU, no
CUDA toolkit, no RDMA fabric, no DGX Spark, **and no secrets**. Contributions
from forks therefore get the full check set.

| Job | What it proves |
|---|---|
| `lint` | `ruff check --select E,F,W --ignore E501` is clean repo-wide. Formatting is reported but **not** enforced. |
| `offline-tests` | Publication roles and evidence references agree, the four CPU-only pytest trees pass, and the sanitized site/preflight/acceptance examples produce an offline dry-run plan. |
| `native-cpu-contract` | SparkCache's host-side layout/parser sources compile warning-clean under `-Werror` and their CTest cases pass. **No CUDA is compiled.** |
| `docs-links` | Every repo-relative Markdown link and heading anchor resolves. External URLs are not fetched. |
| `release-safety` | No tier-1 identifier or credential shape in tracked files. Blocking. |
| `hardware-identifiers` | Reports tier-2 interface names. Advisory, never blocks. |

CI does **not** build `spark_transport`, does not compile a line of CUDA, does
not run a collective, does not touch a fabric, and does not measure anything.
Those are the manual gates in the table above. Do not describe a green CI run
as more than it is.

## Good first issues

Derived from gaps actually observed in this tree, roughly easiest first. None
of them need hardware.

1. **No `.editorconfig`.** Add one matching the existing style.
2. **Ruff policy is split between `.ruff.toml` and CI.** The configuration file
   preserves two reference-snapshot exceptions, while the enforced rule set
   still lives on the CI command line. Move `select = ["E", "F", "W"]` and
   `ignore = ["E501"]` into `.ruff.toml` so `ruff check .` matches CI.
3. **Move the Markdown link checker out of the workflow.** It is currently
   inlined in `.github/workflows/ci.yml` as a heredoc, so contributors cannot
   run it locally. Extract it to `scripts/` and have CI call it.
4. **External links are never checked.** Add a scheduled, non-blocking job
   that checks `http(s)` targets in Markdown, kept out of the merge path so
   third-party downtime cannot block a PR.
5. **Decide a line-length policy.** Pick a limit, record it in the ruff
   configuration, and either fix existing `E501` findings mechanically or
   document the exemption.
6. **Decide a formatting policy.** Either adopt `ruff format` in one
   mechanical, review-focused change or configure it to match the established
   style. Do not reformat files piecemeal inside unrelated pull requests.
7. **Import ordering is unenforced.** Decide whether to enable Ruff's `I`
   rules, preferably in the same mechanical change as the formatting policy.
8. **Let CI cover part of the transport suite.**
    `spark_transport/CMakeLists.txt` declares `LANGUAGES CXX CUDA` and requires
    `CUDAToolkit` and `libibverbs` to configure at all — so a GPU-free runner
    cannot even reach the host-only tests. Five of its test executables were
    verified to build and pass with plain `g++` and no CUDA and no verbs:
    `graph_poll_policy_test`, `eager_staging_timeout_test`,
    `tp4_indexer_graph_c_api_layout_test`, `tp4_graph_command_test`, and
    `tp4_indexer_graph_test`. Add a `SPARK_TRANSPORT_ENABLE_CUDA` option that
    mirrors `SPARK_CACHE_PLACEMENT_ENABLE_CUDA` in `sparkcache/native`, so
    those five can run in CI. *Highest value on this list.*
9. **Repo labels are only the GitHub defaults.** The issue forms would route
    better with `hardware-result`, `regression`, `lane:public-functional`, and
    `lane:reference-performance`. *Maintainer task — needs repo settings.*
10. **Produce the first accepted public-runtime evidence bundle.** Resolving
    source pins is necessary but does not prove that the image builds, serves
    the supported matrix, or passes the seven-stage acceptance gate. Record the
    first clean build/attestation result without borrowing reference-lane
    performance numbers.
