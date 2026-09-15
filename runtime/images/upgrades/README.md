# Bounded image upgrade runner

Status: **research-only**. The controller implements upstream discovery,
bounded patch reconciliation, protected acceptance gates, source-overlay and
native-wheel image construction, TP2 qualification adapters and private run receipts. GPU-free tests exercise its state machine
and failure handling. Neither those tests nor the simulated trial establish GPU
correctness, performance, unattended native builds or serving qualification.

The entry point is `python scripts/image_upgrade.py`, run from the repository
root with Python 3.12 and Git. Windows supports discovery, planning and the
simulated trial. Container gates and image construction require a dedicated,
explicitly leased Linux builder of the target architecture, Docker and a
locally present immutable foundation image. No serving host is selected by
default. No scheduler, deployment change or GitHub write is installed.

## Try the controller without a cluster

```bash
python scripts/image_upgrade.py trial --output .sparkring/upgrade-demo
python -m pytest runtime/images/upgrades -q
```

The demonstration creates a small local Git repository and an independent
page-divisibility oracle. It executes three immediate runs: retain an applicable
patch, skip identical successful inputs, then adapt a conflicting upstream
change. The oracle executes real Python assertions; the LLM response and image
build are simulations, explicitly identified in `TRIAL.json`. This is not a
three-night hardware test. Choose an unused output directory when repeating it.

## Observe upstream changes for three nights

```bash
python scripts/image_upgrade.py init-r37 --output .sparkring/lil-arm64-policy
python scripts/image_upgrade.py loop \
  --policy .sparkring/lil-arm64-policy/policy.json \
  --state .sparkring/lil-arm64-state --nights 3 --interval-hours 24
```

PowerShell accepts the same arguments on one line. The loop runs once
immediately, then waits the specified interval after each completed run. It
requires the terminal process to remain alive. Closing it ends the loop; it
does not create a background service. A separate scheduler can invoke `run`
with these same arguments once per night.

Planning fetches Git objects and writes local artifacts. It does not call an
LLM, run upstream code, build/pull image layers, access GPUs or publish. Discovery
is refreshed even when successful identical inputs can be skipped. Failure is
not remembered as success. Planning success does not suppress a subsequent
execution or image-build run.

The initializer reads the retained
[vLLM/B12X source composition](../compositions/lil-r37-glm-spark/source-lock.json),
copies its carried patches into private policy inputs, and tracks explicit LIL
branch refs. It pins the shared SparkRing foundation by registry digest and
local image ID. Editing controller code, policy, patches or oracle inputs changes
the approval digest. Nothing modifies the retained composition or profile defaults.

The policy excludes `.claude/` and `.agents/` tool metadata from source
snapshots. No runtime, build or test paths may be excluded. Source commits remain
recorded, but snapshot digests describe these explicitly filtered contents, not
the complete upstream Git tree. Other symlinks and submodules are unsupported
and stop materialization. Git long-path support is enabled only for the owned
snapshot repositories on Windows.

## Published ARM64 image discovery

```bash
python scripts/image_upgrade.py discover-arm64
```

This reads the machine-readable publication record behind
[randomvariable's image release page](https://randomvariable.github.io/vllm-multiarch-oci/image-releases/)
and inspects registry configuration using Docker Buildx. It downloads metadata,
not image layers. The adapter requires the expected repository, an immutable
digest and publication tag, a timestamp no older than 36 hours and a Linux
ARM64 image configuration. A publication record is **not serving qualification**.

Discovery emits a foundation candidate; it never edits the approved foundation.
The existing [Bazel foundation evaluation](../compositions/lil-bazel-arm64/README.md)
explains which inherited SparkRing capabilities must be supplied before that
image can serve as a SparkRing composition. Image discovery and source-ref
tracking are independent: approval must bind compatible engine, kernel,
transport and cache inputs rather than combining their newest labels.

## Execute on a reserved builder

First review `policy.json`, its patch files, `REPORT.md`, and any
`reconciliation-request.json`. Pull the policy's exact foundation image onto
the reserved builder manually; the runner refuses an absent foundation. Do not
use a serving stack as an overnight builder without reserving it explicitly.

```bash
python scripts/image_upgrade.py validate --policy .sparkring/lil-arm64-policy/policy.json
```

The output includes `policy_sha256`. Create an operator-owned builder lease:

```json
{
  "schema": "sparkring-upgrade-builder-lease/v1",
  "policy_sha256": "COPY_THE_VALIDATED_SHA256",
  "host": "EXACT_BUILDER_HOSTNAME",
  "exclusive": true,
  "not_before": 0,
  "expires_at": 0
}
```

Times are Unix seconds; replace both zeros with the reserved interval. A lease
is an explicit operator assertion of exclusive use, not a cluster reservation
service or proof that another process is absent. Expired, wrong-host and
wrong-policy leases fail before execution. Use a separate state directory and
filesystem/Docker quotas on that builder.

```bash
python scripts/image_upgrade.py run \
  --policy .sparkring/lil-arm64-policy/policy.json \
  --state .sparkring/lil-arm64-state \
  --execute --approved-policy VALIDATED_SHA256 \
  --builder-lease .sparkring/builder-lease.json --build
```

Omit `--build` to stop after source reconciliation. Use `loop --nights 3` with
the same execution arguments only after a supervised execution has passed.
That command does not reserve the machine: the lease must cover each run.

The reference build adapter reuses the owner's
[candidate-image installer](../candidate_image.py) and
[Dockerfile](../Dockerfile.candidate). It verifies the parent inventory,
packages exact accepted vLLM/B12X source trees, preserves unrelated inherited
files and verifies the installed candidate. Native/build-input changes block
this adapter. A full native rebuild requires a separately reviewed fixed build
adapter; this implementation does not pretend a changed dependency lock or CUDA
extension is a Python-only update.

## Native compilation and verified reuse

Use `init-r37 --native` to select the GB10 wheel compiler. Its private policy
declares SM121a, eight compiler jobs, a CPU/memory limit, the Torch ABI and a
compiler deadline. `foundation.compiler_image_id` may select an immutable
tooling image separately from the serving foundation. The
[tool preparation adapter](tooling.py) supplies checksum-verified Rust 1.95.0
and pinned Python test/build wheels; it does not install tools on serving hosts.
Build the prepared context with [Dockerfile.tooling](Dockerfile.tooling) and
record its actual local image ID in the private policy.

The compiler has no GPU devices, model mounts or Docker socket. It builds wheels
from read-only accepted sources in an owned writable directory. The installer
checks wheel hashes, preserves unrelated foundation files and Torch versions,
rejects introduced dependency conflicts, and records installed files and native
provenance. It creates a distinct boundary-cache runtime attestation; prior
attestations are not evidence for different compiled bytes.

An operator-pinned `foundation.native_cache` manifest can reference an earlier
compiled wheel. Reuse requires exact equality of declared native/build inputs,
native-language files elsewhere in each source tree, compiler image, Torch ABI
and architecture. Repackaging must preserve the complete native-member hash map.
The result reports `native_rebuilt: false` and retains the compiled artifact's
provenance. Changed native inputs require compilation; a Python-only change does
not justify relabeling old binaries as rebuilt.

Carried patches must describe the **complete retained feature closure** relative
to their declared baseline commit. A published image can contain source older
than the operational reference. Mechanical patch success is insufficient if the
selection drops unlisted improvements. Compare source changes and add missing
feature contracts before treating that selection as an upgrade.

## Supervised TP2 qualification

[tp2_suite.py](tp2_suite.py) consumes a private site file using
`sparkring-tp2-qualification/v1`: ordered SSH hosts/hostnames, two saved container
inspection files, model name, dedicated cache parent, test API/rendezvous ports,
and selected cache/media/performance checks. It uses the shared container
renderer, preserving CPU affinity and memory settings. Unsupported snapshot
settings fail rather than being silently discarded.

The suite creates run-owned containers on separate test ports and marked cache
directories. Its checks cover text, persisted-prefix credit after both worker
processes restart, and corrupted synthetic chunks falling back to recomputation.
Fault injection preserves byte-for-byte backups and never targets an unmarked
serving cache. Optional media checks send three solid-color images and one red
video; they do not establish general video accuracy.

The optional performance adapter requires a hash-pinned benchmark script,
hash-pinned baseline JSON records, identical sampling/output/workload settings,
three repetitions and explicit prefill/decode/normalized-step thresholds. It
uses measured throughput and at least 95% observed concurrency. Transient
queue/underfill display badges remain visible in evidence; errors, loops,
readiness failures and inadequate measured fill fail the gate.

[tp2_experiment.py](tp2_experiment.py) orders GLM before Qwen, preserves exact
rollback container IDs, and restores that deployment after unsuccessful trials
or ordinary nightly tests. `--leave-qualified` is a separate supervised-only
action: both model suites must pass before Qwen is started on the declared
serving port. Nightly recipes must not supply that option. Model weights are
reused through read-only binds, not downloaded by the qualification adapter.

These adapters are implemented but require live qualification for the selected
image/site. Keep API access restricted to the trusted management network.

[tp2_gate.py](tp2_gate.py) exposes the ordered experiment as an operator hardware
gate. It requires cache, selected media and matched performance checks for both
model sites, then requires successful restoration of the saved deployment. It
has no promotion option. Its composite receipt retains each model's distinct
performance control; it does not invent a single control-image identity for two
different models. Register the site and control files as hashed gate inputs.

The model suite writes `progress.json` as stages advance. Failure receipts retain
completed responses and identify the failed stage, including a cold-cache answer
when publication fails. Progress records are diagnostic evidence, not passing
qualification receipts.

## Supply semantic reconciliation

Mechanical application is attempted file by file. Clean approved fragments are
retained. Conflicts produce a bounded request containing the behavior contracts,
relevant source views, carried patch and gate results. Oversized context is
rejected, not silently truncated. Large files use explicitly labeled diff views.

Binary assets such as compressed kernel-calibration tables are represented by
their patch and file hashes, sizes, paths and presence in each source snapshot.
Their opaque payloads are not repeated in the LLM request. The complete approved
patch remains hash-bound, textual source views are preserved, and proposals
cannot edit those binary-asset paths. This encoding does not establish semantic
compatibility of a binary asset; an unresolved migration remains a blocker.

Two proposal transports are implemented:

- **Endpoint:** initialize with `--agent-endpoint https://HOST/v1 --agent-model
  MODEL`. The endpoint must support chat completions returning JSON. Optional
  `agent.key_env` names the environment variable holding its API key. No key is
  written into policy or run artifacts. API charges are not estimated; call
  count, request size, output tokens and time are bounded. Review what source
  content will be sent before enabling this mode.
- **Files:** pass `--proposal-dir .sparkring/proposals`. A missing proposal stops
  the run after saving its request. Read that request and follow the
  [LLM operator recipe](nightly-recipe.md). No model service is called by this
  transport.

A file proposal is named `REQUEST_INPUT_SHA256.json`. Compute that identity with
`agent.request_identity`: canonical JSON uses sorted keys, compact separators
and no nonfinite values, excluding only the diagnostic `feedback` field. The
policy digest, source pins, contracts, source views and carried patch remain
bound. Observation timing may change on a retry; every acceptance gate must
still run against the proposed source. Its content is:

```json
{
  "request_sha256": "REQUEST_INPUT_SHA256",
  "proposal": {
    "disposition": "adapt",
    "reason": "Explain the preserved behavior and relevant upstream interface.",
    "patch": "UNIFIED_DIFF_AGAINST_THE_SUPPLIED_CANDIDATE"
  }
}
```

The other dispositions are `retire`, `incompatible` and `unresolved`; they must
carry an empty patch. Patch bytes are data, never shell commands. Agent edits
must stay inside approved source paths and outside native/protected paths.
They cannot change acceptance policy or test files in the protected baseline.
The approved baseline must pass its oracles before a proposal can be accepted.
Retirement requires upstream to pass the same oracles; optimization retirement
also requires a protected performance oracle. Passing a bounded oracle is not
a proof that an LLM preserved every semantic property.

## Qualification boundaries

| Layer | Implemented admission | Evidence not supplied by the initializer |
|---|---|---|
| Source | Baseline, upstream and candidate oracle runs; source-digest binding; bounded repair | Whole-runtime behavioral equivalence |
| Image | Parent inventory, accepted source identities, native wheel/reuse checks, installed inventory, feature/cache bindings | Arbitrary dependency or compiler-family migration |
| GPU/profile | TP2 text/cache/media/performance adapters, rollback and separately scoped hardware leases | Qualification of every profile or TP4 topology |
| Publication | Explicit candidate-only action permission and immutable image receipt | Publisher adapter, provenance/license review or stable promotion |

The supplied source recipe runs the retained recurrent-checkpoint and hybrid
recovery CPU tests and B12X two-checkpoint CPU tests inside the foundation.
These container gates must be calibrated on the builder; a missing dependency,
skipped test or zero-test receipt is not success. Candidate image gates reject
changed feature source preimages and SparkCache lease-contract bindings. They
do not rewrite those hashes to make an incompatible composition appear valid.

The supplied recipe has no hardware gates or publisher command. It cannot
claim GLM/Qwen/DeepSeek, TP2/TP4, cache restart/corruption, multimodal or performance
qualification. Extending the test matrix means registering fixed, hashed gate
inputs and exact resources, not copying serving defaults out of `profiles/`.
SGLang and incompatible vLLM families are not automatically combined into one
image. "All profiles" is a compatibility and evidence matrix, not an image name.

Gate receipts use `sparkring-upgrade-gate/v1`. They identify the gate, input
fingerprint, source-tree digest or image ID (`subject_sha256`), variant, explicit
outcome, positive assertion count and zero skipped tests. Performance metrics
require at least three finite measurements and compare medians against protected
baseline thresholds. The thresholds, tests and compiler recipes are operator
policy; an LLM cannot relax them as part of reconciliation.

An image/hardware performance gate must pin `baseline_image` and return a nested
`baseline` receipt with the same input fingerprint, `variant: "control"`, and
that image as `subject_sha256`. The control must pass with positive assertions,
zero skips and repeated measurements. Missing controls or a candidate median
outside the configured regression tolerance fail acceptance. Source oracles
instead compare against the separately executed protected baseline snapshot.

Hardware gates additionally require `sparkring-upgrade-hardware-lease/v1` with
the policy digest, `exclusive: true`, valid times, `resources` and `gate_ids`.
The fixed adapter must enforce reservation, snapshot/restore, model-weight reuse,
cleanup and matched workload controls. The runner provides no implicit permission
to stop containers or use another task's cluster. A hardware receipt covers only
its named gates; it is not general stability qualification.

## State, limits and recovery

Each run preserves its frozen policy and hashes, discovery, source snapshots,
requests/responses, accepted patches, gate logs, build bundle, `report.json`,
`REPORT.md` and draft `PR.md` under `STATE/runs/RUN_ID`. These are private review
artifacts, not automatically posted PRs. Review logs and paths for sensitive
information before sharing. Only the agent transport reads its named API key;
child commands receive an environment allowlist.

The state directory has an ownership marker and crash-released local process
lock. It must be on a local filesystem, not a shared NAS or network filesystem.
The same directory excludes concurrent controllers. This is not a distributed
lock across different directories or hosts. Failed inputs retry on a later run;
interrupted or uncertain external actions stop all retries, including `--force`.

```bash
python scripts/image_upgrade.py status --state .sparkring/lil-arm64-state
python scripts/image_upgrade.py resolve --state .sparkring/lil-arm64-state \
  --run-id RUN_ID --confirm-owned-work-stopped
```

Resolve only after inspecting every owned Docker/build/hardware action and
confirming it has stopped. Gate directories record owned container names and
run labels. Killing the controller or Docker client does not prove daemon-side
work stopped. The resolution command records the operator's acknowledgement;
it does not stop or delete anything.

Subprocess output, process time, source archive size and LLM attempts are bounded.
Free-space/state-size checks run between stages; they are **soft** checks. Git
fetches, Docker layers and writable gate output need external filesystem/Docker
quotas for a hard disk budget. Gate containers use read-only source mounts,
network isolation, dropped capabilities and CPU/memory/PID limits. Containers
are risk reduction, not a perfect hostile-code sandbox; use a disposable builder
without production secrets. The runner never prunes images, deletes evidence,
downloads model weights or promotes a serving default.

Cache namespace impact: the controller does not change `CacheIdentity`, digest
salts or chunk geometry. Any reconciled runtime/connector change still needs
cache compatibility and corruption-recovery evidence before profile adoption.
