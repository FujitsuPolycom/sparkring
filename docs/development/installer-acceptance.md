# Installer acceptance

A hardware run qualifies the installer only when the documented entrypoint
completes the workflow without operator repair commands. Installing an image or
starting a model through private scripts is component evidence, not installer
acceptance.

The operator supplies credentials, selects an exact profile, optionally chooses
storage paths, and approves the displayed disruptive changes. A noninteractive
caller supplies the same choices as flags or a literal configuration file.
Machine output must be parseable, with explicit success, failure, pending input
and resumable state. Credentials must not enter logs or exported plans.

The workflow owns authenticated discovery, prerequisite and package handling,
network inspection, image/model cache discovery, verified fiber transfers,
storage preflight, managed deployment replacement, readiness checks and logs.
Preparation failures leave the working model up. A failed switch must attempt
verified recovery of the retained deployment and report the recovery result.
Unexpected or unowned workloads require a clear decision rather than being killed.

For an existing-ring upgrade test, prepare these starting conditions:

- The previous owned model is running and its rollback assets are retained.
- At least one worker needs the head's package update.
- At least one worker lacks the selected candidate image.
- Existing complete model weights are available for reuse.
- Network configuration and node order come from discovery, not a test-only IP list.

Run the documented command once. Observe through documented logs/status only.
If an extra SSH command is needed to make installation succeed, record the gap,
fix the program and restart the acceptance test from a declared fixture.
Test preparation itself must be recorded separately from actions taken by the
installer. Simulated low-space, interrupted-transfer, unknown-workload and
rollback cases require the same orchestration as a real invocation.

A successful upgrade on configured nodes does not establish blank-host, single-
uplink, reboot-recovery or cable-reordering qualification. Those require separate
fixtures and hardware evidence. CPU tests never qualify CUDA/RDMA serving.

## Configured TP4 result, 2026-09-24

Installer source `02a873c0a3896957ec2c3332ff31512d2357c32e` completed one
`sparkring install --profile qwen38-flash-next-qad-tp4 --image-lock image-lock.json --yes --json`
invocation without operator repair commands. The image configuration was
`sha256:3db79d3c6bea958aada8855b5bc6ef368484b14b798b04c8b507b8e140925276`:
CUDA 13.4.2, NCCL 2.32.3 and runtime-status 0.3.1.

The fixture began with the old owned model running, three workers on an older
package, the candidate image absent on one worker, and complete cached weights.
The installer updated workers, transferred the image through authenticated
fabric SSH in 171.5 seconds, reused verified checkpoint receipts in 4.0–4.3
seconds per rank, then stopped the old model and launched all four new ranks.
The image's cold kernel tuning and startup took 549.7 seconds after API-rank
start. Readiness and two short generation checks passed. The active pointer
committed only after those checks. JSON stdout parsed successfully.

All four independent host/container/runtime identity and freshness joins matched.
The dashboard, model-list and health routes responded over the management LAN.
These checks establish this installation and basic serving; they do not establish
performance, broad model correctness or long-running stability.

Two earlier attempts failed and remain separate evidence. The first exposed a
Docker listing that hid an imported untagged image; it stopped before downtime.
The second exposed Docker's default io_uring restriction after model startup.
That attempt automatically stopped the candidate, restored the retained previous
deployment, and passed readiness and generation checks. The fixes use exact-ID
image inspection and a container-scoped loader policy with CPU preflight.

The final source passed 1,187 Linux tests. Its ARM64 package passed manifest,
source-bundle, CLI and eight service-definition checks. Test fixture preparation
and read-only observations are separate from installer actions. Private receipts
and logs are retained by the operator; no site addresses or credentials are
published here. The existing mesh, TP2 and published release inputs were unchanged.

## Additional profiles and TP2 preparation, 2026-09-24

`qwen38-flash-next-qad-tp4-sparkcache` completed public installation using its
declared shared-2026.09.3 image, not the external CUDA image above. Readiness
took 444.0 seconds. A planned retained-container restart passed in 229.9 seconds.
An identical 8,889-token retrieval prompt returned the correct opaque identifier
before and after restart. The post-restart API credited 8,640 cached tokens;
each of four physical workers separately logged restoring 8,640 tokens and
91.0 MiB. This was one bounded fixture, not a cache performance benchmark.

Six of seven bounded API smoke gates passed. The forced-tool-call completion
gate failed; the unchanged TP2 baseline exhibited the same failure. Health,
model/context identity, arithmetic, image-input acceptance and prefix-response
checks passed. Image-input acceptance is not a visual-understanding test.

GLM preparation review found a nested workspace incompatible with the managed
backend and service paths already occupied by the current mesh. The controller
now uses the backend's supported workspace, stages before model shutdown, and
checks existing service ownership before mutations. Source
`4663b6c711ffa64b3cc68d659d13de919ec456ba` returned `needs_input` for the occupied
fabric through the public install command, leaving Qwen and the mesh unchanged.
GLM serving is not qualified by this test; existing-mesh integration or an
explicitly reviewed migration remains necessary.

TP2 installer acceptance has not run: SSH access is available, but privileged
installation awaits sudo credentials/root access, and both local disks are
nearly full. The baseline API checks above do not establish installer acceptance.
No TP2 packages, containers, images or network settings were changed.
