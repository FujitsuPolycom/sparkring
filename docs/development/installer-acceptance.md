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

Status: **implemented**. One upgrade of a configured four-Spark ring (TP4),
at installer source `02a873c0a389`, meets this definition (below). Reboot
recovery on a four-Spark ring and on a pair, and upgrades through the published
one-line command on both, completed without operator repair commands on
2026-09-26 (below). Each result holds for its installer revision only.
Blank-host and cable-reordering acceptance have no hardware evidence.

## Configured TP4 result, 2026-09-24

Installer source `02a873c0a3896957ec2c3332ff31512d2357c32e` completed one
`sparkring install --profile qwen38-flash-next-qad-tp4 --image-lock image-lock.json --yes --json`
invocation without operator repair commands. The image configuration was
`sha256:3db79d3c6bea958aada8855b5bc6ef368484b14b798b04c8b507b8e140925276`:
CUDA 13.4.2, NCCL 2.32.3 and runtime-status 0.3.1.

The fixture began with the previous owned model running, three workers on an
earlier package revision, the candidate image absent on one worker, and
complete cached weights.
The installer updated workers, transferred the image through authenticated
fabric SSH in 171.5 seconds, reused verified checkpoint receipts in 4.0–4.3
seconds per rank, then stopped the previous model and launched all four
candidate ranks.
The image's cold kernel tuning and startup took 549.7 seconds after API-rank
start. Readiness and two short generation checks passed. The active pointer
committed only after those checks. JSON stdout parsed successfully.

All four independent host/container/runtime identity and freshness joins matched.
The dashboard, model-list and health routes responded over the management LAN.
These checks establish this installation and basic serving; they do not establish
performance, broad model correctness or long-running stability.

Recovery evidence from the same ring and fixture: in a run whose candidate
failed after model startup, because Docker's default seccomp profile blocked
`io_uring`, the installer stopped the candidate and restored the retained
previous deployment, which passed readiness and generation checks. The
installer inspects images by exact ID and runs the checkpoint loader under a
container-scoped `io_uring` policy with a CPU preflight, as described in
[Install SparkRing: reference](../operations/install-reference.md#serving-image-and-profiles).

Installer source `02a873c0a389` passed 1,187 Linux tests. Its ARM64 package passed manifest,
source-bundle, CLI and eight service-definition checks. Test fixture preparation
and read-only observations are separate from installer actions. Private receipts
and logs are retained by the operator; no site addresses or credentials are
published here. The existing mesh, TP2 and published release inputs were unchanged.

## Reboot recovery and one-line upgrades, 2026-09-26

Reboot recovery. With `qwen38-flash-next-qad-tp4` serving on TP4, rank 3 was
rebooted. When it was back, the other three Sparks' mesh services had failed,
their model containers were still running, and on their ports facing rank 3
the fabric address had left RoCE GID index 3. One
`sudo sparkring install --profile qwen38-flash-next-qad-tp4 --yes --json` at
installer source `d135e35566da`, the revision already installed on every
Spark, found that the installed model did not serve, stopped it on every Spark,
re-added the addresses to GID index 3, started the four mesh services, passed
every ring check and started the model: 269 seconds from command to
`Model ready`. With `qwen38-flash-next-tp2` serving on TP2, the worker was
rebooted; Node A's two fabric ports lost GID index 3 the same way. One
`sudo sparkring install --profile qwen38-flash-next-tp2 --yes --json` at
installer source `a320c87a8c3d` stopped both ranks, re-added Node A's two
addresses and started the model in 288 seconds. Both models answered a short
arithmetic question correctly afterwards.

One-line upgrade. The published command
`curl -fsSL …/one-command-installer/install.sh | bash -s -- --profile PROFILE --yes`
built installer source `21d07dc6670d` on Node A of each cluster and upgraded
every Spark from the revision above: 357 seconds on TP2 and 349 seconds on TP4
from command to `Model ready`, with no operator repair command. The serving
image and checkpoint were already present, so image distribution and
checkpoint transfer were not exercised. On TP4 the same command with
`--checkpoint qad-step-4000`, then without it, switched the served checkpoint
and back in 288 and 253 seconds.

These runs establish recovery and upgrade on these two clusters, not
performance or long-running stability.

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

For the managed GLM backend, the controller uses the backend's supported
workspace, stages assets before model shutdown, and checks existing service
ownership before changing anything. On a ring whose mesh service paths belonged
to another deployment, installer source
`4663b6c711ffa64b3cc68d659d13de919ec456ba` returned `needs_input` (field
`fabric`) through `sparkring install`, leaving the running Qwen model and the
mesh unchanged. This run does not establish GLM serving through the managed
backend; that requires an existing-mesh adapter or an explicitly reviewed
migration.

## Installer-profile installations on the shared image, 2026-09-25

Installer package revision `eb8ec2ba3b17` installed each of the six installer
profiles, as that revision defined them, with `sparkring install` on one
directly cabled pair (TP2) and one four-Spark ring (TP4), on image
`dev-20260925-cuda1342-nccl2323-status031`. At that revision both Qwen
profiles pinned checkpoint revision `60215d26cf5e` (Hugging Face branch
`qad-step5500-ple1000`), with MXFP8 LM-head and hyper-connection quantization
off and greedy drafting. Counting, arithmetic and code checks passed for every
profile, and first-start readiness and single-run throughput were recorded in
the
[six-profile installation record](../../performance/records/images/dev-20260925-installer-profiles-20260925.md).
Installer deployments of `qwen38-flash-next-tp2` and
`qwen38-flash-next-qad-tp4` with checkpoint revision `629bc3218833` on image
`dev-20260925-qwendecode-cuda1342-nccl2323-status031` supplied the serving
containers measured in the
[installer tuning record](../../performance/records/qwen38-flash-next/installer-tuning-20260925.md).
With the settings committed, including probabilistic drafting, source
revisions `e75451a671a3` (pair) and `f0ce5bea531f` (ring) installed the two
profiles again after the deployment and asset-admission state on every node
had been moved aside, reusing cached checkpoint files and images; that record's
Installer deployments section gives the conditions and results.
Their records do not describe the upgrade fixture above, so these runs are
installation evidence, not acceptance under this definition.
