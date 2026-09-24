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
