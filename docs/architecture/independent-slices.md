# Independent slices and dashboard management

This design lets operations on disjoint Sparks proceed together on one
controller. A slice is a set of fabric positions, such as 0–3 or 4–5. A
deployment keeps its existing active pointer, transaction journal and saved
state in that slice's slot. No service or collector is added.

## Locks and shared state

The controller takes a shared `flock` on `install.lock` and exclusive locks
on each affected position in numerical order. A ring operation takes the
exclusive `install.lock`. Existing programs that take that exclusive lock
continue to exclude every operation. Lock files stay in place; the kernel
releases ownership when a process exits, including after a crash. Recorded
PIDs alone never establish ownership.

Installing into a smaller slice also locks every Spark of any deployment
it displaces. Thus replacing 4–7 with 6–7 excludes an operation on 4–5 until
the displaced deployment has stopped or rolled back. Conflicts fail before
mutation, naming the holder and its Sparks; repeating after it finishes
serializes the operations without a polling queue. `--yes` cannot bypass a
lock. A short controller metadata lock protects admission and publication;
it is never held across model loading or network transfer.

Setup, fabric changes, relay configuration, driver reloads, package upgrades
and cluster-wide storage release remain ring-wide. A slice install checks
worker versions and refuses an upgrade, directing the operator to setup.
It does not park another slice's mesh or perform automatic global retention.
Whole-fabric installations retain those maintenance steps.

Transfer work uses deployment-specific directories and an ephemeral local
registry port. Image copies can traverse other Sparks and use Node A's disk
and network even when the image is installed only on the target slice.
Checkpoint downloads share that network too. These are advisory resource
conflicts, not permission to change the forwarding Sparks' configuration.
Source checkouts are protected separately by revision. Atomic file writes
use unique temporary names. Shared dashboard publication takes a short lock
around reading records and writing the snapshot.

Recovery checks the lock scope of each deployment independently and takes
the same scope again before restarting it. The existing timer schedule is
unchanged. Logs have operation identifiers and slice indexes; `sparkring
logs --on 4-5` selects the latest operation for those Sparks. Unqualified
`logs` retains its existing combined log.

## Activity grades

An operation evaluates activity on demand, before mutation, from live file
locks, deployment records, benchmark quiet holders and the fabric document.
It never starts a collector. The response uses `sparkring-activity/v1` with
`grade`, `positions` and a list of reasons containing the slice, activity,
resource and plain-language message.

| Grade | Rule | Example |
|---|---|---|
| `blocked` | A lock overlaps the affected positions, or a ring operation holds the gate | Install on Sparks 4–7 holds Sparks 4–7 needed by restart on 6–7. |
| `may-be-affected` | Another slice installs, a benchmark holds quiet mode, or shared transfer/relay resources may be busy | Install on Sparks 0–3 shares Node A's download disk and network with Sparks 4–7; this operation may be affected, probably not. |
| `clear` | No recorded contention is found | No conflicting activity is recorded for Sparks 4–5. |

Unknown traffic is not proof of an idle model. If traffic is sampled, there
is at most one bounded `/metrics` request per neighbouring deployment; quiet
mode suppresses it. Advisory warnings require confirmation, except with
`--yes`; noninteractive callers receive a request for confirmation. JSON
results include the assessment. Locks remain authoritative if activity
changes between the preview and execution.

## Dashboard action flow and security

Read-only mode remains the default. Manage mode requires explicit enablement
and an authentication token read from a file. The dashboard refuses manage
mode as root. Its service account can connect to the existing least-privilege
helper; it gains neither Docker access nor a controller SSH key.

Each slice offers install/swap with a profile, checkpoint and bounded options,
down, restart, automatic recovery on/off and logs. A preview invokes the
helper's allow-listed planning command and shows its output and activity
assessment. Confirmation invokes the helper once with the same arguments.
The server binds a short-lived, single-use confirmation to the reviewed
arguments. Execution rechecks conflicts. Expired or changed plans need a
fresh preview. The helper validates arguments independently and audits
accepted and rejected requests with the kernel peer UID and exit status.

Management requests require a bearer token, same-origin validation and a
CSRF nonce. Tokens are never put in URLs or local storage. Responses are
uncacheable. Existing Host validation and content security policy apply.
No management polling, websocket, background job or auto-refresh is added.
Long actions occupy their existing HTTP/helper request until completion;
operation logs are available from another terminal. Following logs means
running the shown `sparkring logs --on … --follow` command, without adding
a browser polling loop.

## Compatibility and qualification

Existing deployment and helper protocol identities remain valid. Additional
activity fields are additive. `restart`, lifecycle `--yes`, recovery `--on`
and log `--on`/operation selection extend the CLI. Existing `up` and `down`
plans and the read-only dashboard continue to work. Linux `flock` supplies
the production concurrency contract; non-POSIX development hosts may retain
conservative exclusion.

CPU tests must exercise real competing processes, stale metadata, fake
Sparks and Docker, shared files, allow-list rejection, authentication, CSRF,
confirmation reuse and unchanged collector request counts. CPU evidence
does not qualify serving latency, RDMA or resource interference. The
[hardware test plan](../development/independent-slices-hardware.md) bounds
that qualification to 90 minutes on eight Sparks.
