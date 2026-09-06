# Offline integration checks

Status: implemented, offline-tested. No hardware serving qualification is claimed.

Environment: Linux under local WSL, Python 3.12, Go 1.26.0. The tests use temporary
files and fake command runners. No serving hosts or model endpoints are involved.
Companion CLI: FujitsuPolycom/lil at `329cde801b847294005cb16765692032a6cdf206`.

| Check | Result |
| --- | --- |
| Integration tests, including both profiles, direct-copy routing, and configurable GID indices | 72 passed |
| Companion lil `go test ./...` | all packages passed |
| Python lint | passed |
| MTP3 and DFlash four-rank bundles, SparkCache enabled and disabled, consumed by built lil CLI | validate and render passed |
| Exported MTP3 bundle with simulated SSH hosts | legacy approval, conflicting ownership, immutable container IDs, and partial-host actions passed |

The bridge test runs SparkRing's canonical Bash builder, checks DCP4, SIRCL,
headless ranks, graph sizes and connector presence, and passes its JSON bundle
to the actual locally built lil binary. Lifecycle tests simulate missing hosts,
existing containers, wrong ownership, preflight mismatches, and worker failures.
Distribution tests exercise checksum caching, four destinations, resumability,
bad files, traversal rejection, and SSH command ordering without network traffic.

The separate [hardware record](HARDWARE_VALIDATION.md) covers MTP3 startup,
response, restart restore, and a small direct-copy fixture. Operator directories,
image loading, and managed-mesh installation remain explicit setup steps
documented in README. These offline tests alone do not prove hardware behavior.

CI builds the pinned companion, runs its Go tests and vet, then runs the bridge
with `LIL_TEST_BINARY` set. Local runs without that variable skip real-CLI checks;
set it to an executable built from the pinned revision to reproduce CI coverage.
