# Offline integration checks

Status: implemented, offline-tested. No hardware serving qualification is claimed.

Environment: Linux under local WSL, Python 3.12, Go 1.26.0. The tests use temporary
files and fake command runners. No serving hosts or model endpoints are involved.

| Check | Result |
| --- | --- |
| Integration and canonical launcher tests, including MTP3, direct-copy routing, and configurable GID indices | 74 passed |
| Companion lil `go test ./...` | all packages passed |
| Python lint | passed |
| Exported four-rank bundle consumed by built lil CLI | validate and render passed |

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
