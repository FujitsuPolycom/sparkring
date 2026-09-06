# Integration ownership

Status: implemented in the FujitsuPolycom integration fork; no upstream maintenance commitment is assumed.

Companion implementation: [FujitsuPolycom/lil, `codex/image-runtime-adapter`](https://github.com/FujitsuPolycom/lil/tree/codex/image-runtime-adapter),
based on upstream revision `11df08a793596b0a5b09e72e90d9a1ece51c9306`.
Tested companion revision: `8a3e86c096e8dae2d1e7055a7f55070141653125`;
[fork draft PR #1](https://github.com/FujitsuPolycom/lil/pull/1).
The extension is maintained separately from this SparkRing change. Upstream lil
cannot execute these bundles; the fork provides the implementation for testing
and interface discussion.

| Owner | Responsibility |
| --- | --- |
| SparkRing | Images, model profiles, canonical argument export, SIRCL settings, SparkCache identities, native checks, distribution and support |
| lil extension | Generic bundle validation, command execution, lifecycle order, ownership checks, status and logs |
| Operator | Trusted bundle review, host access, fabric wiring and storage paths |

The fork's lil command consumes `lil-image-bundle/v1`: a bundle ID and ordered
rank records containing a host, container name, Docker argument array, and optional
preflight commands with exact expected output. It has no SparkRing imports or
model-specific conditionals. Image-owned code replaces the source-checkout
assumption for this command only; existing lil commands retain their behavior.

SparkRing owns compatibility with the lil revision it pins. Its images can change
without requiring lil maintainers to understand cache formats or transport kernels.
Any upstream proposal should contain only this reusable image interface and its
tests. SparkRing-specific policy and hardware testing stay in SparkRing.

Reviewed lil baseline `11df08a793596b0a5b09e72e90d9a1ece51c9306` normally clears
Docker entrypoints and mounts vLLM/B12X source directories in
`internal/launcher/builder.go`. Its source parity check in `checks.go` excludes
native extensions. The image command instead preserves the adapter's canonical
arguments and executes its explicit checks. This is a separate fork command,
not a claim that upstream supports `--runtime sparkring`.

The selected descriptor lists canonical runtime/model pin files and their
normalized UTF-8 SHA-256 values. Changed inputs require a descriptor review.
Preflight checks target configuration and index metadata; it does not hash all
target weight shards. Verify model files during staging.
