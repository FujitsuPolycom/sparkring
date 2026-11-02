# Release 2026.10.5's installer fixes on the ring of eight: restart after setup, dashboard with an unlisted checkpoint

Status: **implemented** for the two installer fixes; **research-only** for
the DeepSeek-V4.1-Flash-1017 serving run that carried them. This record
gives the hardware check of two fixes in release 2026.10.5's package and the
deployment they ran on. The release record
([README](../../../runtime/releases/dev-20261016-sparkcache-4a17f1f5198e-status038/README.md))
cites it.

## Conditions

- A ring of eight DGX Sparks (GB10), ConnectX-7 RoCE, cycle-8 fabric, SIRCL
  ring sessions with NCCL off. Installer image
  `dev-20261016-sparkcache-4a17f1f5198e-status038` (configuration ID
  `sha256:f625053225d4…`).
- Package built from the release tree with both fixes (source `4a219e05`,
  the same change as the release's `relays.check_position` rule), installed
  on every Spark by `sudo sparkring setup --yes` on 2026-10-17 at 10:24 CDT.
- The deployment: `deepseek-v41-flash-sparkcache-tp4` on positions 0-3 with
  default settings, serving the checkpoint
  `deepseek-ai/DeepSeek-V4.1-Flash-1017` @ `1bae0e1088993b9b82de08f59de5be5a34e42170`
  from a local copy verified against its pin manifest (a research
  configuration: the SparkCache profile lists no `1017` checkpoint; the
  TP4 profile offers it as `--checkpoint 1017`).

## Measurement

The installer's return codes and messages; the SHA-256 of Node A's fabric
document before and after `setup`; the API's readiness; the dashboard's site
file.

## Result

Restart after setup (the fix: a rank accepts a copy of the fabric document
whose bytes differ from the deployment's reference when it records the same
cabling identity):

| Step | Time (CDT) | Result |
|---|---|---|
| Fabric document digest before | 11:10 | `98a214db572db454…` |
| `sudo sparkring setup --yes` | 11:11 to 11:15 | rc 0; "Fabric verified: 8 cables on 8 Sparks (cycle-8), relays 96 rules and 96 routes, reboot-persistent" |
| Digest after | 11:15 | `65d55ba05e6ac0b8…`; the deployment's recorded reference matches no copy |
| `sudo sparkring check --on 0-3` | 11:15 | rc 0, "All checks passed" |
| `sudo sparkring down --on 0-3 --execute --yes` | 11:15 | rc 0 |
| `sudo sparkring up --on 0-3 --execute --yes` | 11:15 to 11:21 | rc 0; API ready at 11:21:26 |
| `sudo sparkring check --on 0-3` | 11:21 | rc 0, "All checks passed" |

Before the fix, the same restart was refused four times on 2026-10-17
between 01:26 and 03:17 with "This Spark's fabric document differs from the
deployment's; run sudo sparkring setup".

Dashboard with an unlisted checkpoint (the fix: the site file and the
operator view describe a record from its own lock when the installed
package does not list its checkpoint): with stopped records of a research
checkpoint entry (`nvidia-fp4-local`) under the controller, the site file was
written at 11:00 after the installation and lists the two serving
deployments; before the fix every installer run ended with "Dashboard: its
site file was not refreshed (ValueError: Select a checkpoint the profile
lists: …)".

The deployment itself: installed in 1808 s (the first start autotuned all
2,021 b12x kernels for the new revision); `sparkring check` passed; a
three-turn session (12,189-token prompt at turn 3) replayed with 12,032
cached tokens before and after a restart (`restore hit tokens=12032
ms=4266.9` and `ms=3999.9`); KV pool
`GPU KV cache size: 11,308,736 tokens` (one start). No speed was measured.

## Conclusion

A retained SIRCL deployment restarts after `sparkring setup` rewrote the
fabric document, and the dashboard's export survives records of unlisted
checkpoints, on this image and package.

## Limitations

- One deployment, one setup; the other-cabling refusal path is covered by
  the unit test only.
- Deployments installed from earlier packages keep the byte-exact check
  until installed again.
- The 1017 serving run is a research configuration of the SparkCache
  profile; its figures are not the profile's.
