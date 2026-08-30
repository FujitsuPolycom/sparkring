# GLM-5.3 opaque-page base-flight qualification

Status: **implemented, not qualified**.

`qualification.py` publishes one 98,304-token base and sixteen independently
authenticated 131,072-token result roots with private 32,768-token tails. It
records prompt and response hashes without response content. It never starts,
stops, or restarts a service.

Run `semantic` and `publish`, inspect each rank's isolated manifest root, and
retain their receipts. A service restart is a separate operator action and
must not occur without authorization. After that action, run `replay`, capture
bounded rank logs for only that cohort, and run `verify` with four manifest
receipts and four logs. The verifier requires one
`sparkcache-page-base-restore-flight/v1` summary per rank with 16 participants,
one physical base read, 15 avoided reads, and outcome `verified`, plus 16
independently verified result restores on every rank.

The unrelated request submitted after the cohort uses no common prefix and
must complete successfully. This is a bounded isolation check, not a latency
claim.
