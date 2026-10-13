# Independent slices: bounded hardware qualification

Budget: at most 90 minutes on eight Sparks. The coordinator chooses cached
profiles that already serve successfully on 0–3, 4–5 and 6–7. Stop at the
time limit, on an unexpected restart outside the selected slice, or on any
fabric change. Preserve logs and restore the initial deployments. Do not
download a large checkpoint merely to complete this plan.

| Minutes | Action and required evidence |
|---|---|
| 0–10 | Record package revision, fabric, placement, recovery settings and running containers. Warm the models. Save the same fixed decode workload and latency/throughput measurements for every comparison. |
| 10–30 | Run randomized repeated baseline/read-only/manage-open idle comparisons, with the same existing polling and quiet settings. Record controller CPU time, serving process CPU time, model API request counts, GPU utilization, inter-token latency and tokens/s. Manage-open sends zero additional idle model requests and creates zero additional periodic tasks. Report paired differences and instrument resolution; do not claim literal zero CPU from noisy samples. Any repeatable latency or throughput regression blocks qualification. |
| 30–45 | Start cached installs concurrently on 4–5 and 6–7 while 0–3 serves the fixed workload. Capture timestamps proving overlap. Attempt an overlapping restart and a ring-wide setup while they hold locks; both must refuse before host changes and name the holders. Repeat with a pre-existing exclusive install lock. |
| 45–55 | Preview image transfer/download on another slice. Check the advisory names its slice and Node A disk/network or forwarding path. Reject confirmation, then approve only a bounded cached transfer. Compare serving measurements and record resource effects separately from idle overhead. |
| 55–65 | Terminate a test operation on an idle slice, preserving its transaction journal. Verify the kernel releases its locks, stale metadata is ignored and the documented interrupted transaction can resume. Never terminate serving ranks for this check. |
| 65–75 | Hold an operation lock on 4–5 and run one recovery check. Verify 0–3 and 6–7 are still judged independently. Toggle recovery only on 6–7, restart that cached deployment and follow its own operation log. |
| 75–85 | From an unprivileged manage dashboard, preview and confirm each permitted action on 6–7. Verify missing/wrong token, cross-origin POST, missing CSRF, reused confirmation and arbitrary helper options fail. Capture helper audit UID and result. Verify read-only mode accepts no mutation. |
| 85–90 | Restore initial deployments and recovery choices. Record incomplete checks, artifacts, measurement uncertainty and pass/fail against the idle-load criterion. |

Use existing instrumentation where possible. Any additional measurement
sampling must be identical across baseline and manage-open runs. A management
action's one-shot activity check belongs to action cost, not idle overhead.
