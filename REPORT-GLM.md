# Report: the nccl transport, and the cycle contract on the eight-Spark profiles

Branch `claude/installer-nccl-arm`, based on `claude/sircl-kraken-image` at bd2bb4eb. Worktree
`installer-nccl-arm`. No push; no network, SSH or Docker; nothing ran on any Spark;
`spark_transport/sircl/` is unchanged.

## The design

### 1. `--transport nccl`: vLLM's PyNccl alone

`runtime/common/transport.py` gains a third backend of the existing
`sparkring-transport/v1` transport section, `backend: "nccl"`, built by
`nccl_section` and checked by `validate_nccl_section`. A deployment on it:

- loads no SIRCL: `nccl_adapt` strips every `SIRCL_*` variable from each
  rank's container and removes `sircl` from `VLLM_PLUGINS` (the image layer
  never adds it, so in practice the list keeps `b12x_loader,sparkring_status`);
- turns the prepared transports off by applying the SIRCL launcher's
  `DISABLED_TRANSPORTS` (`serve.plan`), which sets
  `VLLM_ENABLE_ROCE_ALLREDUCE=0` (the RoCEnante slot), clears
  `SPARKRING_TRANSPORT_PROFILE` and `SPARKRING_TRANSPORT_MANIFEST_SHA256`, and
  disables the prepared four-rank adapter (`SPARK_TP4_ENABLED=0`,
  `VLLM_SPARK_TP4_MODE`, `VLLM_SPARK_TP4_VOCAB_MODE`); so vLLM's PyNccl
  carries every tensor-parallel and expert-parallel collective;
- keeps every other profile setting, including the profile's other NCCL
  settings (`NCCL_SWITCHLESS_RING_ONLY`, `NCCL_IB_GID_INDEX`, channels, proto).

Admission rules, mirroring SIRCL's refusals in wording:

- a fabric document is required (`choose` refuses without one): the per-rank
  devices and the group's shape come from it;
- the group must satisfy NCCL's own cabling rule (`nccl_cabling_rule`): a
  cabled pair, or a group whose consecutive ranks share cables around the
  whole group (SIRCL's `NcclPolicy.ALL` or `RING`). A path of three or more is
  refused by `nccl_refusal`, which names the rule, the uncabled pairs
  (`fabric.nccl_policy_of`), and the alternative (`--transport sircl`);
- a profile whose decode-context-parallel groups are such paths is refused
  with the same rule: on the whole cycle those groups are lines of
  `decode-context-parallel-size` Sparks. Groups of two (cabled pairs) and the
  whole cycle itself are allowed.

On a whole cycle the section records
`nccl: {settings: {NCCL_ALGO: Ring, NCCL_SKIP_TREE_CONNECT: 1}, reasons: {...}}`;
the settings are `ring_settings()` — the SIRCL serve launcher's `RING_SETTINGS`
(`sparkring_sircl.vllm.serve.bundle`), imported rather than copied. On a cabled
pair the section records no settings. `nccl_adapt` applies the section's
settings and derives each rank's `NCCL_IB_HCA` from the section's devices with
the SIRCL launcher's own `serve.plan.nccl_hca_value`, which keeps the style of
the profile's own value (the `=` prefix and the `:1` port suffixes of the
eight-Spark profiles). The devices are `transport.rank_devices` of the group's
topology — the same mapping the SIRCL adapter mounts into containers — so a
rank whose cabled ports differ by position (Spark 0's port 0 faces Spark 1's
port 1 on the eight-Spark ring) gets its own value instead of the profile's
single global one.

### 2. The cycle contract wherever NCCL may run on a whole cycle

`transport.environment` (the SIRCL deployment's container environment) now
calls `added_ring_settings`: on a group whose recorded cabling is `ring` with
`--nccl auto`, the deployment adds the ring settings the profile leaves unset
and passes the merged environment to the guard, so `guard.effective_policy`
sees `NCCL_ALGO=Ring` with `NCCL_SKIP_TREE_CONNECT=1` and lets NCCL's ring
carry the ring-shaped collectives on that group. A profile that names another
value for either setting is refused (the contract needs exactly these). The
plan prints one line naming both settings and why
(`plan_lines`: "NCCL on this cycle: ... the installer adds what the profile
leaves unset, so NCCL's ring runs but builds no tree connections between
Sparks that share no cable"). Profile JSON is authored, not generated
(`scripts/generate_profiles.py` regenerates compatibility exports and the
deployment table from authored inputs), so nothing was edited there and the
installer adds the settings at deployment.

### 3. Plan and receipt naming

`transport.plan_lines` prints `Transport: nccl on every collective; SIRCL is
not loaded and the RoCEnante slot is off (--transport nccl)`, the group line,
and one line per added NCCL setting with its recorded reason
(`NCCL_SETTING_REASONS`); a cabled pair states that the profile's own settings
need nothing added. The install result's `transport` field
(`install_workflow.transport_summary`) names `backend: "nccl"`, the group, the
fabric and the added settings, with `expected` = `transport.NCCL_EXPECTED`;
the summary card's Transport line reads `nccl`
(`transport_card_line`). The SIRCL receipt check does not run for an nccl
deployment: `install_workflow.execute` and `check.transport_check` gate it on
`backend == "sircl"`, `transport_receipts.host_report` answers
`backend: "nccl"` without receipts, and `sparkring status`
(`controller.transport_view`, `transport_status_line`) shows the backend and
group. What an nccl deployment still verifies on every Spark is its fabric
document (`check_host_document`), because the per-rank devices come from it;
its tuning-table checks apply to SIRCL sections only.

### Consumers that were backend-sensitive

`installer.make_lock` validates a `backend: "nccl"` section with
`validate_nccl_section` (the lock digest covers it, so the request identity
changes with it); `installer.specifications` routes `transport.adapt` by
backend. `install_workflow.transport_choice` builds the nccl section;
`select_deployment` records the nccl request;
`serving` adds the relay-table ring check for SIRCL groups only;
`scripts/installer_host.py` gates the SIRCL layer admission, the ring
operations and the SIRCL receipt directory on `backend == "sircl"`;
`controller up` compares the recorded backend.

## Every changed function (file:line, at this branch's head)

`runtime/common/transport.py`

- module docstring: the nccl backend described (lines 12-24)
- `BACKENDS` gains `"nccl"` (line 127)
- `choose` — accepts `nccl`, refuses it without a fabric document and refuses
  `--nccl` with it (line 512)
- `ring_settings` — the SIRCL launcher's `RING_SETTINGS` (line 676)
- `NCCL_SETTING_REASONS` (line 684), `NCCL_EXPECTED` (line 690)
- `nccl_cabling_rule` (line 694), `nccl_refusal` (line 701)
- `nccl_section` (line 707)
- `validate_nccl_section` (line 746)
- `added_ring_settings` (line 819)
- `policy` — unchanged; now called with the merged environment (line 833)
- `environment` — adds `ring` to the common environment (lines 944-945,
  969-970)
- `adapt` — routes to `nccl_adapt` (line 1080)
- `nccl_adapt` (line 1117)
- `plan_lines` — routes to `nccl_plan_lines`, prints the cycle line (lines
  1147, 1163-1167)
- `nccl_plan_lines` (line 1208)
- `check_host_document` — tuning tables only for sections that record them
  (line 1273)

`runtime/common/installer.py`

- `make_lock` — validates by backend (line 367)

`runtime/host/install_workflow.py`

- `transport_choice` — builds the nccl section (line 517)
- `select_deployment` — records the nccl request (lines 661-667)
- `serving` — ring check for SIRCL groups only (line 932)
- `transport_summary` — the nccl result field (line 1107)
- `transport_card_line` — `nccl` (line 1104)
- `execute` — the receipt check runs on SIRCL only (line 1447)
- `main` — `--transport` and `--nccl` help (lines 1504-1513)

`runtime/host/controller.py`

- `up` argument help (lines 542-547); the recorded-transport comparison of an
  existing deployment (line 769); `transport_view` (line 975);
  `transport_status_line` (line 973)

`runtime/host/transport_receipts.py`

- `host_report` — the nccl answer without receipts (line 154)

`runtime/host/check.py`

- `transport_check` — `{"backend": "nccl"}` (line 79); the tuning-table export
  gate (line 238)

`scripts/installer_host.py`

- `admit_image` — the SIRCL layer admission on SIRCL sections only (line 206)
- `perform` — ring operations on SIRCL sections only (line 1996); the start
  relay-table check (line 2137); the SIRCL receipt directory at create
  (line 2167)

Tests: `runtime/common/test_transport_nccl.py` (new), `runtime/common/test_transport.py`
(`launcher_plan` injects the deployment's ring settings into the launcher
profile, as the SIRCL bundle does; new whole-cycle test),
`runtime/host/test_install_sircl.py` (new host-level nccl install test).

## Test results

All commands run in WSL with `python3 -m pytest <files> -q` (CPU only) and
`python3 -m ruff check --select E,F,W --ignore E501`:

- `runtime/common/test_transport.py`: 97 passed.
- `runtime/common/test_transport_nccl.py` (new): 16 passed — nccl on a pair
  and on a pair arc of the eight-cycle (both directions of a cable), nccl on
  the whole eight-cycle (environment exactly as specified, per-rank
  `NCCL_IB_HCA` for all eight ranks, profile NCCL settings kept), plan lines
  with one line per added setting and its reason, refusal of a path of three
  and of four with the cabling rule, refusal of decode-context-parallel groups
  that are paths (and acceptance of pairs of two), `choose` without a fabric
  document and with `--nccl`, six malformed-section refusals, lock identity,
  and the fabric-document host check.
- `runtime/host/test_install_sircl.py`: 8 tests. First run: 7 passed and
  `test_status_prints_the_last_receipt_verdict` failed — `transport_view` read
  a transport section's `backend` by key, so a section without the field
  (the test's minimal record) stopped `sparkring status` with a KeyError;
  fixed (`value.get("backend")`), that test passes again, and the whole file
  was re-run afterwards (count below). Includes the new
  `test_the_nccl_transport_installs_on_a_pair_without_sircl`, which installs
  `--transport nccl` on the pair cluster and checks the printed plan, the lock
  section, the result field and the `--nccl` refusal.
- `runtime/host/test_placement.py`: 42 passed.
- `runtime/common/test_installer.py` and `runtime/common/test_installer_image.py`
  (the lock's neighbours): 126 passed.
- ruff `--select E,F,W --ignore E501`: all touched files pass.

Byte-identical plans against bd2bb4eb: before any change, a script rendered
the lock, the transport section, every rank's rendered container
(environment, command, mounts) and the plan lines of a SIRCL `never`
deployment on a pair (TP2), a SIRCL `never` deployment on the whole
four-cycle (TP4), and a prepared deployment on a pair; after the change the
same script produced byte-identical output
(SHA-256 `c5fc9ddbf867e013113ea11e3f7c18ea482eb9548bca6b762b44ec5f7e287ed8`
both times).

## What could not be tested

- Nothing ran on hardware: no Spark, no RoCE device, no NCCL build. The
  per-rank `NCCL_IB_HCA` values and the ring settings are checked as
  environment text, not against NCCL's behaviour; whether vLLM's PyNccl then
  carries every collective on the eight-Spark ring is a live-fabric question.
- The receipt scan (`transport_receipts.evaluate`) is untouched and was not
  exercised for nccl: an nccl deployment records no SIRCL receipts, so the
  recorded "receipt" for it is the install result's `transport` field, not a
  per-collective verdict. A live benchmark that wants per-collective proof
  needs its own measurement.
- `sparkring up --transport nccl` on an existing deployment and the
  `sparkring status` nccl lines are covered by unit-level branches of
  `transport_view`/`transport_status_line`'s siblings, not by a host-level
  test run of `up` itself.
- Paths of six or more Sparks are refused at placement level
  (`placement.unsupported`) before the transport is chosen, with the message
  that names SIRCL's relay limit; the nccl cabling rule is named only for
  paths of two to five.
