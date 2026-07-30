# Adaptive-MTP controller status/reset surface

Status: offline-validated, opt-in integration candidate. The launch and
`sitecustomize.py` plumbing exists, but both switches default to zero and no
running model has loaded this package yet.

## Why this surface exists

The deployed adaptive acceptance-length controller is scheduler-global. Its
partial observation window and selected depth survive request boundaries, so
sequential prompt strata are not independent unless the controller is observed
and reset between strata.

This package adds one automatic policy and two scheduler-local actions:

- On the first request of each genuinely idle engine epoch, restore K to the
  configured maximum and clear only the partial observation window. Requests
  arriving concurrently stay in the same epoch. Retained streaming sessions
  are not reset.

- `status`: read raw controller K, floor-snapped K, current window counts,
  reset totals, and reset-safety state.
- `reset`: restore raw K to the configured maximum and clear the controller's
  four window accumulators.

Manual reset runs in vLLM's existing serialized
`EngineCoreRequestType.UTILITY` path. Automatic reset runs immediately before
`EngineCore.add_request()` admits the first request after the engine has become
fully idle. Neither is a worker collective RPC: workers do not own the
scheduler controller. No new HTTP endpoint is added.

## Safety contract

`reset` refuses without mutation unless all inspected state is empty:

- scheduler `running`, `waiting`, and `skipped_waiting` queues;
- scheduler request registry and known deferred-request queues;
- streaming-input waiters and deferred-free fences;
- EngineCore asynchronous batch queue;
- per-request speculative output placeholders, draft-token placeholder rows,
  async discard frames, and in-flight token counts.

Automatic reset uses the same gate and skips retained streaming sessions
instead of treating their next input chunk as a new workload epoch.

The complete controller mutation ABI is validated before raw K changes. A
missing field or non-callable `_reset_window()` is fatal.

## Reported state

The controller is the source of truth for the current window:

```text
_num_observation_steps -> window.observation_steps
_num_drafts            -> window.drafted_rounds
_num_draft_tokens       -> window.attempted_tokens
_num_accepted_tokens    -> window.accepted_tokens
```

Reset totals are split by `manual` and `idle_epoch`.

The reset patch deliberately does **not** attribute verification acceptance to
`SchedulerOutput.num_spec_tokens_to_schedule`: that field selects proposals
for the next worker step, while acceptance belongs to proposals made by the
previous step. Exact depth/acceptance accounting comes from the true-draft
snapshot and interval-bound Observer counters.

At C1, independently validate these depth counts with the existing atomic
`/cache/jit/spark-graph-status-rankN.json` census:

| DCP query shape | Selected depth |
|---|---:|
| Q2 `[2,16,576]` | K1 |
| Q3 `[3,16,576]` | K2 |
| Q5 `[5,16,576]` | K4 |

The delta eager-signature count divided by 78 GLM layers should equal target
rounds at that depth. This external counter is a validation surface, not a
replacement for raw K/window state.

## Deployment ABI pins

`runtime_installer.install()` is inert unless
`SPARK_ADAPTIVE_MTP_CONTROL=1`. Before mutating any class it requires:

- vLLM
  `0.11.2.dev279+eldritch.final.fcc6141.b12x284a2ea.fi25dd814.cu132.20260626`;
- `acceptance_length.py`
  `bbc5176e48827fee1412a7ce95ecd8ef7a57c60ad8b7c66e16447ae3cb133a0e`;
- `scheduler.py`
  `a66b04fa9aaa59ca1c647c7ea3db8b5a3c683577aae9cb553781b825eb298e77`;
- `async_scheduler.py`
  `da6343d7e7c394a1738cf72905cbecc208003ffa461ccb441268333a3eb9f884`;
- `engine/core.py`
  `cd661d0356003026225a83293234acf5d8b668acea48bcfdbee2881b05aa452d`.

These are the file hashes from
`deliverables/glm52-adaptive-mtp-runtime-audit-20260727.md` (private
archive). Local vLLM checkouts do not match them and are not treated as
deployable substitutes.

The installer assumes:

1. `AcceptanceLengthController` exposes the six fields and `_reset_window`
   documented in the audit.
2. `EngineCoreProc` inherits `EngineCore` and dispatches named utility methods
   on itself.
3. `EngineCore.add_request()` is inherited from the pinned `EngineCore` class
   in this TP4/DCP runtime and `EngineCore.has_work()` is the idle authority.
4. The installer runs before `EngineCore` construction. The checked-in
   `sitecustomize.py` performs that composition only when the opt-in is one.

## Composition and calls

After staging the directory as an importable `adaptive_mtp_controller` package,
startup composition is:

```python
from adaptive_mtp_controller.runtime_installer import install

install()
```

Set the opt-in and ladder explicitly:

```bash
SPARK_ADAPTIVE_MTP_CONTROL=1
VLLM_ADAPTIVE_SPEC_DEPTHS=2,4
```

From frontend-local Python, use vLLM's existing utility RPC:

```python
status = await async_llm.engine_core.call_utility_async(
    "spark_adaptive_mtp_control", "status"
)
reset = await async_llm.engine_core.call_utility_async(
    "spark_adaptive_mtp_control", "reset"
)
```

Or use `client.control_async(async_llm, "status")` and
`client.control_async(async_llm, "reset")`. Synchronous offline engines can use
`client.control_sync`.

Automatic idle-epoch reset prevents a completed prompt from poisoning the next
independent prompt. Manual reset remains useful between benchmark strata and
repetitions; invoke it outside timing. A refused manual reset is a hard
experimental gate, not a retry signal while requests remain active.

## Offline validation

```bash
python -m pytest -q \
  spark_transport/experiments/adaptive_mtp_controller
```

No cluster, model, container, commit, or push is performed by these tests.
