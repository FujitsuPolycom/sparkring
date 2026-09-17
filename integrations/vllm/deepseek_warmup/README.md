# DeepSeek DSpark preparation warmup

Status: **implemented**, with [bounded TP2/K5 preparation qualification](../../../performance/records/deepseek-v4-flash/dspark-prepare-warmup-tp2.json).
This optional image-build overlay extends DSpark's startup preparation. Profile
defaults and published images remain unchanged. Packaging remains research-only;
other kernel coverage in [issue 188](https://github.com/FujitsuPolycom/sparkring/issues/188) is still open.

## Behavior

The supported source creates a verification-capacity manager only when
confidence-based draft pruning is enabled. Its startup warmup calls DSpark's
preparation sweep through that manager, so a fixed K5 profile with pruning
disabled skips the explicit sweep. The overlay calls the existing DSpark
warmup in scratch lane 1 before readiness even when the manager is absent.
Capacity-enabled DSpark and other speculative methods retain their call paths.

The preparation kernel chooses its tile from the maximum scheduled tokens
within one request plus that request's draft-query count:

```text
tile = min(256, next_power_of_2(scheduled_tokens + draft_query_rows))
```

For anchor-sampled DSpark K5, five draft-query rows make tiles 8, 16, 32, 64,
128 and 256 reachable. The original explicit representatives cover 16, 64
and 256. The overlay enumerates feasible tiles and scalar specialization
classes: context count 1, ordinary integers and multiples of 16. It selects
positive target lengths and single/batched request counts within the configured
token buffer. Query counts come from the configured speculator;
the sweep does not assume a shared anchor convention for DSpark and DFlash.
Ordinary DFlash's preparation remains unchanged.

Warmup sampled-token IDs use int64, matching `ReqStates.last_sampled_tokens`;
next-prefill IDs are allocated separately as int32. This avoids warming a
pointer-type signature that real requests never use. The application also
checks the request-state source hash that defines those types.

Startup call placement, DSpark warmup shapes and warmup buffer types change.
The GPU kernel, weights, inference path and serving limits retain their source
definitions. Other speculative methods retain their warmup call sequences.

## Prepare an image overlay

The [source manifest](fixtures/source.json) identifies the exact input files
extracted from the DeepSeek image selected by manifest
`sha256:827a8e8c5749b78529cc0015dd174e1b19a0accc116bc142282f8b75428f98bd`.
The corresponding image ID is
`sha256:41fc632d02352a69f59dc18be184488c1f9d51bd9ad0a29e22ae818af4f465fe`.
The application rejects changed input hashes before creating any output,
including the unchanged request-state definition used to verify buffer types.

During a separate image build, with this integration directory available:

```bash
python integrations/vllm/deepseek_warmup/apply.py \
  --source-root /opt/venv/lib/python3.12/site-packages \
  --output-root /tmp/dspark-warmup-overlay
```

The output directory must not exist. It contains two source files under
`vllm/` and `manifest.json`, including resulting hashes. Copy its `vllm/`
contents into the derivative image's site-packages and retain the manifest.
The command leaves the source tree unchanged. Use a distinct image identity
and compiler-cache namespace; this is not a live-worker patch procedure.
The [readable diff](dspark-prepare-warmup.patch) is checked against the application.

## Validation boundary

Run offline checks from the repository root:

```bash
python -m pytest integrations/vllm/deepseek_warmup -q
```

The tests replay the actual startup branch, target-length planner and complete
preparation method from
licensed source fixtures. They cover pruning enabled/disabled, other methods,
query counts, dynamic depths, token budgets, sampled/prefill pointer types,
context scalar specializations, source drift and output preservation.
They do not validate GPU writes, model correctness or startup memory usage.

The recorded GPU component control produced nine additional runtime preparation
events. The corrected warmup captured 13 warmup events and no additional
events across 19 runtime fixtures, with output checks. A separate TP2/K5 launch
from empty private serving caches passed 20 API cases / 36 requests in 30.734
seconds; both ranks logged zero post-readiness preparation-kernel warnings.
The record identifies exact image/source hashes and the component's AST-replay
scope.

This validates the listed preparation signatures. Other kernel warnings remain.
No TP4, all-shapes, long-soak or 1M-token quality qualification is claimed.

Tile coverage is not complete JIT-key coverage. Triton also specializes
argument types and alignment; other attention, MoE and sampling kernels have
their own keys. A vLLM JIT-monitor warning can mean a first use resolved from
disk. Triton's compilation listener reports `cache_hit` and timings, while
B12X logs distinguish `disk-hit` from `miss status=disk-cache-miss`. Use that
evidence to separate compilation from persistent-cache loading. The absence
of additional warnings alone does not establish cold-cache or stability
qualification.

## Source license

The compressed preparation, startup and request-state fixtures retain vLLM's Apache-2.0 contributor headers.
[fixtures/LICENSE](fixtures/LICENSE) preserves the license. The manifest
records image provenance and both compressed and uncompressed hashes; it does
not claim an unmodified upstream source revision for the extracted image files.
