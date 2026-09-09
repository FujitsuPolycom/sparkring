"""Instrument source-pinned MTP3 scheduler decisions without changing cache policy."""

import ast
import hashlib
from pathlib import Path

BEFORE_SHA256 = "ce9460834e08f97dbbfeb3f1238b78ee6a3363dd59a2aef8ceeb715857385895"
AFTER_SHA256 = "9500c99fd5f7d82e4c5247c41b64fd4cfc085ca39303db8991f4dcb7e5196a68"
CONTINUATION_BEFORE_SHA256 = "f46c40c1c41daf2bab4566dd185d320f4ec1fb11e8d632c90c47b0f1bb1808fa"
CONTINUATION_AFTER_SHA256 = "6c784cb6d30d89a078e386081f650e7249269704f2bea06c063c656c71bd9cf2"
SOURCE_TRANSFORMS = {
    BEFORE_SHA256: AFTER_SHA256,
    CONTINUATION_BEFORE_SHA256: CONTINUATION_AFTER_SHA256,
}


METHODS = '''    def _sparkcache_record_event(self, request, event, **fields):
        connector = self.connector
        if not getattr(connector, "request_cache_events_enabled", False):
            return
        callback = getattr(connector, "record_request_cache_event", None)
        if callback is not None:
            fields.setdefault("preemptions", request.num_preemptions)
            callback(request, event, **fields)

    def _sparkcache_capture_prompt_steps(self, scheduler_output):
        if not getattr(self.connector, "request_cache_events_enabled", False):
            return
        # Preserve dispatch positions: async scheduling advances request counters
        # before the matching output arrives, and preemption can reset them.
        steps = {}
        for req_id, count in scheduler_output.num_scheduled_tokens.items():
            request = self.requests[req_id]
            start = min(request.num_computed_tokens, request.num_prompt_tokens)
            end = min(request.num_computed_tokens + count, request.num_prompt_tokens)
            if end > start or getattr(request, "_sparkcache_consumed_generation", None) != request.num_preemptions:
                steps[req_id] = (start, end, request.num_preemptions)
        scheduler_output._sparkcache_prompt_steps = steps

    def _sparkcache_complete_prompt_step(self, scheduler_output, request, stale):
        step = getattr(scheduler_output, "_sparkcache_prompt_steps", {}).pop(
            request.request_id, None
        )
        if step is None or stale or step[2] != request.num_preemptions:
            return
        if step[0] == step[1] and getattr(request, "_sparkcache_consumed_generation", None) == step[2]:
            return
        self._sparkcache_record_event(
            request, "prompt_step_completed", start_token=step[0],
            end_token=step[1], preemptions=step[2], stale=False,
        )
        request._sparkcache_consumed_generation = step[2]

'''

TRANSFORMS = (
    ('    def _preempt_request(\n', METHODS + '    def _preempt_request(\n'),
    ('                local_lease_alternative = None\n',
     '''                local_lease_alternative = None
                cache_trace_lease_attached = (
                    getattr(request, "_sparkcache_pending_lease_generation", None)
                    == request.num_preemptions
                )
'''),
    ('                        if attached_tokens:\n',
     '''                        if attached_tokens:
                            cache_trace_lease_attached = True
                            # Allocation can defer this attached request; retain
                            # attribution until admission in the same attempt.
                            request._sparkcache_pending_lease_generation = request.num_preemptions
'''),
    ('                # Record at admission so unscheduled lookups are not counted.\n',
     '''                if did_prefix_cache_lookup or cache_trace_lease_attached:
                    cache_trace_local = min(
                        num_new_local_computed_tokens if did_prefix_cache_lookup
                        else request.num_computed_tokens, request.num_prompt_tokens
                    )
                    self._sparkcache_record_event(
                        request, "admitted", local_tokens=cache_trace_local,
                        external_tokens=min(num_external_computed_tokens,
                                            request.num_prompt_tokens - cache_trace_local),
                        lease_attached=cache_trace_lease_attached,
                        source="gpu_lease" if cache_trace_lease_attached else "prefix_lookup",
                    )
                    request._sparkcache_pending_lease_generation = None
                    request._sparkcache_consumed_generation = None

                # Record at admission so unscheduled lookups are not counted.
'''),
    ('    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:\n',
     '    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:\n        self._sparkcache_capture_prompt_steps(scheduler_output)\n'),
    ('        request.num_preemptions += 1\n',
     '        request.num_preemptions += 1\n        self._sparkcache_record_event(request, "preempted")\n'),
    ('            req_index = model_runner_output.req_id_to_index[req_id]\n',
     '''            self._sparkcache_complete_prompt_step(
                scheduler_output, request, output_is_stale
            )
            req_index = model_runner_output.req_id_to_index[req_id]
'''),
    ('        if request.request_id in self.failed_recving_kv_req_ids:\n',
     '''        cache_trace_restore_failed = request.request_id in self.failed_recving_kv_req_ids
        if request.request_id in self.failed_recving_kv_req_ids:
'''),
    ('        self.finished_recving_kv_req_ids.remove(request.request_id)\n',
     '''        self._sparkcache_record_event(
            request, "restore_finalized", success=not cache_trace_restore_failed,
            valid_prefix_tokens=min(request.num_computed_tokens, request.num_prompt_tokens),
        )
        self.finished_recving_kv_req_ids.remove(request.request_id)
'''),
    ('        self._inflight_prefills.discard(request)\n        connector_delay_free_blocks, kv_xfer_params = self._connector_finished(request)\n',
     '''        self._sparkcache_record_event(request, "finished", status=request.status.name)
        self._inflight_prefills.discard(request)
        connector_delay_free_blocks, kv_xfer_params = self._connector_finished(request)
'''),
)


def transform(source: bytes) -> bytes:
    if hashlib.sha256(source).hexdigest() not in SOURCE_TRANSFORMS:
        raise ValueError("Cache-attribution scheduler preimage differs")
    text = source.decode().replace("\r\n", "\n")
    for before, after in TRANSFORMS:
        if text.count(before) != 1:
            raise ValueError("Cache-attribution scheduler anchor differs")
        text = text.replace(before, after, 1)
    ast.parse(text)
    return text.encode()


def apply(path: Path) -> dict:
    source = path.read_bytes()
    before = hashlib.sha256(source).hexdigest()
    for original, patched in SOURCE_TRANSFORMS.items():
        if before == patched:
            return {"before_sha256": original, "after_sha256": patched}
    patched = transform(source)
    after = SOURCE_TRANSFORMS[before]
    if hashlib.sha256(patched).hexdigest() != after:
        raise ValueError("Cache-attribution scheduler postimage differs")
    path.write_bytes(patched)
    return {"before_sha256": before, "after_sha256": after}
