"""Attribute accepted target-prompt work from authoritative scheduler events.

Reuse is credited when accepted execution consumes a selected prefix. Counts
accumulate across preemption attempts and can exceed the original prompt size.
They exclude draft work, replay inside kernels, and rejected worker outputs.
"""

from dataclasses import dataclass


def _count(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("Attribution token counts and generations must be nonnegative integers")
    return value


@dataclass
class _Attempt:
    generation: int
    local: int
    offered_external: int
    external: int = 0
    pending_restore: bool = False
    consumed: bool = False
    progress: int = 0
    can_readmit: bool = False


class RequestAttribution:
    def __init__(self, prompt_tokens: int):
        self.prompt_tokens = _count(prompt_tokens)
        self.local_tokens_reused = 0
        self.external_tokens_reused = 0
        self.prompt_tokens_computed = 0
        self.preemptions = 0
        self.valid = True
        self.prompt_completed = False
        self.attempt: _Attempt | None = None

    def record(self, event: str, **fields) -> None:
        generation = _count(fields.get("preemptions", 0))
        if generation < self.preemptions:
            self.valid = False
            return
        if event == "preempted":
            self.preemptions = generation
            self.attempt = None
            return
        if event == "admitted":
            local = min(_count(fields["local_tokens"]), self.prompt_tokens)
            external = min(_count(fields["external_tokens"]), self.prompt_tokens - local)
            if (self.attempt is not None and generation == self.attempt.generation
                    and not self.attempt.can_readmit):
                # Resuming a parked restore reports no newly allocated prefix.
                # Its original allocation and receive outcome remain authoritative.
                return
            if generation != self.preemptions:
                self.valid = False
            self.preemptions = generation
            self.attempt = _Attempt(generation, local, external,
                                    pending_restore=external > 0)
            return
        attempt = self.attempt
        if attempt is None or generation != attempt.generation:
            self.valid = False
            return
        if event == "restore_finalized":
            prefix = min(_count(fields["valid_prefix_tokens"]), self.prompt_tokens)
            if not attempt.pending_restore or attempt.consumed:
                self.valid = False
                return
            attempt.pending_restore = False
            attempt.can_readmit = fields.get("success") is False
            attempt.local = min(attempt.local, prefix)
            accepted = max(0, prefix - attempt.local)
            if accepted > attempt.offered_external:
                self.valid = False
            if fields.get("success") is True:
                attempt.external = min(accepted, attempt.offered_external)
            elif accepted:
                # A failed restore cannot prove that its residual external
                # prefix passed every rank's integrity boundary.
                self.valid = False
            return
        if event != "prompt_step_completed":
            raise ValueError("Unknown request attribution event")
        start = min(_count(fields["start_token"]), self.prompt_tokens)
        end = min(_count(fields["end_token"]), self.prompt_tokens)
        if end < start:
            raise ValueError("Completed prompt interval ends before it starts")
        if fields.get("stale"):
            return
        if attempt.pending_restore:
            self.valid = False
        prefix = attempt.local + attempt.external
        if start > max(prefix, attempt.progress):
            self.valid = False
        if not attempt.consumed:
            # Work that recomputes an admitted prefix must not also claim that
            # same prefix as consumed reuse.
            used_local = min(attempt.local, start)
            used_external = min(attempt.external, max(0, start - used_local))
            self.local_tokens_reused += used_local
            self.external_tokens_reused += used_external
            attempt.consumed = True
        self.prompt_tokens_computed += max(0, end - max(start, attempt.progress))
        attempt.progress = max(attempt.progress, end)
        self.prompt_completed |= end == self.prompt_tokens

    def summary(self, status: str) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "local_tokens_reused": self.local_tokens_reused,
            "external_tokens_reused": self.external_tokens_reused,
            "prompt_tokens_computed": self.prompt_tokens_computed,
            "preemptions": self.preemptions,
            "attribution_complete": self.valid and self.prompt_completed and status in (
                "FINISHED_STOPPED", "FINISHED_LENGTH_CAPPED",
            ),
            "status": status,
            "accounting_scope": "accepted_target_prompt_work_across_attempts",
        }
