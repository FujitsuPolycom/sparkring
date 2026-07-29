# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from math import floor


@dataclass(frozen=True)
class AcceptanceLengthUpdate:
    previous_num_spec_tokens: int
    num_spec_tokens: int
    mean_num_accepted_tokens: float
    mean_num_draft_tokens: float


class AcceptanceLengthController:
    """Adjust speculative depth from the observed accepted draft length."""

    def __init__(self, max_num_spec_tokens: int, observation_window: int) -> None:
        if max_num_spec_tokens <= 0:
            raise ValueError("max_num_spec_tokens must be greater than zero.")
        if observation_window <= 0:
            raise ValueError("observation_window must be greater than zero.")

        self.max_num_spec_tokens = max_num_spec_tokens
        self.observation_window = observation_window
        self.num_spec_tokens = max_num_spec_tokens

        self._num_observation_steps = 0
        self._num_drafts = 0
        self._num_draft_tokens = 0
        self._num_accepted_tokens = 0

    def observe_batch(
        self,
        *,
        num_drafts: int,
        num_draft_tokens: int,
        num_accepted_tokens: int,
    ) -> AcceptanceLengthUpdate | None:
        """Observe one scheduler step and occasionally update the depth."""
        if num_drafts < 0 or num_draft_tokens < 0 or num_accepted_tokens < 0:
            raise ValueError("Speculative decoding counts must be non-negative.")
        if num_accepted_tokens > num_draft_tokens:
            raise ValueError("num_accepted_tokens must not exceed num_draft_tokens.")
        if num_drafts == 0:
            if num_draft_tokens or num_accepted_tokens:
                raise ValueError("Token counts require at least one draft.")
            return None

        self._num_observation_steps += 1
        self._num_drafts += num_drafts
        self._num_draft_tokens += num_draft_tokens
        self._num_accepted_tokens += num_accepted_tokens
        if self._num_observation_steps < self.observation_window:
            return None

        mean_num_accepted_tokens = self._num_accepted_tokens / self._num_drafts
        mean_num_draft_tokens = self._num_draft_tokens / self._num_drafts
        target_num_spec_tokens = min(
            self.max_num_spec_tokens,
            max(1, floor(mean_num_accepted_tokens + 1.5)),
        )

        # Ratchet-down fix (boot AA found it, BB refined it): when the depth is
        # snapped to a ladder point, floor(mean+1.5) can never exceed the snapped
        # depth, so a highly-accepted window stays stuck low forever (depth drops
        # under high-entropy load and never recovers on return to code). Use an
        # acceptance RATIO, not an absolute gap: at depth 2 code accepts ~1.87/2
        # (93%) — clearly headroom to go deeper — but an absolute-gap test
        # (accepted >= drafted-0.05) misses it. Climb when >=80% of drafts land.
        if mean_num_draft_tokens > 0 and (
            mean_num_accepted_tokens >= 0.8 * mean_num_draft_tokens
        ):
            target_num_spec_tokens = max(
                target_num_spec_tokens,
                min(self.max_num_spec_tokens, self.num_spec_tokens + 1),
            )

        previous_num_spec_tokens = self.num_spec_tokens
        if target_num_spec_tokens < self.num_spec_tokens:
            self.num_spec_tokens = target_num_spec_tokens
        elif target_num_spec_tokens > self.num_spec_tokens:
            self.num_spec_tokens += 1

        self._reset_window()
        return AcceptanceLengthUpdate(
            previous_num_spec_tokens=previous_num_spec_tokens,
            num_spec_tokens=self.num_spec_tokens,
            mean_num_accepted_tokens=mean_num_accepted_tokens,
            mean_num_draft_tokens=mean_num_draft_tokens,
        )

    def _reset_window(self) -> None:
        self._num_observation_steps = 0
        self._num_drafts = 0
        self._num_draft_tokens = 0
        self._num_accepted_tokens = 0
