"""PROTOTYPE: portable state oracle for the SparkCache snapshot ring."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SlotState(str, Enum):
    FREE = "FREE"
    GPU_FILLING = "GPU_FILLING"
    READY = "READY"
    WRITING = "WRITING"


@dataclass
class Slot:
    state: SlotState = SlotState.FREE
    generation: int = 0
    context: int = 0
    discarded: bool = False


@dataclass
class SnapshotRing:
    slot_count: int = 3
    slots: list[Slot] = field(init=False)
    would_block: int = 0

    def __post_init__(self) -> None:
        if self.slot_count not in (2, 3):
            raise ValueError("prototype ring requires two or three slots")
        self.slots = [Slot() for _ in range(self.slot_count)]

    def submit(self, context: int) -> str:
        for index, slot in enumerate(self.slots):
            if slot.state is SlotState.FREE:
                slot.generation += 1
                slot.context = context
                slot.discarded = False
                slot.state = SlotState.GPU_FILLING
                return f"submitted context {context} in slot {index}"
        self.would_block += 1
        return "WOULD_BLOCK: inference continues; snapshot dropped"

    def complete(self, index: int) -> str:
        slot = self.slots[index]
        if slot.state is not SlotState.GPU_FILLING:
            return f"slot {index} is not GPU_FILLING"
        if slot.discarded:
            self._free(slot)
            return f"slot {index} drained and dropped"
        slot.state = SlotState.READY
        return f"slot {index} READY; KV lease may be released"

    def claim(self, index: int) -> str:
        slot = self.slots[index]
        if slot.state is not SlotState.READY or slot.discarded:
            return f"slot {index} cannot be claimed"
        slot.state = SlotState.WRITING
        return f"writer owns immutable slot {index}"

    def release(self, index: int) -> str:
        slot = self.slots[index]
        if slot.state not in (SlotState.READY, SlotState.WRITING):
            return f"slot {index} cannot be released"
        self._free(slot)
        return f"slot {index} FREE"

    def abandon(self, context: int) -> str:
        affected = 0
        for slot in self.slots:
            if slot.state is SlotState.FREE or slot.context != context:
                continue
            affected += 1
            if slot.state in (SlotState.GPU_FILLING, SlotState.WRITING):
                slot.discarded = True
            else:
                self._free(slot)
        return f"abandoned context {context}; affected {affected} slots"

    @staticmethod
    def _free(slot: Slot) -> None:
        generation = slot.generation
        slot.state = SlotState.FREE
        slot.generation = generation
        slot.context = 0
        slot.discarded = False
