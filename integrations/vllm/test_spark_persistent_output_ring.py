from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import spark_tp4_backend
from spark_persistent_output_ring import PersistentOutputRing


class _FakeView:
    def __init__(self, pointer: int) -> None:
        self.pointer = pointer


class _FakeStorage:
    def __init__(self, slots: int) -> None:
        self._views = tuple(_FakeView(0x1000 + index) for index in range(slots))

    def unbind(self, dimension: int) -> tuple[_FakeView, ...]:
        if dimension != 0:
            raise AssertionError(f"unexpected unbind dimension {dimension}")
        return self._views


class _FakeTorch:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], object, object]] = []

    def empty(
        self,
        shape: tuple[int, ...],
        *,
        dtype: object,
        device: object,
    ) -> _FakeStorage:
        self.calls.append((shape, dtype, device))
        return _FakeStorage(shape[0])


class _FakeTensor:
    def __init__(
        self,
        shape: tuple[int, ...] = (5, 6144),
        dtype: object = "torch.bfloat16",
        device: object = "cuda:0",
    ) -> None:
        self.shape = shape
        self.dtype = dtype
        self.device = device


class PersistentOutputRingTest(unittest.TestCase):
    def test_backend_slot_setting_is_strict_and_bounded(self) -> None:
        for value, expected in (("0", 0), ("256", 256), ("4096", 4096)):
            with self.subTest(value=value):
                with patch.dict(
                    os.environ,
                    {"SPARK_TP4_PERSISTENT_OUTPUT_SLOTS": value},
                    clear=True,
                ):
                    self.assertEqual(
                        spark_tp4_backend._persistent_output_slots(),
                        expected,
                    )

        for invalid in ("-1", "1.5", "4097"):
            with self.subTest(invalid=invalid):
                with (
                    patch.dict(
                        os.environ,
                        {"SPARK_TP4_PERSISTENT_OUTPUT_SLOTS": invalid},
                        clear=True,
                    ),
                    self.assertRaisesRegex(
                        ValueError,
                        "SPARK_TP4_PERSISTENT_OUTPUT_SLOTS",
                    ),
                ):
                    spark_tp4_backend._persistent_output_slots()

    def test_rejects_nonpositive_slot_count(self) -> None:
        for slots in (0, -1):
            with self.subTest(slots=slots):
                with self.assertRaisesRegex(ValueError, "must be positive"):
                    PersistentOutputRing(slots)

    def test_allocates_one_storage_and_cycles_prebuilt_views(self) -> None:
        torch_module = _FakeTorch()
        tensor = _FakeTensor()
        ring = PersistentOutputRing(3)

        outputs = [ring.acquire(tensor, torch_module) for _ in range(5)]

        self.assertEqual(
            torch_module.calls,
            [((3, 5, 6144), "torch.bfloat16", "cuda:0")],
        )
        self.assertIs(outputs[0], outputs[3])
        self.assertIs(outputs[1], outputs[4])
        self.assertEqual(ring.acquires, 5)
        self.assertEqual(ring.wraps, 1)

    def test_rejects_signature_change(self) -> None:
        torch_module = _FakeTorch()
        ring = PersistentOutputRing(2)
        ring.acquire(_FakeTensor(), torch_module)

        for changed in (
            _FakeTensor(shape=(1, 6144)),
            _FakeTensor(dtype="torch.float16"),
            _FakeTensor(device="cuda:1"),
        ):
            with self.subTest(changed=changed):
                with self.assertRaisesRegex(RuntimeError, "signature changed"):
                    ring.acquire(changed, torch_module)


if __name__ == "__main__":
    unittest.main()
