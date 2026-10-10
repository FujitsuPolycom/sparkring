#!/usr/bin/env python3
"""Integration checks for the CPU communicator lifecycle and thread contracts."""
from __future__ import annotations

import argparse
import concurrent.futures
import ctypes as C
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import unittest

import test_api as api


SUCCESS, ARGUMENT, USAGE, PROGRESS, TIMEOUT = 0, 4, 5, 7, 8
LIBRARY = Path(__file__).resolve().parents[1] / "build" / "libsircl.so"


class LifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        api.LIBRARY = LIBRARY
        api.initialize_library()
        cls.lib = api.LIB

    def setUp(self):
        self.saved_env = {name: os.environ.get(name) for name in
                          ["LIBSIRCL_BOOTSTRAP_ONLY", "SIRCL_BOOTSTRAP_TIMEOUT_MS",
                           "SIRCL_SITE_HASH"]}
        os.environ["LIBSIRCL_BOOTSTRAP_ONLY"] = "1"
        os.environ["SIRCL_BOOTSTRAP_TIMEOUT_MS"] = "700"
        os.environ.pop("SIRCL_SITE_HASH", None)

    def tearDown(self):
        for name, value in self.saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def uid(self):
        value = api.UniqueId()
        self.assertEqual(self.lib.ncclGetUniqueId(C.byref(value)), SUCCESS)
        return value

    def asynchronous_error(self, handle):
        error = C.c_int(-1)
        self.assertEqual(self.lib.ncclCommGetAsyncError(handle, C.byref(error)), SUCCESS)
        return error.value

    def wait_ready(self, handles):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            errors = [self.asynchronous_error(handle) for handle in handles]
            self.assertTrue(all(error in [SUCCESS, PROGRESS] for error in errors), errors)
            if all(error == SUCCESS for error in errors):
                return
            time.sleep(0.002)
        self.fail("communicators did not become ready")

    def dispose_all(self, handles):
        for handle in handles:
            if handle.value:
                self.assertEqual(self.lib.ncclCommDestroy(handle), SUCCESS)

    def assert_metadata(self, handles, world):
        for rank, handle in enumerate(handles):
            for name, expected in [("Count", world), ("UserRank", rank)]:
                value = C.c_int(-1)
                self.assertEqual(getattr(self.lib, "ncclComm" + name)(handle, C.byref(value)), SUCCESS)
                self.assertEqual(value.value, expected)

    def test_grouped_initialization_world_sizes_two_through_eight(self):
        for world in range(2, 9):
            with self.subTest(world=world):
                value = self.uid()
                handles = [C.c_void_p() for _ in range(world)]
                self.assertEqual(self.lib.ncclGroupStart(), SUCCESS)
                for rank in reversed(range(world)):
                    self.assertEqual(self.lib.ncclCommInitRank(C.byref(handles[rank]),
                                                               world, value, rank), SUCCESS)
                    self.assertEqual(self.asynchronous_error(handles[rank]), PROGRESS)
                self.assertEqual(self.lib.ncclGroupEnd(), SUCCESS)
                try:
                    self.assert_metadata(handles, world)
                    for handle in handles:
                        self.assertEqual(self.lib.ncclCommFinalize(handle), SUCCESS)
                finally:
                    self.dispose_all(handles)

    def test_initall_devices_and_nested_group_deferral(self):
        devices = (C.c_int * 4)(3, 4, 8, 9)
        storage = (C.c_void_p * 4)()
        self.assertEqual(self.lib.ncclGroupStart(), SUCCESS)
        self.assertEqual(self.lib.ncclCommInitAll(storage, 4, devices), SUCCESS)
        handles = [C.c_void_p(pointer) for pointer in storage]
        for handle in handles:
            self.assertEqual(self.asynchronous_error(handle), PROGRESS)
        self.assertEqual(self.lib.ncclGroupEnd(), SUCCESS)
        try:
            self.assert_metadata(handles, 4)
            for rank, handle in enumerate(handles):
                device = C.c_int(-1)
                self.assertEqual(self.lib.ncclCommCuDevice(handle, C.byref(device)), SUCCESS)
                self.assertEqual(device.value, devices[rank])
        finally:
            self.dispose_all(handles)

    def test_nested_error_propagates_to_pending_handles(self):
        value = self.uid()
        handle = C.c_void_p()
        self.assertEqual(self.lib.ncclGroupStart(), SUCCESS)
        self.assertEqual(self.lib.ncclGroupStart(), SUCCESS)
        self.assertEqual(self.lib.ncclCommInitRank(C.byref(handle), 1, value, 0), SUCCESS)
        # A rejected InitAll must preserve its precise error at the outer end.
        storage = (C.c_void_p * 2)()
        devices = (C.c_int * 2)(0, -1)
        self.assertEqual(self.lib.ncclCommInitAll(storage, 2, devices), ARGUMENT)
        self.assertEqual(list(storage), [None, None])
        self.assertEqual(self.lib.ncclGroupEnd(), SUCCESS)
        self.assertEqual(self.lib.ncclGroupEnd(), ARGUMENT)
        try:
            self.assertEqual(self.asynchronous_error(handle), ARGUMENT)
            self.assertIn(b"group", self.lib.ncclGetLastError(handle))
        finally:
            self.dispose_all([handle])

    def test_pending_group_handle_cannot_be_destroyed_from_another_thread(self):
        value = self.uid()
        handle = C.c_void_p()
        self.assertEqual(self.lib.ncclGroupStart(), SUCCESS)
        self.assertEqual(self.lib.ncclCommInitRank(C.byref(handle), 1, value, 0), SUCCESS)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            self.assertEqual(pool.submit(self.lib.ncclCommDestroy, handle).result(timeout=1), USAGE)
        self.assertEqual(self.lib.ncclGroupEnd(), SUCCESS)
        self.dispose_all([handle])

    def test_nonblocking_initialization_polling(self):
        value = self.uid()
        config = api.config(72)
        config.blocking = 0
        handles = [C.c_void_p() for _ in range(8)]
        for rank, handle in enumerate(handles):
            self.assertEqual(self.lib.ncclCommInitRankConfig(C.byref(handle), 8, value,
                                                            rank, C.byref(config)), PROGRESS)
        try:
            self.wait_ready(handles)
            self.assert_metadata(handles, 8)
        finally:
            self.dispose_all(handles)

    def test_grouped_nonblocking_initialization_polling(self):
        value = self.uid()
        config = api.config()
        config.blocking = 0
        handles = [C.c_void_p() for _ in range(3)]
        self.assertEqual(self.lib.ncclGroupStart(), SUCCESS)
        for rank, handle in enumerate(handles):
            self.assertEqual(self.lib.ncclCommInitRankConfig(C.byref(handle), 3, value,
                                                            rank, C.byref(config)), SUCCESS)
        self.assertEqual(self.lib.ncclGroupEnd(), PROGRESS)
        try:
            self.wait_ready(handles)
            self.assert_metadata(handles, 3)
        finally:
            self.dispose_all(handles)

    def test_group_terminal_error_takes_precedence_over_inprogress(self):
        good = self.uid()
        malformed = api.UniqueId.from_buffer_copy(bytes(128))
        config = api.config()
        config.blocking = 0
        handles = [C.c_void_p(), C.c_void_p()]
        self.assertEqual(self.lib.ncclGroupStart(), SUCCESS)
        self.assertEqual(self.lib.ncclCommInitRankConfig(C.byref(handles[0]), 1, good, 0,
                                                        C.byref(config)), SUCCESS)
        self.assertEqual(self.lib.ncclCommInitRank(C.byref(handles[1]), 1, malformed, 0), SUCCESS)
        result = self.lib.ncclGroupEnd()
        try:
            self.assertEqual(result, ARGUMENT)
            self.assertEqual(self.asynchronous_error(handles[1]), ARGUMENT)
        finally:
            self.dispose_all(handles)

    def test_abort_and_destroy_cancel_incomplete_nonblocking_init(self):
        os.environ["SIRCL_BOOTSTRAP_TIMEOUT_MS"] = "5000"
        for operation in ["Abort", "Destroy"]:
            with self.subTest(operation=operation):
                value = self.uid()
                config = api.config()
                config.blocking = 0
                handle = C.c_void_p()
                self.assertEqual(self.lib.ncclCommInitRankConfig(C.byref(handle), 2, value,
                                                                0, C.byref(config)), PROGRESS)
                time.sleep(0.02)
                self.assertEqual(self.asynchronous_error(handle), PROGRESS)
                start = time.monotonic()
                self.assertEqual(getattr(self.lib, "ncclComm" + operation)(handle), SUCCESS)
                self.assertLess(time.monotonic() - start, 0.20)
                output = C.c_int(-1)
                self.assertEqual(self.lib.ncclCommGetAsyncError(handle, C.byref(output)), ARGUMENT)

    def test_incomplete_blocking_init_retains_failure_for_cleanup(self):
        os.environ["SIRCL_BOOTSTRAP_TIMEOUT_MS"] = "120"
        value = self.uid()
        handle = C.c_void_p()
        self.assertEqual(self.lib.ncclCommInitRank(C.byref(handle), 2, value, 0), TIMEOUT)
        try:
            self.assertTrue(handle.value)
            self.assertEqual(self.asynchronous_error(handle), TIMEOUT)
            self.assertIn(b"timed out", self.lib.ncclGetLastError(handle))
        finally:
            self.dispose_all([handle])

    def test_split_orders_members_by_key_and_leaves_nocolor_ranks_out(self):
        world = 4
        value = self.uid()
        handles = [C.c_void_p() for _ in range(world)]
        self.assertEqual(self.lib.ncclGroupStart(), SUCCESS)
        for rank in range(world):
            self.assertEqual(self.lib.ncclCommInitRank(C.byref(handles[rank]), world, value, rank), SUCCESS)
        self.assertEqual(self.lib.ncclGroupEnd(), SUCCESS)
        self.lib.ncclCommSplit.argtypes = [C.c_void_p, C.c_int, C.c_int, C.POINTER(C.c_void_p), C.c_void_p]
        colors, keys = [0, 1, 0, -1], [5, 1, 2, 0]
        children = [C.c_void_p() for _ in range(world)]
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=world) as pool:
                codes = list(pool.map(lambda r: self.lib.ncclCommSplit(handles[r], colors[r], keys[r],
                                                                       C.byref(children[r]), None), range(world)))
            self.assertEqual(codes, [SUCCESS] * world)
            self.assertIsNone(children[3].value)
            # Color 0 holds parent ranks 2 (key 2) and 0 (key 5); color 1 holds parent rank 1 alone.
            for parent, (count, rank) in {0: (2, 1), 1: (1, 0), 2: (2, 0)}.items():
                for name, expected in [("Count", count), ("UserRank", rank)]:
                    got = C.c_int(-1)
                    self.assertEqual(getattr(self.lib, "ncclComm" + name)(children[parent], C.byref(got)), SUCCESS)
                    self.assertEqual(got.value, expected, (parent, name))
        finally:
            self.dispose_all([child for child in children if child.value])
            self.dispose_all(handles)

    def test_split_without_config_inherits_a_nonblocking_parent(self):
        # NULL config: the child takes the parent's blocking mode, so its initialization returns
        # ncclInProgress and completes by polling, as the parent's did.
        value = self.uid()
        config = api.config()
        config.blocking = 0
        handles = [C.c_void_p() for _ in range(2)]
        children = [C.c_void_p() for _ in range(2)]
        self.lib.ncclCommSplit.argtypes = [C.c_void_p, C.c_int, C.c_int, C.POINTER(C.c_void_p), C.c_void_p]
        try:
            for rank, handle in enumerate(handles):
                self.assertEqual(self.lib.ncclCommInitRankConfig(C.byref(handle), 2, value, rank, C.byref(config)),
                                 PROGRESS)
            self.wait_ready(handles)
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                codes = list(pool.map(lambda r: self.lib.ncclCommSplit(handles[r], 0, r, C.byref(children[r]), None),
                                      range(2)))
            self.assertEqual(codes, [PROGRESS] * 2)
            self.wait_ready(children)
            self.assert_metadata(children, 2)
        finally:
            self.dispose_all([child for child in children if child.value])
            self.dispose_all([handle for handle in handles if handle.value])

    def test_queries_can_race_with_destroy(self):
        # Pollers hold registry references. Destroy removes membership first,
        # waits for references, and stale queries return InvalidArgument.
        for _ in range(10):
            handle = api.new_comm()
            start = threading.Barrier(5)
            stop = threading.Event()
            def polling():
                start.wait(timeout=2)
                results = []
                for _ in range(400):
                    value = C.c_int(-1)
                    result = self.lib.ncclCommCount(handle, C.byref(value))
                    results.append(result)
                    if result == SUCCESS:
                        self.assertEqual(value.value, 1)
                    else:
                        self.assertEqual(result, ARGUMENT)
                    self.lib.ncclGetLastError(handle)
                    if stop.is_set() and result == ARGUMENT:
                        break
                return results
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                jobs = [pool.submit(polling) for _ in range(4)]
                start.wait(timeout=2)
                self.assertEqual(self.lib.ncclCommDestroy(handle), SUCCESS)
                stop.set()
                for job in jobs:
                    self.assertTrue(job.result(timeout=2))
            output = C.c_int(-1)
            self.assertEqual(self.lib.ncclCommCount(handle, C.byref(output)), ARGUMENT)

    def test_repeated_init_destroy_releases_file_descriptors(self):
        # Isolate from IDs intentionally left incomplete by other tests.
        script = r'''
import ctypes as C, os, sys, time
sys.path.insert(0, sys.argv[2]); import test_api as api
api.LIBRARY=api.Path(sys.argv[1]); api.initialize_library()
before=len(os.listdir('/proc/self/fd'))
for i in range(100):
    h=api.new_comm()
    assert api.LIB.ncclCommDestroy(h)==0
time.sleep(0.15)
after=len(os.listdir('/proc/self/fd'))
assert after==before, (before, after)
'''
        subprocess.run([sys.executable, "-c", script, str(LIBRARY), str(Path(__file__).parent)],
                       check=True, capture_output=True, text=True, timeout=5)

    def test_dso_unload_joins_workers_and_closes_roots(self):
        script = r'''
import _ctypes, ctypes as C, os, sys, time
sys.path.insert(0, sys.argv[2]); import test_api as api
api.LIBRARY=api.Path(sys.argv[1]); api.initialize_library()
before_fd=len(os.listdir('/proc/self/fd'))
before_threads=len(os.listdir('/proc/self/task'))
ready=api.new_comm()
u=api.UniqueId(); assert api.LIB.ncclGetUniqueId(C.byref(u))==0
c=api.config(); c.blocking=0; h=C.c_void_p()
assert api.LIB.ncclCommInitRankConfig(C.byref(h),2,u,0,C.byref(c))==7
time.sleep(0.02)
start=time.monotonic()
_ctypes.dlclose(api.LIB._handle)
assert time.monotonic()-start<0.5
time.sleep(0.05)
assert len(os.listdir('/proc/self/fd'))==before_fd
assert len(os.listdir('/proc/self/task'))==before_threads
'''
        env = os.environ.copy()
        env["SIRCL_BOOTSTRAP_TIMEOUT_MS"] = "5000"
        subprocess.run([sys.executable, "-c", script, str(LIBRARY), str(Path(__file__).parent)],
                       env=env, check=True, capture_output=True, text=True, timeout=3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, default=LIBRARY)
    options, extra = parser.parse_known_args()
    LIBRARY = options.library.resolve()
    unittest.main(argv=[sys.argv[0]] + extra, verbosity=2)
