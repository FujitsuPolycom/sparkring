#!/usr/bin/env python3
"""Integration checks for the CPU communicator lifecycle and thread contracts."""
from __future__ import annotations

import argparse
import concurrent.futures
import ctypes as C
import json
import os
from pathlib import Path
import subprocess
import struct
import sys
import tempfile
import threading
import time
import unittest

import test_api as api
from test_bootstrap import raw_peer, raw_reply


SUCCESS, ARGUMENT, USAGE, PROGRESS, TIMEOUT = 0, 4, 5, 7, 8
LIBRARY = Path(__file__).resolve().parents[1] / "build" / "libsircl.so"


class LifecycleTests(unittest.TestCase):
    def test_abort_cancels_an_inflight_split(self):
        handles = (C.c_void_p * 2)()
        self.assertEqual(self.lib.ncclCommInitAll(handles, 2, None), SUCCESS)
        old = os.environ.get("LIBSIRCL_SETUP_TIMEOUT_MS")
        os.environ["LIBSIRCL_SETUP_TIMEOUT_MS"] = "2000"
        child = C.c_void_p()
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                split = pool.submit(self.lib.ncclCommSplit, handles[0], 0, 0, C.byref(child), None)
                time.sleep(0.05)
                self.assertFalse(split.done())
                start = time.monotonic()
                self.assertEqual(self.lib.ncclCommAbort(handles[0]), SUCCESS)
                elapsed = time.monotonic() - start
                handles[0] = None
                self.assertNotEqual(split.result(timeout=1), SUCCESS)
                self.assertLess(elapsed, 1)
        finally:
            if old is None:
                os.environ.pop("LIBSIRCL_SETUP_TIMEOUT_MS", None)
            else:
                os.environ["LIBSIRCL_SETUP_TIMEOUT_MS"] = old
            for handle in handles:
                if handle:
                    self.lib.ncclCommAbort(handle)

    def test_lifecycle_call_on_an_initializing_communicator_is_refused(self):
        """ncclCommRevoke or ncclCommFinalize on a communicator whose initialization is still in progress is
        refused with ncclInvalidUsage. ncclInProgress never becomes a group's result, so a group's pending
        initialization never ends FAILED with ncclInProgress, which would read as still in progress forever."""
        os.environ["SIRCL_BOOTSTRAP_TIMEOUT_MS"] = "5000"
        config = api.config()
        config.blocking = 0
        pending, grouped = C.c_void_p(), C.c_void_p()
        first, second = self.uid(), self.uid()
        self.assertEqual(self.lib.ncclCommInitRankConfig(C.byref(pending), 2, first, 0, C.byref(config)), PROGRESS)
        try:
            self.assertEqual(self.lib.ncclGroupStart(), SUCCESS)
            self.assertEqual(self.lib.ncclCommInitRank(C.byref(grouped), 1, second, 0), SUCCESS)
            self.assertEqual(self.lib.ncclCommRevoke(pending, 0), USAGE)
            self.assertEqual(self.lib.ncclCommFinalize(pending), USAGE)
            self.assertEqual(self.lib.ncclGroupEnd(), SUCCESS)
            self.assertEqual(self.asynchronous_error(grouped), SUCCESS)
        finally:
            self.lib.ncclCommAbort(pending)
            if grouped.value:
                self.lib.ncclCommDestroy(grouped)

    def test_sixty_four_live_roots_are_the_limit_and_the_refusal_names_it(self):
        """Each id a process creates keeps its bootstrap root while its communicator lives: with 64 live one-rank
        communicators a 65th id is refused with ncclInvalidUsage naming the limit, and destroying one
        communicator makes room for an id again."""
        script = r'''
import ctypes as C, json, os, sys, time
sys.path.insert(0, sys.argv[2]); import test_api as api
api.LIBRARY=api.Path(sys.argv[1]); api.initialize_library()
os.environ["LIBSIRCL_BOOTSTRAP_ONLY"]="1"
handles=[api.new_comm() for _ in range(64)]
u=api.UniqueId(); refused=api.LIB.ncclGetUniqueId(C.byref(u)); message=api.LIB.ncclGetLastError(None).decode()
assert api.LIB.ncclCommDestroy(handles.pop())==0
time.sleep(0.2)
again=api.LIB.ncclGetUniqueId(C.byref(u))
print(json.dumps([refused, message, again]))
'''
        out = subprocess.run([sys.executable, "-c", script, str(LIBRARY), str(Path(__file__).parent)],
                             capture_output=True, text=True, timeout=30)
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        refused, message, again = json.loads(out.stdout)
        self.assertEqual(refused, USAGE)
        self.assertIn("already holds 64 bootstrap roots", message)
        self.assertEqual(again, SUCCESS)

    def test_split_leader_failure_fails_every_member_at_once(self):
        """The split leader cannot create the new unique id (the process holds all 64 bootstrap roots). It still
        takes part in the unique-id round with a failure record, so both ranks return its result at once
        instead of the member waiting out LIBSIRCL_SETUP_TIMEOUT_MS, and the parent's bootstrap stays usable."""
        script = r'''
import concurrent.futures, ctypes as C, json, os, sys, time
sys.path.insert(0, sys.argv[2]); import test_api as api
api.LIBRARY=api.Path(sys.argv[1]); api.initialize_library()
os.environ.update(LIBSIRCL_BOOTSTRAP_ONLY="1", LIBSIRCL_SETUP_TIMEOUT_MS="5000", SIRCL_BOOTSTRAP_TIMEOUT_MS="30000")
handles=(C.c_void_p*2)(); assert api.LIB.ncclCommInitAll(handles,2,None)==0
held=[]
while True:
    u=api.UniqueId()
    if api.LIB.ncclGetUniqueId(C.byref(u))!=0: break
    held.append(u)
children=[C.c_void_p(), C.c_void_p()]
def split(r):
    code=api.LIB.ncclCommSplit(handles[r],0,r,C.byref(children[r]),None)
    return code, api.LIB.ncclGetLastError(None).decode()
start=time.monotonic()
with concurrent.futures.ThreadPoolExecutor(2) as pool:
    results=list(pool.map(split, range(2)))
elapsed=time.monotonic()-start
none=[C.c_void_p(), C.c_void_p()]
with concurrent.futures.ThreadPoolExecutor(2) as pool:
    again=list(pool.map(lambda r: api.LIB.ncclCommSplit(handles[r],-1,0,C.byref(none[r]),None), range(2)))
print(json.dumps({"results":results,"elapsed":elapsed,"held":len(held),"children":[c.value for c in children],
                  "again":again}))
'''
        out = subprocess.run([sys.executable, "-c", script, str(LIBRARY), str(Path(__file__).parent)],
                             capture_output=True, text=True, timeout=30)
        self.assertEqual(out.returncode, 0, out.stderr)
        data = json.loads(out.stdout)
        self.assertEqual(data["held"], 63)
        self.assertEqual([code for code, _ in data["results"]], [USAGE, USAGE], data)
        self.assertIn("64 bootstrap roots", data["results"][0][1])
        self.assertIn("split leader (parent rank 0)", data["results"][1][1])
        self.assertLess(data["elapsed"], 2)
        self.assertEqual(data["children"], [None, None])
        self.assertEqual(data["again"], [SUCCESS, SUCCESS])

    def test_split_whose_peer_never_arrives_is_a_remote_error(self):
        """A split whose peer never takes part times out in its first all-gather and returns ncclRemoteError, the
        result the engine's setup exchanges give a peer that does not arrive, naming the timeout."""
        handles = (C.c_void_p * 2)()
        self.assertEqual(self.lib.ncclCommInitAll(handles, 2, None), SUCCESS)
        saved = os.environ.get("LIBSIRCL_SETUP_TIMEOUT_MS")
        os.environ["LIBSIRCL_SETUP_TIMEOUT_MS"] = "300"
        try:
            child = C.c_void_p()
            self.assertEqual(self.lib.ncclCommSplit(handles[0], 0, 0, C.byref(child), None), 6)
            self.assertIn(b"timed out", self.lib.ncclGetLastError(None))
        finally:
            if saved is None:
                os.environ.pop("LIBSIRCL_SETUP_TIMEOUT_MS", None)
            else:
                os.environ["LIBSIRCL_SETUP_TIMEOUT_MS"] = saved
            for handle in handles:
                self.lib.ncclCommAbort(handle)

    def test_split_refuses_short_records_in_both_rounds(self):
        for short_round in (0, 1):
            with self.subTest(round=short_round):
                value = self.uid()
                peer = raw_peer(bytes(value), 2, 1)
                handle, child = C.c_void_p(), C.c_void_p()
                try:
                    self.assertEqual(self.lib.ncclCommInitRank(C.byref(handle), 2, value, 0), SUCCESS)
                    self.assertEqual(raw_reply(peer), SUCCESS)
                    def respond():
                        for round_number in range(short_round + 1):
                            payload = b"\x00" if round_number == short_round else struct.pack("=ii", 0, 1)
                            peer.sendall(b"SCF1" + struct.pack("!III", round_number, len(payload), 0) + payload)
                            header = peer.recv(16, __import__("socket").MSG_WAITALL)
                            self.assertEqual(len(header), 16)
                            size = struct.unpack("!I", header[12:])[0]
                            self.assertEqual(len(peer.recv(size, __import__("socket").MSG_WAITALL)), size)
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        response = pool.submit(respond)
                        result = self.lib.ncclCommSplit(handle, 0, 0, C.byref(child), None)
                        response.result(timeout=2)
                    self.assertEqual(result, 6)
                    self.assertFalse(child.value)
                finally:
                    peer.close()
                    if child.value:
                        self.lib.ncclCommAbort(child)
                    if handle.value:
                        self.lib.ncclCommAbort(handle)

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
            # The group's end and the pending handle name the call that failed the group.
            self.assertIn(b"group initialization failed: negative device number", self.lib.ncclGetLastError(None))
            self.assertEqual(self.asynchronous_error(handle), ARGUMENT)
            self.assertIn(b"group initialization rejected: negative device number", self.lib.ncclGetLastError(handle))
        finally:
            self.dispose_all([handle])

    def test_thread_exiting_inside_a_group_releases_its_pending_handles(self):
        """A thread opens a group, initializes a communicator in it and exits without ncclGroupEnd: the
        handle fails with ncclInvalidUsage, naming the reason, and another thread can destroy it."""
        value = self.uid()
        handle = C.c_void_p()
        codes = []

        def grouped():
            codes.append(self.lib.ncclGroupStart())
            codes.append(self.lib.ncclCommInitRank(C.byref(handle), 1, value, 0))
        thread = threading.Thread(target=grouped)
        thread.start()
        thread.join(timeout=5)
        self.assertEqual(codes, [SUCCESS, SUCCESS])
        self.assertTrue(handle.value)
        self.assertEqual(self.asynchronous_error(handle), USAGE)
        self.assertIn(b"exited before its ncclGroupEnd", self.lib.ncclGetLastError(handle))
        self.assertEqual(self.lib.ncclCommDestroy(handle), SUCCESS)

    def test_calls_that_enqueue_nothing_leave_the_group_as_it_was(self):
        """Inside a group, failed calls that enqueue nothing (a query of an unknown handle, a NULL output to
        ncclGetUniqueId, an unsupported function, revoke flags) return their errors and leave the group's
        initialization and result alone."""
        value = self.uid()
        handle = C.c_void_p()
        count = C.c_int(-1)
        self.lib.ncclCommShrink.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.POINTER(C.c_void_p), C.c_void_p,
                                            C.c_int]
        self.assertEqual(self.lib.ncclGroupStart(), SUCCESS)
        self.assertEqual(self.lib.ncclCommInitRank(C.byref(handle), 1, value, 0), SUCCESS)
        self.assertEqual(self.lib.ncclCommCount(C.c_void_p(16), C.byref(count)), ARGUMENT)
        self.assertEqual(self.lib.ncclGetUniqueId(None), ARGUMENT)
        self.assertEqual(self.lib.ncclCommShrink(handle, None, 0, None, None, 0), USAGE)
        self.assertEqual(self.lib.ncclCommRevoke(handle, 1), ARGUMENT)
        try:
            self.assertEqual(self.lib.ncclGroupEnd(), SUCCESS)
            self.assertEqual(self.asynchronous_error(handle), SUCCESS)
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
            # A query of the failed handle returns its result with its text for this thread.
            self.assertEqual(self.lib.ncclGetVersion(None), ARGUMENT)
            count = C.c_int(-1)
            self.assertEqual(self.lib.ncclCommCount(handle, C.byref(count)), TIMEOUT)
            self.assertIn(b"timed out", self.lib.ncclGetLastError(None))
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

    def test_unload_waits_for_a_call_still_inside_the_library(self):
        """Another thread is inside ncclCommSplit, waiting for a peer, when the library unloads (dlclose, or
        the destructors that exit runs). The unload cancels the call and waits for it to leave before freeing
        the communicator, so the call returns an error instead of using freed memory or unmapped code. The
        exit case is decisive under `make check-asan`, where a use of the freed communicator is reported.

        The parent communicator's bootstrap root is a stand-in broker in the child process that admits rank 0
        and never answers its all-gather, so the split waits in its socket poll until cancellation reaches
        it. In the exit case a helper library, loaded after libsircl and finalized after it, keeps the
        exiting process alive for 0.3 s after libsircl's destructor, long enough for the split thread to wake."""
        script = r'''
import _ctypes, ctypes as C, os, socket, struct, sys, threading, time
sys.path.insert(0, sys.argv[2]); import test_api as api
api.LIBRARY=api.Path(sys.argv[1]); api.initialize_library()
if sys.argv[3]=="exit": C.CDLL(sys.argv[4])
os.environ["LIBSIRCL_BOOTSTRAP_ONLY"]="1"; os.environ["LIBSIRCL_SETUP_TIMEOUT_MS"]="5000"
server=socket.socket(); server.bind(("127.0.0.1",0)); server.listen()
peers=[]
def broker():
    peer,_=server.accept(); peers.append(peer)
    hello=b""
    while len(hello)<136: hello+=peer.recv(136-len(hello))
    peer.sendall(b"SCC1"+struct.pack("!I",0))
threading.Thread(target=broker, daemon=True).start()
u=api.UniqueId(); assert api.LIB.ncclGetUniqueId(C.byref(u))==0
u.internal[16:18]=struct.pack("!H",server.getsockname()[1])
handle=C.c_void_p(); assert api.LIB.ncclCommInitRank(C.byref(handle),2,u,0)==0
child=C.c_void_p(); result=[]
def split(): result.append(api.LIB.ncclCommSplit(handle,0,0,C.byref(child),None))
thread=threading.Thread(target=split, daemon=True); thread.start(); time.sleep(0.1)
assert not result
if sys.argv[3]=="exit":
    sys.stdout.flush(); C.CDLL(None).exit(0)
_ctypes.dlclose(api.LIB._handle)
thread.join(2)
assert result and result[0]!=0, result
print("split returned", result[0])
'''
        with tempfile.TemporaryDirectory() as directory:
            source, helper = Path(directory) / "linger.c", Path(directory) / "liblinger.so"
            source.write_text("#include <unistd.h>\n__attribute__((destructor)) static void linger(void) "
                              "{ usleep(300000); }\n")
            subprocess.run([os.environ.get("CC", "cc"), "-shared", "-fPIC", str(source), "-o", str(helper)],
                           check=True)
            for mode in ("dlclose", "exit"):
                with self.subTest(mode=mode):
                    out = subprocess.run([sys.executable, "-c", script, str(LIBRARY), str(Path(__file__).parent),
                                          mode, str(helper)], capture_output=True, text=True, timeout=10)
                    self.assertEqual(out.returncode, 0, out.stderr)
                    self.assertNotIn("AddressSanitizer", out.stderr)
                    if mode == "dlclose":
                        self.assertIn("split returned", out.stdout)

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
