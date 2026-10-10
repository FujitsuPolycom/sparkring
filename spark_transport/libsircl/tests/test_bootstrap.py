#!/usr/bin/env python3
"""CPU-only loopback bootstrap tests. No CUDA, RDMA or Spark access.

The test compiles the internal bootstrap separately so it does not require its
symbols to be exported from the production libnccl ABI.
"""
import concurrent.futures
import ctypes
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time
import unittest
import warnings


class Bootstrap:
    def __init__(self, path):
        self.path = str(path)
        self.lib = ctypes.CDLL(self.path)
        self.lib.sccl_bootstrap_id.argtypes = [ctypes.c_void_p]
        self.lib.sccl_bootstrap_id.restype = ctypes.c_int
        self.lib.sccl_bootstrap_join.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                                ctypes.c_int,
                                                ctypes.POINTER(ctypes.c_void_p)]
        self.lib.sccl_bootstrap_join.restype = ctypes.c_int
        self.lib.sccl_bootstrap_close.argtypes = [ctypes.c_void_p]
        self.lib.sccl_bootstrap_fd.argtypes = [ctypes.c_void_p]
        self.lib.sccl_bootstrap_fd.restype = ctypes.c_int
        self.lib.sccl_bootstrap_error.restype = ctypes.c_char_p
        self.lib.sccl_bootstrap_process_valid.restype = ctypes.c_int
        self.lib.test_cancel_join.argtypes = self.lib.sccl_bootstrap_join.argtypes
        self.lib.test_cancel_join.restype = ctypes.c_int
        self.lib.test_already_cancelled_join.argtypes = self.lib.sccl_bootstrap_join.argtypes
        self.lib.test_already_cancelled_join.restype = ctypes.c_int
        self.lib.sccl_bootstrap_allgather.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint,
                                                      ctypes.POINTER(ctypes.c_void_p),
                                                      ctypes.POINTER(ctypes.c_uint), ctypes.c_void_p]
        self.lib.sccl_bootstrap_allgather.restype = ctypes.c_int
        self.lib.sccl_bootstrap_allgather_within.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint,
                                                             ctypes.POINTER(ctypes.c_void_p),
                                                             ctypes.POINTER(ctypes.c_uint), ctypes.c_int,
                                                             ctypes.c_void_p]
        self.lib.sccl_bootstrap_allgather_within.restype = ctypes.c_int
        self.libc = ctypes.CDLL(None)
        self.libc.free.argtypes = [ctypes.c_void_p]

    def allgather(self, handle, data, world):
        out = ctypes.c_void_p()
        lengths = (ctypes.c_uint * 8)()
        result = self.lib.sccl_bootstrap_allgather(handle, data, len(data), ctypes.byref(out), lengths, None)
        if result:
            return result, None
        total = sum(lengths[:world])
        blob = ctypes.string_at(out.value, total) if total else b""
        self.libc.free(out)
        parts, at = [], 0
        for k in range(world):
            parts.append(blob[at:at + lengths[k]])
            at += lengths[k]
        return 0, parts

    def unique_id(self):
        value = ctypes.create_string_buffer(128)
        result = self.lib.sccl_bootstrap_id(value)
        if result:
            raise RuntimeError((result, self.lib.sccl_bootstrap_error()))
        return value.raw

    def join(self, value, world, rank):
        handle = ctypes.c_void_p()
        result = self.lib.sccl_bootstrap_join(value, world, rank,
                                             ctypes.byref(handle))
        return result, handle

    def close(self, handle):
        self.lib.sccl_bootstrap_close(handle)


def raw_peer(value, world, rank):
    peer = socket.create_connection(("127.0.0.1", struct.unpack("!H", value[16:18])[0]),
                                    timeout=1)
    peer.sendall(value + struct.pack("!II", world, rank))
    return peer


def raw_reply(peer):
    response = b""
    while len(response) < 8:
        part = peer.recv(8 - len(response))
        if not part:
            raise RuntimeError("lost bootstrap reply")
        response += part
    if response[:4] != b"SCC1":
        raise RuntimeError("invalid bootstrap reply")
    return struct.unpack("!I", response[4:])[0]


def fork_for_safety_probe():
    # These probes intentionally verify refusal after fork with active workers.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return os.fork()


class BootstrapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = tempfile.TemporaryDirectory(prefix="sircl-bootstrap-")
        source = Path(__file__).resolve().parents[1] / "src" / "bootstrap.c"
        library = Path(cls.build.name) / "bootstrap.so"
        cancellation_probe = Path(cls.build.name) / "cancel_probe.c"
        cancellation_probe.write_text(r'''
#define _POSIX_C_SOURCE 200809L
#include "bootstrap.h"
#include <pthread.h>
#include <time.h>
static void *cancel_after_delay(void *arg) {
  struct timespec delay = {0, 50000000};
  nanosleep(&delay, NULL);
  atomic_store_explicit((atomic_int *)arg, 1, memory_order_relaxed);
  return NULL;
}
int test_cancel_join(const unsigned char id[128], int world, int rank,
                     sccl_bootstrap **out) {
  atomic_int cancelled;
  atomic_init(&cancelled, 0);
  pthread_t thread;
  if (pthread_create(&thread, NULL, cancel_after_delay, &cancelled)) return 2;
  int result = sccl_bootstrap_join_cancel(id, world, rank, out, &cancelled);
  pthread_join(thread, NULL);
  return result;
}
int test_already_cancelled_join(const unsigned char id[128], int world, int rank,
                               sccl_bootstrap **out) {
  atomic_int cancelled;
  atomic_init(&cancelled, 1);
  return sccl_bootstrap_join_cancel(id, world, rank, out, &cancelled);
}
''', encoding="utf-8")
        subprocess.run([os.environ.get("CC", "cc"), "-std=c11", "-Wall", "-Wextra",
                        "-Werror", "-pedantic", "-O2", "-D_FORTIFY_SOURCE=2",
                        "-fPIC", "-shared", "-pthread", "-I", str(source.parent),
                        str(source), str(cancellation_probe), "-o", str(library)], check=True)
        cls.bootstrap = Bootstrap(library)

    @classmethod
    def tearDownClass(cls):
        cls.build.cleanup()

    def setUp(self):
        self.old_timeout = os.environ.get("SIRCL_BOOTSTRAP_TIMEOUT_MS")
        self.old_hash = os.environ.get("SIRCL_SITE_HASH")
        os.environ["SIRCL_BOOTSTRAP_TIMEOUT_MS"] = "1200"
        os.environ.pop("SIRCL_SITE_HASH", None)

    def tearDown(self):
        for name, value in [("SIRCL_BOOTSTRAP_TIMEOUT_MS", self.old_timeout),
                            ("SIRCL_SITE_HASH", self.old_hash)]:
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def assert_group(self, value, world):
        with concurrent.futures.ThreadPoolExecutor(max_workers=world) as pool:
            futures = [pool.submit(self.bootstrap.join, value, world, rank)
                       for rank in reversed(range(world))]
            results = [f.result(timeout=3) for f in futures]
        try:
            self.assertEqual([result for result, _ in results], [0] * world)
            for _, handle in results:
                self.assertGreaterEqual(self.bootstrap.lib.sccl_bootstrap_fd(handle), 0)
        finally:
            for _, handle in results:
                self.bootstrap.close(handle)

    def test_two_processes_id_creator_is_rank_one(self):
        value = self.bootstrap.unique_id()
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                                  "--child", self.bootstrap.path, value.hex(), "2", "0"],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        result, handle = self.bootstrap.join(value, 2, 1)
        try:
            self.assertEqual(result, 0)
            stdout, stderr = child.communicate(timeout=3)
            self.assertEqual(child.returncode, 0, stderr)
            self.assertEqual(stdout.strip(), "0")
        finally:
            self.bootstrap.close(handle)
            if child.poll() is None:
                child.kill()
                child.communicate()

    def test_eight_ranks(self):
        self.assert_group(self.bootstrap.unique_id(), 8)

    def test_single_rank(self):
        self.assert_group(self.bootstrap.unique_id(), 1)

    def test_simultaneous_ids(self):
        ids = [self.bootstrap.unique_id() for _ in range(4)]
        self.assertEqual(len(set(ids)), 4)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            jobs = [pool.submit(self.assert_group, value, 3) for value in ids]
            for job in jobs:
                job.result(timeout=4)

    def test_duplicate_rank_does_not_poison_group(self):
        value = self.bootstrap.unique_id()
        first = raw_peer(value, 2, 0)
        try:
            # Waiting briefly lets the first complete handshake establish rank 0.
            time.sleep(0.03)
            result, duplicate = self.bootstrap.join(value, 2, 0)
            self.assertEqual(result, 5)
            self.assertFalse(duplicate.value)
            result, second = self.bootstrap.join(value, 2, 1)
            try:
                self.assertEqual(result, 0)
                self.assertEqual(raw_reply(first), 0)
            finally:
                self.bootstrap.close(second)
        finally:
            first.close()

    def test_wrong_world_does_not_poison_group(self):
        value = self.bootstrap.unique_id()
        first = raw_peer(value, 2, 0)
        try:
            time.sleep(0.03)
            result, rejected = self.bootstrap.join(value, 3, 1)
            self.assertEqual(result, 5)
            self.assertFalse(rejected.value)
            result, second = self.bootstrap.join(value, 2, 1)
            try:
                self.assertEqual(result, 0)
                self.assertEqual(raw_reply(first), 0)
            finally:
                self.bootstrap.close(second)
        finally:
            first.close()

    def test_bad_nonce_does_not_poison_group(self):
        value = self.bootstrap.unique_id()
        altered = bytearray(value)
        altered[20] ^= 1
        result, handle = self.bootstrap.join(bytes(altered), 2, 0)
        self.assertEqual(result, 6)
        self.assertFalse(handle.value)
        self.assert_group(value, 2)

    def test_site_hash_and_external_address(self):
        os.environ["SIRCL_SITE_HASH"] = "a5" * 32
        value = self.bootstrap.unique_id()
        self.assertEqual(value[36:68], bytes.fromhex("a5" * 32))
        os.environ["SIRCL_SITE_HASH"] = "b6" * 32
        result, handle = self.bootstrap.join(value, 1, 0)
        self.assertEqual(result, 5)
        self.assertFalse(handle.value)
        os.environ["SIRCL_SITE_HASH"] = "a5" * 32
        altered = bytearray(value)
        altered[12:16] = bytes([203, 0, 113, 1])
        result, handle = self.bootstrap.join(bytes(altered), 1, 0)
        self.assertEqual(result, 5)
        self.assertFalse(handle.value)
        self.assert_group(value, 1)

    def test_incomplete_init_timeout(self):
        os.environ["SIRCL_BOOTSTRAP_TIMEOUT_MS"] = "180"
        value = self.bootstrap.unique_id()
        start = time.monotonic()
        result, handle = self.bootstrap.join(value, 2, 0)
        elapsed = time.monotonic() - start
        self.assertEqual(result, 8)
        self.assertFalse(handle.value)
        self.assertGreater(elapsed, 0.10)
        self.assertLess(elapsed, 1.0)

    def test_cancelled_init_returns_without_waiting_for_world(self):
        value = self.bootstrap.unique_id()
        handle = ctypes.c_void_p()
        start = time.monotonic()
        result = self.bootstrap.lib.test_cancel_join(value, 2, 0,
                                                     ctypes.byref(handle))
        self.assertEqual(result, 5)
        self.assertFalse(handle.value)
        self.assertLess(time.monotonic() - start, 0.30)
        self.assertIn(b"cancelled", self.bootstrap.lib.sccl_bootstrap_error())
        time.sleep(0.03)
        self.assert_group(value, 2)

    def test_already_cancelled_init_does_not_connect(self):
        value = self.bootstrap.unique_id()
        handle = ctypes.c_void_p()
        result = self.bootstrap.lib.test_already_cancelled_join(value, 2, 0,
                                                                ctypes.byref(handle))
        self.assertEqual(result, 5)
        self.assertFalse(handle.value)
        self.assert_group(value, 1)

    def test_unused_id_closes_listener_and_descriptors(self):
        # Prior brokers see rank shutdowns asynchronously; wait for their polls.
        time.sleep(0.15)
        before = len(os.listdir("/proc/self/fd"))
        os.environ["SIRCL_BOOTSTRAP_TIMEOUT_MS"] = "100"
        value = self.bootstrap.unique_id()
        self.assertGreater(len(os.listdir("/proc/self/fd")), before)
        time.sleep(0.20)
        self.assertEqual(len(os.listdir("/proc/self/fd")), before)
        result, handle = self.bootstrap.join(value, 1, 0)
        self.assertEqual(result, 6)
        self.assertFalse(handle.value)

    def test_fork_child_libc_exit_does_not_stop_parent_listener(self):
        value = self.bootstrap.unique_id()
        child = fork_for_safety_probe()
        if child == 0:
            signal.alarm(2)
            # libc exit runs DSO destructors; os._exit would hide this defect.
            libc = ctypes.CDLL(None)
            libc.exit.argtypes = [ctypes.c_int]
            libc.exit(0)
            os._exit(99)
        waited, status = os.waitpid(child, 0)
        self.assertEqual(waited, child)
        self.assertTrue(os.WIFEXITED(status))
        self.assertEqual(os.WEXITSTATUS(status), 0)
        self.assert_group(value, 2)

    def test_fork_child_refuses_api_and_preserves_parent_communicator(self):
        value = self.bootstrap.unique_id()
        result, handle = self.bootstrap.join(value, 1, 0)
        self.assertEqual(result, 0)
        parent_fd = self.bootstrap.lib.sccl_bootstrap_fd(handle)
        read_end, write_end = os.pipe()
        child = fork_for_safety_probe()
        if child == 0:
            signal.alarm(2)
            os.close(read_end)
            try:
                buffer = ctypes.create_string_buffer(128)
                id_result = self.bootstrap.lib.sccl_bootstrap_id(buffer)
                join_result, rejected = self.bootstrap.join(value, 1, 0)
                invalid_process = self.bootstrap.lib.sccl_bootstrap_process_valid() == 0
                descriptor = self.bootstrap.lib.sccl_bootstrap_fd(handle)
                self.bootstrap.close(handle)
                inherited_fd_retained = os.fstat(parent_fd).st_mode > 0
                diagnostic = self.bootstrap.lib.sccl_bootstrap_error()
                good = (id_result == 5 and join_result == 5 and not rejected.value
                        and invalid_process and descriptor == -1
                        and inherited_fd_retained and b"exec" in diagnostic)
                os.write(write_end, b"1" if good else b"0")
            except BaseException:
                os.write(write_end, b"0")
            os.close(write_end)
            libc = ctypes.CDLL(None)
            libc.exit.argtypes = [ctypes.c_int]
            libc.exit(0)
            os._exit(99)
        os.close(write_end)
        try:
            self.assertEqual(os.read(read_end, 1), b"1")
            _, status = os.waitpid(child, 0)
            self.assertTrue(os.WIFEXITED(status))
            self.assertEqual(os.WEXITSTATUS(status), 0)
            self.assertEqual(self.bootstrap.lib.sccl_bootstrap_process_valid(), 1)
            self.assertEqual(self.bootstrap.lib.sccl_bootstrap_fd(handle), parent_fd)
            # The broker's TCP endpoint is still live after inherited cleanup.
            peer = socket.fromfd(parent_fd, socket.AF_INET, socket.SOCK_STREAM)
            try:
                peer.setblocking(False)
                with self.assertRaises(BlockingIOError):
                    peer.recv(1, socket.MSG_PEEK)
            finally:
                peer.close()
        finally:
            os.close(read_end)
            self.bootstrap.close(handle)

    def test_partial_hello_does_not_block_valid_ranks(self):
        value = self.bootstrap.unique_id()
        partial = socket.create_connection(("127.0.0.1", struct.unpack("!H", value[16:18])[0]))
        try:
            partial.sendall(b"S")
            self.assert_group(value, 2)
            self.assertEqual(raw_reply(partial), 6)
        finally:
            partial.close()

    def test_disconnected_rank_can_rejoin(self):
        value = self.bootstrap.unique_id()
        peer = raw_peer(value, 2, 0)
        time.sleep(0.03)
        peer.close()
        time.sleep(0.03)
        self.assert_group(value, 2)

    def joined_group(self, world, value=None):
        value = value or self.bootstrap.unique_id()
        with concurrent.futures.ThreadPoolExecutor(max_workers=world) as pool:
            futures = [pool.submit(self.bootstrap.join, value, world, rank) for rank in range(world)]
            results = [f.result(timeout=5) for f in futures]
        self.assertEqual([r for r, _ in results], [0] * world)
        return [handle for _, handle in results]

    def test_allgather_rounds(self):
        world = 3
        handles = self.joined_group(world)
        try:
            for round_number, sizes in enumerate(([0, 0, 0], [5, 1, 300], [70000, 0, 2])):
                payloads = [bytes([round_number * 16 + rank]) * sizes[rank] for rank in range(world)]
                with concurrent.futures.ThreadPoolExecutor(max_workers=world) as pool:
                    futures = [pool.submit(self.bootstrap.allgather, handles[rank], payloads[rank], world)
                               for rank in range(world)]
                    results = [f.result(timeout=10) for f in futures]
                for result, parts in results:
                    self.assertEqual(result, 0)
                    self.assertEqual(parts, payloads)
        finally:
            for handle in handles:
                self.bootstrap.close(handle)

    def test_allgather_fails_fast_when_a_rank_leaves(self):
        handles = self.joined_group(3)
        self.bootstrap.close(handles[2])
        start = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(self.bootstrap.allgather, handles[rank], b"x", 3) for rank in range(2)]
            results = [f.result(timeout=10) for f in futures]
        self.assertTrue(all(result == 6 for result, _ in results), results)
        self.assertLess(time.monotonic() - start, 2.0)
        for handle in handles[:2]:
            self.bootstrap.close(handle)

    def test_bounded_allgather_times_out_without_every_rank(self):
        # The teardown rounds of ncclCommDestroy: a round bounded by its own timeout returns the timeout
        # result when a rank never sends it; with every rank it completes like an unbounded round.
        handles = self.joined_group(2)
        try:
            out, lengths = ctypes.c_void_p(), (ctypes.c_uint * 8)()
            start = time.monotonic()
            result = self.bootstrap.lib.sccl_bootstrap_allgather_within(handles[0], b"x", 1, ctypes.byref(out),
                                                                        lengths, 300, None)
            self.assertEqual(result, 8)
            self.assertGreaterEqual(time.monotonic() - start, 0.25)
            self.assertLess(time.monotonic() - start, 3.0)
        finally:
            for handle in handles:
                self.bootstrap.close(handle)
        handles = self.joined_group(2)
        try:
            def bounded(rank):
                out, lengths = ctypes.c_void_p(), (ctypes.c_uint * 8)()
                result = self.bootstrap.lib.sccl_bootstrap_allgather_within(handles[rank], b"y", 1,
                                                                            ctypes.byref(out), lengths, 5000, None)
                if out.value:
                    self.bootstrap.libc.free(out)
                return result, list(lengths[:2])
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                results = [f.result(timeout=10) for f in [pool.submit(bounded, rank) for rank in range(2)]]
            self.assertEqual(results, [(0, [1, 1])] * 2)
        finally:
            for handle in handles:
                self.bootstrap.close(handle)

    def test_allgather_refuses_oversized_payload(self):
        handles = self.joined_group(1)
        try:
            result, _ = self.bootstrap.allgather(handles[0], b"x" * ((1 << 20) + 1), 1)
            self.assertEqual(result, 4)
            result, parts = self.bootstrap.allgather(handles[0], b"ok", 1)
            self.assertEqual((result, parts), (0, [b"ok"]))
        finally:
            self.bootstrap.close(handles[0])

    def test_interface_address(self):
        interfaces = [name for _, name in socket.if_nameindex() if name != "lo"]
        address = None
        for name in interfaces:
            try:
                import fcntl
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
                    packed = fcntl.ioctl(probe.fileno(), 0x8915, struct.pack("256s", name.encode()[:15]))
                address = socket.inet_ntoa(packed[20:24])
                break
            except OSError:
                continue
        if address is None:
            self.skipTest("no IPv4 interface besides loopback")
        os.environ["SIRCL_BOOTSTRAP_IFNAME"] = "=" + name
        try:
            value = self.bootstrap.unique_id()
            self.assertEqual(socket.inet_ntoa(value[12:16]), address)
            handles = self.joined_group(2, value)
            for handle in handles:
                self.bootstrap.close(handle)
        finally:
            os.environ.pop("SIRCL_BOOTSTRAP_IFNAME")
        os.environ["SIRCL_BOOTSTRAP_ADDR"] = "not-an-address"
        try:
            buffer = ctypes.create_string_buffer(128)
            self.assertEqual(self.bootstrap.lib.sccl_bootstrap_id(buffer), 4)
        finally:
            os.environ.pop("SIRCL_BOOTSTRAP_ADDR")

    def test_invalid_env_and_arguments(self):
        for invalid in ["0", "-1", "nan", "600001"]:
            os.environ["SIRCL_BOOTSTRAP_TIMEOUT_MS"] = invalid
            buffer = ctypes.create_string_buffer(128)
            self.assertEqual(self.bootstrap.lib.sccl_bootstrap_id(buffer), 4)
            self.assertEqual(buffer.raw, bytes(128))
        os.environ["SIRCL_BOOTSTRAP_TIMEOUT_MS"] = "1200"
        os.environ["SIRCL_SITE_HASH"] = "xyz"
        buffer = ctypes.create_string_buffer(128)
        self.assertEqual(self.bootstrap.lib.sccl_bootstrap_id(buffer), 4)
        os.environ.pop("SIRCL_SITE_HASH")
        value = self.bootstrap.unique_id()
        for world, rank in [(0, 0), (9, 0), (2, -1), (2, 2)]:
            result, handle = self.bootstrap.join(value, world, rank)
            self.assertEqual(result, 4)
            self.assertFalse(handle.value)
        self.assert_group(value, 1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        bootstrap = Bootstrap(sys.argv[2])
        result, handle = bootstrap.join(bytes.fromhex(sys.argv[3]),
                                        int(sys.argv[4]), int(sys.argv[5]))
        print(result, flush=True)
        bootstrap.close(handle)
        sys.exit(0 if result == 0 else 1)
    unittest.main(verbosity=2)
