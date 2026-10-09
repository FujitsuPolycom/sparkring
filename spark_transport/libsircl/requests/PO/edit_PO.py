"""Edit SIRCL's package for an externally driven progress loop: one thread may serve every session of a
process (request PO from libsircl).

python edit_PO.py <package root: spark_transport/sircl> [<impl docs root>]

Changes, each an exact replacement that must find its old text once:

- oneshot/_roce_proxy.c: the progress loop's body becomes `progress_pass` (one pass: 1 when it posted,
  forwarded or credited something, 0 when idle, -1 after a failure, with the session's `failed` set); the
  thread `progress_main` loops over it exactly as before. Two exported functions:
  `roce_start_external(c)` marks a connected session running without a thread of its own, with the
  same start-up of the posted sequences as `roce_start`, and `roce_progress(c)` runs one pass of such a
  session from the caller's thread. `roce_stop` joins a thread only when the session has one. Native
  ABI 10.
- oneshot/_proxy.py: ABI_VERSION 10; the two functions declared; `Proxy.start_external()` and
  `Proxy.progress()`.
- README.md: the binding's ABI and function list.
- tests/test_progress_external.py: a group of four on the verbs stand-in, every session driven by one
  shared Python thread through `progress()`, one-shot and two-shot ops checked byte for byte; a session
  that is not started externally refuses `progress()`.

The wire protocol, the record, the arena layout and the command ring do not change; sessions started
with `roce_start` behave as before.
"""

import sys
from pathlib import Path


def replace_once(path: Path, pairs) -> None:
    raw = path.read_bytes()
    crlf = b"\r\n" in raw and raw.count(b"\r\n") == raw.count(b"\n")
    text = raw.decode("utf-8").replace("\r\n", "\n")
    for old, new in pairs:
        count = text.count(old)
        if count != 1:
            raise SystemExit(f"{path}: found {count} of {old[:90]!r}")
        text = text.replace(old, new)
    path.write_bytes((text.replace("\n", "\r\n") if crlf else text).encode("utf-8"))


PROXY_C = [
    ("#define ROCE_ABI_VERSION 9\n", "#define ROCE_ABI_VERSION 10\n"),
    ("    int connected;\n    int started;\n",
     "    int connected;\n    int started;\n"
     "    int external;              /* roce_start_external: passes come from roce_progress, no thread */\n"
     "    uint64_t external_idle;    /* consecutive idle passes of an external session */\n"),
    ('''static void *progress_main(void *arg) {
    roce_ctx_t *c = (roce_ctx_t *)arg;
    volatile uint32_t *ctrl = ctrl_words(c);
    uint64_t idle = 0;
    const struct timespec nap = {0, ROCE_NAP_NS};
    while (atomic_load_explicit(&c->running, memory_order_relaxed)) {
        int work = 0;
        if (c->chain_slots) {''', '''/* One pass of the progress loop: 1 when it posted, forwarded or credited something, 0 when it found
 * nothing to do, -1 after a failure (the session's `failed` is set). `idle` counts consecutive idle
 * passes; every 64th drains the completion queues. */
static int progress_pass(roce_ctx_t *c, uint64_t *idle) {
    volatile uint32_t *ctrl = ctrl_words(c);
    {
        int work = 0;
        if (c->chain_slots) {'''),
    ('''            work = 1;
        }
        if (work) {
            idle = 0;
            continue;
        }
        idle++;
        if (idle % 64 == 0 && drain_all(c) != 0) goto failed;
        if (idle >= ROCE_IDLE_SPINS) nanosleep(&nap, NULL);
    }
    return NULL;
failed:
    atomic_store(&c->failed, 1);
    return NULL;
}''', '''            work = 1;
        }
        if (work) {
            *idle = 0;
            return 1;
        }
        ++*idle;
        if (*idle % 64 == 0 && drain_all(c) != 0) goto failed;
        return 0;
    }
failed:
    atomic_store(&c->failed, 1);
    return -1;
}

static void *progress_main(void *arg) {
    roce_ctx_t *c = (roce_ctx_t *)arg;
    uint64_t idle = 0;
    const struct timespec nap = {0, ROCE_NAP_NS};
    while (atomic_load_explicit(&c->running, memory_order_relaxed)) {
        int pass = progress_pass(c, &idle);
        if (pass < 0) break;
        if (pass == 0 && idle >= ROCE_IDLE_SPINS) nanosleep(&nap, NULL);
    }
    return NULL;
}'''),
    ('''    if (!c->started) {
        /* A restart keeps the newest posted sequences, so ops that rang while
         * the thread was stopped are posted when it resumes. */
        volatile uint32_t *ctrl = ctrl_words(c);
        c->last_seq = ctrl[ROCE_CTRL_DOORBELL];
        c->posting_seq = c->last_seq;
        for (int k = 1; k < ROCE_MAX_PHASES; k++) c->last_phase_seq[k] = ctrl[ROCE_CTRL_PHASE + k];
        atomic_store(&c->posted_seq, c->last_seq);
        c->started = 1;
    }
    atomic_store(&c->failed, 0);
    atomic_store(&c->running, 1);
    int rc = pthread_create(&c->thread, use, progress_main, c);''', '''    begin_posting(c);
    c->external = 0;
    atomic_store(&c->failed, 0);
    atomic_store(&c->running, 1);
    int rc = pthread_create(&c->thread, use, progress_main, c);'''),
    ('''int roce_start(roce_ctx_t *c) {
    if (!c->connected) {''', '''/* Before the first start: post from the newest sequences in the command ring. A restart keeps the
 * newest posted sequences, so ops that rang while the session was stopped are posted when it resumes. */
static void begin_posting(roce_ctx_t *c) {
    if (c->started) return;
    volatile uint32_t *ctrl = ctrl_words(c);
    c->last_seq = ctrl[ROCE_CTRL_DOORBELL];
    c->posting_seq = c->last_seq;
    for (int k = 1; k < ROCE_MAX_PHASES; k++) c->last_phase_seq[k] = ctrl[ROCE_CTRL_PHASE + k];
    atomic_store(&c->posted_seq, c->last_seq);
    c->started = 1;
}

/* Mark a connected session running without a progress thread of its own: the caller's thread drives
 * it with roce_progress, one pass per call, so one thread can serve every session of a process. One
 * thread drives a session at a time; stop the driving thread's calls before roce_stop. */
int roce_start_external(roce_ctx_t *c) {
    if (!c->connected) {
        FAIL(c, "the progress loop needs a connected session");
        return -1;
    }
    if (atomic_load(&c->running)) return c->external ? 0 : -1;
    link_set_windowed(c);
    begin_posting(c);
    c->external = 1;
    c->external_idle = 0;
    atomic_store(&c->failed, 0);
    atomic_store(&c->running, 1);
    return 0;
}

/* One pass of an externally driven session (roce_start_external): 1 when it did work, 0 when idle,
 * -1 when the session failed (roce_error says why) or is not running externally. */
int roce_progress(roce_ctx_t *c) {
    if (!c->external || !atomic_load_explicit(&c->running, memory_order_relaxed)) return -1;
    return progress_pass(c, &c->external_idle);
}

int roce_start(roce_ctx_t *c) {
    if (!c->connected) {'''),
    ('''void roce_stop(roce_ctx_t *c) {
    if (c != NULL && atomic_exchange(&c->running, 0)) pthread_join(c->thread, NULL);
}''', '''void roce_stop(roce_ctx_t *c) {
    if (c != NULL && atomic_exchange(&c->running, 0) && !c->external) pthread_join(c->thread, NULL);
}'''),
]

PROXY_PY = [
    ("ABI_VERSION = 9\n", "ABI_VERSION = 10\n"),
    ('''        "roce_start": (i32, [p]),
''', '''        "roce_start": (i32, [p]),
        "roce_start_external": (i32, [p]),
        "roce_progress": (i32, [p]),
'''),
    ('''    def stop(self) -> None:
        if self._ctx:
            self._lib.roce_stop(self._ctx)''', '''    def start_external(self) -> None:
        """Mark the session running without a progress thread of its own; :meth:`progress` drives it,
        so one thread can serve every session of a process."""
        if self._lib.roce_start_external(self._handle()) != 0:
            raise RuntimeError(f"SIRCL session failed to start for external progress: {self.error()}")

    def progress(self) -> int:
        """One pass of the progress loop of a session started with :meth:`start_external`: 1 when it did
        work, 0 when idle; raises when the session failed or is not driven externally."""
        result = int(self._lib.roce_progress(self._handle()))
        if result < 0:
            raise RuntimeError(f"SIRCL native progress failed: {self.error() or 'not started externally'}")
        return result

    def stop(self) -> None:
        if self._ctx:
            self._lib.roce_stop(self._ctx)'''),
]

README = [
    ("`ABI_VERSION` (9), `load(path=None)`", "`ABI_VERSION` (10), `load(path=None)`"),
    ("`set_links(prev, next, index, slots, slot_bytes, link_offset, *, ring_prev=-1,\nring_next=-1, ring_window=0)`, "
     "`set_trace(capacity)`, `take_trace(max_records)`,\n`start`, `stop`,",
     "`set_links(prev, next, index, slots, slot_bytes, link_offset, *, ring_prev=-1,\nring_next=-1, ring_window=0)`, "
     "`set_trace(capacity)`, `take_trace(max_records)`,\n`start`, `start_external` and `progress` (one thread "
     "may drive several sessions: `progress` runs one pass of a session started with `start_external`), `stop`,"),
    ("`roce_set_trace`, `roce_trace_take`, `roce_start`, `roce_stop`,",
     "`roce_set_trace`, `roce_trace_take`, `roce_start`, `roce_start_external`, `roce_progress`, `roce_stop`,"),
]

TEST = '''"""Sessions driven by one shared progress loop instead of a thread each (``Proxy.start_external``).

A group of four with two lanes runs on the in-memory verbs stand-in; no session starts a progress thread
of its own: one Python thread calls ``progress()`` on every session in turn, as a process-wide progress
thread would. One-shot and two-shot ops move exact bytes. A session that was started with its own thread
refuses ``progress()``.
"""

import threading

import pytest

from sparkring_sircl import routes
from sparkring_sircl.testing import fabric


def _connect_external(session, lane_check_ms=2000):
    session.fabric.start()
    blobs = [proxy.local_blob() for proxy in session.proxies]
    for proxy in session.proxies:
        proxy.connect(blobs)
    for proxy in session.proxies:
        proxy.lane_check(lane_check_ms)
    for proxy in session.proxies:
        proxy.start_external()


def test_one_thread_drives_every_session(simulator_library):
    session = fabric.LocalSession(str(simulator_library), routes.Layout.parse("path:0-3"), lanes=2)
    stop = threading.Event()
    errors = []
    passes = [0, 0]

    def drive():
        try:
            while not stop.is_set():
                for proxy in session.proxies:
                    passes[proxy.progress()] += 1
        except Exception as error:  # noqa: BLE001 - reported below
            errors.append(error)

    try:
        _connect_external(session)
        thread = threading.Thread(target=drive, daemon=True)
        thread.start()
        fabric.run_oneshot_ops(session, [16, 4096, 16384, 48, 16384])
        fabric.run_twoshot_ops(session, [64, 4096, 16384], first_seq=6)
        stop.set()
        thread.join(10)
        assert not errors, errors
        assert passes[1] > 0
        assert all(proxy.stats()["ops_posted"] == 8 for proxy in session.proxies)
    finally:
        stop.set()
        session.close()


def test_progress_refused_without_external_start(simulator_library):
    session = fabric.LocalSession(str(simulator_library), routes.Layout.parse("path:0-1"), lanes=1)
    try:
        session.connect()
        with pytest.raises(RuntimeError):
            session.proxies[0].progress()
    finally:
        session.close()
'''


def main() -> int:
    root = Path(sys.argv[1])
    replace_once(root / "sparkring_sircl" / "oneshot" / "_roce_proxy.c", PROXY_C)
    replace_once(root / "sparkring_sircl" / "oneshot" / "_proxy.py", PROXY_PY)
    replace_once(root / "README.md", README)
    (root / "tests" / "test_progress_external.py").write_text(TEST, encoding="utf-8")
    print("edited", root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
