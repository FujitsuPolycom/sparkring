"""Sessions configured from several threads at once (``tests/test_session_environment.py``), in a process of its own.

``python construction_driver.py`` runs the ring session's real ``_configure`` (``oneshot/runtime.py``) on objects
built without ``__init__``, every rank of a ``ring:4`` session a thread, with CUDA Python and the CuTe DSL mocked
for this process (``teardown_driver._CudaStandIn``; ``_configure`` calls neither). Each rank runs a constructor's
two environment steps: ``_configure`` reads ``SIRCL_POST_ORDER`` and ``SIRCL_PROGRESS_CPU``, then
``_environment`` holds the rank's resolved peer list and its CPU in the process environment for a while, as
while a native context is created and started. It prints one JSON object:

- ``held``: rank 1 configures while rank 2 holds ``SIRCL_POST_ORDER=0,1,3`` and ``SIRCL_PROGRESS_CPU=7``; whether
  rank 1 finished while the values were held, and the order and CPU it took (or its error);
- ``rounds``: every rank of the session at once, 10 times: rank ``r`` configures ``5 r`` ms after the round
  starts and then holds its values for 30 ms, so ranks 1-3 configure while rank 0 holds; the orders and CPUs each
  took, and any error.
"""

from __future__ import annotations

import json
import sys
import threading
import time


def main() -> None:
    from teardown_driver import _CudaStandIn

    sys.meta_path.insert(0, _CudaStandIn())

    from sparkring_sircl import routes
    from sparkring_sircl.oneshot import runtime

    layout = routes.Layout.parse("ring:4")
    derived = routes.derive_routes(layout, 2)

    def configured(rank: int):
        s = runtime.RoceOneshotAllReduce.__new__(runtime.RoceOneshotAllReduce)
        s.rank, s.world_size = rank, layout.world
        s.startup_wait_s, s.serving_wait_s = runtime.DEFAULT_STARTUP_WAIT_S, runtime.DEFAULT_SERVING_WAIT_S
        for name in ("chain_order", "chain_index", "relay_safe_bytes", "ring_problem", "poll_rate_per_s"):
            setattr(s, name, None)
        s.wait_regime = "startup"
        peer_routes = derived.route_map(rank)
        devices = sorted({device for lanes in peer_routes.values() for device in lanes})
        s._configure(2 << 20, 64 << 10, devices, peer_routes, 3, None, None, None, None, None, None, None,
                     "ring:4")
        return s

    out: dict = {}
    inside, release = threading.Event(), threading.Event()

    def hold() -> None:
        with runtime._environment(SIRCL_POST_ORDER="0,1,3", SIRCL_PROGRESS_CPU="7"):
            inside.set()
            release.wait(10)

    held: dict = {}

    def configure_one() -> None:
        try:
            session = configured(1)
            held["peers"], held["cpu"] = list(session.post_order_peers), session._progress_cpu
        except Exception as exc:  # noqa: BLE001
            held["error"] = f"{type(exc).__name__}: {exc}"

    holder = threading.Thread(target=hold)
    holder.start()
    inside.wait(10)
    worker = threading.Thread(target=configure_one)
    worker.start()
    worker.join(0.5)
    held["finished_while_held"] = not worker.is_alive()
    release.set()
    holder.join()
    worker.join(10)
    out["held"] = held

    rounds: list = []
    errors: list = []

    def rank_thread(rank: int, round_: int) -> None:
        try:
            time.sleep(0.005 * rank)
            session = configured(rank)
            rounds.append((round_, rank, list(session.post_order_peers), session._progress_cpu))
            with runtime._environment(SIRCL_POST_ORDER=session._post_order_text, SIRCL_PROGRESS_CPU=str(10 + rank)):
                time.sleep(0.03)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"round {round_} rank {rank}: {type(exc).__name__}: {exc}")

    for round_ in range(10):
        threads = [threading.Thread(target=rank_thread, args=(rank, round_)) for rank in range(layout.world)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)
    out["rounds"] = {"taken": rounds, "errors": errors}
    out["expected"] = {rank: list(configured(rank).post_order_peers) for rank in range(layout.world)}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
