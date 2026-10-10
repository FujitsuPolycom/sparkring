"""Run llm-inference-bench's llm_decode_bench.py with its self-update check turned off and every cell started
from an idle server.

python -m performance.harnesses.serving_ab.bench_run BENCH_DIR ARGS...

The benchmark's main() asks GitHub for a newer version and offers to pull it; a campaign keeps one
benchmark revision, so this wrapper loads the script as a module, replaces check_for_update with a function
that returns False, and calls main() with ARGS.

It also makes the benchmark's two request steps start only once the server reports no running and no
waiting request (the benchmark's own ``wait_server_idle``, read from vLLM's Prometheus metrics):

- each decode cell (``run_one_cell``), the hidden pre-decode warm-up cell included;
- the cell's prefix-cache scout (``wait_prefill_task_with_live``): the cell's streams start once the scout's
  own request has left the server.

The benchmark's readiness gate opens a cell's measured window only while the server queues no request, and
it counts every request the server holds. Where the profile's ``--max-num-seqs`` equals the cell's
concurrency, one request outside the cell waits behind the cell's streams; the gate then opens only when a
first-wave stream completes, and the window covers that wave's last tokens and the next wave's prefills
instead of steady decode. Starting from an idle server keeps earlier requests out of a cell; a request that
another client sends during the cell still counts, so measurements need an API that no other client uses.

A server without the metrics returns at once, as the benchmark's ``wait_server_idle`` does; a server that
does not become idle within IDLE_TIMEOUT_SECONDS lets the step start anyway.
"""
import functools
import importlib.util
import sys
from pathlib import Path

IDLE_TIMEOUT_SECONDS = 900.0
IDLE_STABLE_SECONDS = 1.0


def patch(module):
    """Turn off ``module``'s self-update check and start its cells and scouts from an idle server."""
    module.check_for_update = lambda console: False
    run_one_cell, wait_prefill = module.run_one_cell, module.wait_prefill_task_with_live

    async def idle(client, base_url, engine, state=None, live=None, status=""):
        return await module.wait_server_idle(client, base_url, engine, stable_seconds=IDLE_STABLE_SECONDS,
                                             timeout_seconds=IDLE_TIMEOUT_SECONDS, state=state, live=live,
                                             status=status)

    @functools.wraps(run_one_cell)
    async def cell_from_idle(*args, **kwargs):
        options = dict(zip(("client", "base_url"), args), **kwargs)
        await idle(options["client"], options["base_url"], options.get("engine", module.ENGINE_SGLANG),
                   state=options.get("state"), live=options.get("live"),
                   status="waiting for server idle before the cell")
        return await run_one_cell(*args, **kwargs)

    @functools.wraps(wait_prefill)
    async def scout_then_idle(request_task, client, base_url, engine, state, live, status):
        result = await wait_prefill(request_task, client, base_url, engine, state, live, status)
        await idle(client, base_url, engine, state=state, live=live, status="waiting for the scout to leave the server")
        return result

    module.run_one_cell = cell_from_idle
    module.wait_prefill_task_with_live = scout_then_idle
    return module


def main() -> None:
    bench = Path(sys.argv[1]) / "llm_decode_bench.py"
    spec = importlib.util.spec_from_file_location("llm_decode_bench_campaign", bench)
    module = importlib.util.module_from_spec(spec)
    sys.argv = [str(bench)] + sys.argv[2:]
    spec.loader.exec_module(module)
    patch(module).main()


if __name__ == "__main__":
    main()
