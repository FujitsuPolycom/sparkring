"""Free the page cache and wait for settled memory on a deployment's Sparks before vLLM starts.

On GB10 the CPU and GPU share one memory, so the page cache that checkpoint reads and the previous model's
teardown leave behind counts against the free memory vLLM checks at startup: vLLM refuses to start when the
free memory is below ``--gpu-memory-utilization`` of the total. Before every start of a deployment,
``settle`` therefore

1. writes back dirty pages and drops the page, dentry and inode caches on each of its Sparks
   (``sync; echo 3 > /proc/sys/vm/drop_caches``, which frees caches only and changes no setting);
2. reads ``MemTotal`` and ``MemAvailable`` from ``/proc/meminfo`` every 5 s until two readings differ by
   less than 256 MiB on every Spark, for at most 180 s;
3. refuses the start, naming each Spark's settled available memory and the share vLLM asks for, when any
   Spark has less available than that share.

Before it waits, it also refuses a start whose vLLM share (``--gpu-memory-utilization`` of the total) plus
the pinned host memory of a SparkCache connector (``pinned_bytes``: its page-capture slots and CUDA restore
arenas, which it allocates beside vLLM's share) exceeds a Spark's total less HEADROOM_GIB for the system.

The serving A/B runner (performance/harnesses/serving_ab) applies the same steps before each of its starts.
"""
import json
import time

DROP_CACHES = ["sudo", "-n", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"]
MEMINFO = ["awk", "/^MemTotal:|^MemAvailable:/ {print $2}", "/proc/meminfo"]
SETTLE_SECONDS, SETTLE_STEP, STABLE_KIB = 180, 5, 256 * 1024
# Memory a Spark keeps for DGX OS, Docker and SparkRing's host services beside a model. With caches dropped and
# no model running, GB10 Sparks report 116.4-116.7 GiB MemAvailable of their 121.69 GiB MemTotal (5.0-5.3 GiB
# held); this rounds that up.
HEADROOM_GIB = 6.0


class MemoryShort(RuntimeError):
    """A Spark has less settled available memory than vLLM asks for at startup."""


def requested(command):
    """``(utilization, kv_cache_bytes)`` of a vLLM command's ``--gpu-memory-utilization`` and
    ``--kv-cache-memory-bytes``, each None when the command does not set it."""
    tokens = [str(token) for token in command]
    values = {}
    for index, token in enumerate(tokens):
        flag, equals, value = token.partition("=")
        if flag in ("--gpu-memory-utilization", "--kv-cache-memory-bytes"):
            values[flag] = value if equals else (tokens[index + 1] if index + 1 < len(tokens) else None)
    utilization = values.get("--gpu-memory-utilization")
    kv_bytes = values.get("--kv-cache-memory-bytes")
    return (float(utilization) if utilization else None), (int(kv_bytes) if kv_bytes else None)


def pinned_bytes(command):
    """Pinned host memory, in bytes, that the SparkCache connector of a vLLM command allocates beside vLLM's
    share: with ``spark_cache_async_page_capture`` on, its capture slots
    (``spark_cache_async_page_capture_slot_bytes`` times ``spark_cache_async_page_capture_slot_count``, 2 by
    default); with ``spark_cache_cuda_restore`` on, its restore arenas
    (``spark_cache_cuda_restore_arena_budget_bytes``, else one lane's two placement arenas). 0 for a command
    without ``--kv-transfer-config`` or with another connector."""
    tokens = [str(token) for token in command]
    raw = None
    for index, token in enumerate(tokens):
        flag, equals, value = token.partition("=")
        if flag == "--kv-transfer-config":
            raw = value if equals else (tokens[index + 1] if index + 1 < len(tokens) else None)
    if not raw:
        return 0
    try:
        extra = json.loads(raw).get("kv_connector_extra_config") or {}
    except (ValueError, AttributeError):
        return 0
    total = 0
    if extra.get("spark_cache_async_page_capture"):
        total += int(extra.get("spark_cache_async_page_capture_slot_bytes") or 0) * int(
            extra.get("spark_cache_async_page_capture_slot_count") or 2)
    if extra.get("spark_cache_cuda_restore"):
        budget = extra.get("spark_cache_cuda_restore_arena_budget_bytes")
        total += int(budget) if budget else 2 * int(extra.get("spark_cache_cuda_placement_arena_bytes") or 0)
    return total


def _gib(kib):
    return round(kib / 2 ** 20, 2)


def budget_problem(rows, utilization, pinned):
    """The refusal text when vLLM's share plus ``pinned`` bytes exceed a Spark's total less HEADROOM_GIB, or
    None. ``rows`` are ``(host, total KiB)``."""
    if not utilization:
        return None
    over = []
    for host, total in rows:
        total_gib = total / 2 ** 20
        share, pinned_gib = utilization * total_gib, pinned / 2 ** 30
        if share + pinned_gib > total_gib - HEADROOM_GIB:
            over.append(f"{host} (vLLM's share {utilization:g} of {total_gib:.2f} GiB is {share:.2f} GiB, "
                        f"SparkCache pins {pinned_gib:.2f} GiB, together {share + pinned_gib:.2f} GiB, above "
                        f"{total_gib - HEADROOM_GIB:.2f} GiB: the total less {HEADROOM_GIB:g} GiB for the system)")
    if not over:
        return None
    return ("Not starting: the model's memory does not fit on " + ", ".join(over) + ". Lower the profile's "
            "--gpu-memory-utilization or its SparkCache capture slots (spark_cache_async_page_capture_slot_bytes).")


def settle(hosts, utilization, *, run, pinned_bytes=0, sleep=time.sleep, now=time.monotonic, say=print):
    """Drop caches on each of ``hosts`` (in rank order, each once), wait for settled memory and return one row
    per host: ``host``, ``total_gib``, ``available_gib``, ``asks_gib`` (None without ``utilization``) and
    whether the readings ``settled``. ``run(host, argv)`` runs a command on that Spark and returns its standard
    output; ``pinned_bytes`` is the SparkCache connector's pinned host memory (``pinned_bytes()``). Raises
    MemoryShort, before waiting, when ``utilization`` of a Spark's total plus ``pinned_bytes`` exceeds that
    total less HEADROOM_GIB (``budget_problem``), and after settling when a Spark has less available memory
    than ``utilization`` of its total."""
    hosts = list(dict.fromkeys(hosts))
    for host in hosts:
        run(host, DROP_CACHES)

    def reading():
        values = []
        for host in hosts:
            total, available = run(host, MEMINFO).split()
            values.append((int(total), int(available)))
        return values

    previous, deadline = reading(), now() + SETTLE_SECONDS
    problem = budget_problem([(host, total) for host, (total, _) in zip(hosts, previous)], utilization, pinned_bytes)
    if problem:
        raise MemoryShort(problem)
    while True:
        sleep(SETTLE_STEP)
        current = reading()
        settled = all(abs(a[1] - b[1]) < STABLE_KIB for a, b in zip(previous, current))
        previous = current
        if settled or now() > deadline:
            break
    rows = [{"host": host, "total_gib": _gib(total), "available_gib": _gib(available),
             "asks_gib": _gib(utilization * total) if utilization else None, "settled": settled}
            for host, (total, available) in zip(hosts, current)]
    say("Memory before the start (" + ("settled" if settled else f"not settled after {SETTLE_SECONDS} s")
        + (f"; vLLM asks for {utilization:g} of the total" if utilization else "") + "): "
        + ", ".join(f"{row['host']} {row['available_gib']:.2f} GiB available" for row in rows) + ".")
    short = [row for row in rows if row["asks_gib"] is not None and row["available_gib"] < row["asks_gib"]]
    if short:
        raise MemoryShort(
            "Not starting: available memory after dropping caches is below what vLLM asks for at startup "
            "(--gpu-memory-utilization of the total) on " + ", ".join(
                f"{row['host']} ({row['available_gib']:.2f} of {row['total_gib']:.2f} GiB available, vLLM asks "
                f"{row['asks_gib']:.2f} GiB)" for row in short)
            + ". A process outside this deployment holds the memory; stop it and install again.")
    return rows
