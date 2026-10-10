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

The serving A/B runner (performance/harnesses/serving_ab) applies the same steps before each of its starts.
"""
import time

DROP_CACHES = ["sudo", "-n", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"]
MEMINFO = ["awk", "/^MemTotal:|^MemAvailable:/ {print $2}", "/proc/meminfo"]
SETTLE_SECONDS, SETTLE_STEP, STABLE_KIB = 180, 5, 256 * 1024


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


def _gib(kib):
    return round(kib / 2 ** 20, 2)


def settle(hosts, utilization, *, run, sleep=time.sleep, now=time.monotonic, say=print):
    """Drop caches on each of ``hosts`` (in rank order, each once), wait for settled memory and return one row
    per host: ``host``, ``total_gib``, ``available_gib``, ``asks_gib`` (None without ``utilization``) and
    whether the readings ``settled``. ``run(host, argv)`` runs a command on that Spark and returns its standard
    output. Raises MemoryShort when a Spark has less available memory than ``utilization`` of its total."""
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
