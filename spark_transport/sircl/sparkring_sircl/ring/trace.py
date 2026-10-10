"""Stage times of traced chain and link ops (``SIRCL_EVENT_TRACE``), torch-free.

A session with an event trace records, per rank, what its progress thread did
with every chunk of a chain op or item of a link op (``protocol.TraceEvent``:
first seen ready, posted, done, consumed, credits) and, for the chain
all-reduce, when its kernel saw an inbound chunk's lane flags and published a
chunk staged or consumed. Kernel times are the GPU's ``%globaltimer`` mapped
to the host's ``CLOCK_REALTIME`` by a ping-pong probe (``offset_error_ns``
bounds the mapping), so stages that cross from the kernel to the progress
thread carry that error; stages within one source do not.

:func:`stages` turns one rank's records into per-chunk stage times of every
outbound stream:

- ``kernel``: the kernel's inbound flags to its staged chunk (the chain
  reduce, or the end rank's turn from partial to result);
- ``notice``: the staged chunk to the progress thread seeing it ready;
- ``credit``: ready to posted (waiting for the downstream rank's credit or
  send-queue room);
- ``wire``: posted to done (every lane's writes completed);
- ``gap``: one chunk's post to the next one's on the same stream.

:func:`summary` gives the median and 90th percentile of each stage per
stream; ``python -m sparkring_sircl.ring trace --results <dir>`` prints them
for every traced case of a run.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

STAGES = ("kernel", "notice", "credit", "wire", "gap")
CHAIN_STREAMS = ("A partials", "A results", "B partials", "B results")
LINK_STREAM = 4


def stream_name(stream: int) -> str:
    """``A partials`` (chain stream 0) to ``B results`` (3), or ``link <k>`` (stream 4 + k)."""
    if 0 <= stream < len(CHAIN_STREAMS):
        return CHAIN_STREAMS[stream]
    return f"link {stream - LINK_STREAM}"


def _partials_stream(stream: int) -> int | None:
    """The chain stream whose inbound chunk the kernel turned into a chunk of ``stream``."""
    if stream in (0, 2):
        return stream
    if stream in (1, 3):
        return stream - 1
    return None


def stages(records: Iterable[Sequence]) -> dict[int, dict[str, list[float]]]:
    """Per outbound stream, the stage times in microseconds of every posted chunk or item.

    ``records`` are ``(ns, source, event, stream, value)`` as
    ``AllReduce.event_trace_records`` returns them (``event`` a ``protocol.TraceEvent`` name).
    """
    first: dict[tuple[str, int, int], int] = {}
    for ns, _source, event, stream, value in records:
        first.setdefault((str(event), int(stream), int(value)), int(ns))
    result: dict[int, dict[str, list[float]]] = {}
    posted = sorted((stream, tag, ns) for (event, stream, tag), ns in first.items() if event == "POSTED")
    previous: dict[int, int] = {}
    for stream, tag, ns in posted:
        times = result.setdefault(stream, {name: [] for name in STAGES})
        ready = first.get(("READY", stream, tag))
        done = first.get(("DONE", stream, tag))
        staged = first.get(("KERNEL_READY", stream, tag))
        source = _partials_stream(stream)
        flag = first.get(("KERNEL_FLAG", source, tag)) if source is not None else None
        if staged is not None and flag is not None:
            times["kernel"].append((staged - flag) / 1e3)
        if staged is not None and ready is not None:
            times["notice"].append((ready - staged) / 1e3)
        if ready is not None:
            times["credit"].append((ns - ready) / 1e3)
        if done is not None:
            times["wire"].append((done - ns) / 1e3)
        if stream in previous:
            times["gap"].append((ns - previous[stream]) / 1e3)
        previous[stream] = ns
    return result


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def summary(records: Iterable[Sequence]) -> dict[str, dict[str, dict[str, float]]]:
    """Per stream name, per stage: chunks, median and 90th percentile in microseconds."""
    result: dict[str, dict[str, dict[str, float]]] = {}
    for stream, times in sorted(stages(records).items()):
        result[stream_name(stream)] = {
            name: {"chunks": len(values), "p50_us": round(_percentile(values, 0.5), 2),
                   "p90_us": round(_percentile(values, 0.9), 2)}
            for name, values in times.items() if values
        }
    return result


def table(results: Mapping) -> str:
    """Stage medians of every traced case of a merged run (``summary.merge``)."""
    lines = []
    for case in results.get("cases", ()):
        traces = case.get("event_traces") or {}
        if not traces:
            continue
        lines.append(f"{case.get('collective')} {case.get('bytes')} B, {case.get('algorithm')}:")
        for rank in sorted(traces, key=int):
            trace = traces[rank]
            error = trace.get("offset_error_ns")
            lines.append(f"  rank {rank} (kernel clock within {error} ns, lost {trace.get('lost')}):")
            for stream, stage_times in summary(trace.get("records", ())).items():
                text = ", ".join(f"{name} {values['p50_us']:.1f}/{values['p90_us']:.1f}"
                                 for name, values in stage_times.items())
                lines.append(f"    {stream}: {text} (us, median/p90)")
    return "\n".join(lines) if lines else "no traced cases (run with --session-env SIRCL_EVENT_TRACE=<records>)"


__all__ = ["CHAIN_STREAMS", "STAGES", "stages", "stream_name", "summary", "table"]
