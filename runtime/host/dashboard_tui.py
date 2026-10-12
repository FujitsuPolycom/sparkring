"""``sparkring dashboard tui``: the dashboard as a terminal view, on Node A, over SSH if need be.

Status: Experimental. Covered by offline tests only.

It shows what the web page shows, from the same collected state: it reads the dashboard service's
``/api/status`` on this Spark and nothing else, so it sends nothing to any Spark's model API and
opens no SSH session. Without ``--url`` it reads ``http://127.0.0.1:8090`` (the service listens on
loopback beside Node A's LAN address) and, when nothing answers there, the service's own LAN
address from the fabric document (``default_urls``). A URL whose host is not this Spark is refused before any
socket opens (``require_local``), so the view cannot become a second collector. Like the web page,
an open view counts as a viewer and the service reads each model's ``/metrics`` every 5 s instead
of every 60 s; ``--passive`` leaves the service's reads as they are.

The screen, top to bottom:

- the header: models serving, Sparks busy, requests running, total output tok/s, the age of the
  last read and the read period;
- one line per ring in cable order, each model's arc in its own colour;
- one card per model: state, TP and arc, transport, uptime, tok/s, running requests, KV use and
  capacity, accept length and TTFT. ``d`` or space expands the selected card into its detail rows
  (model, endpoint, layout, arc, transport, head container, throughput with prompt rate, tok/s and
  KV sparklines, requests, KV cache with concurrency at the maximum context, latency with prefix
  hits, totals, settings, parsers, runtime image, the log command) and one row per Spark: rank, LAN
  address, RAM and disk free, disk free red below DISK_RED_FRACTION of the disk or DISK_RED_BYTES.

Logs stay on demand: the card shows the ``docker logs`` command to run on the head Spark.

Keys: up/down or j/k select · d/space details · e events · r read now · -/= slower/faster · q quit.
``--once`` prints one frame as plain text (``--details`` expands every card).
"""
from __future__ import annotations

import argparse
import base64
import ipaddress
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from runtime.host import dashboard

STATE_COLORS = {"serving": 1, "not_ready": 2, "offline": 3, "stopped": 4, "idle": 4}
STATE_SHORT = {"serving": "serving", "not_ready": "start", "stopped": "stopped", "offline": "OFFLINE", "idle": "idle"}
# A segment's style: the low 4 bits are a colour pair, BOLD and DIM are flags above them.
BOLD, DIM = 1 << 8, 1 << 9
GOOD, WARN, CRIT, INFO, ACCENT, BLUE = 1, 2, 3, 4, 5, 6
# Models on the ring are told apart by colour; the collector numbers them (arc.color) in the web
# page's palette order (sky, violet, teal, pink, ...).
RING_PALETTE = (INFO, ACCENT, GOOD, WARN, BLUE)
SPARK_BLOCKS = "▁▂▃▄▅▆▇█"
DISK_RED_FRACTION = 0.02          # disk free below this share of the root filesystem renders red
DISK_WARN_FRACTION = 0.05         # ... and below this share yellow
READ_TIMEOUT = 8.0
LOOPBACK_URL = f"http://127.0.0.1:{dashboard.DEFAULT_PORT}"


def style(pair: int = 0, bold: bool = False, dim: bool = False) -> int:
    return pair | (BOLD if bold else 0) | (DIM if dim else 0)


def hi_color(value: float, warn: float, crit: float) -> int:
    """Colour of a higher-is-worse value: green below `warn`, yellow, then red from `crit`."""
    return CRIT if value >= crit else WARN if value >= warn else GOOD


def dotted(text: str, color: int = 0) -> list:
    """'a · b · c' as segments whose separators render dim."""
    out = []
    for index, part in enumerate(text.split(" · ")):
        if index:
            out.append((" · ", style(dim=True)))
        out.append((part, color))
    return out


def dur_short(seconds) -> str:
    if seconds is None or seconds < 0:
        return ""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, hours = seconds // 60, seconds // 3600
    if hours < 1:
        return f"{minutes}m"
    days = hours // 24
    return f"{days}d{hours % 24}h" if days else f"{hours}h{minutes % 60:02d}m"


def tokfmt(tokens) -> str:
    if tokens is None:
        return "–"
    if tokens >= 1e6:
        return f"{tokens / 1e6:.1f}M"
    if tokens >= 1e3:
        return f"{round(tokens / 1e3)}k"
    return str(round(tokens))


def bytesfmt(value) -> str:
    if value is None:
        return "–"
    for unit, size in (("T", 1 << 40), ("G", 1 << 30), ("M", 1 << 20)):
        if value >= size:
            return f"{value / size:.0f}{unit}" if value >= 100 * size else f"{value / size:.1f}{unit}"
    return f"{value:.0f}B"


def sparkline(values: list, width: int) -> str:
    """`values` in `width` columns (the maximum of each bucket) as block characters."""
    values = [v or 0 for v in values]
    if not values or width <= 0:
        return ""
    if len(values) > width:
        step = len(values) / width
        values = [max(values[int(i * step):max(int(i * step) + 1, int((i + 1) * step))]) for i in range(width)]
    peak = max(values)
    if peak <= 0:
        return SPARK_BLOCKS[0] * len(values)
    return "".join(SPARK_BLOCKS[min(7, int(v / peak * 7.999))] for v in values)


def span_text(positions) -> str:
    """'4–5' for an arc, '7–2' when it wraps past the last Spark, '3' for one position."""
    if not positions:
        return ""
    return str(positions[0]) if len(positions) == 1 else f"{positions[0]}–{positions[-1]}"


def group_color(member: dict) -> int:
    arc = member.get("arc")
    return RING_PALETTE[(arc.get("color") or 0) % len(RING_PALETTE)] if arc else 0


def rings_of(snapshot: dict) -> list:
    rings = snapshot.get("rings")
    if rings is None:
        rings = [snapshot["ring"]] if snapshot.get("ring") else []
    return rings


def members_of(snapshot: dict) -> list:
    return [m for group in snapshot.get("groups") or [] for m in group.get("members") or []]


def disk_style(free, total) -> int:
    """Red below DISK_RED_FRACTION of the disk, yellow below DISK_WARN_FRACTION, plain otherwise."""
    if free is None:
        return style(dim=True)
    fraction = free / total if total else 1.0
    if fraction < DISK_RED_FRACTION:
        return style(CRIT, bold=True)
    if fraction < DISK_WARN_FRACTION:
        return style(WARN)
    return 0


# -- lines -------------------------------------------------------------------------------------
def header_segments(snapshot: dict, *, age, interval: float, error=None) -> list:
    """Models serving, Sparks busy, requests running, total tok/s, the read's age and period."""
    members = members_of(snapshot)
    rings = rings_of(snapshot)
    serving = [m for m in members if m.get("state") == "serving"]
    running = sum(((m.get("live") or {}).get("running") or 0) for m in serving)
    tps = sum(((m.get("live") or {}).get("output_tps") or 0) for m in serving)
    positions = [p for ring in rings for p in ring.get("positions") or []]
    size = sum(ring.get("size", len(ring.get("positions") or [])) for ring in rings)
    busy = sum(1 for p in positions if p.get("member") or p.get("run"))
    down = sum(1 for p in positions if p.get("up") is False)
    sep = (" · ", style(dim=True))
    segs = [(" " + (snapshot.get("title") or "SparkRing"), style(bold=True)), sep]
    if len(rings) > 1:
        segs += [(f"{len(rings)} rings", style(dim=True)), sep]
    segs += [(f"{len(serving)} serving", style(GOOD, bold=True) if serving else style(WARN, bold=True)), sep,
             (f"{busy}/{size} Sparks busy", style(INFO, bold=True) if busy else style(dim=True))]
    if down:
        segs += [sep, (f"{down} down", style(CRIT, bold=True))]
    segs += [sep, (f"{round(running)} reqs", style(WARN, bold=True) if running else style(dim=True)),
             sep, (f"{tps:.1f} tok/s", style(GOOD, bold=True) if tps >= 0.05 else style(dim=True))]
    # `age` is the time since this view's last read of the service, `period` how often the service
    # reads each model's /metrics (5 s while viewed, 60 s otherwise).
    collector = snapshot.get("collector") or {}
    period = collector.get("metrics_seconds") or interval
    stale = age is not None and age > max(15.0, 2 * interval)
    segs += [sep, (f"live {dur_short(age) if age is not None else '?'} ago", style(WARN) if stale else style(dim=True)),
             (f" · poll {period:.0f}s", style(dim=True))]
    if collector.get("quiet"):
        segs += [sep, ("quiet: no request is sent", style(WARN, bold=True))]
    if error:
        segs.append((f" · STALE: {str(error)[:40]}", style(CRIT, bold=True)))
    return segs


def ring_owners(ring: dict, members: list) -> dict:
    owners: dict = {}
    for member in members:
        if member.get("ring", ring.get("id")) != ring.get("id"):
            continue
        for position in (member.get("arc") or {}).get("positions") or []:
            owners.setdefault(position, (member["id"], group_color(member), member.get("state")))
    return owners


def ring_segments(ring: dict, members: list, width: int = 1000) -> list:
    """The ring in cable order: '●0 spark0 ━ ●1 spark1 ─ ○2 spark2 … ─ ↺'.

    ● serving, ◐ starting or stopped, ○ no model, ✗ the head's API does not answer; '━' joins
    neighbours of one model, and on a closed ring the last cable returns to position 0. Host names
    are left out when the line would not fit in `width` columns."""
    positions = ring.get("positions") or []
    size = len(positions)
    closed = ring.get("closed", True)
    owners = ring_owners(ring, members)
    full = sum(len(f"●{i} {p.get('short') or p.get('name') or ''} ─ ") for i, p in enumerate(positions)) + 1
    names = full <= width
    segs = []
    for index, spark in enumerate(positions):
        owner = owners.get(index)
        if spark.get("up") is False:
            glyph, st = "✗", style(CRIT, bold=True)
        elif owner:
            glyph, st = ("●" if owner[2] == "serving" else "◐"), style(owner[1], bold=True)
        else:
            glyph, st = "○", style(dim=True)
        segs.append((f"{glyph}{index}" + (f" {spark.get('short') or spark.get('name')}" if names else ""), st))
        if index == size - 1 and not closed:
            break
        following = owners.get((index + 1) % size)
        joined = bool(owner and following and following[0] == owner[0] and size > 1)
        cable = (" ━ " if joined else " ─ ") + ("↺" if index == size - 1 else "")
        segs.append((cable, style(owner[1], bold=True) if joined else style(dim=True)))
    return segs


def ring_label(ring: dict, rings: list) -> str:
    return f"{ring.get('short') or ring.get('id')} ring " if len(rings) > 1 else ""


def model_line(member: dict, now: float, ring_tag: bool = False) -> tuple[str, list]:
    """(name, segments) of one model's card line: state, TP and arc, transport, uptime, tok/s,
    running requests, KV use and capacity, accept length and TTFT."""
    state = member.get("state") or "checking"
    bits = []
    b = lambda text, st=0: bits.append((text, st))  # noqa: E731
    b(STATE_SHORT.get(state, state), style(STATE_COLORS.get(state, 4), bold=True))
    arc = member.get("arc") or {}
    if arc:
        b(f"TP{arc.get('size')}{'?' if arc.get('inferred') else ''}", BLUE)
        where = f"{member.get('ring')}:" if ring_tag and member.get("ring") else ""
        b(f"@{where}{span_text(arc.get('positions'))}", style(group_color(member), bold=True))
        transport = (member.get("transport") or {}).get("label")
        if transport and state == "serving":
            b(transport, style(dim=True))
    if state == "serving":
        if member.get("state_since") and dur_short(now - member["state_since"]):
            b("up " + dur_short(now - member["state_since"]), style(dim=True))
        live = member.get("live") or {}
        capacity = member.get("capacity") or {}
        tps = live.get("output_tps") or 0
        b(f"{tps:.1f} t/s", style(GOOD, bold=True) if tps >= 0.05 else style(GOOD, dim=True))
        run = round(live.get("running") or 0)
        b(f"run {run}", style(WARN, bold=True) if run else style(dim=True))
        if live.get("waiting"):
            b(f"wait {round(live['waiting'])}", style(WARN, bold=True))
        if live.get("kv") is not None:
            pct = live["kv"] * 100
            text = f"kv {round(pct)}%"
            if capacity.get("kv_tokens"):
                text += f" ({tokfmt(capacity['kv_tokens'])} tok)"
            b(text, hi_color(pct, 50, 80))
        if live.get("accept_length_5m") is not None:
            b(f"acc {live['accept_length_5m']:.2f}", 0)
        if live.get("ttft_5m") is not None:
            ttft = live["ttft_5m"]
            b(f"ttft {ttft:.2f}s", hi_color(ttft, 1, 3) if ttft > 1 else 0)
    elif state in ("stopped", "offline", "not_ready") and member.get("model"):
        b("last " + str(member["model"]), style(dim=True))
    elif state == "idle":
        b("another deployment holds the port", style(dim=True))
    return member.get("name") or member.get("id"), bits


def model_details(member: dict, now: float, width: int = 100) -> list:
    """The detail rows of one model's card; each row is a list of (text, style) segments."""
    rows = []
    add = lambda *segs: rows.append(list(segs))  # noqa: E731
    lbl = lambda label: (f"{label:<10} ", style(INFO, bold=True))  # noqa: E731
    state = member.get("state") or "checking"
    layout = member.get("layout") or {}
    live = member.get("live") or {}
    capacity = member.get("capacity") or {}
    settings = member.get("settings") or {}
    totals = member.get("totals") or {}
    since = member.get("state_since")
    head = member.get("state_label") or state
    if since:
        lasted = dur_short(now - since)      # empty when the start lies ahead of this host's clock
        head += (f" for {lasted}" if lasted else "") + f" (since {time.strftime('%m-%d %H:%M', time.localtime(since))})"
    add((head, style(STATE_COLORS.get(state, 4), bold=True)))
    if member.get("model"):
        add(lbl("model"), (member["model"], style(bold=True)),
            *([("  (last seen)", style(dim=True))] if state != "serving" else []))
    if member.get("endpoint"):
        add(lbl("endpoint"), (member["endpoint"], INFO))
    sparks = member.get("sparks") or []
    segs = [lbl("layout")]
    segs += dotted(" · ".join(b for b in (layout.get("label"), member.get("profile"),
                                          ", ".join(s.get("name", "?") for s in sparks)) if b))
    workers = member.get("workers") or {}
    if workers.get("expected"):
        got, want = workers.get("received", "?"), workers["expected"]
        segs += [(" · workers ", style(dim=True)), (f"{got}/{want}", style(GOOD if got == want else WARN, bold=True))]
    add(*segs)
    arc = member.get("arc") or {}
    if arc:
        names = {s.get("position"): s.get("name", "?") for s in sparks}
        positions = arc.get("positions") or []
        segs = [lbl("arc"), (f"positions {span_text(positions)}", style(group_color(member), bold=True))]
        if len(positions) > 1:
            segs.append((f" · {names.get(positions[0], '?')} → {names.get(positions[-1], '?')}", 0))
        segs += [(" · from ", style(dim=True)), (arc.get("source_label") or "?", 0)]
        add(*segs)
    transport = member.get("transport") or {}
    label = transport.get("label")
    if label and transport.get("sircl_version"):
        label += f" {transport['sircl_version']}"
    # The transport that carries the collectives, by its display name; the image's b12x bundle
    # names what the image carries whichever transport runs, so the view does not show it.
    bits = [label, "NCCL off" if transport.get("nccl") is False and not transport.get("collective") else None]
    if any(bits):
        add(lbl("transport"), *dotted(" · ".join(b for b in bits if b)))
    head_row = member.get("head") or {}
    if head_row:
        segs = [lbl("head"), (f"{head_row.get('name')}:{head_row.get('port')}", INFO)]
        if head_row.get("container"):
            segs += [(" · container ", style(dim=True)), (head_row["container"], ACCENT)]
        add(*segs)
    if state == "serving":
        out_tps, prompt_tps = live.get("output_tps") or 0, live.get("prompt_tps") or 0
        add(lbl("throughput"), ("out ", style(dim=True)),
            (f"{out_tps:.1f} t/s", style(GOOD, bold=True) if out_tps >= 0.05 else style(GOOD, dim=True)),
            (f" (5m {live.get('output_tps_5m') or 0:.1f})", style(dim=True)), (" · prompt ", style(dim=True)),
            (f"{prompt_tps:.1f} t/s", style(GOOD) if prompt_tps >= 0.05 else style(dim=True)))
        history = member.get("history") or []
        if len(history) > 1:
            span = dur_short(history[-1][0] - history[0][0])
            peak = max((h[1] or 0) for h in history)
            kv_peak = max((h[4] or 0) for h in history)
            graph = max(10, min(60, width - 32))
            add(("tok/s      ", style(INFO, bold=True)), (f"{span:>4} ", style(dim=True)),
                (sparkline([h[1] for h in history], graph), GOOD), (f" peak {peak:.0f}", style(GOOD, bold=True)))
            add(("kv         ", style(INFO, bold=True)), (f"{span:>4} ", style(dim=True)),
                (sparkline([h[4] for h in history], graph), WARN),
                (f" peak {kv_peak * 100:.0f}%", style(hi_color(kv_peak * 100, 50, 75), bold=True)))
        run, wait = round(live.get("running") or 0), round(live.get("waiting") or 0)
        segs = [lbl("requests"), ("run ", style(dim=True)), (f"{run}", style(WARN, bold=True) if run else style(dim=True)),
                (" · wait ", style(dim=True)), (f"{wait}", style(WARN, bold=True) if wait else style(dim=True))]
        if capacity.get("max_num_seqs") or settings.get("max_num_seqs"):
            segs += [(" · max seqs ", style(dim=True)), (f"{capacity.get('max_num_seqs') or settings.get('max_num_seqs')}", 0)]
        add(*segs)
        kv = live.get("kv")
        segs = [lbl("kv cache"), (f"{round(kv * 100)}%", style(hi_color(kv * 100, 50, 80), bold=True)) if kv is not None
                else ("–", 0)]
        if capacity.get("kv_tokens"):
            segs.append((f" of {tokfmt(capacity['kv_tokens'])} tok", style(dim=True)))
        if capacity.get("kv_bytes"):
            segs.append((f" ({bytesfmt(capacity['kv_bytes'])})", style(dim=True)))
        if capacity.get("concurrency"):
            context = member.get("max_model_len") or settings.get("max_model_len")
            segs.append((f" · {capacity['concurrency']:.2f}× concurrency"
                         + (f" @ {tokfmt(context)} ctx" if context else ""), 0))
        add(*segs)
        bits = []
        if live.get("ttft_5m") is not None:
            ttft = live["ttft_5m"]
            bits.append([("ttft ", style(dim=True)), (f"{ttft:.2f}s", style(hi_color(ttft, 1, 3)) if ttft > 1 else 0)])
        if live.get("accept_length_5m") is not None:
            bits.append([("accept len ", style(dim=True)), (f"{live['accept_length_5m']:.2f}", 0)])
        if live.get("prefix_hit_5m") is not None:
            hit = live["prefix_hit_5m"] * 100
            bits.append([("prefix hit ", style(dim=True)), (f"{hit:.0f}%", style(GOOD, bold=True) if hit >= 50 else 0)])
        if bits:
            segs = [lbl("latency")]
            for index, part in enumerate(bits):
                if index:
                    segs.append((" · ", style(dim=True)))
                segs += part
            add(*segs, ("  (5m)", style(dim=True)))
        finished = totals.get("finished") or {}
        if finished or totals.get("output_tokens") is not None:
            segs = [lbl("totals")]
            for index, (key, value) in enumerate((k, v) for k, v in finished.items() if v or k == "stop"):
                if index:
                    segs.append((" ", style(dim=True)))
                segs += [(f"{key} ", style(dim=True)),
                         (f"{round(value)}", style(CRIT, bold=True) if key in ("error", "abort") and value else 0)]
            preempt = round(totals.get("preemptions") or 0)
            segs += [(" · preempt ", style(dim=True)), (f"{preempt}", style(CRIT, bold=True) if preempt else style(dim=True)),
                     (" · in ", style(dim=True)), (tokfmt(totals.get("prompt_tokens")), 0),
                     (" out ", style(dim=True)), (tokfmt(totals.get("output_tokens")), 0), (" tok", style(dim=True))]
            add(*segs)
    if settings:
        bits = [f"TP{settings['tensor_parallel_size']}" if settings.get("tensor_parallel_size") else None,
                f"ctx {tokfmt(settings.get('max_model_len'))}" if settings.get("max_model_len") else None,
                f"batch {settings['max_num_batched_tokens']}" if settings.get("max_num_batched_tokens") else None,
                f"kv {settings['cache_dtype']}" if settings.get("cache_dtype") else None,
                settings.get("quantization"),
                (f"spec {settings['speculative_method']}×{settings.get('num_speculative_tokens') or '?'}"
                 if settings.get("speculative_method") else None),
                "prefix-cache" if settings.get("prefix_caching_enabled") else None,
                f"gpu {capacity['gpu_memory_utilization']:.2f}" if capacity.get("gpu_memory_utilization") else None]
        if any(bits):
            add(lbl("settings"), *dotted(" · ".join(b for b in bits if b)))
        parsers = [f"{k.split('_')[0]} {settings[k]}" for k in ("reasoning_parser", "tool_call_parser") if settings.get(k)]
        if parsers:
            add(lbl("parsers"), *dotted(" · ".join(parsers)))
    if member.get("image"):
        add(lbl("runtime"), ("image ", style(dim=True)), (member["image"], INFO))
    logs = member.get("logs") or {}
    if logs.get("commands"):
        add(lbl("logs"), (logs["commands"][0], ACCENT), (f"  on {logs.get('host') or 'the head Spark'}", style(dim=True)))
    nodes = {n.get("node"): n for n in member.get("nodes") or []}
    for spark in sparks:
        node = nodes.get(spark.get("name")) or {}
        up = spark.get("up")
        position = f"{spark['position']} " if spark.get("position") is not None else ""
        segs = [("  ", 0), ("▲" if up else "▼" if up is False else "·",
                            style(GOOD if up else CRIT if up is False else 4, bold=True)),
                (f" {position}{spark.get('name', '?'):<10}", style(bold=True) if up else style(CRIT) if up is False else 0),
                (f" {spark.get('lan_address') or spark.get('address') or ''}", INFO)]
        if node:
            used = (node.get("memory_total") or 0) - (node.get("memory_available") or 0)
            ram = used / node["memory_total"] if node.get("memory_total") else 0
            segs += [("  rank ", style(dim=True)), (f"{node.get('rank', '?')}", 0), (" · ram ", style(dim=True)),
                     (f"{bytesfmt(used)}/{bytesfmt(node.get('memory_total'))}", style(hi_color(ram * 100, 85, 95))),
                     (" · disk free ", style(dim=True)),
                     (bytesfmt(node.get("disk_available")), disk_style(node.get("disk_available"), node.get("disk_total")))]
        add(*segs)
    return rows


# -- the frame -----------------------------------------------------------------------------------
def frame(snapshot, view: dict, *, width: int, height: int | None, age, interval: float, error=None) -> list:
    """The screen as rows of (segments, selected) pairs; `height` None for plain text (no scrolling,
    no key bar). `view` holds the selection, the expanded card and the events pane."""
    if snapshot is None:
        return [([(f" dashboard: {error or 'connecting…'}", style(CRIT) if error else style(dim=True))], False)]
    now = snapshot.get("generated_at", time.time())
    rings = rings_of(snapshot)
    members = members_of(snapshot)
    rows = [(header_segments(snapshot, age=age, interval=interval, error=error), False),
            ([("─" * max(1, width - 1), style(dim=True))], False)]
    for ring in rings:
        label = ring_label(ring, rings)
        rows.append(([(" " + label, style(bold=True))] + ring_segments(ring, members, width - 2 - len(label)), False))
    topology = snapshot.get("topology") or {}
    if topology.get("detail"):
        rows.append(([(" " + topology["detail"], style(WARN))], False))
    rows.append(([], False))
    selected = max(0, min(view.get("selected", 0), len(members) - 1)) if members else 0
    view["selected"] = selected
    name_width = max([9] + [len(str(m.get("name") or m.get("id") or "")) for m in members])
    cards = []
    for index, member in enumerate(members):
        name, bits = model_line(member, now, ring_tag=len(rings) > 1)
        state = member.get("state") or ""
        expanded = view.get("details") == "all" or (view.get("details") and index == selected)
        marker = " ▾" if expanded else (" ●" if state == "serving" else " ○")
        segs = [(marker, style(STATE_COLORS.get(state, 4), bold=state == "serving")),
                (f" {name:<{name_width}} ", style(group_color(member), bold=True))]
        for position, (text, st) in enumerate(bits):
            segs += ([(" ", 0)] if position else []) + [(text, st)]
        cards.append((segs, index == selected and height is not None))
        if expanded:
            details = model_details(member, now, width)
            for row, detail in enumerate(details):
                cards.append(([("   └ " if row == len(details) - 1 else "   │ ", style(dim=True))] + detail, False))
    if not members:
        cards.append(([("   no model deployment is recorded for this fabric", style(dim=True))], False))
    if height is None:
        return rows + cards + _events(snapshot, view, 10 if view.get("events") else 0)
    events = _events(snapshot, view, 7) if view.get("events") else []
    room = max(1, height - len(rows) - len(events) - 1)
    first_selected = next((i for i, (_, sel) in enumerate(cards) if sel), 0)
    top = view.get("top", 0)
    block_end = first_selected + 1 + sum(1 for segs, _ in cards[first_selected + 1:] if segs and segs[0][0].startswith("   "))
    if block_end - top > room:
        top = block_end - room
    top = max(0, min(top, first_selected, max(0, len(cards) - room)))
    view["top"] = top
    shown = cards[top:top + room]
    return rows + shown + [([], False)] * (room - len(shown)) + events + [(_key_bar(width), False)]


def _events(snapshot, view, count) -> list:
    if not count:
        return []
    rows = [([(" events (e to hide)", style(dim=True, bold=True))], False)]
    for event in (snapshot.get("events") or [])[:count]:
        stamp = time.strftime("%H:%M", time.localtime(event["at"]))
        rows.append(([(" " + stamp + "  ", style(dim=True)), (f"{event['stack']:<11} ", ACCENT), (event["text"], 0)], False))
    return rows


def _key_bar(width) -> list:
    hints = " ↑↓ select · d details · e events · r read now · -/= period · q quit"
    return [(hints.ljust(max(0, width - 1))[:max(0, width - 1)], style(4, dim=True))]


def clip(segments: list, width: int) -> list:
    """`segments` cut at `width - 1` columns: the last column is never written, because some
    consoles wrap there and push the next row onto the end of this one."""
    out, column = [], 0
    for text, st in segments:
        if column >= width - 1:
            break
        part = text[:width - 1 - column]
        out.append((part, st))
        column += len(part)
    return out


def text_lines(rows: list, width: int) -> list:
    return ["".join(text for text, _ in clip(segs, width)) for segs, _ in rows]


# -- reading the local dashboard -----------------------------------------------------------------
class NotLocal(ValueError):
    """The terminal view reads the dashboard on this Spark only."""


def local_addresses(fabric_path: Path = dashboard.FABRIC_PATH) -> set:
    """Loopback and this Spark's management and LAN addresses from the fabric document.

    No name is resolved: a lookup could send a packet to a name server."""
    found = {"127.0.0.1", "::1"}
    try:
        fabric = dashboard.fabric_from_document(dashboard.read_json(fabric_path))
    except (OSError, ValueError):
        return found
    hostname = socket.gethostname()
    for row in fabric["positions"]:
        if row["hostname"] == hostname:
            found.update(a for a in (row.get("address"), row.get("lan_address")) if a)
    return found


def require_local(url: str, addresses: set) -> str:
    """`url` when it names the dashboard on this Spark over HTTP by an address in `addresses`, a
    loopback address or 'localhost'; NotLocal otherwise, before any socket opens."""
    parts = urllib.parse.urlsplit(url)
    host = parts.hostname or ""
    if parts.scheme != "http" or parts.path not in ("", "/") or parts.query or parts.username:
        raise NotLocal(f"{url}: give the dashboard's own address, http://ADDRESS:{dashboard.DEFAULT_PORT}")
    try:
        literal = ipaddress.ip_address(host)
        local = literal.is_loopback or str(literal) in addresses
    except ValueError:
        local = host == "localhost"
    if not local:
        raise NotLocal(f"{host} is not this Spark: the terminal view reads the dashboard on this Spark only "
                       "(run it on Node A, over SSH if need be)")
    return url.rstrip("/")


def local_url(fabric_path: Path = dashboard.FABRIC_PATH) -> str:
    """The dashboard service's address on this Spark, by the service's own default bind rule."""
    return dashboard.service_url(fabric_path)


def default_urls(fabric_path: Path = dashboard.FABRIC_PATH) -> list:
    """Loopback first, then the service's own listen address on this Spark when it differs."""
    urls = [LOOPBACK_URL]
    own = local_url(fabric_path)
    return urls + ([own] if own not in urls else [])


class Reader:
    """Reads the local dashboard's /api/status every `interval` seconds in a background thread.

    `base` is one URL or a list of them, tried in order: a refused or unreachable address passes to
    the next, and the one that answered is read from then on. An HTTP error is an answer.

    A read counts as a viewer, as the web page's does: the service then reads each model's
    /metrics every 5 s instead of every 60 s. A `passive` reader asks the service not to count it,
    so the view changes nothing the service sends."""

    def __init__(self, base, *, password=None, opener=None, passive=False):
        self.bases = [base] if isinstance(base, str) else list(base)
        self.base = self.bases[0]
        self.path = "/api/status?viewer=passive" if passive else "/api/status"
        self.opener = opener or urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.headers = {"User-Agent": "sparkring-dashboard-tui"}
        if password:
            self.headers["Authorization"] = "Basic " + base64.b64encode(f"sparkring:{password}".encode()).decode()
        self.snapshot, self.error, self.updated = None, None, 0.0
        self.interval = dashboard.METRICS_VIEWED_SECONDS
        self.lock = threading.Lock()
        self.stop, self.nudge = threading.Event(), threading.Event()

    def read(self) -> dict:
        ordered = [self.base] + [b for b in self.bases if b != self.base]
        for index, base in enumerate(ordered):
            request = urllib.request.Request(base + self.path, headers=self.headers, method="GET")
            try:
                with self.opener.open(request, timeout=READ_TIMEOUT) as response:
                    found = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError:
                raise
            except OSError:
                if index == len(ordered) - 1:
                    raise
                continue
            self.base = base
            return found
        raise OSError("no dashboard address to read")

    def once(self) -> None:
        try:
            snapshot = self.read()
            with self.lock:
                self.snapshot, self.error, self.updated = snapshot, None, time.time()
        except (OSError, ValueError) as error:
            with self.lock:
                self.error = str(error) or "unreachable"

    def loop(self) -> None:
        while not self.stop.is_set():
            self.once()
            self.nudge.wait(self.interval)
            self.nudge.clear()

    def state(self):
        with self.lock:
            age = time.time() - self.updated if self.updated else None
            return self.snapshot, self.error, age


# -- curses ------------------------------------------------------------------------------------------
def _attributes(curses, st: int) -> int:
    attrs = curses.color_pair(st & 0xF) if st & 0xF else curses.A_NORMAL
    if st & BOLD:
        attrs |= curses.A_BOLD
    if st & DIM:
        attrs |= curses.A_DIM
    return attrs


def _draw(curses, screen, rows) -> None:
    height, width = screen.getmaxyx()
    screen.erase()
    for y, (segs, selected) in enumerate(rows[:height]):
        column = 0
        for text, st in clip(segs, width):
            try:
                screen.addnstr(y, column, text, len(text), _attributes(curses, st) | (curses.A_REVERSE if selected else 0))
            except curses.error:
                pass
            column += len(text)
    screen.refresh()


def interactive(reader: Reader) -> int:
    try:
        import curses
    except ImportError:
        print("The terminal view needs Python's curses module; sparkring dashboard tui --once prints one frame.",
              file=sys.stderr)
        return 2

    def run(screen):
        curses.curs_set(0)
        if curses.has_colors():
            curses.start_color()
            try:
                curses.use_default_colors()
                background = -1
            except curses.error:
                background = curses.COLOR_BLACK
            for number, color in ((1, curses.COLOR_GREEN), (2, curses.COLOR_YELLOW), (3, curses.COLOR_RED),
                                  (4, curses.COLOR_CYAN), (5, curses.COLOR_MAGENTA), (6, curses.COLOR_BLUE)):
                curses.init_pair(number, color, background)
        screen.timeout(150)
        view = {"selected": 0, "top": 0, "details": False, "events": False}
        while True:
            snapshot, error, age = reader.state()
            height, width = screen.getmaxyx()
            _draw(curses, screen, frame(snapshot, view, width=width, height=height, age=age,
                                        interval=reader.interval, error=error))
            key = screen.getch()
            count = len(members_of(snapshot or {}))
            if key in (ord("q"), 27):
                return
            if key in (curses.KEY_UP, ord("k")):
                view["selected"] = max(0, view["selected"] - 1)
            elif key in (curses.KEY_DOWN, ord("j")):
                view["selected"] = min(max(0, count - 1), view["selected"] + 1)
            elif key in (ord("d"), ord(" ")):
                view["details"] = not view["details"]
            elif key == ord("e"):
                view["events"] = not view["events"]
            elif key == ord("r"):
                reader.nudge.set()
            elif key == ord("-"):
                reader.interval = min(60.0, reader.interval + 1.0)
            elif key in (ord("="), ord("+")):
                reader.interval = max(1.0, reader.interval - 1.0)

    threading.Thread(target=reader.loop, daemon=True, name="dashboard-reader").start()
    try:
        curses.wrapper(run)
    except curses.error as error:
        if "setupterm" in str(error):
            print("The terminal view needs a terminal; sparkring dashboard tui --once prints one frame.",
                  file=sys.stderr)
            return 2
        raise
    finally:
        reader.stop.set()
        reader.nudge.set()
    return 0


def main(args) -> int:
    """`sparkring dashboard tui [--once [--details] [--events] [--width N]] [--passive] [--url URL]
    [--password-file FILE]`."""
    addresses = local_addresses()
    urls = [require_local(url, addresses) for url in ([args.url] if args.url else default_urls())]
    password = Path(args.password_file).read_text(encoding="utf-8").strip() if args.password_file else None
    reader = Reader(urls, password=password, passive=args.passive)
    if not args.once:
        return interactive(reader)
    reader.once()
    snapshot, error, age = reader.state()
    if snapshot is None:
        print(f"SparkRing: the dashboard at {' or '.join(urls)} did not answer: {error} (the service runs on "
              "Node A; run the terminal view there)", file=sys.stderr)
        return 1
    width = args.width or shutil.get_terminal_size((120, 40)).columns
    view = {"selected": 0, "details": "all" if args.details else False, "events": args.events}
    if os.name == "nt":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass
    for line in text_lines(frame(snapshot, view, width=width, height=None, age=age, interval=reader.interval), width):
        print(line)
    return 0


def add_parser(commands) -> None:
    tui = commands.add_parser("tui", help="the dashboard in this terminal, read from the dashboard service on this "
                                          "Spark (no request to any model, no SSH)")
    tui.add_argument("--once", action="store_true", help="print one frame as plain text and exit")
    tui.add_argument("--details", action="store_true", help="with --once: expand every model's card")
    tui.add_argument("--events", action="store_true", help="with --once: add the most recent events")
    tui.add_argument("--width", type=int, help="with --once: columns (default: the terminal's)")
    tui.add_argument("--passive", action="store_true",
                     help="do not count as a viewer: the service keeps reading /metrics once a minute")
    tui.add_argument("--url", help="the dashboard service on this Spark (default: http://127.0.0.1:8090, then the "
                                   "service's LAN address)")
    tui.add_argument("--password-file", help="the dashboard's Basic-authentication password, if it has one")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="sparkring dashboard tui")
    add_parser(parser.add_subparsers(dest="command", required=True))
    raise SystemExit(main(parser.parse_args(["tui", *sys.argv[1:]])))
