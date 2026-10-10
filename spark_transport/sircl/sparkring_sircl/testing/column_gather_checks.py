"""GPU emulation checks of the vLLM adapter's column gathers (:class:`sparkring_sircl.vllm.executor.ColumnGather`).

A column gather (an all-gather along a dimension with more than one row in front of it) that the session
would carry on its ring or chain as a dimension-0 gather of the same shard runs as that gather into a
staging buffer plus one local copy. :func:`column_gather_checks` runs every case through the executor on
every rank twice, staged and as ``all_gather_large`` along the column dimension (tiles), and compares the
two outputs bit for bit with each other and with the concatenation; it checks the route each case took, the
staging buffer kept across eager calls of different sizes, and a CUDA graph capture of the staged gather
replayed with new inputs. ``python -m sparkring_sircl.testing.gpu_emulation --column-gather-only`` runs only
these checks after preparing the sessions.

Every rank of the emulation is a thread on one GPU, and the ranks' streams share the GPU's hardware queues:
a kernel waiting for its peers holds back whatever another rank queued behind it on the same queue. Each
collective here therefore runs in a phase of its own, which every rank finishes before the next begins, and
the inputs are placed on the device in a phase before it.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any


def _inputs(torch, world: int, shape, dtype, seed: int):
    tensors = []
    for rank in range(world):
        generator = torch.Generator().manual_seed(seed * 1009 + rank)
        tensors.append(torch.randn(tuple(shape), generator=generator).to(dtype))
    return tensors


def _same_bits(torch, got, want) -> bool:
    if got.shape != want.shape or got.dtype != want.dtype:
        return False
    view = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[want.element_size()]
    return bool(torch.equal(got.contiguous().view(view), want.contiguous().view(view)))


def _placed(torch, tensor, layout: str, device):
    """``tensor`` on ``device`` in ``layout``: contiguous, transposed (a non-contiguous view of a
    contiguous transpose), or offset (contiguous, 2 bytes past a 16-byte boundary)."""
    if layout == "transposed":
        return tensor.to(device).transpose(-1, -2).contiguous().transpose(-1, -2)
    if layout == "offset":
        base = torch.empty(tensor.numel() + 8, dtype=tensor.dtype, device=device)
        view = base[1:1 + tensor.numel()].view(tensor.shape)
        view.copy_(tensor.to(device))
        return view
    return tensor.to(device)


def _set_schedule(group, schedule: str) -> None:
    for session in group.sessions:
        session.gather_schedule = schedule


def _poisoned(group) -> list[int]:
    return [rank for rank, session in enumerate(group.sessions) if session.poisoned]


def column_gather_checks(group: Any) -> Iterator[tuple[str, bool, str]]:
    """Column gathers of 4 to 8 MiB shards, staged and tiled, eager and captured (see the module); each
    check is yielded when it completes."""
    torch = group.torch
    from ..vllm import executor, planner

    session0 = group.sessions[0]
    if not callable(getattr(session0, "gather_uses_chain", None)) or not session0.link_available:
        yield ("column gathers", True, "not available here (no chain links)")
        return
    world = group.world
    limits = [planner.SessionLimits.of(session, reduce_dtypes=("bfloat16",)) for session in group.sessions]
    policy = planner.Policy(None, "never", "sircl")
    bf16, fp16, fp32 = torch.bfloat16, torch.float16, torch.float32
    ring_route = "ring" if session0.ring_available else "chain"
    # (schedule, shard shape, dim, dtype, layout, expected route). 4 to 8 MiB shards gathered along a
    # dimension with rows in front of it; [8192, 328] BF16 is GLM-5.3's column-split projection at an
    # 8,192-token prefill chunk (5,373,952 bytes per rank). Under ``auto`` the expected route is the session's
    # own answer for a dimension-0 gather of the shard's bytes, asked as the executor asks it ("session"):
    # the ring of a cabled pair's built-in plan, the chain from the chain minimum on paths and cycles.
    cases = [
        ("ring", (8192, 256), -1, bf16, "contiguous", ring_route),
        ("ring", (8192, 328), -1, bf16, "contiguous", ring_route),
        ("ring", (4096, 1024), 1, bf16, "contiguous", ring_route),
        ("ring", (2, 4096, 328), 2, bf16, "contiguous", ring_route),
        ("ring", (4096, 8, 64), 1, bf16, "contiguous", ring_route),
        ("ring", (8192, 257), -1, bf16, "contiguous", ring_route),     # rows of 514 bytes: 2-byte copy view
        ("ring", (2048, 520), -1, fp32, "contiguous", ring_route),
        ("ring", (8192, 328), -1, bf16, "transposed", ring_route),
        ("ring", (8192, 328), -1, bf16, "offset", ring_route),
        ("auto", (8192, 328), -1, bf16, "contiguous", "session"),
        ("auto", (2048, 1024), -1, fp16, "contiguous", "session"),
        ("pieces", (8192, 328), -1, bf16, "contiguous", None),
    ]
    saved = session0.gather_schedule
    staged = [executor.ColumnGather(True) for _ in range(world)]
    tiled = [executor.ColumnGather(False) for _ in range(world)]
    seed = 4100
    try:
        for schedule, shape, dim, dtype, layout, expected_route in cases:
            seed += 1
            _set_schedule(group, schedule)
            inputs = _inputs(torch, world, shape, dtype, seed)
            reference = torch.cat(inputs, dim=dim)
            nbytes = inputs[0].numel() * inputs[0].element_size()
            if expected_route == "session":
                probe = torch.empty((nbytes,), dtype=torch.uint8, device="meta")
                expected_route = ("ring" if session0.gather_uses_ring(probe, 0) else
                                  "chain" if session0.gather_uses_chain(probe, 0) else None)
            label = (f"column gather {list(shape)} {str(dtype).replace('torch.', '')} dim {dim} ({layout}), "
                     f"{nbytes} B per rank, schedule {schedule}")
            if _poisoned(group):
                yield (label, False, f"not run: poisoned ranks {_poisoned(group)}")
                continue
            try:
                placed = group.each(lambda rank, session, inputs=inputs, layout=layout:
                                    _placed(torch, inputs[rank], layout, session.device))
                plans = [planner.plan_all_gather(planner.TensorMeta.of(x), dim, limits[rank], policy, capturing=False)
                         for rank, x in enumerate(placed)]
                timed = {}
                outputs = {}
                for name, objects in (("staged", staged), ("tiled", tiled)):
                    started = time.perf_counter()
                    outputs[name] = group.each(
                        lambda rank, session, objects=objects, dim=dim:
                        executor.all_gather(session, plans[rank], placed[rank], dim, world,
                                            column_gather=objects[rank]))
                    timed[name] = (time.perf_counter() - started) * 1e3
                new = [output.cpu() for output in outputs["staged"]]
                old = [output.cpu() for output in outputs["tiled"]]
            except Exception as error:  # noqa: BLE001 - reported as a failed check
                yield (label, False, f"{type(error).__name__}: {error}")
                continue
            problems = []
            routes = {gather.last_route for gather in staged}
            if routes != {expected_route}:
                problems.append(f"routes {sorted(map(str, routes))}, expected {expected_route}")
            if {gather.last_route for gather in tiled} != {None}:
                problems.append("the switched-off object staged a call")
            for rank in range(world):
                if not _same_bits(torch, new[rank], old[rank]):
                    problems.append(f"rank {rank}: staged and tiled outputs differ")
                elif not _same_bits(torch, new[rank], reference):
                    problems.append(f"rank {rank}: output differs from the concatenation")
            detail = (f"plan {plans[0].method}, route {expected_route}, staged {timed['staged']:.1f} ms, "
                      f"tiled {timed['tiled']:.1f} ms (emulated, all ranks)")
            yield (label, not problems, "; ".join(problems[:4]) or detail)

        # Eager calls of different sizes on one object per rank: the kept buffer grows to the largest
        # world * shard and every later call reuses it.
        _set_schedule(group, "ring")
        sizes = [(8192, 328), (8192, 256), (4096, 1024), (8192, 256), (4096, 1024), (8192, 328)]
        label = "column gathers of 5.25, 4, 8, 4, 8 and 5.25 MiB shards in turn on one staging buffer per rank"
        kept = [executor.ColumnGather(True) for _ in range(world)]
        problems = []
        largest = 0
        pointers: list[set[int]] = [set() for _ in range(world)]
        try:
            if _poisoned(group):
                raise RuntimeError(f"poisoned ranks {_poisoned(group)}")
            for index, shape in enumerate(sizes):
                inputs = _inputs(torch, world, shape, bf16, 4300 + index)
                placed = group.each(lambda rank, session, inputs=inputs: inputs[rank].to(session.device))
                outputs = group.each(
                    lambda rank, session, placed=placed:
                    executor.all_gather(session, planner.plan_all_gather(
                        planner.TensorMeta.of(placed[rank]), -1, limits[rank], policy, capturing=False),
                        placed[rank], -1, world, column_gather=kept[rank]))
                largest = max(largest, world * shape[0] * shape[1] * 2)
                reference = torch.cat(inputs, dim=-1)
                for rank in range(world):
                    if kept[rank].staging_bytes != largest:
                        problems.append(f"call {index} rank {rank}: staging {kept[rank].staging_bytes} B, "
                                        f"expected {largest}")
                    if index >= 2:
                        pointers[rank].add(kept[rank]._buffer.data_ptr())
                    if not _same_bits(torch, outputs[rank].cpu(), reference):
                        problems.append(f"call {index} rank {rank}: differs from the concatenation")
            moved = [rank for rank in range(world) if len(pointers[rank]) != 1]
            if moved:
                problems.append(f"ranks {moved}: the buffer moved after it reached the largest size")
            yield (label, not problems, "; ".join(problems[:4])
                   or f"staging {largest} B kept per rank, calls {kept[0].calls}")
        except Exception as error:  # noqa: BLE001
            yield (label, False, f"{type(error).__name__}: {error}")

        # One CUDA graph per rank holding a staged column gather, replayed with new inputs.
        shape, dim = (8192, 328), -1
        label = f"graph column gather {list(shape)} bfloat16 dim {dim}, schedule ring"
        state = []
        try:
            if _poisoned(group):
                raise RuntimeError(f"poisoned ranks {_poisoned(group)}")
            # Captures run one rank after another: nothing executes while capturing.
            for rank, session in enumerate(group.sessions):
                with torch.cuda.stream(group.streams[rank]):
                    x = torch.zeros(shape, dtype=bf16, device=session.device)
                    plan = planner.plan_all_gather(planner.TensorMeta.of(x), dim, limits[rank], policy, capturing=True)
                    gather = executor.ColumnGather(True)
                    graph = torch.cuda.CUDAGraph()
                    with session.capture():
                        with torch.cuda.graph(graph, stream=group.streams[rank]):
                            y = executor.all_gather(session, plan, x, dim, world, column_gather=gather)
                state.append((x, y, graph, gather))
            problems = []
            if {entry[3].last_route for entry in state} != {ring_route}:
                problems.append(f"captured routes {sorted(str(entry[3].last_route) for entry in state)}")
            if any(entry[3].staging_bytes for entry in state):
                problems.append("a capture used the kept staging buffer")
            for replay_seed in (4401, 4402):
                inputs = _inputs(torch, world, shape, bf16, replay_seed)
                group.each(lambda rank, session, inputs=inputs: state[rank][0].copy_(inputs[rank].to(session.device)))
                group.each(lambda rank, session: state[rank][2].replay())
                reference = torch.cat(inputs, dim=dim)
                wrong = [rank for rank in range(world) if not _same_bits(torch, state[rank][1].cpu(), reference)]
                if wrong:
                    problems.append(f"replay seed {replay_seed}: ranks {wrong} differ")
            yield (label, not problems, "; ".join(problems) or f"2 replays, route {ring_route}")
        except Exception as error:  # noqa: BLE001
            yield (label, False, f"{type(error).__name__}: {error}")
    finally:
        _set_schedule(group, saved)
