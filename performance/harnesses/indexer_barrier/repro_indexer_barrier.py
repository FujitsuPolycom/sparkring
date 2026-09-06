"""CPU interleaving model of the actual B12X group-barrier helper.

This does not execute CUDA. It extracts the helper from the given source and
models three blocks, each with a leader and a histogram-publishing warp.
The adversarial schedule delays block B's publisher until block A can scan.
"""

import argparse
import ast
import json
from pathlib import Path


class LowerBarrier(ast.NodeTransformer):
    def visit_Expr(self, node):
        if not isinstance(node.value, ast.Call):
            return node
        call = node.value
        name = ast.unparse(call.func)
        if name == "cute.arch.sync_threads":
            args = [ast.Constant("sync")]
        elif name == "red_add_global_release_i32":
            args = [ast.Constant("arrival")]
        elif name == "spin_wait_global_ge_i32":
            args = [ast.Constant("wait"), call.args[1]]
        else:
            return node
        return ast.copy_location(ast.Expr(ast.Yield(ast.Tuple(args, ast.Load()))), node)


def load_barrier(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "_fused_group_barrier")
    function.decorator_list = []
    function.returns = None
    for arg in function.args.args:
        arg.annotation = None
    function = LowerBarrier().visit(function)
    namespace = {"Int32": int, "_fused_state_ptr": lambda *args: 0,
                 "_FUSED_STATE_ARRIVAL": 0}
    exec(compile(ast.fix_missing_locations(ast.Module([function], [])),
                 str(path), "exec"), namespace)
    return namespace["_fused_group_barrier"]


def simulate(barrier):
    # top-k=512, 513 candidates. A publishes 512 candidates in an abstract lower
    # bin, B publishes one in a higher bin, C publishes none. The two-bin model
    # collapses the matching coarse/fine histogram updates into one operation.
    histogram = {2: 0, 3: 0}
    arrival = 0
    decisions = {}
    trace = []
    sync_epoch = {block: 0 for block in "ABC"}
    sync_arrivals = {}
    waiting_sync = {}

    def actor(block, tx):
        if tx == 32:
            yield ("publish", block)
        phase = yield from barrier(None, 0, 0, 3, tx)
        if tx == 0:
            yield ("scan", block)
        yield ("decision", block)
        if decisions[block] == "another_round":
            yield from barrier(None, 0, phase, 3, tx)

    generators = {f"{b}{t}": actor(b, t) for b in "ABC" for t in (0, 32)}
    pending = {key: next(gen) for key, gen in generators.items()}

    def advance(key):
        try:
            pending[key] = next(generators[key])
        except StopIteration:
            pending.pop(key)

    # B32 represents a different warp and has lowest priority: hardware may delay it
    # warp while block leaders publish arrival and other blocks consume it.
    while pending:
        progressed = False
        for key in ("A0", "A32", "B0", "C0", "C32", "B32"):
            if key not in pending:
                continue
            op = pending[key]
            block = key[0]
            if op[0] == "wait" and arrival < op[1]:
                continue
            if op[0] == "decision" and block not in decisions:
                continue
            if op[0] == "sync":
                if key in waiting_sync:
                    continue
                epoch = sync_epoch[block]
                waiting_sync[key] = epoch
                reached = sync_arrivals.setdefault((block, epoch), set())
                reached.add(key)
                if len(reached) == 2:
                    sync_epoch[block] += 1
                    for peer in sorted(reached):
                        waiting_sync.pop(peer)
                        advance(peer)
            else:
                if op[0] == "publish":
                    histogram[2 if block == "A" else 3] += {"A": 512, "B": 1, "C": 0}[block]
                    trace.append(f"{key}: publish -> {histogram.copy()}")
                elif op[0] == "arrival":
                    arrival += 1
                    trace.append(f"{key}: arrival={arrival}")
                elif op[0] == "scan":
                    # The pinned kernel stops refining when pivot-bin count
                    # equals the remaining top-k slots.
                    count_above = histogram[3]
                    remaining = 512 - count_above
                    decisions[block] = (
                        "done" if histogram[2] == remaining else "another_round"
                    )
                    trace.append(f"{key}: scan {histogram.copy()} -> {decisions[block]}")
                advance(key)
            progressed = True
            break
        if not progressed:
            break
    return {"deadlocked": bool(pending), "decisions": decisions,
            "arrival": arrival, "pending": pending, "trace": trace}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--expect", choices=("deadlock", "complete"), required=True)
    args = parser.parse_args()
    result = simulate(load_barrier(args.source))
    print(json.dumps(result, indent=2))
    assert result["deadlocked"] == (args.expect == "deadlock"), result
