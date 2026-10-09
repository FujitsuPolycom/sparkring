"""What the plugin relies on in the pinned SIRCL build, read from its source (nothing is imported).

Each test names one interface fact that a reason in ``FILE_CHECKS`` states; together with the pins they
show that the runtime's attribute reads and the packed kernel's protocol are those of the pinned SIRCL build.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from glm_dcp_decode_comm import runtime

PROJECT = Path(__file__).resolve().parents[1]
PACK_KERNEL = PROJECT / "glm_dcp_decode_comm" / "_scatter_pack_cute.py"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    found = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name]
    assert len(found) == 1, name
    return found[0]


def _methods(cls: ast.ClassDef) -> dict[str, ast.FunctionDef]:
    return {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}


def _self_attributes(cls: ast.ClassDef) -> set[str]:
    names = set()
    for node in ast.walk(cls):
        targets = (node.targets if isinstance(node, ast.Assign)
                   else [node.target] if isinstance(node, (ast.AnnAssign, ast.AugAssign)) else [])
        for target in targets:
            for item in ast.walk(target):
                if isinstance(item, ast.Attribute) and isinstance(item.value, ast.Name) and item.value.id == "self":
                    names.add(item.attr)
    return names


def test_sircls_communicator_has_every_collective_the_stream_wrapper_moves(sircl_root):
    cls = _class(_tree(sircl_root / "vllm" / "communicator.py"), "SirclCudaCommunicator")
    assert set(runtime._WRAPPED_METHODS) <= set(_methods(cls))
    assert "sircl" in _self_attributes(cls)


def test_a_dcp_groups_adapter_holds_its_collectives_and_their_runtime(sircl_root):
    tree = _tree(sircl_root / "vllm" / "adapter.py")
    build = _methods(_class(tree, "GroupAdapter"))["_build_dcp"]
    source = ast.unparse(build)
    assert "self.dcp = dcp" in source and "self.session = dcp._runtime" in source
    assert any(isinstance(node, ast.FunctionDef) and node.name == "live_adapters" for node in tree.body)
    assert "shared_from" in _self_attributes(_class(tree, "GroupAdapter"))


def test_the_dcp_collectives_keep_what_the_runtime_reads(sircl_root):
    cls = _class(_tree(sircl_root / "vllm" / "dcp_collectives.py"), "SirclDcpCollectives")
    assert {"_runtime", "gather_chunk", "gather_max", "scatter_op_bytes"} <= _self_attributes(cls)
    limit = ast.unparse(_methods(cls)["_scatter_op_limit"])
    assert "min(self.scatter_op_bytes, self._runtime.max_size)" in limit
    gather = ast.unparse(_methods(cls)["all_gather"])
    assert "rt.all_gather(view[r0:r1], dim=-1, out=out_rows[r0:r1])" in gather


def test_the_session_has_the_arena_view_and_launch_surface_the_packed_all_to_all_uses(sircl_root):
    cls = _class(_tree(sircl_root / "oneshot" / "runtime.py"), "RoceOneshotAllReduce")
    methods = _methods(cls)
    assert {"all_gather", "_counter_addresses", "_order_stream", "_mark_stream", "_tuned_op",
            "check_health"} <= set(methods)
    assert "out" in [arg.arg for arg in methods["all_gather"].args.kwonlyargs]
    assert {"_recv_base", "_flag_base", "_send_base", "_ctrl_base", "_slot_bytes", "_epoch_address",
            "_poison_address", "_threads", "_blocks", "_large_blocks", "_packs_per_thread", "_layout", "_lock",
            "lane_count", "spin_limit", "scatter_available", "max_size", "max_gather_bytes", "large_piece_bytes",
            "relay_safe_bytes", "world_size", "rank", "device"} <= _self_attributes(cls)
    tuned = ast.unparse(methods["_tuned_op"])
    assert "with self._tuning_lock:" in tuned and "yield" in tuned


def test_the_scatter_launch_takes_the_tuned_op_before_the_session_lock(sircl_root):
    session = ast.unparse(_methods(_class(_tree(sircl_root / "oneshot" / "runtime.py"),
                                          "RoceOneshotAllReduce"))["all_to_all"])
    assert "with self._tuned_op('all_to_all'" in session
    ops = (sircl_root / "oneshot" / "_scatter_ops.py").read_text(encoding="utf-8")
    assert "blocks = getattr(session, \"_large_blocks\", session._blocks)" in ops
    assert "grid = proto.grid_blocks(packs, session._threads, blocks, session._packs_per_thread)" in ops


def test_sircls_dcp_combine_shim_is_marked_and_exchanges_through_the_communicator(sircl_root):
    tree = _tree(sircl_root / "vllm" / "shims.py")
    marker = next(node for node in tree.body if isinstance(node, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "MARKER" for t in node.targets))
    assert ast.literal_eval(marker.value) == runtime.SIRCL_SHIM_MARKER
    install = ast.unparse(next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                               and node.name == "_install_dcp_all_to_all"))
    assert "setattr(dcp_a2a_lse_reduce, MARKER, original)" in install
    assert "communicator.all_to_all_single(recv_buffer.view(-1), send_buffer.view(-1))" in install
    assert "module._dcp_a2a_unpack_combine(recv_buffer, head_dim, lse_pack_dim, return_lse" in install


def test_a_dcp_all_to_all_within_the_op_limit_is_one_session_all_to_all(sircl_root):
    executor = ast.unparse(next(node for node in _tree(sircl_root / "vllm" / "executor.py").body
                                if isinstance(node, ast.FunctionDef) and node.name == "all_to_all_single"))
    assert "if plan.ops == 1:\n            session.all_to_all(src, output)" in executor


def test_the_protocol_constants_of_the_scatter_op(sircl_root):
    tree = _tree(sircl_root / "protocol.py")
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    assert values["OP_SHIFT"] == 30 and values["PACK_BYTES"] == 16
    enums = {cls.name: {n.targets[0].id: ast.literal_eval(n.value) for n in cls.body if isinstance(n, ast.Assign)}
             for cls in tree.body if isinstance(cls, ast.ClassDef) and cls.name in ("Op", "Ctrl")}
    assert enums["Op"]["SCATTER"] == 3
    assert {k: enums["Ctrl"][k] for k in ("DOORBELL", "NBYTES", "ERROR_SEQ", "MISSING_PEER", "OP_WORD",
                                          "MISSING_LANE", "WAIT_LIMIT_US")} == {
        "DOORBELL": 0, "NBYTES": 1, "ERROR_SEQ": 2, "MISSING_PEER": 3, "OP_WORD": 4, "MISSING_LANE": 6,
        "WAIT_LIMIT_US": 7}


def test_every_device_function_the_packed_kernel_imports_exists(sircl_root):
    kernel = _tree(PACK_KERNEL)
    imported = {(node.module, alias.name) for node in kernel.body if isinstance(node, ast.ImportFrom)
                and (node.module or "").startswith("sparkring_sircl") for alias in node.names}
    assert imported
    for module, name in imported:
        path = sircl_root.joinpath(*module.split(".")[1:]).with_suffix(".py")
        defined = {node.name for node in _tree(path).body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        defined |= {t.id for node in _tree(path).body if isinstance(node, ast.Assign)
                    for t in node.targets if isinstance(t, ast.Name)}
        assert name in defined, (module, name)


def _code_lines(text: str, start: str, end: str) -> list[str]:
    """Non-comment lines from the line containing ``start`` up to (not including) the line containing ``end``."""
    lines = text.splitlines()
    first = next(i for i, line in enumerate(lines) if start in line)
    last = next(i for i in range(first + 1, len(lines)) if end in lines[i])
    return [line.strip() for line in lines[first:last] if line.strip() and not line.strip().startswith("#")]


@pytest.mark.parametrize("start,end", [
    ("epoch = ld_relaxed_gpu_u32(epoch_ptr)", "zero = Uint32(size_packs)"),
    ("# 2. the last block to finish staging", "failed = ld_relaxed_gpu_u32(poison_ptr)"),
    ("# 5. the last block to finish publishes", "st_release_gpu_u32(epoch_ptr, seq)"),
])
def test_the_packed_kernel_states_sircls_scatter_protocol(sircl_root, start, end):
    # The epoch and slot, the doorbell (op word with op code 3, byte count, sequence), the timed flag waits
    # with their failure record and poison, and the epoch publication are the statements of the pinned SIRCL build's
    # scatter kernel; only the staging and the copy phase differ.
    ours = _code_lines(PACK_KERNEL.read_text(encoding="utf-8"), start, end)
    sircl = _code_lines((sircl_root / "oneshot" / "_scatter_cute.py").read_text(encoding="utf-8"), start, end)
    assert ours == sircl
    assert len(ours) >= 4


def test_the_copy_phase_reads_peers_where_their_progress_threads_wrote(sircl_root):
    ours = PACK_KERNEL.read_text(encoding="utf-8")
    sircl = (sircl_root / "oneshot" / "_scatter_cute.py").read_text(encoding="utf-8")
    # SIRCL's helper takes the slot count as an argument, the packed kernel's reads it from its launch object.
    assert "recv_base + (Int64(source) * Int64(slots) + slot) * slot_bytes + offset" in sircl
    assert "recv_base + (Int64(source) * Int64(launch._slots) + slot) * slot_bytes + offset" in ours
    assert "output_base + Int64(source) * dst_stride + Int64(q) * Int64(PACK_BYTES)" in ours
    assert "output_base + Int64(source) * dst_stride + Int64(q) * Int64(PACK_BYTES)" in sircl
