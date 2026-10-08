"""The ring harness's ring-swing configuration offline: plan, cases, refusals and bounds."""

from __future__ import annotations

import json

import pytest

from sparkring_sircl.ring import cli, plan, remote, summary, worker
from sparkring_sircl.ring.site import Site


def _site_document(size: int = 8) -> dict:
    return {
        "schema": "sircl-ring-site/v1", "image": "sha256:0123456789ab", "lan_interface": "lan0",
        "control_port": 29650, "remote_dir": "/tmp/sircl-ring",
        "ring": [{"name": f"spark{i}", "ssh": f"op@192.0.2.{10 + i}", "lan_address": f"192.0.2.{10 + i}"}
                 for i in range(size)],
    }


def _ring_swing(name: str = "ring-swing", size: int = 8, **overrides) -> plan.ConfigurationPlan:
    options = plan.configuration_options(name, plan.Options(**overrides))
    return plan.build_plan(Site.from_json(_site_document(size)), name, "run1", options=options, digest="d" * 16)


def test_ring_swing_runs_swing_beside_the_default_all_reduce_on_the_whole_ring():
    built = _ring_swing()
    assert [group.positions for group in built.groups] == [tuple(range(8))]
    assert built.options.swing_sizes == plan.RING_SWING_SIZES == (1 << 20, (1 << 20) + 16, 3 << 19, 2 << 20)
    assert built.options.large_only and built.capacity == 2 << 20
    assert _ring_swing("ring8-swing").options.swing_sizes == plan.RING_SWING_SIZES
    text = plan.render_text(built)
    assert "Swing all-reduce of BF16 messages of [1048576, 1048592, 1572864, 2097152] bytes" in text
    harness = worker.Harness.__new__(worker.Harness)
    harness.options = json.loads(json.dumps(built.to_json()["options"]))
    cases = harness.swing_cases()
    assert cases[:2] == [("all_reduce_swing", (1 << 19,), 0), ("all_reduce", (1 << 19,), 0)]
    assert len(cases) == 2 * len(plan.RING_SWING_SIZES)


def test_ring_swing_refusals(tmp_path, monkeypatch, capsys):
    with pytest.raises(plan.PlanError, match="power of two"):
        plan.builtin_groups("ring-swing", 6)
    with pytest.raises(plan.PlanError, match="transport-only"):
        plan.configuration_options("ring-swing", plan.Options(transport_only=True))
    with pytest.raises(plan.PlanError, match="exceed the capacity"):
        _ring_swing(swing_sizes=(4 << 20,))
    with pytest.raises(plan.PlanError, match="multiples of 16"):
        plan.Options(swing_sizes=(1000,))
    monkeypatch.setattr(remote, "ssh", lambda *a, **k: (_ for _ in ()).throw(AssertionError("contacted a host")))
    site_file = tmp_path / "site.json"
    site_file.write_text(json.dumps(_site_document()))
    assert cli.main(["plan", "--site", str(site_file), "--config", "ring-swing", "--json",
                     "--swing-sizes", "1048576,2097152"]) == 0
    assert json.loads(capsys.readouterr().out)["options"]["swing_sizes"] == [1048576, 2097152]


def test_the_summary_bounds_swing_and_two_shot_rows_on_the_ring_of_eight():
    built = _ring_swing().to_json()
    nbytes = 2 << 20
    runs = [{"collective": "all_reduce_swing", "mode": "graph", "shape": [nbytes // 2], "dim": 0, "bytes": nbytes,
             "correct": True, "mismatched_calls": 0, "checked": 4, "counters": {}, "traffic": "swing",
             "algorithm": "swing through the session", "times_us": [306.0] * 3},
            {"collective": "all_reduce", "mode": "graph", "shape": [nbytes // 2], "dim": 0, "bytes": nbytes,
             "correct": True, "mismatched_calls": 0, "checked": 4, "counters": {}, "traffic": "twoshot",
             "algorithm": "twoshot", "times_us": [350.0] * 3}]
    results = [{"global_rank": rank, "group": 0, "host": f"spark{rank}", "runs": runs, "error": None,
                "exit_code": 0} for rank in range(8)]
    merged = summary.merge(built, results)
    swing_case, twoshot_case = merged["cases"]
    # Swing: 1.75 of the message through every host interface at 24 GB/s; two-shot: 2 of it on a cable.
    assert swing_case["bound_ms"] == pytest.approx(1.75 * nbytes / 24e9 * 1e3, abs=1e-4)
    assert twoshot_case["bound_ms"] == pytest.approx(2 * nbytes / 24e9 * 1e3, abs=1e-4)
    assert swing_case["of_bound"] == pytest.approx(swing_case["bound_ms"] * 1e3 / 306.0, abs=1e-3)
    assert "[swing through the session]  [bound 0.153 ms" in summary.table(merged)


def test_emulated_groups_cap_the_large_grid_so_every_rank_stays_resident():
    from sparkring_sircl.testing.kernel_gpu_checks import emulation_large_blocks

    # Eight ranks on one 170-multiprocessor GPU: 16 blocks each (128 in all); smaller groups keep the default.
    assert [emulation_large_blocks(world, 170, 32) for world in (2, 4, 6, 8, 16)] == [32, 32, 16, 16, 8]
    assert emulation_large_blocks(8, 48, 32) == 4 and emulation_large_blocks(1, 48, 32) == 32


def test_the_worker_swing_reference_matches_the_numpy_reference():
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    from sparkring_sircl.testing import collective_models

    harness = worker.Harness.__new__(worker.Harness)
    harness.torch = torch
    generator = torch.Generator().manual_seed(5)
    inputs = [torch.randn(4096 + 8, generator=generator).to(torch.bfloat16) for _ in range(8)]
    got = harness._swing_reference(inputs)
    words = [tensor.view(torch.int16).numpy().view(np.uint16) for tensor in inputs]
    want = collective_models.swing_reference(words, "bfloat16")
    assert got.dtype == torch.bfloat16 and got.shape == inputs[0].shape
    assert got.view(torch.int16).numpy().view(np.uint16).tobytes() == want.tobytes()
    # Swing rounds after every step, so its bits differ from the rank-ordered sum somewhere.
    ordered = collective_models.rank_order_sum(words, "bfloat16")
    assert ordered.tobytes() != want.tobytes()
