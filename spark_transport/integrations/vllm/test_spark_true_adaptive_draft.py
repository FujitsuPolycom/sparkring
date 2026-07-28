from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

import spark_true_adaptive_draft as adapter


@dataclass
class SchedulerOutput:
    num_spec_tokens_to_schedule: int
    total_num_scheduled_tokens: int = 1


class InputBatch:
    def __init__(self, rows: int) -> None:
        self.rows = rows


class FakeHandler:
    def __init__(self) -> None:
        self.widths: list[int] = []

    def set_draft_tokens(self, input_batch, draft_tokens):
        self.widths.append(int(draft_tokens.shape[1]))
        return draft_tokens


class FakeSpeculator:
    def __init__(self, rows: int = 2) -> None:
        self.num_speculative_steps = 4
        self.draft_tokens = np.full((rows, 4), 99, dtype=np.int64)
        self.steps_executed: list[int] = []

    def propose(
        self,
        input_batch,
        attn_metadata=None,
        slot_mappings=None,
    ):
        self.steps_executed.append(self.num_speculative_steps)
        for column in range(self.num_speculative_steps):
            self.draft_tokens[:, column] = column + 10
        if self.num_speculative_steps == 1:
            return self.draft_tokens[:, :1]
        return self.draft_tokens


class FakeRunner:
    def __init__(self, rows: int = 2) -> None:
        self.num_speculative_steps = 4
        self.speculator = FakeSpeculator(rows)
        self.draft_tokens_handler = FakeHandler()
        self.last_result = None
        self.execute_pending = False
        self.return_intermediate = False

    def execute_model(
        self,
        scheduler_output,
        intermediate_tensors=None,
        dummy_run=False,
        skip_attn_for_dummy_run=False,
        is_profile=False,
    ):
        if scheduler_output.total_num_scheduled_tokens <= 0:
            self.execute_pending = False
            return "no-forward"
        if self.return_intermediate:
            self.execute_pending = False
            return "intermediate"
        self.execute_pending = True
        return None

    def sample_tokens(self, grammar_output):
        del grammar_output
        assert self.execute_pending
        self.execute_pending = False
        result = self.speculator.propose(InputBatch(2))
        self.last_result = self.draft_tokens_handler.set_draft_tokens(
            InputBatch(2), result
        )
        return self.last_result


@pytest.fixture(autouse=True)
def reset_adapter():
    adapter._reset_for_tests()
    yield
    adapter._reset_for_tests()


def patch_fakes() -> None:
    adapter._patch_classes(FakeRunner, FakeSpeculator, FakeHandler)


@pytest.mark.parametrize("selected", [1, 2, 3, 4])
def test_selected_depth_controls_compute_and_scheduler_width(selected: int) -> None:
    patch_fakes()
    runner = FakeRunner()

    assert runner.execute_model(SchedulerOutput(selected)) is None
    assert runner.speculator.steps_executed == []
    result = runner.sample_tokens(None)

    assert runner.speculator.steps_executed == [selected]
    assert runner.speculator.num_speculative_steps == 4
    assert result.shape == (2, selected)
    assert runner.draft_tokens_handler.widths == [selected]
    np.testing.assert_array_equal(
        runner.speculator.draft_tokens[:, :selected],
        np.tile(np.arange(10, 10 + selected), (2, 1)),
    )
    if selected < 4:
        np.testing.assert_array_equal(
            runner.speculator.draft_tokens[:, selected:],
            np.full((2, 4 - selected), -1),
        )


def test_k2_snapshot_reports_four_saved_request_steps() -> None:
    patch_fakes()
    runner = FakeRunner(rows=2)
    runner.execute_model(SchedulerOutput(2))
    runner.sample_tokens(None)

    snapshot = adapter.true_adaptive_draft_snapshot()

    assert snapshot["execute_calls"] == 1
    assert snapshot["sample_calls"] == 1
    assert snapshot["proposal_calls"] == 1
    assert snapshot["handler_calls"] == 1
    assert snapshot["proposal_batches_by_depth"] == {"2": 1}
    assert snapshot["proposal_requests_by_depth"] == {"2": 2}
    assert snapshot["saved_draft_steps"] == 4
    assert snapshot["failures"] == 0


@pytest.mark.parametrize("selected", [-1, 5, 2.0, "2", True])
def test_invalid_scheduler_depth_fails_before_proposal(selected) -> None:
    patch_fakes()
    runner = FakeRunner()

    with pytest.raises(
        RuntimeError,
        match="num_spec_tokens_to_schedule",
    ):
        runner.execute_model(SchedulerOutput(selected))

    assert runner.speculator.steps_executed == []
    assert adapter.true_adaptive_draft_snapshot()["failures"] == 1


@pytest.mark.parametrize("selected", [0, None])
def test_non_drafted_prefill_depth_uses_stock_path(selected) -> None:
    patch_fakes()
    runner = FakeRunner()

    assert runner.execute_model(SchedulerOutput(selected)) is None
    assert not hasattr(runner, "_spark_pending_proposal_steps")
    result = runner.sample_tokens(None)

    assert result.shape == (2, 4)
    assert runner.speculator.steps_executed == [4]


def test_dummy_and_profile_runs_keep_full_depth() -> None:
    patch_fakes()
    runner = FakeRunner()

    runner.execute_model(SchedulerOutput(1), dummy_run=True)
    runner.sample_tokens(None)
    runner.execute_model(SchedulerOutput(1), is_profile=True)
    runner.sample_tokens(None)

    assert runner.speculator.steps_executed == [4, 4]
    assert runner.draft_tokens_handler.widths == [4, 4]


def test_runner_maximum_must_remain_four() -> None:
    patch_fakes()
    runner = FakeRunner()
    runner.num_speculative_steps = 5

    with pytest.raises(RuntimeError, match="maximum depth 4"):
        runner.execute_model(SchedulerOutput(2))


def test_nested_selected_state_is_restored() -> None:
    patch_fakes()
    runner = FakeRunner()
    runner.speculator._spark_selected_proposal_steps = 3
    runner.draft_tokens_handler._spark_selected_handler_steps = 3

    runner.execute_model(SchedulerOutput(2))
    runner.sample_tokens(None)

    assert runner.speculator._spark_selected_proposal_steps == 3
    assert runner.draft_tokens_handler._spark_selected_handler_steps == 3


def test_second_execute_before_sample_fails_closed() -> None:
    patch_fakes()
    runner = FakeRunner()

    runner.execute_model(SchedulerOutput(2))
    with pytest.raises(RuntimeError, match="unconsumed execute/sample"):
        runner.execute_model(SchedulerOutput(4))

    # The original pending step remains consumable after the refused overlap.
    result = runner.sample_tokens(None)
    assert result.shape == (2, 2)


def test_empty_scheduler_step_does_not_poison_next_execute() -> None:
    patch_fakes()
    runner = FakeRunner()

    assert runner.execute_model(SchedulerOutput(0, 0)) == "no-forward"
    assert not hasattr(runner, "_spark_pending_proposal_steps")

    runner.execute_model(SchedulerOutput(2))
    result = runner.sample_tokens(None)
    assert result.shape == (2, 2)


def test_non_sampling_execute_return_clears_pending_depth() -> None:
    patch_fakes()
    runner = FakeRunner()
    runner.return_intermediate = True

    assert runner.execute_model(SchedulerOutput(2)) == "intermediate"
    assert not hasattr(runner, "_spark_pending_proposal_steps")

    runner.return_intermediate = False
    runner.execute_model(SchedulerOutput(4))
    result = runner.sample_tokens(None)
    assert result.shape == (2, 4)


def test_contract_requires_exact_environment(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_SPARK_MTP_TOKENS", "4")
    monkeypatch.setenv("VLLM_SPARK_MTP_ADAPTIVE_WINDOW", "32")
    monkeypatch.setenv("VLLM_ADAPTIVE_SPEC_DEPTHS", "2,4")
    assert adapter._attested_contract() == (4, 32, (2, 4))

    monkeypatch.setenv("VLLM_ADAPTIVE_SPEC_DEPTHS", "1,2,4")
    with pytest.raises(RuntimeError, match="exactly 2,4"):
        adapter._attested_contract()


def test_enable_flag_is_strict(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_SPARK_TRUE_ADAPTIVE_DRAFT", raising=False)
    assert adapter._strict_enabled() is False
    monkeypatch.setenv("VLLM_SPARK_TRUE_ADAPTIVE_DRAFT", "1")
    assert adapter._strict_enabled() is True
    monkeypatch.setenv("VLLM_SPARK_TRUE_ADAPTIVE_DRAFT", "yes")
    with pytest.raises(RuntimeError, match="exactly 0 or 1"):
        adapter._strict_enabled()
