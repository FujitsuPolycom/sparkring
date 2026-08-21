from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from . import live_installer
from .live_installer import LivePins, LiveQ2RSession, LiveTypes
from .vllm_adapter import AdapterValidationError, source_sha256


@dataclass
class FakeEvent:
    ready: bool = True

    def record(self, stream: Any) -> None:
        assert stream == "stream"

    def query(self) -> bool:
        return self.ready

    def elapsed_time(self, end: Any) -> float:
        del end
        return 1.0


class CudaGraphManager:
    def __init__(
        self,
        vllm_config: Any,
        device: Any,
        cudagraph_mode: Any,
        decode_query_len: int,
        lora_capture_cases: Any = None,
    ) -> None:
        del vllm_config, cudagraph_mode, lora_capture_cases
        self.device = device
        self.decode_query_len = decode_query_len

    def run_fullgraph(self, descriptor: str) -> str:
        return f"full:{descriptor}"

    def run_pw_graph(self, model: Any, model_inputs: str) -> str:
        del model
        return f"piecewise:{model_inputs}"


class ModelCudaGraphManager(CudaGraphManager):
    def run_fullgraph(self, descriptor: str) -> str:
        # Match the deployed override: target dispatch enters this method,
        # which delegates the actual replay to the base implementation.
        return f"model-{super().run_fullgraph(descriptor)}"


class PrefillSpeculatorCudaGraphManager(CudaGraphManager):
    pass


class DecodeSpeculatorCudaGraphManager(CudaGraphManager):
    pass


@dataclass(frozen=True)
class FakeMode:
    name: str


@dataclass(frozen=True)
class FakeBatchDescriptor:
    cg_mode: FakeMode


class AutoRegressiveSpeculator:
    def __init__(self, num_speculative_steps: int) -> None:
        self.device = "cuda:0"
        self.num_speculative_steps = num_speculative_steps
        self.prefill_cudagraph_manager: CudaGraphManager | None = None
        self.decode_cudagraph_manager: CudaGraphManager | None = None
        self.generated: list[int] = []

    def init_cudagraph_manager(self, cudagraph_mode: Any) -> None:
        self.prefill_cudagraph_manager = PrefillSpeculatorCudaGraphManager(
            None,
            "cuda:0",
            cudagraph_mode,
            self.num_speculative_steps + 1,
        )
        self.decode_cudagraph_manager = DecodeSpeculatorCudaGraphManager(
            None, "cuda:0", cudagraph_mode, 1
        )

    def _multi_step_decode(
        self,
        num_reqs: int,
        skip_attn: bool,
        batch_desc: FakeBatchDescriptor,
        num_tokens_across_dp: Any,
    ) -> None:
        del skip_attn, num_tokens_across_dp
        for step in range(1, self.num_speculative_steps):
            if batch_desc.cg_mode.name == "FULL":
                assert self.decode_cudagraph_manager is not None
                self.decode_cudagraph_manager.run_fullgraph(batch_desc)
            else:
                self._generate_draft(
                    num_reqs,
                    num_reqs,
                    None,
                    None,
                    None,
                    batch_desc.cg_mode,
                )

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: Any,
        slot_mappings: Any,
        num_tokens_across_dp: Any,
        cudagraph_runtime_mode: Any,
    ) -> None:
        del (
            num_reqs,
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        self.generated.append(len(self.generated) + 1)


class GPUModelRunner:
    def __init__(self) -> None:
        self.device = "cuda:0"
        self.speculator = AutoRegressiveSpeculator(num_speculative_steps=5)
        self.cudagraph_manager: CudaGraphManager | None = None

    def initialize_kv_cache(self, kv_cache_config: Any) -> None:
        del kv_cache_config
        self.cudagraph_manager = ModelCudaGraphManager(
            None, self.device, "FULL", 6
        )
        self.speculator.init_cudagraph_manager("FULL")

    def execute_model(
        self,
        scheduler_output: Any,
        intermediate_tensors: Any = None,
    ) -> str:
        del scheduler_output, intermediate_tensors
        assert self.cudagraph_manager is not None
        self.cudagraph_manager.run_fullgraph("target")
        return "target-done"

    def sample_tokens(self, grammar_output: Any = None) -> str:
        del grammar_output
        assert self.speculator.prefill_cudagraph_manager is not None
        self.speculator.prefill_cudagraph_manager.run_fullgraph("draft-prefill")
        return "sample-done"


def pins(**overrides: str) -> LivePins:
    values = {
        "vllm_version": "test",
        "cuda_init": source_sha256(CudaGraphManager.__init__),
        "run_fullgraph": source_sha256(CudaGraphManager.run_fullgraph),
        "run_pw_graph": source_sha256(CudaGraphManager.run_pw_graph),
        "model_run_fullgraph": source_sha256(
            ModelCudaGraphManager.run_fullgraph
        ),
        "autoregressive_init": source_sha256(
            AutoRegressiveSpeculator.init_cudagraph_manager
        ),
        "initialize_kv_cache": source_sha256(
            GPUModelRunner.initialize_kv_cache
        ),
        "execute_model": source_sha256(GPUModelRunner.execute_model),
        "sample_tokens": source_sha256(GPUModelRunner.sample_tokens),
        "multi_step_decode": source_sha256(
            AutoRegressiveSpeculator._multi_step_decode
        ),
        "generate_draft": source_sha256(
            AutoRegressiveSpeculator._generate_draft
        ),
    }
    values.update(overrides)
    return LivePins(**values)


def session(
    custom_pins: LivePins | None = None,
    *,
    expected_speculative_steps: int = 5,
    attested_round_depths: tuple[int, ...] | None = None,
    adaptive_window: int | None = None,
) -> LiveQ2RSession:
    return LiveQ2RSession(
        types=LiveTypes(
            cuda_graph_manager=CudaGraphManager,
            model_cuda_graph_manager=ModelCudaGraphManager,
            autoregressive_speculator=AutoRegressiveSpeculator,
            initialization_runner=GPUModelRunner,
            execution_runner=GPUModelRunner,
        ),
        pins=custom_pins or pins(),
        event_factory=FakeEvent,
        current_stream=lambda device: "stream",
        capacity=16,
        expected_speculative_steps=expected_speculative_steps,
        attested_round_depths=attested_round_depths,
        adaptive_window=adaptive_window,
    )


def test_live_session_separates_same_q_target_and_draft() -> None:
    live = session()
    live.install()
    try:
        runner = GPUModelRunner()
        runner.initialize_kv_cache(None)
        live.arm("same-q")
        assert runner.execute_model(None) == "target-done"
        assert runner.sample_tokens() == "sample-done"
        live.disarm()
        assert live.drain().completed == 4
        snapshot = live.snapshot()
        roles = {
            item["role"]
            for item in snapshot["manager_roles"]["identities"]
        }
        assert roles == {
            "target_verify",
            "draft_prefill",
            "draft_decode",
        }
        measured = {
            key: value["count"]
            for key, value in snapshot["phase_timing"][
                "descriptors"
            ].items()
            if value["count"]
        }
        assert measured["step_envelope:execute_model"] == 1
        assert measured["step_envelope:sample_tokens"] == 1
        assert (
            sum(
                value
                for key, value in measured.items()
                if key.startswith("target_full_graph:")
            )
            == 1
        )
        assert (
            sum(
                value
                for key, value in measured.items()
                if key.startswith("draft_multistep_graph:")
            )
            == 1
        )
        assert snapshot["coverage"]["step_envelope"] == {
            "implemented": True,
            "samples": 2,
            "components": {
                "execute_model": 1,
                "sample_tokens": 1,
            },
            "additive": True,
        }
        assert snapshot["coverage"]["graph_methods"]["counts"] == {
            "target_verify": 1,
            "draft_prefill": 1,
            "draft_decode": 0,
        }
        assert [
            sample["descriptor"]
            for sample in snapshot["phase_timing"]["samples"]
        ] == [
            "step_envelope:execute_model",
            next(
                key
                for key in measured
                if key.startswith("target_full_graph:")
            ),
            "step_envelope:sample_tokens",
            next(
                key
                for key in measured
                if "draft_multistep_graph:stage=prefill," in key
            ),
        ]
    finally:
        live.uninstall()


def test_snapshot_finalizes_bound_graph_descriptors_before_arm() -> None:
    live = session()
    live.install()
    try:
        runner = GPUModelRunner()
        runner.initialize_kv_cache(None)

        snapshot = live.snapshot()

        assert snapshot["phase_timing"]["armed"] is False
        assert snapshot["phase_timing"]["epoch"] == ""
        assert snapshot["coverage"]["graph_methods"]["counts"] == {
            "target_verify": 0,
            "draft_prefill": 0,
            "draft_decode": 0,
        }
    finally:
        live.uninstall()


def test_arm_fails_before_explicit_target_and_draft_binding() -> None:
    live = session()
    live.install()
    try:
        with pytest.raises(RuntimeError, match="target manager"):
            live.arm("too-soon")
    finally:
        live.uninstall()


def test_all_source_pins_validate_before_any_patch() -> None:
    original_initialize = GPUModelRunner.initialize_kv_cache
    original_execute = GPUModelRunner.execute_model
    original_sample = GPUModelRunner.sample_tokens
    live = session(pins(run_pw_graph="0" * 64))
    with pytest.raises(AdapterValidationError, match="source mismatch"):
        live.install()
    assert GPUModelRunner.initialize_kv_cache is original_initialize
    assert GPUModelRunner.execute_model is original_execute
    assert GPUModelRunner.sample_tokens is original_sample


def test_sample_tokens_source_pin_fails_before_any_patch() -> None:
    original_base = CudaGraphManager.run_fullgraph
    original_initialize = GPUModelRunner.initialize_kv_cache
    original_execute = GPUModelRunner.execute_model
    original_sample = GPUModelRunner.sample_tokens
    live = session(pins(sample_tokens="0" * 64))
    with pytest.raises(AdapterValidationError, match="source mismatch"):
        live.install()
    assert CudaGraphManager.run_fullgraph is original_base
    assert GPUModelRunner.initialize_kv_cache is original_initialize
    assert GPUModelRunner.execute_model is original_execute
    assert GPUModelRunner.sample_tokens is original_sample


def test_draft_loop_source_pin_fails_before_any_patch() -> None:
    original_base = CudaGraphManager.run_fullgraph
    original_initialize = GPUModelRunner.initialize_kv_cache
    original_generate = AutoRegressiveSpeculator._generate_draft
    original_loop = AutoRegressiveSpeculator._multi_step_decode
    live = session(pins(multi_step_decode="0" * 64))
    with pytest.raises(AdapterValidationError, match="source mismatch"):
        live.install()
    assert CudaGraphManager.run_fullgraph is original_base
    assert GPUModelRunner.initialize_kv_cache is original_initialize
    assert AutoRegressiveSpeculator._generate_draft is original_generate
    assert AutoRegressiveSpeculator._multi_step_decode is original_loop


def test_actual_draft_decode_calls_are_counted_without_assuming_any() -> None:
    live = session()
    live.install()
    try:
        runner = GPUModelRunner()
        runner.initialize_kv_cache(None)
        assert runner.speculator.decode_cudagraph_manager is not None
        live.arm("observed-decode")
        runner.speculator.decode_cudagraph_manager.run_fullgraph("observed")
        live.disarm()
        assert live.drain().completed == 1
        snapshot = live.snapshot()
        assert snapshot["coverage"]["graph_methods"]["counts"][
            "draft_decode"
        ] == 1
    finally:
        live.uninstall()


def test_direct_generate_call_is_unscoped_not_assigned_a_position() -> None:
    live = session()
    live.install()
    try:
        runner = GPUModelRunner()
        runner.initialize_kv_cache(None)
        live.arm("unscoped")
        runner.speculator._generate_draft(
            1, 1, None, None, None, FakeMode("NONE")
        )
        live.disarm()
        assert live.drain().completed == 1
        observed = live.snapshot()["coverage"][
            "draft_decode_generation"
        ]["observed"]
        assert observed[
            "draft_generation:position=unscoped,dispatch=eager"
        ] == 1
        assert not any(
            count
            for key, count in observed.items()
            if "position=unscoped" not in key
        )
    finally:
        live.uninstall()


@pytest.mark.parametrize(
    ("mode", "dispatch"),
    [("NONE", "eager"), ("FULL", "full_graph")],
)
def test_real_draft_generation_calls_get_positions_not_synthetic_steps(
    mode: str, dispatch: str
) -> None:
    live = session()
    live.install()
    try:
        runner = GPUModelRunner()
        runner.initialize_kv_cache(None)
        live.arm(f"draft-{dispatch}")
        runner.speculator._multi_step_decode(
            1,
            False,
            FakeBatchDescriptor(FakeMode(mode)),
            None,
        )
        live.disarm()
        assert live.drain().completed == 4
        snapshot = live.snapshot()
        coverage = snapshot["coverage"]["draft_decode_generation"]
        assert coverage["configured_speculative_steps"] == 5
        assert coverage["position_zero"] == "draft_prefill"
        assert coverage["expected_loop_positions"] == [1, 2, 3, 4]
        assert coverage["complete_iteration"] is False
        observed = {
            key: value
            for key, value in coverage["observed"].items()
            if value
        }
        assert observed == {
            f"draft_generation:position={position},dispatch={dispatch}": 1
            for position in range(1, 5)
        }
        assert not any("position=5," in key for key in coverage["observed"])
    finally:
        live.uninstall()


def test_mtp4_records_exactly_three_continuation_positions() -> None:
    live = session(expected_speculative_steps=4)
    live.install()
    try:
        runner = GPUModelRunner()
        runner.speculator.num_speculative_steps = 4
        runner.initialize_kv_cache(None)
        live.arm("draft-mtp4")
        runner.speculator._multi_step_decode(
            1,
            False,
            FakeBatchDescriptor(FakeMode("NONE")),
            None,
        )
        live.disarm()
        assert live.drain().completed == 3
        coverage = live.snapshot()["coverage"]["draft_decode_generation"]
        assert coverage["configured_speculative_steps"] == 4
        assert coverage["expected_loop_positions"] == [1, 2, 3]
        assert {
            key: value
            for key, value in coverage["observed"].items()
            if value
        } == {
            f"draft_generation:position={position},dispatch=eager": 1
            for position in range(1, 4)
        }
    finally:
        live.uninstall()


def test_adaptive_mtp2_round_records_only_its_real_continuation() -> None:
    live = session(
        expected_speculative_steps=4,
        attested_round_depths=(2, 4),
        adaptive_window=32,
    )
    live.install()
    try:
        runner = GPUModelRunner()
        runner.speculator.num_speculative_steps = 4
        runner.initialize_kv_cache(None)
        live.arm("adaptive-mtp2")
        runner.speculator.num_speculative_steps = 2
        runner.speculator._multi_step_decode(
            1,
            False,
            FakeBatchDescriptor(FakeMode("NONE")),
            None,
        )
        live.disarm()
        assert live.drain().completed == 1
        coverage = live.snapshot()["coverage"]["draft_decode_generation"]
        assert coverage["configured_speculative_steps"] == 4
        assert coverage["attested_round_depths"] == [2, 4]
        assert coverage["adaptive_window"] == 32
        assert coverage["completed_rounds_by_depth"] == {"2": 1, "4": 0}
        assert {
            key: value
            for key, value in coverage["observed"].items()
            if value
        } == {
            "draft_generation:position=1,dispatch=eager": 1,
        }
    finally:
        live.uninstall()


def test_reset_clears_adaptive_round_depth_counts() -> None:
    live = session(
        expected_speculative_steps=4,
        attested_round_depths=(2, 4),
        adaptive_window=32,
    )
    live.install()
    try:
        runner = GPUModelRunner()
        runner.speculator.num_speculative_steps = 4
        runner.initialize_kv_cache(None)
        live.arm("adaptive-before-reset")
        runner.speculator.num_speculative_steps = 2
        runner.speculator._multi_step_decode(
            1,
            False,
            FakeBatchDescriptor(FakeMode("NONE")),
            None,
        )
        live.disarm()
        assert live.drain().completed == 1
        live.reset()
        coverage = live.snapshot()["coverage"]["draft_decode_generation"]
        assert coverage["completed_rounds_by_depth"] == {"2": 0, "4": 0}
    finally:
        live.uninstall()


def test_unarmed_warmup_does_not_enter_adaptive_depth_evidence() -> None:
    live = session(
        expected_speculative_steps=4,
        attested_round_depths=(2, 4),
        adaptive_window=32,
    )
    live.install()
    try:
        runner = GPUModelRunner()
        runner.speculator.num_speculative_steps = 4
        runner.initialize_kv_cache(None)
        runner.speculator.num_speculative_steps = 2
        runner.speculator._multi_step_decode(
            1,
            False,
            FakeBatchDescriptor(FakeMode("NONE")),
            None,
        )
        coverage = live.snapshot()["coverage"]["draft_decode_generation"]
        assert coverage["completed_rounds_by_depth"] == {"2": 0, "4": 0}
    finally:
        live.uninstall()


def test_adaptive_depth_mix_records_only_observed_positions() -> None:
    live = session(
        expected_speculative_steps=4,
        attested_round_depths=(2, 4),
        adaptive_window=32,
    )
    live.install()
    try:
        runner = GPUModelRunner()
        runner.speculator.num_speculative_steps = 4
        runner.initialize_kv_cache(None)
        live.arm("adaptive-depth-mix")
        for depth in (2, 4):
            runner.speculator.num_speculative_steps = depth
            runner.speculator._multi_step_decode(
                1,
                False,
                FakeBatchDescriptor(FakeMode("NONE")),
                None,
            )
        live.disarm()
        assert live.drain().completed == 4
        coverage = live.snapshot()["coverage"]["draft_decode_generation"]
        assert coverage["completed_rounds_by_depth"] == {"2": 1, "4": 1}
        assert {
            key: value
            for key, value in coverage["observed"].items()
            if value
        } == {
            "draft_generation:position=1,dispatch=eager": 2,
            "draft_generation:position=2,dispatch=eager": 1,
            "draft_generation:position=3,dispatch=eager": 1,
        }
    finally:
        live.uninstall()


def test_adaptive_round_fails_closed_before_unattested_depth_runs() -> None:
    live = session(
        expected_speculative_steps=4,
        attested_round_depths=(2, 4),
        adaptive_window=32,
    )
    live.install()
    try:
        runner = GPUModelRunner()
        runner.speculator.num_speculative_steps = 4
        runner.initialize_kv_cache(None)
        live.arm("adaptive-invalid-depth")
        runner.speculator.num_speculative_steps = 3
        with pytest.raises(
            RuntimeError, match="unattested adaptive draft depth 3"
        ):
            runner.speculator._multi_step_decode(
                1,
                False,
                FakeBatchDescriptor(FakeMode("NONE")),
                None,
            )
        live.disarm()
        assert live.drain().completed == 0
        coverage = live.snapshot()["coverage"]["draft_decode_generation"]
        assert coverage["completed_rounds_by_depth"] == {"2": 0, "4": 0}
        assert not any(coverage["observed"].values())
    finally:
        live.uninstall()


def test_unattested_depth_fails_closed_before_startup_binding() -> None:
    live = session(
        expected_speculative_steps=4,
        attested_round_depths=(2, 4),
        adaptive_window=32,
    )
    live.install()
    try:
        runner = GPUModelRunner()
        runner.speculator.num_speculative_steps = 3
        with pytest.raises(
            RuntimeError, match="unattested adaptive draft depth 3"
        ):
            runner.speculator._multi_step_decode(
                1,
                False,
                FakeBatchDescriptor(FakeMode("NONE")),
                None,
            )
    finally:
        live.uninstall()


@pytest.mark.parametrize(("value", "expected"), [("4", 4), ("5", 5)])
def test_expected_speculative_steps_comes_from_launch_environment(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
    expected: int,
) -> None:
    monkeypatch.setenv("VLLM_SPARK_MTP_TOKENS", value)
    assert live_installer._expected_speculative_steps() == expected


@pytest.mark.parametrize("depth", [4, 5])
def test_fixed_depth_attestation_remains_single_depth(
    monkeypatch: pytest.MonkeyPatch, depth: int
) -> None:
    monkeypatch.setenv("VLLM_SPARK_MTP_TOKENS", str(depth))
    monkeypatch.setenv("VLLM_SPARK_MTP_ADAPTIVE_WINDOW", "0")
    monkeypatch.delenv("VLLM_ADAPTIVE_SPEC_DEPTHS", raising=False)
    attestation = live_installer._depth_attestation()
    assert attestation.configured_speculative_steps == depth
    assert attestation.attested_round_depths == (depth,)
    assert attestation.adaptive_window is None


def test_adaptive_depth_attestation_requires_2_4_and_window_32(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_SPARK_MTP_TOKENS", "4")
    monkeypatch.setenv("VLLM_SPARK_MTP_ADAPTIVE_WINDOW", "32")
    monkeypatch.setenv("VLLM_ADAPTIVE_SPEC_DEPTHS", "2,4")
    attestation = live_installer._depth_attestation()
    assert attestation.configured_speculative_steps == 4
    assert attestation.attested_round_depths == (2, 4)
    assert attestation.adaptive_window == 32


@pytest.mark.parametrize(
    ("configured", "window", "depths", "message"),
    [
        ("4", "31", "2,4", "must be 0 or 32"),
        ("5", "32", "2,4", "requires VLLM_SPARK_MTP_TOKENS=4"),
        ("4", "32", "2,3,4", "must attest exactly 2,4"),
        ("4", "32", "4,2", "must attest exactly 2,4"),
        ("4", "32", "", "must attest exactly 2,4"),
    ],
)
def test_adaptive_depth_attestation_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    configured: str,
    window: str,
    depths: str,
    message: str,
) -> None:
    monkeypatch.setenv("VLLM_SPARK_MTP_TOKENS", configured)
    monkeypatch.setenv("VLLM_SPARK_MTP_ADAPTIVE_WINDOW", window)
    monkeypatch.setenv("VLLM_ADAPTIVE_SPEC_DEPTHS", depths)
    with pytest.raises(RuntimeError, match=message):
        live_installer._depth_attestation()


@pytest.mark.parametrize("value", ["", "3", "6", "five"])
def test_expected_speculative_steps_fails_closed(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("VLLM_SPARK_MTP_TOKENS", value)
    with pytest.raises(RuntimeError, match="must be 4 or 5"):
        live_installer._expected_speculative_steps()


def test_model_override_is_pinned_but_not_double_wrapped() -> None:
    original_base = CudaGraphManager.run_fullgraph
    original_model = ModelCudaGraphManager.run_fullgraph
    live = session(pins(model_run_fullgraph="0" * 64))
    with pytest.raises(AdapterValidationError, match="source mismatch"):
        live.install()
    assert CudaGraphManager.run_fullgraph is original_base
    assert ModelCudaGraphManager.run_fullgraph is original_model


def test_load_types_uses_one_deployed_gpu_runner_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported: list[str] = []
    cuda_module = SimpleNamespace(
        CudaGraphManager=CudaGraphManager,
        ModelCudaGraphManager=ModelCudaGraphManager,
    )
    autoregressive_module = SimpleNamespace(
        AutoRegressiveSpeculator=AutoRegressiveSpeculator
    )
    runner_module = SimpleNamespace(GPUModelRunner=GPUModelRunner)
    modules = {
        "vllm.v1.worker.gpu.cudagraph_utils": cuda_module,
        (
            "vllm.v1.worker.gpu.spec_decode.autoregressive.speculator"
        ): autoregressive_module,
        "vllm.v1.worker.gpu.model_runner": runner_module,
    }

    def fake_import(name: str) -> Any:
        imported.append(name)
        return modules[name]

    monkeypatch.setattr(live_installer.importlib, "import_module", fake_import)
    types = live_installer._load_types()
    assert types.initialization_runner is GPUModelRunner
    assert types.execution_runner is GPUModelRunner
    assert "vllm.v1.worker.gpu_model_runner" not in imported
