from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _function_body(source: str, signature: str) -> str:
    start = source.index(signature)
    opening = source.index("{", start)
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[opening + 1 : index]
    raise AssertionError(f"unterminated function: {signature}")


def test_striped_progress_joins_each_dual_port_phase_before_advancing() -> None:
    source = (ROOT / "src" / "tp4_session.cpp").read_text(encoding="utf-8")
    body = _function_body(source, "void progress_dual_port_striped(")

    ordered_markers = (
        "&phase1_endpoint0_control->producer_sequence",
        "&phase1_endpoint1_control->producer_sequence",
        "post_striped_transfer(*endpoint0_",
        "post_striped_transfer(*endpoint1_",
        "endpoint0_->wait_for_send_through(phase1_work_id)",
        "endpoint1_->wait_for_send_through(phase1_work_id)",
        "&phase1_endpoint0_control->consumer_sequence",
        "&phase1_endpoint1_control->consumer_sequence",
        "post_striped_credit(*endpoint0_",
        "post_striped_credit(*endpoint1_",
        "&phase2_endpoint0_control->producer_sequence",
        "&phase2_endpoint1_control->producer_sequence",
        "post_striped_transfer(*endpoint0_",
        "post_striped_transfer(*endpoint1_",
        "endpoint0_->wait_for_send_through(phase2_work_id)",
        "endpoint1_->wait_for_send_through(phase2_work_id)",
        "&phase2_endpoint0_control->consumer_sequence",
        "&phase2_endpoint1_control->consumer_sequence",
        "post_striped_credit(*endpoint0_",
        "post_striped_credit(*endpoint1_",
        "last_credit_work_id0_ = phase2_credit_work_id",
        "last_credit_work_id1_ = phase2_credit_work_id",
    )
    cursor = 0
    for marker in ordered_markers:
        position = body.find(marker, cursor)
        assert position >= 0, marker
        cursor = position + len(marker)

    graph_progress = _function_body(source, "void progress_graph_commands()")
    assert graph_progress.index("progress(command.sequence") < graph_progress.index(
        "tp4_graph_command_complete(graph_commands_host_, command.sequence)"
    )


def test_striped_setup_uses_a_distinct_exact_arena_handshake() -> None:
    source = (ROOT / "src" / "tp4_session.cpp").read_text(encoding="utf-8")
    constructor = _function_body(source, "explicit Impl(Tp4AllreduceOptions options)")
    handshake = _function_body(source, "void exchange_and_connect_endpoint(")

    assert "tp4_allreduce_schedule_valid(options_.schedule)" in constructor
    assert "tp4_dual_port_striped_options_valid(options_)" in constructor
    assert "make_tp4_striped_endpoint_layout(options_.payload_bytes)" in constructor
    assert "arena_bytes_ = striped_layout_.total_bytes" in constructor
    assert constructor.count("options_.protocol, options_.schedule") == 2
    assert "options_.graph_kernel_strategy, options_.schedule" in constructor

    assert "kTp4DualPortStripedEndpointVersion" in handshake
    assert "kTp4DualPortStripedEndpointTag" in handshake
    assert "remote.buffer_bytes != local.buffer_bytes" in handshake
    assert handshake.index("remote.version != local.version") < handshake.index(
        "endpoint.connect(remote, local.version)"
    )


def test_striped_teardown_reaps_both_qps_then_all_lane_generations() -> None:
    source = (ROOT / "src" / "tp4_session.cpp").read_text(encoding="utf-8")
    body = _function_body(source, "void drain_striped_deferred_credits()")

    endpoint0_reap = body.index("endpoint0_->wait_for_send(last_credit_work_id0_)")
    endpoint1_reap = body.index("endpoint1_->wait_for_send(last_credit_work_id1_)")
    generation_loop = body.index("generation < kTp4StripedGenerationCount")
    assert endpoint0_reap < endpoint1_reap < generation_loop
    assert body.count("tp4_latest_slot_sequence(") == 2
    assert "Tp4TensorStripe::kLowerHalf" in body
    assert "Tp4TensorStripe::kUpperHalf" in body
    assert body.count("->acknowledgement_sequence") == 2
    assert body.count("striped deferred-credit retirement") == 2


def test_striped_schedule_has_an_additive_attested_c_abi() -> None:
    header = (ROOT / "include" / "spark_transport" / "tp4_c_api.h").read_text(
        encoding="utf-8"
    )
    source = (ROOT / "src" / "tp4_c_api.cpp").read_text(encoding="utf-8")
    constructor = _function_body(
        source,
        "spark_tp4_create_with_protocol_graph_kernel_and_schedule(",
    )
    legacy_constructor = _function_body(
        source,
        'extern "C" spark_tp4_handle\n'
        "spark_tp4_create_with_protocol_and_graph_kernel(",
    )
    status = _function_body(source, 'extern "C" int spark_tp4_get_graph_status(')

    assert "SPARK_TP4_WIRE_SCHEDULE_SEQUENTIAL = 0" in header
    assert "SPARK_TP4_WIRE_SCHEDULE_DUAL_PORT_STRIPED = 1" in header
    assert "SPARK_TP4_GRAPH_STATUS_DUAL_PORT_STRIPED = 1U << 10" in header
    assert "options.schedule = schedule_from_wire(wire_schedule)" in constructor
    assert "SPARK_TP4_WIRE_SCHEDULE_SEQUENTIAL" in legacy_constructor
    assert "SPARK_TP4_GRAPH_STATUS_DUAL_PORT_STRIPED" in status
