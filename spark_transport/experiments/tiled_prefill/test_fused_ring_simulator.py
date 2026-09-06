from .fused_ring_simulator import classify_token, simulate


def test_randomized_dual_rail_skew_closes_every_flow() -> None:
    for seed in range(500):
        complete, flows = simulate(seed)
        assert complete
        assert flows == 8


def test_token_faults_fail_closed() -> None:
    assert classify_token(8, 9, 7) == "pending"
    assert classify_token(9, 9, 7) == "ready"
    assert classify_token(10, 9, 7) == "fatal_future"
    assert classify_token(6, 9, 7) == "fatal_regression"
