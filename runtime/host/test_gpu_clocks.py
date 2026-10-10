"""Each rank's GPU SM clock and clock event reasons in sparkring check; offline."""
from runtime.host import gpu_clocks


def test_a_query_line_names_its_clocks_and_active_reasons():
    assert gpu_clocks.parse("2411, 3003, 0x0000000000000000\n") == {"sm_mhz": 2411, "max_sm_mhz": 3003,
                                                                    "reasons": []}
    assert gpu_clocks.parse("721, [N/A], 0x0000000000000088") == {
        "sm_mhz": 721, "max_sm_mhz": None, "reasons": ["hw_slowdown", "hw_power_brake_slowdown"]}
    assert gpu_clocks.parse("208, 3003, 0x0000000000010001")["reasons"] == ["gpu_idle", "0x10000"]


def test_reasons_other_than_idle_and_a_low_busy_clock_need_attention():
    assert gpu_clocks.attention({"sm_mhz": 2405, "max_sm_mhz": 3003, "reasons": []}) == []
    # An idle GPU lowers its own clock.
    assert gpu_clocks.attention({"sm_mhz": 208, "max_sm_mhz": 3003, "reasons": ["gpu_idle"]}) == []
    assert gpu_clocks.attention({"sm_mhz": 721, "max_sm_mhz": 3003, "reasons": []}) == [
        "SM clock 721 MHz, below half of its 3003 MHz maximum"]
    assert gpu_clocks.attention({"sm_mhz": 2405, "max_sm_mhz": None, "reasons": ["sw_thermal_slowdown"]}) == [
        "clock event reasons sw_thermal_slowdown"]


def test_an_unreadable_gpu_is_unknown_not_needing_attention():
    def run(host, argv):
        if host == "operator@192.0.2.11":
            raise RuntimeError("ssh: connect to host 192.0.2.11 port 22: Connection refused")
        return "2405, 3003, 0x0000000000000000"
    rows = gpu_clocks.check([{"rank": 0, "host": "operator@192.0.2.10"}, {"rank": 1, "host": "operator@192.0.2.11"}],
                            run=run)
    assert rows[1] == {"rank": 1, "error": "ssh: connect to host 192.0.2.11 port 22: Connection refused",
                       "attention": []}
    assert gpu_clocks.lines(rows) == ["GPU clocks: SM 2405 MHz on 1 ranks; no clock event reason other than idle",
                                      "GPU clocks unknown on rank 1: ssh: connect to host 192.0.2.11 port 22: "
                                      "Connection refused"]
