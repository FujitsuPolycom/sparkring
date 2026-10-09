#!/usr/bin/env python3
"""CPU tests of tools/site_routes.py's point-to-point windows, against SIRCL's route planner and budget.

The tool imports SIRCL's package, so the tests need it: SIRCL_PACKAGE names the directory that holds
``sparkring_sircl`` (for example the implementation tree's ``spark_transport/sircl``). Without it every test
is skipped, so a host without SIRCL's tree still runs ``make check``; the tests never use another copy of the
package that happens to be importable.

Cases, two lanes each:

- path:4-7 without ``--ring-schedules``: all 6 relayed ordered pairs have a point-to-point window on every
  lane (3 channels); with it, the ring window of rank 3's ring lanes is reserved and 3 ordered pairs have none;
- ring:8 with the session's forward windows of today: none of the 40 relayed ordered pairs has a window;
  capped at one 32 KiB chunk (``--max-window 32768``), and with ``--session-share`` 0.25, 0.5 and 0.95, all 40
  do; at 0.95 every relayed session lane is one chunk smaller (32 to 96 KiB) and every channel window 32 KiB;
- ``--session-share 1`` prints exactly what the default prints; shares outside (0, 1] are refused;
- in every case, no relay hairpin queue holds more than its share (75 % of 512 KiB): the session's windows
  through it (forward windows, and with ``--ring-schedules`` the ring windows) plus the channels' windows;
  and, as the check's negative control, ``--p2p-reserve none`` on ring:8 overfills some queue;
- the ring:8 files of tests/data equal the tool's output (default, and ``--p2p-reserve none``).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = os.environ.get("SIRCL_PACKAGE", "")
SHARE = 393216  # 75 % of SIRCL's 512 KiB hairpin queue (routes.RELAY_QUEUE_SHARE x DEFAULT_HAIRPIN_QUEUE)


def plan(*arguments: str) -> dict:
    environment = dict(os.environ, PYTHONPATH=PACKAGE, PYTHONIOENCODING="utf-8")
    process = subprocess.run([sys.executable, str(ROOT / "tools" / "site_routes.py"), *arguments, "--json"],
                             env=environment, capture_output=True, text=True, timeout=120)
    if process.returncode:
        raise AssertionError(f"site_routes.py {' '.join(arguments)}: {process.stderr}")
    return json.loads(process.stdout)


def windows(text: str) -> dict[int, list[int]]:
    """``<position>=<bytes>[/<bytes>],...`` as {position: [bytes per lane]}."""
    found = {}
    for entry in filter(None, text.split(",")):
        peer, _, values = entry.partition("=")
        found[int(peer)] = [int(value) for value in values.split("/")]
    return found


@unittest.skipUnless(PACKAGE and (Path(PACKAGE) / "sparkring_sircl").is_dir(),
                     "SIRCL_PACKAGE does not name a directory holding sparkring_sircl")
class SiteRoutesP2PTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, PACKAGE)
        from sparkring_sircl import routes

        cls.routes = routes

    def queues_within_share(self, result: dict, layout_text: str, ring_reserved: bool) -> None:
        """No relay queue holds more than its share: session windows (forward, and ring when reserved) plus the
        channels' windows of every lane through it."""
        routes = self.routes
        layout = routes.Layout.parse(layout_text)
        group = routes.derive_routes(layout, 2)
        maps = [group.route_map(rank) for rank in range(layout.world)]
        forward, p2p, ring = {}, {}, {}
        for row in result["ranks"]:
            rank, env = row["rank"], row["env"]
            for peer, lanes in windows(env.get("LIBSIRCL_FORWARD_WINDOWS", "")).items():
                for lane, value in enumerate(lanes):
                    forward[(rank, peer, lane)] = value
            for peer, lanes in windows(env.get("LIBSIRCL_P2P_WINDOWS", "")).items():
                for lane, value in enumerate(lanes):
                    p2p[(rank, peer, lane)] = value
            ring[rank] = int(env.get("LIBSIRCL_RING_WINDOW", "0"))
        reserved = {key: sum(forward.get(member, 0) for member in members)
                    for key, members in routes.relay_queues(layout, maps).items()}
        if ring_reserved:
            order = routes.chain_order(layout, maps)
            for key, members in routes.ring_queues(layout, maps, order).items():
                reserved[key] = max(reserved.get(key, 0), sum(ring[rank] for rank, _, _ in members))
        for key, members in routes.relay_queues(layout, maps).items():
            held = reserved.get(key, 0) + sum(p2p.get(member, 0) for member in members)
            self.assertLessEqual(held, SHARE, f"{layout_text}: relay queue {key} holds {held} bytes")

    def test_path_of_four_without_ring_schedules_gives_every_relayed_pair_a_channel(self):
        result = plan("--layout", "path:4-7", "--lanes", "2")
        self.assertEqual(result["p2p_relayed_pairs"], 6)
        self.assertEqual(result["p2p_unavailable"], [])
        self.assertFalse(result["ring_schedules"])
        for row in result["ranks"]:
            env = row["env"]
            relayed = set(windows(env.get("LIBSIRCL_FORWARD_WINDOWS", "")))
            granted = windows(env.get("LIBSIRCL_P2P_WINDOWS", ""))
            self.assertEqual(set(granted), relayed, f"rank {row['rank']}")
            self.assertTrue(all(value >= 32768 for lanes in granted.values() for value in lanes))
        self.queues_within_share(result, "path:4-7", ring_reserved=False)

    def test_path_of_four_with_ring_schedules_reserves_the_ring_window(self):
        result = plan("--layout", "path:4-7", "--lanes", "2", "--ring-schedules")
        self.assertTrue(result["ring_schedules"])
        self.assertEqual(sorted((a, b) for a, b, _ in result["p2p_unavailable"]), [(2, 0), (3, 0), (3, 1)])
        self.assertEqual(result["ranks"][3]["env"]["LIBSIRCL_RING_WINDOW"], "393216")
        self.queues_within_share(result, "path:4-7", ring_reserved=True)

    def test_ring_of_eight_today_leaves_no_relayed_pair_a_channel(self):
        result = plan("--layout", "ring:8", "--lanes", "2")
        self.assertEqual(result["p2p_relayed_pairs"], 40)
        self.assertEqual(len(result["p2p_unavailable"]), 40)
        self.assertTrue(all("LIBSIRCL_P2P_WINDOWS" not in row["env"] for row in result["ranks"]))
        self.queues_within_share(result, "ring:8", ring_reserved=False)

    def test_ring_of_eight_with_session_windows_of_one_chunk_gives_all_forty(self):
        for arguments in (("--max-window", "32768"), ("--session-share", "0.25"), ("--session-share", "0.5"),
                          ("--session-share", "0.95")):
            with self.subTest(arguments=arguments):
                result = plan("--layout", "ring:8", "--lanes", "2", *arguments)
                self.assertEqual(result["p2p_relayed_pairs"], 40)
                self.assertEqual(result["p2p_unavailable"], [])
                forward = {value for row in result["ranks"]
                           for lanes in windows(row["env"].get("LIBSIRCL_FORWARD_WINDOWS", "")).values()
                           for value in lanes}
                self.assertEqual(min(forward), 32768)
                if arguments in (("--max-window", "32768"), ("--session-share", "0.25")):
                    self.assertEqual(forward, {32768})
                if arguments == ("--session-share", "0.95"):
                    self.assertEqual(forward, {32768, 65536, 98304})
                    granted = {value for row in result["ranks"]
                               for lanes in windows(row["env"].get("LIBSIRCL_P2P_WINDOWS", "")).values()
                               for value in lanes}
                    self.assertEqual(granted, {32768})
                self.queues_within_share(result, "ring:8", ring_reserved=False)

    def test_the_share_check_sees_queues_that_reserve_nothing_for_the_session(self):
        # Negative control: --p2p-reserve none sizes the channels as the only relayed traffic, so the session's
        # forward windows and the channels' windows together exceed some relay queue's share.
        result = plan("--layout", "ring:8", "--lanes", "2", "--p2p-reserve", "none")
        with self.assertRaises(AssertionError):
            self.queues_within_share(result, "ring:8", ring_reserved=False)

    def test_a_session_share_of_one_is_the_default(self):
        for layout in ("path:4-7", "ring:8"):
            with self.subTest(layout=layout):
                default = plan("--layout", layout, "--lanes", "2")
                whole = plan("--layout", layout, "--lanes", "2", "--session-share", "1")
                self.assertEqual(default["ranks"], whole["ranks"])
                self.assertEqual(default["p2p_unavailable"], whole["p2p_unavailable"])

    def test_session_shares_outside_zero_to_one_are_refused(self):
        for value in ("0", "1.5", "-0.25"):
            with self.subTest(value=value):
                process = subprocess.run([sys.executable, str(ROOT / "tools" / "site_routes.py"), "--layout", "ring:8",
                                          "--session-share", value],
                                         env=dict(os.environ, PYTHONPATH=PACKAGE), capture_output=True, text=True,
                                         timeout=120)
                self.assertEqual(process.returncode, 2)
                self.assertIn("--session-share", process.stderr)

    def test_the_ring_of_eight_data_files_are_the_tools_output(self):
        for name, arguments in (("site_routes_ring8_l2.json", ()),
                                ("site_routes_ring8_l2_p2p_alone.json", ("--p2p-reserve", "none"))):
            with self.subTest(file=name):
                recorded = json.loads((ROOT / "tests" / "data" / name).read_text())
                self.assertEqual(recorded, plan("--layout", "ring:8", "--lanes", "2", *arguments))


if __name__ == "__main__":
    unittest.main(verbosity=2)
