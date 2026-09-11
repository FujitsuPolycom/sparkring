import contextlib
import io
import subprocess
import sys
import unittest
from unittest.mock import patch
from memory_guard import GuardState, parse_args, terminate_guarded_containers


class GuardTests(unittest.TestCase):
    def test_nonfinite_durations_are_rejected_before_monitoring(self):
        for option in ("--poll-seconds", "--term-grace-seconds", "--trip-cooldown-seconds"):
            for value in ("nan", "inf", "-inf"):
                with self.subTest(option=option, value=value):
                    errors = io.StringIO()
                    with patch.object(sys, "argv", ["memory_guard.py", f"{option}={value}"]), contextlib.redirect_stderr(errors):
                        with self.assertRaises(SystemExit) as raised:
                            parse_args()
                    self.assertEqual(raised.exception.code, 2)
                    self.assertIn(f"{option} must be finite", errors.getvalue())

    def test_finite_poll_and_zero_grace_cooldown_are_supported(self):
        with patch.object(sys, "argv", ["memory_guard.py", "--poll-seconds=0.25",
                                       "--term-grace-seconds=0", "--trip-cooldown-seconds=0"]):
            args = parse_args()
        self.assertEqual((args.poll_seconds, args.term_grace_seconds,
                          args.trip_cooldown_seconds), (0.25, 0, 0))

    def test_requires_consecutive_low_samples(self):
        state = GuardState()
        def observe(n):
            return state.observe(n, floor_bytes=100, required_samples=2)
        self.assertFalse(observe(99))
        self.assertFalse(observe(100))
        self.assertFalse(observe(99))
        self.assertTrue(observe(99))

    def test_only_labeled_containers_and_bounded_stop(self):
        calls = []
        def run(command):
            calls.append(command)
            return subprocess.CompletedProcess(command, 0, 'abc\n' if command[1] == 'ps' else '', '')
        self.assertEqual(terminate_guarded_containers(grace_seconds=0, run=run), ('abc',))
        self.assertIn('label=org.sparkring.memory-guard=true', calls[0])
        self.assertEqual(calls[1], ('docker', 'kill', '--signal=TERM', 'abc'))
        self.assertEqual(calls[2], ('docker', 'kill', '--signal=KILL', 'abc'))


if __name__ == '__main__':
    unittest.main()
