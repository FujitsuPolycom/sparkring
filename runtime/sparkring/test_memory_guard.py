import subprocess
import unittest
from memory_guard import GuardState, terminate_guarded_containers


class GuardTests(unittest.TestCase):
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
