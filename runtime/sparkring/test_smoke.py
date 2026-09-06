import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / 'profiles/glm53-flash-spark-tp2/smoke.py'
spec = importlib.util.spec_from_file_location('tp2_smoke', path)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class SmokeTests(unittest.TestCase):
    def test_png_fixture_is_deterministic(self):
        data = smoke.png_blue()
        self.assertTrue(data.startswith(b'\x89PNG\r\n\x1a\n'))
        self.assertEqual(data, smoke.png_blue())
        self.assertLess(len(data), 4096)


if __name__ == '__main__':
    unittest.main()
