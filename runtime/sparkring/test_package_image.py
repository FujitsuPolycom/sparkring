import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from package_image import dockerfile, prepare, verify_assets


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.assets = self.root / 'assets'
        self.assets.mkdir()
        (self.assets / 'a.so').write_bytes(b'fixture')
        self.manifest = {'files': [{'name': 'a.so', 'destination': '/opt/a.so',
                                   'sha256': hashlib.sha256(b'fixture').hexdigest()}]}

    def test_tamper_rejected(self):
        (self.assets / 'a.so').write_bytes(b'bad')
        with self.assertRaises(ValueError):
            verify_assets(self.manifest, self.assets)

    def test_missing_rejected(self):
        (self.assets / 'a.so').unlink()
        with self.assertRaises(ValueError):
            verify_assets(self.manifest, self.assets)

    def test_prepares_without_overwriting(self):
        path = self.root / 'manifest.json'
        path.write_text(json.dumps(self.manifest))
        context = self.root / 'context'
        prepare(path, self.assets, context)
        self.assertEqual((context / 'assets/a.so').read_bytes(), b'fixture')
        with self.assertRaises(FileExistsError):
            prepare(path, self.assets, context)

    def test_model_neutral_command_and_no_broken_healthcheck(self):
        content = dockerfile(self.manifest)
        self.assertIn('HEALTHCHECK NONE', content)
        self.assertIn('CMD ["--help"]', content)
        self.assertNotIn('/models/target', content)


if __name__ == '__main__':
    unittest.main()
