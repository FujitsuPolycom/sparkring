"""Real Linux descriptor checks; no NIC, GPU, mount or service changes."""
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).parent))
from sparkring_runtime_status import resources


@unittest.skipUnless(sys.platform == 'linux' and hasattr(os, 'O_PATH'), 'Linux O_PATH required')
class DescriptorSamplingTest(unittest.TestCase):
    def test_root_samples_an_open_local_descriptor(self):
        data, device, kind = resources.descriptor_stats('/')
        self.assertIn(kind, resources.LOCAL_FILESYSTEMS)
        self.assertEqual(device, os.stat('/').st_dev)
        self.assertGreater(data.f_blocks, 0)

    def test_symlink_components_never_reach_fstatvfs(self):
        with tempfile.TemporaryDirectory(prefix='sparkring-status-test-') as tmp:
            base = Path(tmp)
            (base / 'target/child').mkdir(parents=True)
            (base / 'link').symlink_to(base / 'target', target_is_directory=True)
            for path in (base / 'link', base / 'link/child'):
                with self.subTest(path=str(path)), self.assertRaises(resources.UnprovenFilesystem):
                    resources.descriptor_stats(str(path))

    def test_path_replacement_after_admission_cannot_redirect_the_sample(self):
        with tempfile.TemporaryDirectory(prefix='sparkring-status-race-') as tmp:
            base = Path(tmp)
            original, replacement, moved = base / 'original', base / 'replacement', base / 'moved'
            original.mkdir()
            replacement.mkdir()
            inode = original.stat().st_ino
            flags = {key: getattr(os, key) for key in ('O_PATH', 'O_DIRECTORY', 'O_NOFOLLOW', 'O_CLOEXEC')}
            def sample(fd):
                original.rename(moved)
                original.symlink_to(replacement, target_is_directory=True)
                self.assertEqual(os.fstat(fd).st_ino, inode)
                self.assertNotEqual(original.stat().st_ino, inode)
                return os.fstatvfs(fd)
            ops = SimpleNamespace(**flags, open=os.open, close=os.close, fstat=os.fstat, fstatvfs=sample)
            before = len(list(Path('/proc/self/fd').iterdir()))
            resources.descriptor_stats(str(original), ops=ops)
            self.assertEqual(len(list(Path('/proc/self/fd').iterdir())), before)


if __name__ == '__main__':
    unittest.main()
