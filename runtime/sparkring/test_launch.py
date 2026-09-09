import importlib.util
from pathlib import Path
import tempfile
import unittest

path = Path(__file__).resolve().parents[1] / 'profiles/glm53-flash-spark-tp2/launch.py'
spec = importlib.util.spec_from_file_location('tp2_launch', path)
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


class LaunchTests(unittest.TestCase):
    def test_create_only_pinned_rank(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            env = root / 'runtime.env'
            env.write_text('VLLM_HOST_IP=example\nNCCL_SOCKET_IFNAME=example\nGLOO_SOCKET_IFNAME=example\n')
            model, cache = root / 'model', root / 'cache'
            model.mkdir()
            cache.mkdir()
            (model / 'config.json').write_text('{}')
            image = 'ghcr.io/fujitsupolycom/sparkring@sha256:' + 'a'*64
            plan = launch.render(1, 'master.example', model, cache, env, image)
            name, command = plan['name'], plan['command']
            self.assertEqual(command[:2], ['docker', 'create'])
            self.assertIn('--headless', command)
            self.assertIn('master.example', command)
            self.assertNotIn('rm', command)
            self.assertNotIn('${NODE_RANK}', command)
            self.assertEqual(name, 'sparkring-glm53-tp2-r1')
            with self.assertRaises(ValueError):
                launch.render(1, 'master.example', model, cache, env, 'latest')
            env.write_text('VLLM_HOST_IP=<replace>')
            with self.assertRaises(ValueError):
                launch.render(1, 'master.example', model, cache, env, image)


if __name__ == '__main__':
    unittest.main()
