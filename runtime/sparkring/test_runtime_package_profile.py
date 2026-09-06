import json
from pathlib import Path
import unittest


class RuntimePackageProfileTests(unittest.TestCase):
    def test_profile_matches_final_settings_and_is_sanitized(self):
        root = Path(__file__).resolve().parents[1]
        path = root / 'profiles/glm53-flash-spark-tp2/profile.json'
        source = path.read_text()
        profile = json.loads(source)
        args = profile['vllm_args']
        def value(flag):
            return args[args.index(flag) + 1]
        self.assertEqual(value('--max-num-batched-tokens'), '8192')
        self.assertEqual(value('--prefill-schedule-interval'), '2')
        self.assertEqual(value('--kv-cache-memory-bytes'), '5368709120')
        self.assertEqual(value('--max-num-seqs'), '8')
        self.assertEqual(value('--master-addr'), '${MASTER_ADDR}')
        self.assertEqual(profile['status'], 'research-only')
        self.assertNotIn('192.168.', source)
        cache = json.loads(value('--kv-transfer-config'))
        self.assertEqual(cache['kv_load_failure_policy'], 'recompute')
        self.assertTrue(cache['kv_connector_extra_config']['spark_cache_cuda_restore'])
        manifest = json.loads((root / 'sparkring/manifest.json').read_text())
        self.assertIsNone(manifest['registry_digest'])
        self.assertTrue(manifest['publication_blockers'])

    def test_publication_and_sbom_are_content_bound(self):
        import gzip
        import hashlib
        root = Path(__file__).resolve().parent
        receipt = json.loads((root / 'publication.json').read_text())
        self.assertTrue(receipt['published'])
        self.assertFalse(receipt['checks']['gpu_serving_qualified_child'])
        self.assertEqual(receipt['visibility'], 'public')
        with gzip.open(root / receipt['sbom']['path'], 'rb') as stream:
            content = stream.read()
        self.assertEqual(hashlib.sha256(content).hexdigest(), receipt['sbom']['uncompressed_sha256'])
        self.assertEqual(len(json.loads(content)['packages']), receipt['sbom']['package_entries'])


if __name__ == '__main__':
    unittest.main()
