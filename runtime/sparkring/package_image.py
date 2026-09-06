"""Build a hash-checked packaging candidate without touching serving containers."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def verify_assets(manifest, assets):
    for item in manifest['files']:
        path = assets / item['name']
        if path.name != item['name'] or not path.is_file():
            raise ValueError(f'Missing or invalid asset: {item["name"]}')
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError(f'Hash mismatch: {item["name"]}')


def dockerfile(manifest):
    # Clear inherited support/provenance claims that are not qualifications
    # of this child. The manifest is authoritative; parent history is retained.
    lines = ['ARG BASE_IMAGE=local/sparkring-package-base:required', 'FROM ${BASE_IMAGE}',
             'HEALTHCHECK NONE', 'COPY manifest.json /opt/sparkring/runtime-manifest.json',
             'COPY DISTRIBUTION.md /usr/share/licenses/SparkRing/DISTRIBUTION.md',
             'COPY verify_image.py /opt/sparkring/bin/verify-runtime-package.py']
    for item in manifest['files']:
        lines.append('COPY ' + json.dumps(['assets/' + item['name'], item['destination']]))
    lines.extend([
        'RUN python3 /opt/sparkring/bin/verify-runtime-package.py',
        'LABEL org.opencontainers.image.title="SparkRing runtime"',
        'LABEL org.opencontainers.image.source="https://github.com/FujitsuPolycom/sparkring"',
        'LABEL org.opencontainers.image.description="Experimental ARM64 SM121 runtime; support is profile-specific"',
        'LABEL org.sparkring.runtime.status="research-only"',
        'LABEL org.sparkring.runtime.manifest="/opt/sparkring/runtime-manifest.json"',
        'LABEL org.sparkcache.dcp-layouts="unqualified-in-packaged-child"',
        'LABEL org.sparkring.nccl.topology="profile-defined"',
        'LABEL org.sparkring.loader="profile-defined"',
        'LABEL org.sparkcache.cuda-placement.status="packaged-hash-verified"',
        'LABEL org.sparkcache.source-revision="' + manifest.get('sparkcache_source_revision', 'unrecorded') + '"',
        'LABEL org.sparkcache.cuda-placement-sha256="' + next((x['sha256'] for x in manifest['files'] if x['name'] == 'libspark_cache_placement.so'), 'unrecorded') + '"',
        'LABEL org.sparkcache.cuda-snapshot-sha256="' + next((x['sha256'] for x in manifest['files'] if x['name'] == 'libspark_cache_snapshot.so'), 'unrecorded') + '"',
        'ENTRYPOINT ["vllm", "serve"]',
        'CMD ["--help"]',
    ])
    return '\n'.join(lines) + '\n'


def prepare(manifest_path, assets, context):
    manifest = json.loads(manifest_path.read_text())
    verify_assets(manifest, assets)
    # Exclusive creation prevents overwriting a prior package/receipt.
    context.mkdir(parents=True, exist_ok=False)
    (context / 'assets').mkdir()
    for item in manifest['files']:
        shutil.copyfile(assets / item['name'], context / 'assets' / item['name'])
    shutil.copyfile(manifest_path, context / 'manifest.json')
    shutil.copyfile(Path(__file__).with_name('verify_image.py'), context / 'verify_image.py')
    shutil.copyfile(Path(__file__).with_name('DISTRIBUTION.md'), context / 'DISTRIBUTION.md')
    (context / 'Dockerfile').write_text(dockerfile(manifest))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--context', type=Path, required=True)
    parser.add_argument('--build', action='store_true')
    args = parser.parse_args()
    manifest = prepare(Path(__file__).with_name('manifest.json'), args.assets, args.context)
    local_tag = 'local/sparkring:' + manifest['candidate_tag']
    base_tag = 'local/sparkring-package-base:' + manifest['parent_image_id'].split(':')[1]
    command = ['docker', 'build', '--network=none', '--build-arg',
               'BASE_IMAGE=' + base_tag, '-t', local_tag, str(args.context)]
    print(json.dumps({'command': command, 'publishes': False, 'starts_serving': False}))
    if args.build:
        observed = subprocess.check_output(['docker', 'image', 'inspect', '--format', '{{.Id}}', manifest['parent_image_id']], text=True).strip()
        if observed != manifest['parent_image_id']:
            raise ValueError('Parent image drift')
        # BuildKit parses a raw sha256:... image ID as a repository name.
        # A content-named local tag keeps FROM resolvable without claiming a
        # registry digest. Verify the tag resolves to the exact parent ID.
        subprocess.run(['docker', 'tag', observed, base_tag], check=True)
        tagged = subprocess.check_output(['docker', 'image', 'inspect', '--format', '{{.Id}}', base_tag], text=True).strip()
        if tagged != observed:
            raise ValueError('Local parent tag drift')
        subprocess.run(command, check=True)
        image_id = subprocess.check_output(['docker', 'image', 'inspect', '--format', '{{.Id}}', local_tag], text=True).strip()
        receipt = {'schema': 'sparkring-package-build/v1', 'image_id': image_id,
                   'local_tag': local_tag, 'manifest_sha256': hashlib.sha256((args.context / 'manifest.json').read_bytes()).hexdigest(),
                   'status': 'research-only', 'gpu_qualified': False, 'published': False}
        (args.context / 'build-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        print(json.dumps(receipt))


if __name__ == '__main__':
    main()
