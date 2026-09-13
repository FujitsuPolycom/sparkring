"""Verify inherited bytes, install candidate admission inputs and record the payload."""
import base64
import csv
import hashlib
from importlib import metadata
import json
from pathlib import Path
import shutil

ROOT = Path('/opt/sparkring')
SITE = Path('/opt/venv/lib/python3.12/site-packages')
CONTEXT = Path('/r35-context')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')


def main():
    stage = json.loads((ROOT/'receipts/r35-source-stage.json').read_text())
    parent = json.loads((ROOT/'receipts/source-lock.json').read_text())
    payload = json.loads((ROOT/'receipts/installed-python-files.json').read_text())
    allowed = {str(SITE/name):sha for name, sha in stage['installed_source_files'].items()}
    inherited = {'/'+name:sha for name, sha in parent['installed_files'].items()}
    inherited.update({str(Path('/opt/venv')/name):sha for name, sha in payload['files'].items()})
    for name, expected in inherited.items():
        actual = digest(Path(name))
        if actual != allowed.get(name, expected):
            raise ValueError(f'unexpected inherited payload change: {name}')
    for name, expected in allowed.items():
        if digest(Path(name)) != expected:
            raise ValueError(f'candidate source mismatch: {name}')
    removed = []
    for component in ('vllm', 'b12x'):
        for name in (CONTEXT/f'r33-{component}-files.txt').read_text().splitlines():
            path = (SITE/name).resolve()
            if not path.is_relative_to(SITE/component):
                raise ValueError('baseline source path escapes component')
            if name not in stage['installed_source_files'] and path.is_file():
                path.unlink()
                removed.append(str(path))

    # Frozen records remain available as parent provenance, not candidate proof.
    shutil.copy2(ROOT/'receipts/source-lock.json', ROOT/'receipts/r33-parent-source-lock.json')
    components = stage['components']
    version = '0.26.1rc0+sparkring.r35.' + components['vllm']['tree'][:8]
    (SITE/'vllm/_version.py').write_text(
        f'__version__ = version = {version!r}\n'
        f'__version_tuple__ = version_tuple = (0, 26, 1, "rc0", "sparkring.r35.{components["vllm"]["tree"][:8]}")\n'
        '__commit_id__ = commit_id = None\n')
    distribution = metadata.distribution('vllm')
    metadata_path = Path(distribution._path)/'METADATA'
    metadata_path.write_text(metadata_path.read_text().replace('Version: '+distribution.version+'\n', 'Version: '+version+'\n'))

    shutil.copytree(CONTEXT/'profile-contract', ROOT/'profile-contract', dirs_exist_ok=True)
    # Keep parent qualification records as provenance, separate from R35 evidence.
    shutil.move(str(ROOT/'profile-contract/evidence'), str(ROOT/'receipts/r33-parent-profile-evidence'))
    shutil.copytree(CONTEXT/'capability-evidence', ROOT/'profile-contract/evidence')
    if (CONTEXT/'tp2-sparkcache-capabilities.json').exists():
        shutil.copy2(CONTEXT/'tp2-sparkcache-capabilities.json', ROOT/'profile-contract/tp2-sparkcache-capabilities.json')
    contract_path = ROOT/'profile-contract/profile-contract.json'
    contract = json.loads(contract_path.read_text())
    artifact_path = ROOT/'image/artifact-lock.json'
    artifact = json.loads(artifact_path.read_text())
    for sources in (contract['image']['required_sources'], artifact['source_identities']):
        sources.update(vllm_head=components['vllm']['base_commit'],
                       vllm_integrated_tree=components['vllm']['tree'],
                       b12x_commit=components['b12x']['base_commit'], b12x_tree=components['b12x']['tree'])
    artifact['r35_source_composition'] = components
    save(artifact_path, artifact)
    contract['image']['artifact_lock_sha256'] = digest(artifact_path)
    contract['sparkcache_native']['lease_contract'] = '/opt/sparkring/contracts/vllm-connector-jobs-r35.json'
    contract['candidate_qualification'] = 'Runtime qualification pending; no R33 serving evidence is transferred.'
    save(contract_path, contract)
    shutil.copy2(CONTEXT/'vllm-connector-jobs.json', ROOT/'contracts/vllm-connector-jobs-r35.json')
    shutil.copy2(ROOT/'bin/sparkring-r33', ROOT/'bin/r35-profile-admission.py')
    shutil.copy2(CONTEXT/'entrypoint.py', ROOT/'bin/sparkring')
    shutil.copy2(CONTEXT/'verify_image.py', ROOT/'bin/verify-r35.py')

    # Recompute installed wheel RECORD hashes after replacing authored payloads.
    for component in ('vllm', 'b12x'):
        dist = metadata.distribution(component)
        record = Path(dist._path)/'RECORD'
        with record.open(newline='') as stream:
            names = {row[0] for row in csv.reader(stream)}
        names.update(name for name in stage['installed_source_files'] if name.startswith(component+'/'))
        with record.open('w', newline='') as stream:
            writer = csv.writer(stream)
            for name in sorted(names):
                path = (SITE/name).resolve()
                if not path.is_relative_to(Path('/opt/venv')):
                    raise ValueError('wheel RECORD escapes virtual environment')
                if path == record.resolve():
                    writer.writerow((name, '', ''))
                elif path.is_file():
                    raw = bytes.fromhex(digest(path))
                    writer.writerow((name, 'sha256='+base64.urlsafe_b64encode(raw).decode().rstrip('='), path.stat().st_size))

    names = set(inherited) | set(allowed)
    names.update(str(path) for path in (ROOT/'bin').iterdir() if path.is_file())
    names.update(str(path) for path in (ROOT/'profile-contract').rglob('*') if path.is_file())
    names.update(str(path) for path in (ROOT/'receipts/r33-parent-profile-evidence').rglob('*') if path.is_file())
    names.update((str(contract_path), str(artifact_path), str(ROOT/'contracts/vllm-connector-jobs-r35.json'),
                  str(metadata_path), str(SITE/'vllm/_version.py')))
    for component in ('vllm', 'b12x'):
        names.add(str(Path(metadata.distribution(component)._path)/'RECORD'))
    save(ROOT/'receipts/r35-installed.json', {'schema':'sparkring-r35-installed/v1',
         'components':components, 'parent_source_lock_sha256':digest(ROOT/'receipts/r33-parent-source-lock.json'),
         'files':{name:digest(Path(name)) for name in sorted(names) if Path(name).is_file()},
         'removed_authored_files':removed,
         'versions':{**parent['python_closure'], 'vllm':version},
         'qualification':'candidate; model and distributed qualification required'})


if __name__ == '__main__':
    main()
