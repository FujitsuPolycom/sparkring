"""Install authenticated Python source payloads into an R33-derived build stage.

This stage is not serving-ready: the final image must install its R35 profile
and complete installed-file receipt before enabling the serving entrypoint.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

CONTEXT = Path('/context')
SITE = Path('/opt/venv/lib/python3.12/site-packages')


def main():
    lock = json.loads((CONTEXT/'source-composition.json').read_text())
    installed = {}
    for component, record in lock['components'].items():
        if component not in ('vllm', 'b12x'):
            raise ValueError('unexpected source component')
        archive = CONTEXT/record['archive']
        if archive.parent != CONTEXT or hashlib.sha256(archive.read_bytes()).hexdigest() != record['archive_sha256']:
            raise ValueError(f'{component}: archive identity mismatch')
        with tarfile.open(archive) as source:
            members = [m for m in source.getmembers() if m.name.startswith(component+'/') and not m.isdir()]
            for member in members:
                relative = Path(member.name)
                if relative.is_absolute() or '..' in relative.parts or not member.isfile():
                    raise ValueError(f'unsupported source member: {member.name}')
                target = SITE/relative
                if not target.resolve().is_relative_to(SITE.resolve()):
                    raise ValueError('source destination escapes package directory')
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.extractfile(member) as incoming, target.open('wb') as output:
                    shutil.copyfileobj(incoming, output)
                target.chmod(member.mode)
                installed[member.name] = hashlib.sha256(target.read_bytes()).hexdigest()
    (Path('/opt/sparkring/receipts')/'r35-source-stage.json').write_text(json.dumps({
        'schema':'sparkring-r35-source-stage/v1', 'status':'integration-tests-required',
        'components':lock['components'], 'installed_source_files':installed}, indent=2)+'\n')
    print(f'Installed {len(installed)} authenticated source files; serving admission remains disabled')


if __name__ == '__main__':
    main()
