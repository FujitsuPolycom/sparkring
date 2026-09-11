"""Check concrete repository contracts without running hosts, builds or model code."""
from __future__ import annotations
import ast
import hashlib
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime.common.profiles import ROOT, catalog, load, local_path, read_json, resolve  # noqa: E402
from scripts.generate_profiles import generate  # noqa: E402


def validate_preserved(root=ROOT):
    record = read_json(root/'runtime/releases/preserved-inputs.json')
    for relative, digest in record['files'].items():
        data = local_path(relative, root).read_bytes()
        # Git's ordinary text checkout conversion is not a release-input edit.
        # Publication-bound documents have explicit eol=lf and stay byte-exact.
        if hashlib.sha256(data).hexdigest() != digest and relative.startswith('performance/') and not Path(relative).name.startswith('r33-image020-'):
            try:
                data.decode('utf-8')
                data = data.replace(b'\r\n', b'\n')
            except UnicodeDecodeError:
                pass
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f'{relative}: preserved release input changed; restore it or introduce a distinct release')
    return len(record['files'])


def validate_imports(root=ROOT):
    paths = []
    for directory in ('runtime', 'integrations', 'scripts', 'spark_transport/fabric'):
        paths.extend((root/directory).rglob('*.py'))
    for path in paths:
        tree = ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
        for node in ast.walk(tree):
            modules = [node.module or ''] if isinstance(node, ast.ImportFrom) else [x.name for x in node.names] if isinstance(node, ast.Import) else []
            if any(m == 'experiments' or m.startswith(('experiments.', 'spark_transport.experiments')) for m in modules):
                raise ValueError(f'{path.relative_to(root)}:{node.lineno}: maintained code must import its maintained owner, not experiments')
    return len(paths)


def validate_build_contracts(root=ROOT):
    builders = read_json(root/'runtime/images/builders.json')
    names = set()
    for row in builders['builders']:
        if row['id'] in names or row['kind'] not in ('python', 'bash') or not row['reason']:
            raise ValueError('Image builders require unique IDs, a supported interpreter and a composition reason')
        names.add(row['id'])
        local_path(row['path'], root)
    # Inspect the literal allowlist without importing lifecycle code.
    install = root/'runtime/glm53-spark-mtp3-mesh/managed_install.py'
    tree = ast.parse(install.read_text(encoding='utf-8-sig'))
    allowlist = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SOURCE_FILES' for t in node.targets):
            allowlist = ast.literal_eval(node.value)
    if not allowlist or len(allowlist) != len(set(allowlist)):
        raise ValueError('Managed installer must have a unique literal source allowlist')
    for relative in allowlist:
        local_path(relative, root)
    required = {'spark_transport/fabric/cx7_hairpin_diagonal/fabric.py', 'integrations/vllm/rocenante/build_bundle.py'}
    if not required <= set(allowlist):
        raise ValueError('Managed service snapshot omits maintained fabric/bundle dependencies')
    for relative in read_json(root/'runtime/public-overlay-files.json')['files']:
        local_path(relative, root)
    return len(names)


def validate_artifacts(root=ROOT):
    paths = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')
    for relative in filter(None, paths):
        path = root/relative
        if path.suffix.lower() in {'.safetensors', '.gguf', '.ggml', '.ckpt', '.pt', '.pth', '.p12', '.pfx'}:
            # Python .pth startup hooks are small source files, not model tensors.
            if path.suffix == '.pth' and path.stat().st_size < 4096:
                continue
            raise ValueError(f'{relative}: model/checkpoint or credential bundle is not a repository source')


def main():
    try:
        for id in catalog():
            load(id)
            resolve(id)
        generated = generate(check=True)
        frozen = validate_preserved()
        imports = validate_imports()
        builders = validate_build_contracts()
        validate_artifacts()
        print(f'Validated {len(catalog())} profiles, {generated} exports, {frozen} preserved inputs, {imports} Python sources and {builders} builders')
    except (ValueError, OSError, KeyError, SyntaxError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
