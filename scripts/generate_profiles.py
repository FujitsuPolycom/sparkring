"""Regenerate compatibility exports and the deployment table from authored inputs."""
from __future__ import annotations
import argparse
import os
import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime.common.profiles import ROOT, catalog, load, local_path, read_json, resolve  # noqa: E402

from runtime.common.environment import render_environment  # noqa: E402

START = '<!-- BEGIN GENERATED PROFILES -->'
END = '<!-- END GENERATED PROFILES -->'


def profile_table(root=ROOT):
    rows = [(load(id, root)[0], resolve(id, root=root)) for id in catalog(root)]
    lines = [START, '', 'Configured context is a per-request limit, not measured KV capacity or a completed long-context test.',
             'Status describes the evidence scope; recommendation describes deployment navigation.', '']
    for title, predicate in (
        ('Four Sparks', lambda p, r: p['recommendation'] != 'retired' and r['serving']['node_count'] == 4),
        ('Two Sparks', lambda p, r: p['recommendation'] != 'retired' and r['serving']['node_count'] == 2),
        ('Retired GLM-5.3 profiles', lambda p, r: p['recommendation'] == 'retired'),
    ):
        lines += ['### '+title, '', '| Model / features | Layout | Configured context (tokens) | Status | Navigation | Quickstart |', '|---|---|---:|---|---|---|']
        for p, r in sorted(rows, key=lambda pair: (pair[0]['recommendation'] != 'recommended', pair[0]['id'])):
            if not predicate(p, r):
                continue
            s = r['serving']
            lines.append(f"| {p['title']} | TP{s['tensor_parallel_size']}/DCP{s['decode_context_parallel_size']} | {s.get('max_model_len', '—')} | {p['status']} | {p['recommendation']} | [Guide](profiles/{p['id']}/README.md) |")
        lines.append('')
    lines += ['Additional pinned historical variants, including original NVFP4 TP2, are in the [retained deployment index](docs/history/deployment-variants.md).', '', 'Qualification applies only to the exact image, checkpoint, topology and workload in the selected guide.',
              'Switched deployments have no switched-hardware qualification. Qwen with SparkCache is unsupported;',
              'six-node work remains research-only and is outside this deployment catalog.', '', END]
    return '\n'.join(lines)


def generate(check=False, root=ROOT):
    expected = {}
    manifest = read_json(root/'profiles/compatibility.json')
    seen = set()
    for row in manifest['mirrors']:
        if set(row) != {'source', 'destination', 'kind'} or row['destination'] in seen:
            raise ValueError('Compatibility destinations must be unique and name source, destination and kind')
        seen.add(row['destination'])
        target = (root/row['destination']).resolve()
        if not target.is_relative_to(root.resolve()) or target == local_path(row['source'], root):
            raise ValueError('Compatibility export must stay within the checkout and differ from source')
        source = local_path(row['source'], root)
        content = source.read_bytes()
        if source.suffix == '.md':
            def relative_link(match):
                value = match[1]
                if re.match(r'(?:[a-zA-Z]+:|#|/)', value):
                    return match[0]
                path, marker, anchor = value.partition('#')
                relative = os.path.relpath((source.parent/path).resolve(), target.parent).replace('\\', '/')
                return ']('+relative+('#'+anchor if marker else '')+')'
            content = re.sub(r'\]\(([^\s)]+)\)', relative_link, content.decode('utf-8')).encode('utf-8')
        expected[target] = content
    for row in read_json(root/'profiles/environment-exports.json')['exports']:
        target = (root/row['destination']).resolve()
        if not target.is_relative_to(root.resolve()) or target in expected:
            raise ValueError('Environment exports must have unique destinations within the checkout')
        expected[target] = render_environment(row['profile'], root=root, template_only=True).encode('utf-8')
    readme = root/'README.md'
    text = readme.read_text(encoding='utf-8-sig')
    if START not in text or END not in text:
        raise ValueError('README.md requires generated profile region markers')
    before, tail = text.split(START, 1)
    _, after = tail.split(END, 1)
    expected[readme] = (before+profile_table(root)+after).encode()
    stale = []
    for path, content in expected.items():
        if not path.exists() or path.read_bytes().replace(b'\r\n', b'\n') != content.replace(b'\r\n', b'\n'):
            stale.append(path.relative_to(root).as_posix())
            if not check:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
    if check and stale:
        raise ValueError('Run python scripts/generate_profiles.py; stale exports: '+', '.join(stale))
    return len(expected)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    try:
        print(f'Checked {generate(args.check)} generated files')
    except (ValueError, OSError) as error:
        parser.exit(1, str(error)+'\n')
