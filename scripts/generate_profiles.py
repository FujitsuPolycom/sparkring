"""Regenerate compatibility exports and the deployment table from authored inputs."""
from __future__ import annotations
import argparse
import os
import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime.common.profiles import ROOT, catalog, load, local_path, read_json, resolve, legacy_recipe_bytes  # noqa: E402

from runtime.common.environment import render_environment  # noqa: E402

START = '<!-- BEGIN GENERATED PROFILES -->'
END = '<!-- END GENERATED PROFILES -->'
STATUS_LABELS = {
    'qualified': 'Validated',
    'implemented': 'Development',
    'research-only': 'Experimental',
    'unsupported': 'Unsupported',
}


def compact_tokens(tokens):
    """Round display counts in decimal millions or thousands; retain exact source values."""
    if tokens >= 1_000_000:
        return f"{tokens / 1_000_000:.1f}".rstrip('0').rstrip('.') + 'M'
    if tokens >= 1_000:
        return f"{tokens / 1_000:.0f}K"
    return str(tokens)


def profile_table(root=ROOT, *, compact=False):
    rows = [(load(id, root)[0], resolve(id, root=root)) for id in catalog(root)]
    capacity = read_json(root/'performance/profile-capacity.json')['profiles']
    if not set(capacity) <= {p['id'] for p, _ in rows}:
        raise ValueError('Capacity records must name catalog profiles')
    for record in capacity.values():
        if type(record['tokens']) is not int or record['tokens'] <= 0 or not record['conditions']:
            raise ValueError('Capacity records require positive token counts and measurement conditions')
        if record['witness'] not in local_path(record['source'], root).read_text(encoding='utf-8-sig'):
            raise ValueError(f"Capacity evidence changed: {record['source']}")
    lines = [START, '', 'Configured context is a per-request limit, not measured KV capacity or a completed long-context test.',
             'Development profiles are under active development; validated profiles have documented checks for the selected configuration. See each guide for the exact testing scope.', '']
    if compact:
        lines = [START, '']
    for title, predicate in (
        ('Four Sparks', lambda p, r: p['recommendation'] != 'retired' and r['serving']['node_count'] == 4),
        ('Two Sparks', lambda p, r: p['recommendation'] != 'retired' and r['serving']['node_count'] == 2),
        ('Retired profiles', lambda p, r: p['recommendation'] == 'retired'),
    ):
        if compact and title == 'Retired profiles':
            continue
        lines += ['### '+title, '', '| Model / features | Layout | Configured context (tokens) | KV (tokens) | Status | Navigation | Quickstart |', '|---|---|---:|---:|---|---|---|']
        if compact:
            lines[-2:] = ['| Model / features | Layout | Context (tokens) | KV (tokens) | Status | Quickstart |', '|---|---|---:|---:|---|---|']
        for p, r in sorted(rows, key=lambda pair: (pair[0]['recommendation'] != 'recommended', pair[0]['id'])):
            if not predicate(p, r):
                continue
            s = r['serving']
            context = f"{s['max_model_len']:,}" if 'max_model_len' in s else '—'
            record = capacity.get(p['id'])
            kv = f"[{record['tokens']:,}]({record['source']})" if record else '—'
            if compact:
                context = compact_tokens(s['max_model_len']) if 'max_model_len' in s else '—'
                kv = f"[{compact_tokens(record['tokens'])}]({record['source']})" if record else '—'
            if record and not compact:
                kv = f"[{record['tokens']:,}](../{record['source']})"
            if record and record.get('approximate'):
                kv = kv.replace('[', '[~', 1)
            if compact:
                title = f"**{p['title']}**" if p['recommendation'] == 'recommended' else p['title']
                lines.append(f"| {title} | TP{s['tensor_parallel_size']}/DCP{s['decode_context_parallel_size']} | {context} | {kv} | {STATUS_LABELS[p['status']]} | [Guide](profiles/{p['id']}/README.md) |")
                continue
            lines.append(f"| {p['title']} | TP{s['tensor_parallel_size']}/DCP{s['decode_context_parallel_size']} | {context} | {kv} | {STATUS_LABELS[p['status']]} | {p['recommendation']} | [Guide](profiles/{p['id']}/README.md) |")
        lines.append('')
    if compact:
        return '\n'.join(lines + [END])
    lines += ['Additional pinned historical variants, including original NVFP4 TP2, are in the [retained deployment index](docs/history/deployment-variants.md).', '', 'Qualification applies only to the exact image, checkpoint, topology and workload in the selected guide.',
              'Switched deployments have no switched-hardware qualification. Qwen with SparkCache is unsupported;',
              'six-node work remains research-only and is outside this deployment catalog.', '', END]
    text = '\n'.join(lines)
    return text.replace('](profiles/', '](').replace('](docs/', '](../docs/')


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
        content = legacy_recipe_bytes(row["source"], row["destination"], root) if row["kind"] == "recipe" else source.read_bytes()
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
    for relative, compact in (('README.md', True), ('profiles/README.md', False)):
        readme = root/relative
        text = readme.read_text(encoding='utf-8-sig')
        if START not in text or END not in text:
            raise ValueError(f'{relative} requires generated profile region markers')
        before, tail = text.split(START, 1)
        _, after = tail.split(END, 1)
        expected[readme] = (before+profile_table(root, compact=compact)+after).encode()
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
