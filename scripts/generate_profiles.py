"""Regenerate compatibility exports and the deployment table from authored inputs."""
from __future__ import annotations
import argparse
import os
import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime.common.profiles import ROOT, catalog, load, local_path, read_json, resolve, legacy_recipe_bytes, quickstart_status  # noqa: E402

from runtime.common.environment import render_environment  # noqa: E402
from runtime.common import generic_model, thinking  # noqa: E402

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


def deployment_family(resolved):
    return (resolved['model']['repository'], resolved['topology'],
            resolved['serving']['node_count'], resolved['runtime'].get('engine', 'vllm'))


def quickstart_path(profile):
    """Use authored serving guides while retaining release-specific entry pages."""
    if profile['configuration']['format'] == 'serving-profile':
        return profile['guide']
    # Recipe/release-profile guide fields can name retained release documents.
    # Their per-profile README is the maintained user entry point.
    return f"profiles/{profile['id']}/README.md"


def developing_cache_profiles(root=ROOT):
    """Read table annotations without changing a profile's serving settings."""
    path = root / 'profiles/capabilities.json'
    if not path.is_file():
        return set()
    data = read_json(path)
    if (set(data) != {'schema', 'profiles'} or data['schema'] != 'sparkring-capability-status/v1'
            or not isinstance(data['profiles'], dict) or not set(data['profiles']) <= set(catalog(root))
            or any(value != {'sparkcache': 'development'} for value in data['profiles'].values())):
        raise ValueError('Capability annotations require catalog profiles and an explicit development status')
    return set(data['profiles'])


def compact_profile_rows(rows, root=ROOT):
    """Group explicit cache compositions with their base, retaining the default's values."""
    recipe_ids = {p['configuration']['path']: p['id'] for p, _ in rows
                  if p['configuration']['format'] == 'recipe'}
    groups = {}
    rows_by_id = {p['id']: (p, resolved) for p, resolved in rows}
    for p, resolved in rows:
        config = p['configuration']
        key, cached = p['id'], False
        if config['format'] == 'recipe':
            recipe = read_json(local_path(config['path'], root))
            if recipe.get('base_recipe'):
                key = recipe_ids[recipe['base_recipe']]
                cached = True
        elif config['format'] == 'release-profile':
            key = (config['path'], config['key'].removesuffix('-sparkcache'))
            cached = config['key'].endswith('-sparkcache')
            # Cache options may be promoted to different releases independently.
            # Pair named variants for navigation without transferring their evidence.
            base_id = p['id'].removesuffix('-sparkcache') if cached else p['id']
            other_id = base_id if cached else base_id + '-sparkcache'
            if other_id in rows_by_id:
                other, other_resolved = rows_by_id[other_id]
                other_config = other['configuration']
                expected_key = config['key'].removesuffix('-sparkcache') if cached else config['key'] + '-sparkcache'
                if (other_config['format'] == 'release-profile'
                        and other['recommendation'] != 'retired'
                        and p['recommendation'] != 'retired'
                        and other_config['key'] == expected_key
                        and deployment_family(other_resolved) == deployment_family(resolved)
                        and other_resolved['serving']['decode_context_parallel_size'] == resolved['serving']['decode_context_parallel_size']):
                    key = ('release-cache-pair', base_id)
        elif config['format'] == 'serving-profile':
            data = read_json(local_path(config['path'], root))
            cached = resolved['serving'].get('sparkcache', False)
            base = data.get('base_profile')
            if base:
                if base not in rows_by_id or not cached:
                    raise ValueError('A cache variant must identify a catalog base profile')
                base_profile, base_resolved = rows_by_id[base]
                if (base_profile['configuration']['format'] != 'serving-profile'
                        or deployment_family(base_resolved) != deployment_family(resolved)
                        or base_resolved['model'] != resolved['model']):
                    raise ValueError('A cache variant must retain its base deployment family')
                key = base
        groups.setdefault(key, []).append((p, resolved, cached))
    result, cache_cells = [], {}
    developing = developing_cache_profiles(root)
    for variants in groups.values():
        variants.sort(key=lambda item: (item[0]['recommendation'] != 'recommended', item[2]))
        p, resolved, cached = variants[0]
        alternative = next((v for v in variants[1:] if v[2] != cached), None)
        cell = 'No'
        if alternative:
            cache_profile = p if cached else alternative[0]
            cell = f"[Optional]({quickstart_path(cache_profile)})"
        elif cached:
            cell = 'Included'
        elif p['id'] in developing:
            cell = '(in dev)'
        cache_cells[p['id']] = cell
        result.append((p, resolved))
    # A recommended configuration represents its model/topology on the landing
    # page. Other DCP choices remain in the full catalog and profile guide.
    preferred = {deployment_family(v)
                 for p, v in result if p['recommendation'] == 'recommended'}
    result = [(p, v) for p, v in result if p['recommendation'] == 'recommended'
              or deployment_family(v) not in preferred]
    return result, cache_cells


def check_capacity_records(capacity, root=ROOT):
    """Refuse a KV capacity measurement that its cited evidence does not state.

    ``capacity`` maps profile IDs to records of performance/profile-capacity.json.
    A record, and each measurement in its ``checkpoints``, needs a positive
    token count, its measurement conditions, and a ``source`` file in this
    repository that contains its ``witness`` text. A record may state the KV
    bytes per rank at measurement; a checkpoint measurement must, because the
    Install Builder scales it to the KV cache size a user chooses.
    """
    def positive(value):
        return type(value) is int and value > 0

    def evidenced(entry, label):
        if not positive(entry.get('tokens')) or not entry.get('conditions'):
            raise ValueError(f'{label}: capacity records require positive token counts and measurement conditions')
        witness = entry.get('witness')
        if not isinstance(witness, str) or not witness:
            raise ValueError(f'{label}: capacity records require a source file and the witness text it contains')
        if witness not in local_path(entry.get('source'), root).read_text(encoding='utf-8-sig'):
            raise ValueError(f"{label}: capacity evidence changed: {entry['source']}")

    for profile, record in capacity.items():
        evidenced(record, profile)
        if 'kv_bytes_per_rank' in record and not positive(record['kv_bytes_per_rank']):
            raise ValueError(f'{profile}: capacity records state KV bytes per rank as a positive integer')
        for name, entry in (record.get('checkpoints') or {}).items():
            label = f'{profile} checkpoint {name}'
            evidenced(entry, label)
            if not positive(entry.get('kv_bytes_per_rank')):
                raise ValueError(f'{label}: checkpoint capacity records require KV bytes per rank')


def profile_table(root=ROOT, *, compact=False):
    # The generic templates serve a model named at install time, so they have no row.
    rows = [(load(id, root)[0], resolve(id, root=root)) for id in catalog(root) if not generic_model.is_generic(id)]
    model_labels = read_json(root/'profiles/model-names.json')
    names = model_labels['models']
    quant_labels = model_labels['quant_labels']
    for _, resolved in rows:
        repository = resolved['model']['repository']
        if repository not in names or repository not in quant_labels or not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repository):
            raise ValueError(f'Model repository needs a standard display name: {repository}')
        if repository.split('/')[0] not in model_labels.get('publishers', {}):
            raise ValueError(f'Model repository needs a credited publisher: {repository}')
    capacity = read_json(root/'performance/profile-capacity.json')['profiles']
    if not compact:
        return profile_catalog_table(rows, names, root)
    if not set(capacity) <= {p['id'] for p, _ in rows}:
        raise ValueError('Capacity records must name catalog profiles')
    check_capacity_records(capacity, root)
    lines = [START, '', 'Configured context is a per-request limit, not measured KV capacity or a completed long-context test.',
             'Development profiles are under active development; validated profiles have documented checks for the selected configuration. See each guide for the exact testing scope.', '']
    if compact:
        dcp_options = {}
        dcp_capacity = {}
        for profile, resolved in rows:
            if profile['recommendation'] == 'retired':
                continue
            key = deployment_family(resolved)
            dcp = resolved['serving']['decode_context_parallel_size']
            dcp_options.setdefault(key, set()).add(dcp)
            if profile['id'] in capacity:
                rank = (profile['recommendation'] == 'recommended', profile['status'] == 'qualified')
                existing = dcp_capacity.get((key, dcp))
                if existing is None or rank > existing[0]:
                    dcp_capacity[(key, dcp)] = (rank, capacity[profile['id']])
        rows, cache_cells = compact_profile_rows(rows, root)
        lines = [START, '']
    for title, predicate in (
        ('Four Sparks', lambda p, r: p['recommendation'] != 'retired' and r['serving']['node_count'] == 4),
        ('Two Sparks', lambda p, r: p['recommendation'] != 'retired' and r['serving']['node_count'] == 2),
        ('Retired profiles', lambda p, r: p['recommendation'] == 'retired'),
    ):
        if compact and title == 'Retired profiles':
            continue
        lines += ['### '+title, '', '| Model | Quant | Runtime | Layout | Context / KV* (tokens) | Status | Navigation | Quickstart |', '|---|---|---|---|---|---|---|---|']
        if compact:
            lines[-2:] = ['| Model | Quant | DCP | Context / KV* | SparkCache | Status |', '|---|---|---|---|---|---|']
        for p, r in sorted(rows, key=lambda pair: (pair[0]['recommendation'] != 'recommended', pair[0]['id'])):
            if not predicate(p, r) or (compact and r['topology'] == 'switched'):
                continue
            s = r['serving']
            engine = r['runtime'].get('engine', 'vllm')
            engine_title = {'vllm': 'vLLM', 'sglang': 'SGLang'}[engine]
            repository = r['model']['repository']
            model_name = names[repository]
            variant = model_labels.get('quant_variants', {}).get(repository + '@' + r['model'].get('revision', ''))
            quant_url = f"https://huggingface.co/{repository}"
            if variant:
                quant_url += '/tree/' + r['model']['revision']
            # The checkpoint's publisher is credited under its link.
            publisher = model_labels['publishers'][repository.split('/')[0]]
            quant = f"[{variant or quant_labels[repository]}]({quant_url})<br>by {publisher}"
            layout = f"TP{s['tensor_parallel_size']}/DCP{s['decode_context_parallel_size']}"
            if engine == 'sglang':
                layout = f"TP{s['tensor_parallel_size']}/EP{s['expert_parallel_size']}"
            if r['topology'] == 'switched':
                layout += ' · switched'
            context = f"{s['max_model_len']:,}" if 'max_model_len' in s else '—'
            record = capacity.get(p['id'])
            kv = f"[{record['tokens']:,}]({record['source']})" if record else '—'
            if compact:
                key = deployment_family(r)
                default_dcp = s['decode_context_parallel_size']
                choices = [default_dcp, *sorted(dcp_options[key] - {default_dcp})]
                layout = '/'.join(str(value) for value in choices)
                if engine == 'sglang':
                    layout = '—'
                context = compact_tokens(s['max_model_len']) if 'max_model_len' in s else '—'
                kv = f"[{compact_tokens(record['tokens'])}]({record['source']})" if record else '—'
            if record and not compact:
                kv = f"[{record['tokens']:,}](../{record['source']})"
            if compact:
                if len(choices) > 1:
                    counts = []
                    for dcp in choices:
                        selected = dcp_capacity.get((key, dcp))
                        entry = selected[1] if selected else None
                        counts.append(f"[{compact_tokens(entry['tokens'])}]({entry['source']})" if entry else '—')
                    kv = '(' + '/'.join(counts) + ')'
                title = f"[{model_name}]({quickstart_path(p)})"
                if p['recommendation'] == 'recommended':
                    title = f"**{title}**"
                lines.append(f"| {title}<br>{engine_title} | {quant} | {layout} | {context} / {kv} | {cache_cells[p['id']]} | {STATUS_LABELS[quickstart_status(p)]} |")
                continue
            lines.append(f"| {model_name} | {quant} | {engine_title} | {layout} | {context} / {kv} | {STATUS_LABELS[p['status']]} | {p['recommendation']} | [Guide]({quickstart_path(p)}) |")
        lines.append('')
    if compact:
        return '\n'.join(lines + [END])
    lines += ['Additional pinned historical variants, including original NVFP4 TP2, are in the [retained deployment index](docs/history/deployment-variants.md).', '', 'Qualification applies only to the exact image, checkpoint, topology and workload in the selected guide.',
              'Switched deployments have no switched-hardware qualification. Qwen Flash Next TP2 has a bounded-validation SparkCache option;',
              'six-node work remains research-only and is outside this deployment catalog.', '', END]
    text = '\n'.join(lines)
    return text.replace('](profiles/', '](').replace('](docs/', '](../docs/')


def profile_catalog_table(rows, names, root):
    """Present deployment choices first and retain exact variant discovery below."""
    summary = profile_table(root, compact=True).removesuffix(END)
    lines = [summary, 'Status and context describe the linked default. DCP and KV figures follow the same order;',
             'capacity depends on enabled features. Expand a deployment below for each option’s own status and guide.',
             'Switched support is a separate network configuration and has no switched-hardware qualification.', '',
             '## Configuration variants', '',
             'Profile IDs identify saved configurations. Guide status describes the primary quickstart;',
             'record links preserve configuration evidence when the guide selects a different release.', '']
    groups, retired = {}, []
    developing = developing_cache_profiles(root)
    for p, r in rows:
        if p['recommendation'] == 'retired':
            retired.append((p, r))
            continue
        key = (r['model']['repository'], r['serving']['node_count'], r['runtime'].get('engine', 'vllm'))
        groups.setdefault(key, []).append((p, r))
    for (model, nodes, engine), variants in sorted(groups.items()):
        engine_title = {'vllm': 'vLLM', 'sglang': 'SGLang'}[engine]
        lines += ['<details>', f'<summary>{names[model]} · {nodes} Sparks · {engine_title}</summary>', '',
                  '| Parallelism | Network | SparkCache | Guide status | Configuration and guide |',
                  '|---|---|---|---|---|']
        for p, r in sorted(variants, key=lambda item: (item[0]['recommendation'] != 'recommended', item[0]['id'])):
            s = r['serving']
            parallel = f"DCP{s['decode_context_parallel_size']}" if engine == 'vllm' else f"EP{s['expert_parallel_size']}"
            config = p['configuration']
            cached = (config['key'].endswith('-sparkcache') if config['format'] == 'release-profile'
                      else r['serving'].get('sparkcache', False) if config['format'] == 'serving-profile'
                      else bool(read_json(local_path(config['path'], root)).get('base_recipe')))
            label = p['id'] + (' (default)' if p['recommendation'] == 'recommended' else '')
            cache_cell = 'On' if cached else 'Off (in dev)' if p['id'] in developing else 'Off'
            record_link = f" · [record](profiles/{p['id']}/profile.json)" if 'quickstart_status' in p else ''
            lines.append(f"| {parallel} | {r['topology']} | {cache_cell} | {STATUS_LABELS[quickstart_status(p)]} | [{label}]({quickstart_path(p)}){record_link} |")
        lines += ['', '</details>', '']
    lines += ['### Retired profiles', '', '<details>', '<summary>Retired configurations</summary>', '',
              'Retained for compatibility and historical evidence; use an active deployment above for setup.', '']
    for p, r in sorted(retired, key=lambda item: item[0]['id']):
        lines.append(f"- [{p['id']}]({quickstart_path(p)}) — {names[r['model']['repository']]}; {STATUS_LABELS[p['status']]}")
    lines += ['', '[Additional historical variants](docs/history/deployment-variants.md)', '', '</details>', '',
              'Qwen Flash Next TP2 offers optional SparkCache with bounded text/media persistence validation. Six-node deployments remain experimental and are outside this catalog.', '', END]
    text = '\n'.join(lines)
    return re.sub(r'\]\((?!https?://|#)([^)]+)\)', lambda m: '](' + ('../' + m[1]) + ')', text)


INSTALLER_TABLE = '| Model | Checkpoint | Sparks | `--profile` value |'
THINKING = 'Thinking'


def thinking_column(text, root=ROOT):
    """``text`` with the Thinking cell of each installer-table row set from its profile's thinking record.

    The repository README's installer table, whose header starts with
    INSTALLER_TABLE, is maintained by hand except this column: each row's
    cell is thinking.summary of the profile in its `--profile` value column,
    such as ``on · xhigh``. A table without the column, or a row whose profile
    has no record, is refused.
    """
    lines = text.split('\n')
    header = next((number for number, line in enumerate(lines) if line.startswith(INSTALLER_TABLE)), None)
    if header is None:
        raise ValueError('README.md requires the installer profile table')
    names = [cell.strip() for cell in lines[header].strip().strip('|').split('|')]
    if THINKING not in names:
        raise ValueError('README.md installer profile table requires a Thinking column')
    column, profile_column = names.index(THINKING), names.index('`--profile` value')
    number = header + 2
    while number < len(lines) and lines[number].startswith('|'):
        cells = [cell.strip() for cell in lines[number].strip().strip('|').split('|')]
        if len(cells) != len(names):
            raise ValueError('README.md installer profile table rows require every column')
        profile = cells[profile_column].strip('`')
        record = thinking.of(profile, root=root)
        if record is None:
            raise ValueError(f'README.md installer profile {profile} has no thinking record in {thinking.CATALOG}')
        cells[column] = thinking.summary(record)
        lines[number] = '| ' + ' | '.join(cells) + ' |'
        number += 1
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
    # The repository README lists only the installer's profiles, maintained by
    # hand except their Thinking column (thinking_column); every catalog
    # profile appears in the generated catalog page.
    readme = root/'profiles/README.md'
    text = readme.read_text(encoding='utf-8-sig')
    if START not in text or END not in text:
        raise ValueError('profiles/README.md requires generated profile region markers')
    before, tail = text.split(START, 1)
    _, after = tail.split(END, 1)
    expected[readme] = (before+profile_table(root)+after).encode()
    installer_readme = root/'README.md'
    expected[installer_readme] = thinking_column(installer_readme.read_text(encoding='utf-8-sig'), root).encode()
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
