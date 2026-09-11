"""Parse explicit site assignments without sourcing shell code or reading process env."""
from pathlib import Path


def assignment_digest(text: str) -> str:
    """Hash ordered ENV assignments, excluding only blank and comment lines.

    Values, quoting and assignment order remain significant. This checks
    compatibility, not shell evaluation or a published artifact's identity.
    """
    import hashlib
    import re

    assignments = []
    seen = set()
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        key, separator, _ = line.partition('=')
        if not separator or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key) or key in seen:
            raise ValueError('ENV baseline requires unique literal assignments')
        seen.add(key)
        assignments.append(line)
    return hashlib.sha256(('\n'.join(assignments) + '\n').encode('utf-8')).hexdigest()


def read_assignments(path: Path, keys: set[str]) -> dict[str, str]:
    result = {}
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if (not separator or key not in keys or key in result or not value
                or any(c.isspace() for c in value) or any(c in value for c in "<>\"'")):
            raise ValueError(f"{path}:{number}: use one resolved assignment for each site setting")
        result[key] = value
    if set(result) != keys:
        raise ValueError("Site settings must include " + ", ".join(sorted(keys)))
    return result


def render_environment(profile_id, site_values=None, overrides=None, *, root=None, template_only=False):
    """Render adapter input from authoritative knobs and explicit private fields."""
    import re
    from runtime.common.profiles import ROOT, local_path, read_json, resolve

    root = ROOT if root is None else root
    resolved = resolve(profile_id, overrides, root=root)
    from runtime.common.profiles import load
    if not template_only and load(profile_id, root)[0]['launcher']['kind'] != 'bash':
        raise ValueError('This profile uses a staged configuration procedure; follow its guide')
    rows = read_json(root/'profiles/environment-exports.json')['exports']
    matches = [row for row in rows if row['profile'] == profile_id]
    if len(matches) != 1:
        raise ValueError(f'{profile_id}: use the staged or frozen configuration procedure in its guide')
    text = local_path(matches[0]['template'], root).read_text(encoding='utf-8-sig')
    def knob(match):
        key = match[1]
        if key not in resolved['serving'] or type(resolved['serving'][key]) is not int:
            raise ValueError(f'{profile_id}: unknown integer serving field {key}')
        return str(resolved['serving'][key])
    text = re.sub(r'\{\{([a-z_]+)\}\}', knob, text)
    if template_only:
        return text
    site_values = {} if site_values is None else site_values
    if not isinstance(site_values, dict):
        raise ValueError('Per-rank site values must be a JSON object')
    required = set()
    lines = []
    for line in text.splitlines():
        key, separator, value = line.partition('=')
        if separator and not line.lstrip().startswith('#') and '<' in value:
            required.add(key)
            value = site_values.get(key)
            if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_./:@=,+-]+', value):
                raise ValueError(f'{key}: provide a resolved literal site value without whitespace or shell syntax')
            if key == 'IMAGE_ID' and not re.fullmatch(r'sha256:[0-9a-f]{64}', value):
                raise ValueError('IMAGE_ID requires the exact local sha256 image identity')
            line = key+'='+value
        lines.append(line)
    if set(site_values) != required:
        raise ValueError('Site values may fill only template placeholders; use --set for supported serving overrides')
    values = dict(line.split('=', 1) for line in lines if '=' in line and not line.lstrip().startswith('#'))
    rank = values.get('NODE_RANK', values.get('RANK'))
    if rank is not None and (not rank.isdigit() or not 0 <= int(rank) < resolved['serving']['node_count']):
        raise ValueError('Site rank is outside the selected topology')
    if values.get('MODEL_HOST_PATH') == values.get('CACHE_HOST_PATH'):
        raise ValueError('Model and writable cache directories must be distinct')
    return '\n'.join(lines)+'\n'
