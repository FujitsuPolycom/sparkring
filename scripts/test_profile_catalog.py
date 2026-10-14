"""Deployment navigation groups variants without losing their individual guides."""
from runtime.common.profiles import catalog, load
from runtime.common.profiles import resolve
from scripts.generate_profiles import profile_table


def test_qwen_cache_variant_is_visible_without_replacing_native_profile():
    table = profile_table()
    summary, variants = table.split('## Configuration variants', 1)
    pair_summary = summary.split('### Two Sparks', 1)[1]
    rows = [line for line in pair_summary.splitlines() if line.replace('**', '').startswith('| [Qwen3.8-Flash-Next](')]
    assert len(rows) == 1
    # The SparkCache profile pins revision 629bc3218833, not the installer
    # profile's checkpoint, so the installer row offers no cache option.
    assert '| No | Development |' in rows[0]
    assert 'Qwen with SparkCache is unsupported' not in table
    assert resolve('qwen38-flash-next-tp2')['serving']['sparkcache'] is False
    cached = resolve('qwen38-flash-next-tp2-sparkcache')
    assert cached['serving']['sparkcache'] is True
    assert 'SparkCache is disabled' not in str(cached['evidence'])
    cache_row = next(line for line in variants.splitlines() if '[qwen38-flash-next-tp2-sparkcache](' in line)
    assert '| On | Validated |' in cache_row


def test_qad_quant_link_identifies_the_pinned_checkpoint_for_tp2_and_tp4():
    summary = profile_table().split('## Configuration variants', 1)[0]
    ring, pair = summary.split('### Two Sparks', 1)
    link = '[NVFP4 QAD step 5500 PLE](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4/tree/60215d26cf5e42c2db6128774032d57fc62678da)'
    assert link in ring
    # On the install default image the two-Spark profile installs its preferred CSF checkpoint.
    pair_row = next(line for line in pair.splitlines() if '](../profiles/qwen38-flash-next-tp2/README.md)' in line)
    assert '[NVFP4/MXFP8 CSF](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD)' in pair_row
    assert '[NVFP4](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4)' not in pair
    qad = next(line for line in ring.splitlines() if '| [Qwen3.8-Flash-Next](' in line.replace('**', ''))
    # The SparkCache profile is a standalone catalog entry, so the installer row
    # offers no cache option.
    assert '[Optional](' not in qad
    assert resolve('qwen38-flash-next-qad-tp4')['serving']['sparkcache'] is False
    assert resolve('qwen38-flash-next-qad-tp4-sparkcache')['serving']['sparkcache'] is True


def test_installer_rows_show_the_checkpoint_installed_on_the_install_default_image():
    from scripts.generate_profiles import install_checkpoints, installed_defaults
    installs = install_checkpoints()
    # On release 2026.10.4's image the GLM-5.3-Flash pair and ring and the Qwen pair install CSF.
    assert {profile for profile, checkpoint in installs.items() if checkpoint} == {
        'glm53-flash-nvfp4-spark-tp2', 'glm53-flash-nvfp4-spark-tp4', 'qwen38-flash-next-tp2'}
    summary = profile_table(compact=True)
    rows = {'glm53-flash-nvfp4-spark-tp2': ('Two Sparks', 'docs/operations/install.md'),
            'glm53-flash-nvfp4-spark-tp4': ('Four Sparks', 'docs/operations/install.md'),
            'qwen38-flash-next-tp2': ('Two Sparks', 'profiles/qwen38-flash-next-tp2/README.md')}
    for profile, (section, guide) in rows.items():
        part = summary.split('### ' + section, 1)[1].split('###', 1)[0]
        row = next(line for line in part.splitlines() if line.startswith('| **[') and f']({guide})' in line
                   and 'CSF' in line)
        facts = installed_defaults(profile)
        assert '-CSF-QAD)' in row and f"]({facts['kv']['source']})" in row


def test_catalog_groups_glm_choices_and_preserves_every_profile_link():
    table = profile_table()
    summary, variants = table.split('## Configuration variants', 1)
    glm_rows = [line for line in summary.splitlines() if line.startswith('| **[GLM-5.3-Flash](')]
    # Each topology lists the installer's shared-image profile and the
    # SparkCache DCP1 profile.
    assert len(glm_rows) == 4
    cache_rows = [row for row in glm_rows if '[Optional](' in row]
    # DCP4 profiles are retired, so each topology offers DCP1 only.
    assert len(cache_rows) == 2 and all('| 1 |' in row for row in cache_rows)
    for row, nodes in zip(cache_rows, (4, 2)):
        assert f'[Optional](../profiles/glm53-flash-spark-tp{nodes}-dcp1-sparkcache/README.md)' in row
    installer_rows = [row for row in glm_rows if '](../docs/operations/install.md)' in row]
    assert len(installer_rows) == 2 and all(row.endswith('| Development |') for row in installer_rows)
    assert '1,048,576' not in summary
    for profile_id in catalog():
        profile, _ = load(profile_id)
        label = profile_id + (' (default)' if profile['recommendation'] == 'recommended' else '')
        guide = (profile['guide'] if profile['configuration']['format'] == 'serving-profile'
                 else f'profiles/{profile_id}/README.md')
        assert f"[{label}](../{guide})" in variants
    assert '| DCP1 | switched | Off | Experimental |' in variants


def test_dcp4_profiles_are_listed_only_as_retired_configurations():
    dcp4_profiles = (
        'glm53-flash-spark-tp4-dcp4',
        'glm53-flash-spark-tp4-dcp4-sparkcache',
        'glm52-exl3-r7-3.5bpw',
        'sparkcache-glm52-exl3-r7-3.5bpw-sparkcache-tp4-dcp4',
    )
    table = profile_table()
    summary, variants = table.split('## Configuration variants', 1)
    active, retired = variants.split('### Retired profiles', 1)
    assert '[GLM-5.2](' not in summary
    # Decode-context parallelism 4 on four Sparks is listed only as retired; GLM-5.3's eight-Spark
    # profiles, one per checkpoint, run four decode-context-parallel groups of four inside tensor parallelism 8.
    assert [line for line in active.splitlines() if line.startswith('| DCP4 |')] == [
        '| DCP4 | ring of 8 | Off | Experimental | [glm53-nvfp4-tp8 (default)](../profiles/glm53-nvfp4-tp8/README.md) |',
        '| DCP4 | ring of 8 | Off | Experimental | [glm53-nvfp4-csf-tp8](../profiles/glm53-nvfp4-csf-tp8/README.md) |']
    for profile_id in dcp4_profiles:
        assert load(profile_id)[0]['recommendation'] == 'retired'
        assert resolve(profile_id)['serving']['decode_context_parallel_size'] == 4
        assert f'profiles/{profile_id}/' not in summary + active
        assert f'[{profile_id}](../profiles/{profile_id}/README.md)' in retired
    for profile_id in catalog():
        profile, _ = load(profile_id)
        serving = resolve(profile_id)['serving']
        if profile['recommendation'] != 'retired' and serving['node_count'] == 4:
            assert serving.get('decode_context_parallel_size') != 4


def test_catalog_keeps_separate_deepseek_engines_and_variant_validation():
    table = profile_table()
    summary, variants = table.split('## Configuration variants', 1)
    rows = [line for line in summary.splitlines()
            if line.startswith(('| [DeepSeek-V4.1-Flash](', '| **[DeepSeek-V4.1-Flash]('))]
    assert len(rows) == 2
    assert any('<br>vLLM |' in row for row in rows)
    assert any('<br>SGLang |' in row for row in rows)
    for profile_id, status, cache in (
        ('glm53-flash-spark-tp4-dcp1', 'Experimental', 'Off'),
        ('glm53-flash-spark-tp4-dcp1-sparkcache', 'Validated', 'On'),
    ):
        row = next(line for line in variants.splitlines() if f'](../profiles/{profile_id}/README.md)' in line)
        assert f'| {cache} | {status} |' in row
def test_shared_qwen_guide_is_used_for_cache_links_and_variant_navigation():
    from scripts.generate_profiles import profile_table
    compact = profile_table(compact=True)
    catalog = profile_table()
    assert '[Optional](profiles/qwen38-flash-next-tp2/README.md)' not in compact
    assert '[qwen38-flash-next-tp2-sparkcache](../profiles/qwen38-flash-next-tp2/README.md)' in catalog
    assert 'profiles/qwen38-flash-next-tp2-sparkcache/README.md' not in compact
    assert 'profiles/qwen38-flash-next-tp2-sparkcache/README.md' not in catalog



def test_glm_discovery_promotes_native_cache_profiles_without_relabelling_r33():
    summary = profile_table(compact=True)
    glm = [row for row in summary.splitlines() if row.startswith("| **[GLM-5.3-Flash](")]
    cached = [row for row in glm if '[Optional](' in row]
    assert len(glm) == 4 and len(cached) == 2
    assert all(row.endswith("| Validated |") for row in cached)
    for profile_id in ("glm53-flash-spark-tp2-dcp1-sparkcache", "glm53-flash-spark-tp4-dcp1-sparkcache"):
        resolved = resolve(profile_id)
        assert resolved["status"] == "qualified"
        assert resolved["quickstart_status"] == "qualified"
        assert resolved["release"]["id"] == "shared-2026.09.3"
    for profile_id in ("glm53-flash-spark-tp2-dcp1", "glm53-flash-spark-tp4-dcp1"):
        assert resolve(profile_id)["release"]["id"] == "sparkring-r33-dcp4"


def test_cross_release_cache_pair_requires_matching_model_and_layout():
    from copy import deepcopy
    from scripts.generate_profiles import compact_profile_rows

    ids = ('glm53-flash-spark-tp4-dcp1', 'glm53-flash-spark-tp4-dcp1-sparkcache')
    original = [(load(profile_id)[0], resolve(profile_id)) for profile_id in ids]
    assert original[0][0]['configuration']['path'] != original[1][0]['configuration']['path']
    rows, cells = compact_profile_rows(original)
    assert len(rows) == 1 and rows[0][0]['id'] == ids[1]
    assert cells[ids[1]].startswith('[Optional](')

    for section, field, value in (
        ('model', 'repository', 'unrelated/model'),
        ('serving', 'node_count', 2),
        ('serving', 'decode_context_parallel_size', 4),
        ('runtime', 'engine', 'sglang'),
    ):
        mismatched = deepcopy(original)
        mismatched[0][1][section][field] = value
        _, cells = compact_profile_rows(mismatched)
        assert cells[ids[1]] == 'Included'

    retired = deepcopy(original)
    retired[0][0]['recommendation'] = 'retired'
    _, cells = compact_profile_rows(retired)
    assert cells[ids[1]] == 'Included'


def installer_rows(text):
    """The README installer table's rows as cell lists: it starts at its header and ends at the first line outside it."""
    from scripts.generate_profiles import INSTALLER_TABLE
    lines = text.splitlines()
    header = next(number for number, line in enumerate(lines) if line.startswith(INSTALLER_TABLE))
    rows = []
    for line in lines[header + 2:]:
        if not line.startswith('| '):
            break
        rows.append([cell.strip() for cell in line.strip().strip('|').split('|')])
    return rows


def test_readme_installer_columns_follow_each_profiles_installed_defaults():
    import pytest
    from runtime.common.profiles import ROOT
    from scripts.generate_profiles import installer_columns
    text = (ROOT / 'README.md').read_text(encoding='utf-8-sig')
    assert installer_columns(text) == text
    assert '| Model | Checkpoint | Sparks | `--profile` value | Context | KV tokens | Thinking |' in text
    rows = {cells[3].strip('`'): cells[4:] for cells in installer_rows(text)}
    assert {name: cells[2] for name, cells in rows.items()} == {
        'qwen38-flash-next-tp2': 'on · xhigh', 'qwen38-flash-next-tp4': 'on · xhigh',
        'glm53-flash-tp2': 'always · high', 'glm53-flash-tp2-dcp2': 'always · high', 'glm53-flash-tp4': 'always · high',
        'glm53-nvfp4-tp8': 'always · high', 'mimo-v26-flash-mopd-tp2': 'on', 'mimo-v26-flash-mopd-tp4': 'on',
        'deepseek-v41-flash-tp4': 'on · high', 'swift15-qwen38-flash-next-tp2': 'on · xhigh',
        'swift15-qwen38-flash-next-tp4': 'on · xhigh'}
    assert rows['glm53-flash-tp2'][0] == '1M' and rows['qwen38-flash-next-tp2'][0] == '262K'
    # Each recorded pool links its record.
    assert rows['glm53-flash-tp2'][1].startswith('[1.5M](performance/records/')
    assert rows['qwen38-flash-next-tp2'][1].startswith('[2.5M](performance/records/')
    assert all(cells[1].startswith('[') for cells in rows.values())
    # A stale cell is rewritten; the hand-maintained cells stay as written.
    current = next(line for line in text.splitlines() if line.startswith('| Qwen3.8-Flash-Next |') and '`qwen38-flash-next-tp2`' in line)
    stale = text.replace(current, current.split('`qwen38-flash-next-tp2`')[0] + '`qwen38-flash-next-tp2` | 8K | 9M | off |')
    assert stale != text and installer_columns(stale) == text
    # A profile without a recorded pool for its installed checkpoint gets a dash.
    unmeasured = text.replace('`qwen38-flash-next-tp2`', '`glm53-flash-csf-tp8`')
    row = next(line for line in installer_columns(unmeasured).splitlines() if line.startswith('| ') and '`glm53-flash-csf-tp8`' in line)
    assert row.endswith('| 1M | — | always · high |')
    for column in ('Context', 'KV tokens', 'Thinking'):
        with pytest.raises(ValueError, match=f'requires a {column} column'):
            installer_columns(text.replace(f'| {column} |', '| Other |', 1))
    with pytest.raises(ValueError, match='qwen38-flash-next-tp2-sparkcache has no thinking record'):
        installer_columns(text.replace('`qwen38-flash-next-tp2` |', '`qwen38-flash-next-tp2-sparkcache` |'))


def test_recorded_kv_counts_only_a_measurement_of_the_configured_checkpoint_and_kv_size(tmp_path):
    import json
    import shutil
    from runtime.common.profiles import ROOT
    from scripts.generate_profiles import CAPACITY, recorded_kv
    # glm53-flash-tp2's record measured its own nvfp4-spark checkpoint and, separately, csf.
    assert recorded_kv('glm53-flash-nvfp4-spark-tp2')['tokens'] == 1530566
    assert recorded_kv('glm53-flash-nvfp4-spark-tp2', 'csf')['tokens'] == 1472548
    # Qwen's pair has a csf measurement; its record's own is of qad-step-4000, which sizes no other checkpoint.
    assert recorded_kv('qwen38-flash-next-tp2', 'csf')['tokens'] == 2534058
    assert recorded_kv('qwen38-flash-next-tp2', 'qad-step5500-ple1000') is None
    assert recorded_kv('glm53-flash-csf-tp8') is None
    # A measurement at another KV size than the profile configures is not shown.
    root = tmp_path / 'checkout'
    shutil.copytree(ROOT / 'profiles', root / 'profiles')
    capacity = json.loads((ROOT / CAPACITY).read_text(encoding='utf-8'))
    capacity['profiles']['glm53-flash-nvfp4-spark-tp2']['checkpoints']['csf']['kv_bytes_per_rank'] = 5368709120
    (root / 'performance').mkdir()
    (root / CAPACITY).write_text(json.dumps(capacity), encoding='utf-8')
    assert recorded_kv('glm53-flash-nvfp4-spark-tp2', 'csf', root) is None


def test_readme_installer_columns_read_the_profile_an_alias_selects():
    from runtime.common.profiles import ROOT
    from runtime.host.models import ALIASES
    from scripts.generate_profiles import compact_tokens, installed_defaults, installer_columns
    text = (ROOT / 'README.md').read_text(encoding='utf-8-sig')
    assert {'glm53-flash-tp2', 'glm53-flash-tp4', 'qwen38-flash-next-tp4', 'qwen38-flash-next-tp8'} <= set(ALIASES)
    original = next('| ' + ' | '.join(cells) + ' |' for cells in installer_rows(text) if cells[3] == '`qwen38-flash-next-tp4`')
    # Each alias and its profile ID give the same cells.
    for alias, ident in ALIASES.items():
        facts = installed_defaults(ident)
        for name in (alias, ident):
            edited = text.replace(original, f'| Model | Checkpoint | 4 | `{name}` | stale | stale | stale |')
            assert edited != text
            kv = f'[{compact_tokens(facts["kv"]["tokens"])}]({facts["kv"]["source"]})' if facts['kv'] else '—'
            row = f'| Model | Checkpoint | 4 | `{name}` | {compact_tokens(facts["context"])} | {kv} | {facts["thinking"]} |'
            assert row in installer_columns(edited)


def test_the_network_column_names_each_topology_in_plain_words():
    from runtime.common import profiles
    from scripts.generate_profiles import TOPOLOGY_LABELS, topology_label
    assert set(TOPOLOGY_LABELS) == profiles.TOPOLOGIES
    variants = profile_table().split('## Configuration variants', 1)[1]
    rows = [line.split(' | ')[1] for line in variants.splitlines() if line.startswith('| DCP') or line.startswith('| EP')]
    assert rows and set(rows) <= set(TOPOLOGY_LABELS.values())
    assert {'pair', 'ring of 4', 'ring of 8', 'RoCEnante mesh (SIRCL carries most collectives)'} <= set(rows)
    assert not any(topology in variants for topology in TOPOLOGY_LABELS if topology != 'switched')
    import pytest
    with pytest.raises(ValueError, match='needs a Network label'):
        topology_label('direct-line-3')
