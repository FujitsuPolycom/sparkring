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
    assert link in ring and link in pair
    assert '[NVFP4](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4)' not in pair
    qad = next(line for line in ring.splitlines() if '| [Qwen3.8-Flash-Next](' in line.replace('**', ''))
    # The SparkCache profile is a standalone catalog entry, so the installer row
    # offers no cache option.
    assert '[Optional](' not in qad
    assert resolve('qwen38-flash-next-qad-tp4')['serving']['sparkcache'] is False
    assert resolve('qwen38-flash-next-qad-tp4-sparkcache')['serving']['sparkcache'] is True


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
    # profile runs four decode-context-parallel groups of four inside tensor parallelism 8.
    assert [line for line in active.splitlines() if line.startswith('| DCP4 |')] == [
        '| DCP4 | direct-cycle-8 | Off | Experimental | [glm53-nvfp4-tp8 (default)](../profiles/glm53-nvfp4-tp8/README.md) |']
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


def test_readme_thinking_column_follows_each_profiles_thinking_record():
    import pytest
    from runtime.common.profiles import ROOT
    from scripts.generate_profiles import thinking_column
    text = (ROOT / 'README.md').read_text(encoding='utf-8-sig')
    assert thinking_column(text) == text
    rows = {line.split('|')[4].strip().strip('`'): line.split('|')[6].strip()
            for line in text.splitlines() if line.startswith('| ') and line.split('|')[4].strip().startswith('`')
            and line.split('|')[6].strip() != 'Thinking'}
    assert rows == {'qwen38-flash-next-tp2': 'on · xhigh', 'qwen38-flash-next-qad-tp4': 'on · xhigh',
                    'glm53-flash-nvfp4-spark-tp2': 'always · max', 'glm53-flash-nvfp4-spark-tp4': 'always · max',
                    'mimo-v26-flash-mopd-tp2': 'on', 'mimo-v26-flash-mopd-tp4': 'on', 'deepseek-v41-flash-tp4': 'on · high',
                    'swift15-qwen38-flash-next-tp2': 'on · xhigh', 'swift15-qwen38-flash-next-tp4': 'on · xhigh'}
    # A stale cell is rewritten; the hand-maintained cells stay as written.
    stale = text.replace('`qwen38-flash-next-tp2` | 8000 | on · xhigh |', '`qwen38-flash-next-tp2` | 8000 | off |')
    assert stale != text and thinking_column(stale) == text
    with pytest.raises(ValueError, match='requires a Thinking column'):
        thinking_column(text.replace('| API port | Thinking |', '| API port | Effort |'))
    with pytest.raises(ValueError, match='qwen38-flash-next-tp2-sparkcache has no thinking record'):
        thinking_column(text.replace('`qwen38-flash-next-tp2` |', '`qwen38-flash-next-tp2-sparkcache` |'))
