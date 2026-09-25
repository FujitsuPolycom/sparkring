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
    assert '[Optional](../profiles/qwen38-flash-next-tp2/README.md)' in rows[0]
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
    assert '[NVFP4 QAD](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4/tree/629bc3218833a38b475b719f34aa571666f4a03e)' in ring
    assert '[NVFP4 QAD step 5500 PLE](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4/tree/60215d26cf5e42c2db6128774032d57fc62678da)' in pair
    assert '[NVFP4](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4)' not in pair
    qad = next(line for line in ring.splitlines() if '| [Qwen3.8-Flash-Next](' in line.replace('**', ''))
    assert '[Optional](../profiles/qwen38-flash-next-qad-tp4-sparkcache/README.md)' in qad
    assert resolve('qwen38-flash-next-qad-tp4')['serving']['sparkcache'] is False
    assert resolve('qwen38-flash-next-qad-tp4-sparkcache')['serving']['sparkcache'] is True


def test_catalog_groups_glm_choices_and_preserves_every_profile_link():
    table = profile_table()
    summary, variants = table.split('## Configuration variants', 1)
    glm_rows = [line for line in summary.splitlines() if line.startswith('| **[GLM-5.3-Flash](')]
    assert len(glm_rows) == 2  # Four-Spark and two-Spark deployments.
    assert '| 1/4 |' in glm_rows[0]
    for row, nodes in zip(glm_rows, (4, 2)):
        assert f'[Optional](../profiles/glm53-flash-spark-tp{nodes}-dcp1-sparkcache/README.md)' in row
    assert '1,048,576' not in summary
    for profile_id in catalog():
        profile, _ = load(profile_id)
        label = profile_id + (' (default)' if profile['recommendation'] == 'recommended' else '')
        guide = (profile['guide'] if profile['configuration']['format'] == 'serving-profile'
                 else f'profiles/{profile_id}/README.md')
        assert f"[{label}](../{guide})" in variants
    assert '| DCP1 | switched | Off | Experimental |' in variants


def test_catalog_keeps_separate_deepseek_engines_and_variant_validation():
    table = profile_table()
    summary, variants = table.split('## Configuration variants', 1)
    rows = [line for line in summary.splitlines() if line.startswith('| [DeepSeek-V4.1-Flash](')]
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
    assert '[Optional](profiles/qwen38-flash-next-tp2/README.md)' in compact
    assert '[qwen38-flash-next-tp2-sparkcache](../profiles/qwen38-flash-next-tp2/README.md)' in catalog
    assert 'profiles/qwen38-flash-next-tp2-sparkcache/README.md' not in compact
    assert 'profiles/qwen38-flash-next-tp2-sparkcache/README.md' not in catalog



def test_glm_discovery_promotes_native_cache_profiles_without_relabelling_r33():
    summary = profile_table(compact=True)
    glm = [row for row in summary.splitlines() if row.startswith("| **[GLM-5.3-Flash](")]
    assert len(glm) == 2
    assert all(row.endswith("| Validated |") for row in glm)
    assert all('[Optional](' in row for row in glm)
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
