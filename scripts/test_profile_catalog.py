"""Deployment navigation groups variants without losing their individual guides."""
from runtime.common.profiles import catalog, load
from runtime.common.profiles import resolve
from scripts.generate_profiles import profile_table


def test_qwen_cache_variant_is_visible_without_replacing_native_profile():
    table = profile_table()
    summary, variants = table.split('## Configuration variants', 1)
    pair_summary = summary.split('### Two Sparks', 1)[1]
    rows = [line for line in pair_summary.splitlines() if line.startswith('| Qwen3.8-Flash-Next |')]
    assert len(rows) == 1
    assert '[Optional](../profiles/qwen38-flash-next-tp2/README.md)' in rows[0]
    assert 'Qwen with SparkCache is unsupported' not in table
    assert resolve('qwen38-flash-next-tp2')['serving']['sparkcache'] is False
    cached = resolve('qwen38-flash-next-tp2-sparkcache')
    assert cached['serving']['sparkcache'] is True
    assert 'SparkCache is disabled' not in str(cached['evidence'])
    cache_row = next(line for line in variants.splitlines() if '[qwen38-flash-next-tp2-sparkcache](' in line)
    assert '| On | Experimental |' in cache_row


def test_qad_quant_link_identifies_its_checkpoint_separately_from_tp2():
    summary = profile_table().split('## Configuration variants', 1)[0]
    ring, pair = summary.split('### Two Sparks', 1)
    assert '[NVFP4 QAD](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4/tree/629bc3218833a38b475b719f34aa571666f4a03e)' in ring
    assert '[NVFP4](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4)' in pair
    assert 'NVFP4 QAD' not in pair
    qad = next(line for line in ring.splitlines() if '| Qwen3.8-Flash-Next |' in line)
    assert '[Optional](../profiles/qwen38-flash-next-qad-tp4-sparkcache/README.md)' in qad
    assert resolve('qwen38-flash-next-qad-tp4')['serving']['sparkcache'] is False
    assert resolve('qwen38-flash-next-qad-tp4-sparkcache')['serving']['sparkcache'] is True


def test_catalog_groups_glm_choices_and_preserves_every_profile_link():
    table = profile_table()
    summary, variants = table.split('## Configuration variants', 1)
    glm_rows = [line for line in summary.splitlines() if line.startswith('| **GLM-5.3-Flash**')]
    assert len(glm_rows) == 2  # Four-Spark and two-Spark deployments.
    assert 'DCP1/DCP4' in glm_rows[0]
    assert all('[Optional]' in row for row in glm_rows)
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
    rows = [line for line in summary.splitlines() if line.startswith('| DeepSeek-V4.1-Flash |')]
    assert len(rows) == 2
    assert any('| vLLM |' in row for row in rows)
    assert any('| SGLang |' in row for row in rows)
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
