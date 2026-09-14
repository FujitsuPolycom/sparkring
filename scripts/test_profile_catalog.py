"""Deployment navigation groups variants without losing their individual guides."""
from runtime.common.profiles import catalog
from runtime.common.profiles import resolve
from scripts.generate_profiles import profile_table


def test_qwen_cache_variant_is_visible_without_replacing_native_profile():
    table = profile_table()
    summary, variants = table.split('## Configuration variants', 1)
    rows = [line for line in summary.splitlines() if line.startswith('| Qwen3.8-Flash-Next |')]
    assert len(rows) == 1
    assert '[Optional](../profiles/qwen38-flash-next-tp2-sparkcache/README.md)' in rows[0]
    assert 'Qwen with SparkCache is unsupported' not in table
    assert resolve('qwen38-flash-next-tp2')['serving']['sparkcache'] is False
    cached = resolve('qwen38-flash-next-tp2-sparkcache')
    assert cached['serving']['sparkcache'] is True
    assert 'SparkCache is disabled' not in str(cached['evidence'])
    cache_row = next(line for line in variants.splitlines() if '](../profiles/qwen38-flash-next-tp2-sparkcache/README.md)' in line)
    assert '| On | Experimental |' in cache_row


def test_catalog_groups_glm_choices_and_preserves_every_profile_link():
    table = profile_table()
    summary, variants = table.split('## Configuration variants', 1)
    glm_rows = [line for line in summary.splitlines() if line.startswith('| **GLM-5.3-Flash**')]
    assert len(glm_rows) == 2  # Four-Spark and two-Spark deployments.
    assert 'DCP1/DCP4' in glm_rows[0]
    assert all('[Optional]' in row for row in glm_rows)
    assert '1,048,576' not in summary
    for profile_id in catalog():
        assert f'../profiles/{profile_id}/README.md' in variants
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
