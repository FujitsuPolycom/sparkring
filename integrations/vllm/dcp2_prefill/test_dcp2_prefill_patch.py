"""CPU admission, source identity and derived-receipt contracts."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest

from integrations.vllm.dcp2_prefill import patch
from runtime.images import derive_dcp2_prefill as image
from runtime.images import derived_layer
from runtime.common import image_lock, transport

FIXTURES = Path(__file__).with_name('fixtures')


def source(path):
    return (FIXTURES / Path(path).name).read_bytes().replace(b'\r\n', b'\n')


def module(path, patched=True):
    data = patch.apply(source(path), path) if patched else source(path)
    name = 'dcp2_fixture_' + Path(path).stem
    result = ModuleType(name)
    sys.modules[name] = result
    exec(compile(data, path, 'exec'), result.__dict__)
    return result


def config(tp=2, dcp=2):
    return NS(model_config=NS(hf_text_config=NS(model_type='glm5_next'), dtype='torch.bfloat16',
                             enable_sleep_mode=False, enable_return_routed_experts=False),
              cache_config=NS(enable_prefix_caching=True, mamba_cache_mode='align', prefix_cache_retention_interval=0),
              parallel_config=NS(tensor_parallel_size=tp, decode_context_parallel_size=dcp,
                                 pipeline_parallel_size=1, data_parallel_size=1, prefill_context_parallel_size=1,
                                 enable_expert_parallel=False, enable_eplb=False),
              scheduler_config=NS(max_num_batched_tokens=8192, max_num_scheduled_tokens=None,
                                  long_prefill_token_threshold=0, prefill_compute_share=None,
                                  max_parallel_prefills=1, enable_chunked_prefill=True),
              additional_config={'kda_prefill_backend': 'b12x'}, speculative_config=None,
              use_v2_model_runner=True, use_request_boundary_checkpoints=False, lora_config=None)


@pytest.fixture
def env(monkeypatch):
    vllm = ModuleType('vllm')
    vllm.envs = NS(VLLM_B12X_KDA_PREFILL_COALESCING=True)
    monkeypatch.setitem(sys.modules, 'vllm', vllm)
    return vllm.envs


@pytest.mark.parametrize('tp,dcp,allowed', [(2, 1, True), (2, 2, True), (4, 1, True), (4, 2, True),
                                          (4, 4, True), (2, 4, False), (8, 2, False), (1, 1, False)])
def test_kda_admission(env, tp, dcp, allowed):
    fn = module(patch.KDA).validate_coalescing_config
    if allowed:
        assert fn(config(tp, dcp))
    else:
        with pytest.raises(ValueError, match='requires GLM5Next'):
            fn(config(tp, dcp))


def test_original_refuses_and_switch_off_remains_off(env):
    with pytest.raises(ValueError):
        module(patch.KDA, False).validate_coalescing_config(config())
    env.VLLM_B12X_KDA_PREFILL_COALESCING = False
    assert module(patch.KDA).validate_coalescing_config(config()) is False


@pytest.mark.parametrize('owner,key,value', [
    ('parallel_config', 'pipeline_parallel_size', 2), ('parallel_config', 'data_parallel_size', 2),
    ('parallel_config', 'prefill_context_parallel_size', 2), ('parallel_config', 'enable_expert_parallel', True),
    ('parallel_config', 'enable_eplb', True), ('scheduler_config', 'max_num_batched_tokens', 4096),
    ('scheduler_config', 'max_parallel_prefills', 2), ('cache_config', 'enable_prefix_caching', False),
    ('cache_config', 'prefix_cache_retention_interval', 2048), ('model_config', 'dtype', 'torch.float16'),
])
def test_other_guards_survive(env, owner, key, value):
    candidate = config()
    setattr(getattr(candidate, owner), key, value)
    with pytest.raises(ValueError):
        module(patch.KDA).validate_coalescing_config(candidate)


@pytest.mark.parametrize('path', [patch.KDA, patch.MHC, patch.PINS, patch.HOOKS])
def test_patches_are_pinned_and_reapplication_or_drift_refuses(path):
    result = patch.apply(source(path), path)
    with pytest.raises(ValueError, match='hash differs'):
        patch.apply(result, path)
    with pytest.raises(ValueError, match='hash differs'):
        patch.apply(source(path)+b'\n', path)


def test_mhc_changes_only_both_admission_tuples_and_diagnostics():
    before = ast.parse(source(patch.MHC))
    after = ast.parse(patch.apply(source(patch.MHC), patch.MHC))
    changed = [old.name for old, new in zip(before.body, after.body)
               if ast.dump(old) != ast.dump(new)]
    assert changed == ['configure', 'validate_model']
    # Execute the real forward-time guard before any model traversal.
    text = patch.apply(source(patch.MHC), patch.MHC).decode()
    tree = ast.parse(text)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'validate_model')
    guard = next(n for n in fn.body if isinstance(n, ast.If))
    code = compile(ast.Expression(guard.test), '<mhc-guard>', 'eval')
    for tp, dcp in ((2, 1), (2, 2), (4, 1), (4, 2), (4, 4)):
        assert not eval(code, dict(config=config(tp, dcp).parallel_config, model=NS(is_sequence_parallel=False)))
    assert eval(code, dict(config=config(2, 4).parallel_config, model=NS(is_sequence_parallel=False)))


def test_checkpoint_targets_and_continuation_physical_ownership():
    kda = module(patch.KDA)
    for start in (0, 8192, 16384):
        plan = kda.prefill_checkpoint_plan(start=start, end=start+8192, prompt=32768, num_tokens=32768,
                                           block_size=2048, publications=(start+2048, start+4096, start+6144))
        assert kda.checkpoint_metadata(plan, start, start+8192, 2048, 4) == (
            [2048, 4096, 6144, 0], [start//2048, start//2048+1, start//2048+2, -1])
    blocks = [NS(block_id=i, is_null=False, ref_cnt=1, block_hash=None) for i in range(4)]
    source_col, layout = kda.continuation_layout(blocks, (2048, 10240, (4096, 6144, 8192)), 2048, 3)
    assert source_col == 0 and layout == (0, 1, 2, 3, -2, -2, -2, -2)
    blocks[2].block_id = blocks[1].block_id
    with pytest.raises(ValueError, match='aliases physical ownership'):
        kda.continuation_layout(blocks, (2048, 10240, (4096, 6144, 8192)), 2048, 3)


def test_new_build_pins_both_features_and_old_build_still_refuses(tmp_path):
    pins = module(patch.PINS)
    build = next(b for b in pins.SUPPORTED if b.name == patch.BUILD)
    assert (2, 2) in pins.mhc_admits(build.name)
    assert (2, 2) not in pins.mhc_admits('sparkring-kraken-beta-20261007-bc9ea774')
    assert build.files[patch.KDA.removeprefix('vllm/')] == hashlib.sha256(patch.apply(source(patch.KDA))).hexdigest()
    assert build.files[patch.MHC.removeprefix('vllm/')] == hashlib.sha256(patch.apply(source(patch.MHC), patch.MHC)).hexdigest()


def test_installer_rejects_parent_lock_even_with_profile_added():
    profile = 'glm53-flash-nvfp4-spark-tp2-dcp2'
    parent = copy.deepcopy(image.PARENT)
    parent['profiles'] = sorted({*parent['profiles'], profile})
    with pytest.raises(ValueError, match='source-pinned'):
        image_lock.for_profile(profile, parent)
    parent['sircl']['vllm_pins'] = [patch.BUILD]
    assert image_lock.for_profile(profile, parent)['sircl']['vllm_pins'] == [patch.BUILD]


def test_mhc_serving_plan_uses_full_ceiling_for_owner_rows():
    # Unmodified _plan_for excerpt, from the pinned image's
    # vllm/models/deepseek_v4/nvidia/b12x.py:206. This is the production
    # selection rule used by the GPU comparison, not a synthetic mirror.
    scope = {'Plan': object}
    exec((FIXTURES/'mhc_plan_selection.py').read_text(), scope)
    for ceiling in (4096, 8192):
        full = object()
        plans = {('post_pre', size): object() for size in (1, 2, 4, 8, 12, 16, 20, 24, 28, 32)}
        plans['post_pre', ceiling] = full
        owner = NS(_plans=plans)
        assert scope['_plan_for'](owner, 'post_pre', ceiling//2) is full
        assert scope['_plan_for'](owner, 'post_pre', ceiling) is full


def test_real_layer_receipt_rewrite_and_lock_admission(tmp_path, monkeypatch):
    # Synthetic parent receipts; source bytes are exact image fixtures.
    files = {derived_layer.SITE+path: source(path) for path in (patch.KDA, patch.MHC, patch.PINS, patch.HOOKS)}
    sircl = copy.deepcopy(image.PARENT['sircl'])
    receipt = {k: copy.deepcopy(sircl[k]) for k in ('version', 'abi_version', 'wheel', 'native', 'p2p', 'tuning_key')}
    receipt.update(schema='sparkring-sircl-layer/v1', files={derived_layer.SITE+p: derived_layer.digest(source(p))
                                                          for p in (patch.PINS, patch.HOOKS)})
    raw = derived_layer.canonical_json(receipt)
    files[image_lock.LAYER_RECEIPT] = raw
    parent = copy.deepcopy(image.PARENT)
    parent['sircl']['receipt']['sha256'] = derived_layer.digest(raw)
    monkeypatch.setattr(image, 'PARENT', parent)
    base = dict(files={p: derived_layer.digest(b) for p, b in files.items()},
                capabilities={'runtime_status': {'version': parent['status_version']}})
    for kind in image_lock.LIBRARY_STEMS:
        base['files'][sircl[kind]['path']] = sircl[kind]['sha256']
    files[derived_layer.BASE_RECEIPT] = derived_layer.canonical_json(base)
    parent['parent_receipt_sha256'] = derived_layer.digest(files[derived_layer.BASE_RECEIPT])
    files[derived_layer.TOOLCHAIN_RECEIPT] = derived_layer.canonical_json(dict(parent_receipt_sha256=parent['parent_receipt_sha256']))
    parent['toolchain_receipt_sha256'] = derived_layer.digest(files[derived_layer.TOOLCHAIN_RECEIPT])
    context = tmp_path/'context'
    derived_layer.prepare_layer(image.LAYER, parent, files.__getitem__, context)
    plan = json.loads((context/'plan.json').read_text())
    assert len(plan['replaced']) == 5
    assert plan['lock_fields']['sircl']['vllm_pins'] == [patch.BUILD]
    assert plan['lock_fields']['sircl']['native'] == parent['sircl']['native']
    out = context/'files'
    base_out = json.loads((out/derived_layer.BASE_RECEIPT.lstrip('/')).read_text())
    layer_out = json.loads((out/image_lock.LAYER_RECEIPT.lstrip('/')).read_text())
    target = derived_layer.SITE+patch.PINS
    assert base_out['files'][target] == layer_out['files'][target] != receipt['files'][target]
    derived = derived_layer.derived_lock(plan, dict(Id='sha256:'+'9'*64, Size=1000), 'dev-dcp2-prefill',
                                                profiles=['glm53-flash-nvfp4-spark-tp2-dcp2'])
    assert image_lock.checkpoint_problem(derived, {'model_repository': 'local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD',
                                                 'model_revision': 'dec48abd33efa73c3bb7c95b74eee10cad34f9be',
                                                 'target_variant': None}) is None
    def read_container(command, **kwargs):
        return NS(stdout=(out/command[-1].lstrip('/')).read_bytes())
    assert transport.check_layer(derived['image_id'], derived['parent_receipt_sha256'], derived['sircl'],
                                 run=read_container)['files_verified'] == 2
    # The layer refuses a different source parent before it can rewrite receipts.
    files[derived_layer.BASE_RECEIPT] += b'\n'
    with pytest.raises(ValueError, match='pinned parent receipt'):
        image.replace(files.__getitem__, base)
