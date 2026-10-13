"""Exact source transformations for TP2/DCP2 prefill admission.

The parallel-size predicates, diagnostics and SIRCL source identities change.
Checkpoint planning, allocation, metadata and arithmetic kernels stay intact.
"""
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / 'runtime/images/compositions/dcp2-prefill/sources.json'
KDA = 'vllm/v1/core/recurrent_prefill_checkpoint.py'
MHC = 'vllm/models/glm5next/nvidia/mhc_prefill_sharding.py'
PINS = 'sparkring_sircl/vllm/pins.py'
HOOKS = 'sparkring_sircl/vllm/hooks.py'
BUILD = 'sparkring-kraken-bc9ea774-tp2-dcp2-prefill'
SWAPS = (
    ('parallel.tensor_parallel_size == 2\n                and parallel.decode_context_parallel_size == 1',
     'parallel.tensor_parallel_size == 2\n                and parallel.decode_context_parallel_size in (1, 2)'),
    ('TP4 with DCP1/2/4 or TP2 with DCP1, PP1/DP1, B12X KDA, ',
     'TP4 with DCP1/2/4 or TP2 with DCP1/2, PP1/DP1, B12X KDA, '),
)


def transform(data, path=KDA):
    if path not in (KDA, MHC, PINS, HOOKS):
        raise ValueError('Unknown prefill source path: '+path)
    text = data.decode('utf-8')
    if path == HOOKS:
        before = 'builds=("lil-image-aba309e4610c", "sparkring-kraken-beta-20261007-bc9ea774"),'
        if text.count(before) != 2:
            raise ValueError('Expected the mHC and Qwen hook build lists')
        return text.replace(before, before[:-2] + f', "{BUILD}"),').encode('utf-8')
    if path == MHC:
        before = 'not in ((2, 1), (4, 1), (4, 2), (4, 4))'
        if text.count(before) != 2:
            raise ValueError('Expected both mHC configuration and forward admission gates')
        return text.replace(before, 'not in ((2, 1), (2, 2), (4, 1), (4, 2), (4, 4))').replace(
            'TP2/DCP1 or TP4/DCP1,2,4', 'TP2/DCP1,2 or TP4/DCP1,2,4').encode('utf-8')
    swaps = SWAPS if path == KDA else pins_swaps()
    for before, after in swaps:
        if text.count(before) != 1:
            raise ValueError('Prefill source does not match the pinned substitution')
        text = text.replace(before, after)
    return text.encode('utf-8')


def pins_swaps():
    rows = {row['path']: row for row in json.loads(MANIFEST.read_text(encoding='utf-8'))['files']}
    admission = 'MHC_ADMITS: dict[str, tuple[tuple[int, int], ...]] = {}'
    addition = f'''{admission}

# Source-patched CSF build: only parallel-size admission changes. The KDA
# file is also pinned so this identity requires both prefill admissions.
_DCP2_PARENT = next(build for build in SUPPORTED
                    if build.name == "sparkring-kraken-beta-20261007-bc9ea774")
SUPPORTED += (dataclasses.replace(
    _DCP2_PARENT, name="{BUILD}",
    source="CSF sources with SparkRing dcp2-prefill composition",
    files={{**_DCP2_PARENT.files,
           "{MHC.removeprefix('vllm/')}": "{rows[MHC]['sha256']}",
           "{KDA.removeprefix('vllm/')}": "{rows[KDA]['sha256']}"}},
),)
MHC_ADMITS["{BUILD}"] = ((2, 1), (2, 2), (4, 1), (4, 2), (4, 4))'''
    return ((admission, addition), ('    actual = hashes(root)\n',
        '    actual = hashes(root, tuple(dict.fromkeys(\n'
        '        name for build in SUPPORTED for name in build.files)))\n'))


def apply(data, path=KDA):
    row = next(row for row in json.loads(MANIFEST.read_text(encoding='utf-8'))['files'] if row['path'] == path)
    if sha256(data).hexdigest() != row['inherited_sha256']:
        raise ValueError('Prefill source hash differs from the composition')
    result = transform(data, path)
    if sha256(result).hexdigest() != row['sha256']:
        raise ValueError('Prefill result hash differs from the composition')
    compile(result, path, 'exec')
    return result
