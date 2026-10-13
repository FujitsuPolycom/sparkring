"""Derive a CSF image with TP2/DCP2 KDA coalescing and mHC row ownership.

Uses the shared prepare/build/record lifecycle and exact preimage/result pins.
KDA and mHC arithmetic are unchanged. No action publishes an image.
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from integrations.vllm.dcp2_prefill import patch
from runtime.images.derived_layer import SITE, Layer, main, canonical_json, digest
from runtime.common import image_lock

RECORD = json.loads(patch.MANIFEST.read_text(encoding='utf-8'))
PARENT = json.loads((patch.ROOT / 'runtime/releases' / RECORD['parent_release'] / 'installer-image.json').read_text())


def replace(read, receipt):
    from runtime.images.derived_layer import BASE_RECEIPT
    if digest(read(BASE_RECEIPT)) != PARENT['parent_receipt_sha256']:
        raise ValueError('The dcp2-prefill layer requires its pinned parent receipt')
    written = {SITE + row['path']: patch.apply(read(SITE + row['path']), row['path']) for row in RECORD['files']}
    # The wheel remains the source artifact. Record the source overlay and
    # its resulting installed bytes separately instead of claiming a rebuilt wheel.
    raw = read(image_lock.LAYER_RECEIPT)
    if digest(raw) != PARENT['sircl']['receipt']['sha256']:
        raise ValueError('SIRCL receipt differs from the pinned parent composition')
    sircl = json.loads(raw)
    changed = {SITE+path: digest(written[SITE+path]) for path in (patch.PINS, patch.HOOKS)}
    for target, sha in changed.items():
        if sircl['files'].get(target) != digest(read(target)):
            raise ValueError('SIRCL receipt does not cover the parent source: '+target)
        sircl['files'][target] = sha
    sircl['source_overlays'] = [dict(name='dcp2-prefill', inherited_receipt_sha256=digest(raw),
                                   files=changed)]
    written[image_lock.LAYER_RECEIPT] = canonical_json(sircl)
    return written


def update_receipt(receipt, replaced):
    sircl = dict(PARENT['sircl'], vllm_pins=[patch.BUILD],
                 receipt=dict(path=image_lock.LAYER_RECEIPT, sha256=digest(replaced[image_lock.LAYER_RECEIPT])))
    return {'sircl': sircl}


LAYER = Layer(
    name='dcp2-prefill',
    purpose='Admit KDA coalescing and mHC row ownership at TP2/DCP2 under their existing switches',
    replace=replace,
    update_receipt=update_receipt,
    provenance='/opt/sparkring/receipts/derived-dcp2-prefill.json',
    pins={SITE + row['path']: (row['inherited_sha256'], row['sha256']) for row in RECORD['files']},
)

if __name__ == '__main__':
    main(LAYER)
