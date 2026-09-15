"""Validate serving activation receipts against their retained profile contract."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VERIFIER = ROOT / 'runtime/sparkring/jovian-r33/profiles/verify_profile.py'


def validate(document: dict) -> dict:
    """Check rank and cache declarations, then apply the source-bound verifier."""
    if not isinstance(document, dict) or document.get('schema') != 'sparkring-r33-activation-receipt/v1':
        raise ValueError('Expected a sparkring-r33-activation-receipt/v1 object')
    spec = importlib.util.spec_from_file_location('retained_activation_verifier', VERIFIER)
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    _, selected = verifier.profile(document.get('profile'))
    ranks = document.get('ranks')
    if (not isinstance(ranks, list) or len(ranks) != selected['node_count']
            or any(not isinstance(rank, dict) or type(rank.get('rank')) is not int for rank in ranks)
            or {rank['rank'] for rank in ranks} != set(range(selected['node_count']))):
        raise ValueError('Activation receipt must contain every integer rank exactly once')
    cache = document.get('sparkcache')
    if not isinstance(cache, dict) or cache.get('enabled') is not selected['sparkcache']:
        raise ValueError('Activation receipt SparkCache state differs from the selected profile')
    return verifier.validate_activation(document)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = validate(json.loads(args.receipt.read_text(encoding='utf-8')))
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
