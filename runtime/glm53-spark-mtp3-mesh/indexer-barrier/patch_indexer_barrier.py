"""Apply histogram publication ordering to the pinned mesh indexer source."""
import hashlib
from pathlib import Path

RELATIVE = 'b12x/attention/dsa_indexer/fused_indexer.py'
BEFORE = '893fbcade135b7e1d146b8fb6530cde0650be515f69bf9a17ced0a9c61a141e2'
AFTER = '35a7564ec8bf18f1b7dc5562b21616437b65e7b0689c56341b6436c8d202637b'


def patch_bytes(source: bytes) -> bytes:
    digest = hashlib.sha256(source).hexdigest()
    if digest == AFTER:
        return source
    if digest != BEFORE:
        raise ValueError(f'Unsupported mesh indexer source SHA-256: {digest}')
    arrival = b'    arrival_ptr = _fused_state_ptr(state, group_id, Int32(_FUSED_STATE_ARRIVAL))'
    barrier = (b'    # Publish only after every warp has finished its histogram writes.\r\n'
               b'    cute.arch.sync_threads()\r\n' + arrival)
    cache = b'"attention.indexer.fused_indexer", 1, cache_key, labels=labels'
    if source.count(arrival) != 1 or source.count(cache) != 1:
        raise ValueError('Mesh indexer barrier or compile-revision anchor differs')
    result = source.replace(arrival, barrier, 1).replace(
        cache, b'"attention.indexer.fused_indexer", 2, cache_key, labels=labels', 1)
    if hashlib.sha256(result).hexdigest() != AFTER:
        raise ValueError('Mesh indexer transform produced unexpected source bytes')
    return result


def apply(path: Path) -> None:
    result = patch_bytes(path.read_bytes())
    path.write_bytes(result)
