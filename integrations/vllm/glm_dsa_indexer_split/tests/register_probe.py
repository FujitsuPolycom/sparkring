"""One probe process for ``test_register.py``: register, import, report.

Prints one JSON line with the plugin's registration state and whether
``B12xSparseIndexer.forward`` is this plugin's wrapper. Any refusal exits
non-zero with its message.
"""

import split_image_env  # noqa: F401  (must precede every vllm import)
import json
import os
import sys

import glm_dsa_indexer_split


def main() -> int:
    if os.environ.get("PROBE_IMPORT_FIRST"):
        import vllm.v1.attention.backends.mla.b12x_indexer as bi  # noqa: F401
    glm_dsa_indexer_split.register()
    import vllm.v1.attention.backends.mla.b12x_indexer as bi

    forward = vars(bi.B12xSparseIndexer)["forward"]
    print(json.dumps({
        "registered": glm_dsa_indexer_split.status()["settings"] is not None,
        "installed": sorted(glm_dsa_indexer_split.status()["installed"]),
        "pending": glm_dsa_indexer_split.status()["pending_modules"],
        "forward_is_wrapper": getattr(forward, glm_dsa_indexer_split.MARKER, None)
        == "B12xSparseIndexer.forward",
    }))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except glm_dsa_indexer_split.PatchRefused as error:
        print(f"PatchRefused: {error}", file=sys.stderr)
        sys.exit(1)
